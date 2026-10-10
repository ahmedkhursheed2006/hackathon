"""Camera acquisition + MediaPipe FaceMesh runtime for the Vision_Engine (Req 1).

This is the I/O layer that sits *on top of* the pure, testable vision core in
:mod:`medusa.vision.core` (``order_faces`` / ``proximity_fraction`` /
``compute_gaze`` / ``gaze_within_tolerance`` / ``classify_frame``). It owns the
two heavy, native-wheel dependencies -- OpenCV (``cv2.VideoCapture``) for frame
capture and MediaPipe FaceMesh for landmark extraction -- and turns raw frames
into the :class:`~medusa.core.models.FrameMeta` the pure core consumes.

Responsibilities (Requirement 1)
--------------------------------
* **Acquire within budget (Req 1.1):** open the configured device index via
  OpenCV within a 5s budget and target >=10 fps.
* **Never transmit continuous video (Req 1.2):** capture is purely local. Nothing
  here opens a socket or sends frames anywhere. A single raw JPEG frame can be
  grabbed on demand (:meth:`FaceMeshExtractor.last_jpeg` / the capture's
  :meth:`CameraCapture.read_jpeg`) for the Alert_Service's *single* triggered
  frame, but there is no streaming path.
* **Acquisition failure keeps the system running (Req 1.3):** on failure a
  camera-access error is surfaced/logged, the configuration is retained, and the
  engine keeps running without frames.
* **Distinct permission message (Req 1.4):** when the OS denies camera access the
  error carries a distinct "grant camera permission" message, separate from a
  generic acquisition failure.
* **Bounded retry (Req 1.5):** while not capturing after a prior failure,
  re-acquisition is attempted at intervals not exceeding 10s.
* **Fast resume (Req 1.6):** a successful re-acquire resumes capture immediately,
  well within the 2s budget, at the configured target fps.

Design for a wheel-less interpreter
------------------------------------
``cv2`` and ``mediapipe`` have **no wheels** on some interpreters (e.g. Python
3.14 here), so:

* **Nothing at import time touches ``cv2``/``mediapipe``.** ``import
  medusa.vision.camera`` always succeeds. The default backends import the native
  libraries *lazily*, only when a real device is actually opened.
* **Dependency injection** is the primary seam: :class:`CameraCapture` takes a
  ``capture_factory`` and :class:`FaceMeshExtractor` takes a ``detector`` /
  ``detector_factory``. Tests (task 12.3) inject fakes implementing the
  :class:`CaptureBackend` / :class:`FaceMeshDetector` protocols and never need
  ``cv2``/``mediapipe`` installed.
* **The FrameMeta-building logic is pure** (:func:`landmarks_to_bounding_box`,
  :func:`face_from_landmarks`, :func:`build_frame_meta`): it depends only on the
  core models and plain landmark sequences, so it is unit-/property-testable
  without any native dependency.

:func:`default_capture_factory` and :func:`default_facemesh_detector_factory`
provide the real OpenCV / MediaPipe backends via guarded lazy imports; if the
wheels are unavailable they raise :class:`DependencyUnavailableError` carrying
the exact import error rather than failing at module import.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from medusa.core.models import BoundingBox, Face, FrameMeta

__all__ = [
    # Errors / outcomes
    "CameraError",
    "CameraPermissionError",
    "DependencyUnavailableError",
    "AcquireResult",
    "CaptureState",
    # Protocols / injected types
    "CaptureBackend",
    "CaptureFactory",
    "FaceMeshDetector",
    "FaceMeshDetectorFactory",
    "NowFn",
    "SleepFn",
    # Pure FrameMeta helpers
    "landmarks_to_bounding_box",
    "face_from_landmarks",
    "build_frame_meta",
    # Runtime classes
    "CameraCapture",
    "FaceMeshExtractor",
    # Defaults / constants
    "ACQUIRE_BUDGET_S",
    "RETRY_INTERVAL_S",
    "TARGET_FPS",
    "default_capture_factory",
    "default_facemesh_detector_factory",
]

_log = logging.getLogger(__name__)

# --- Requirement-driven timing constants -------------------------------------
#: Max time allowed to acquire the device at startup (Req 1.1 / 1.3).
ACQUIRE_BUDGET_S = 5.0
#: Upper bound on the interval between re-acquisition attempts (Req 1.5).
RETRY_INTERVAL_S = 10.0
#: Target capture rate (Req 1.1 / 1.6).
TARGET_FPS = 10


# A monotonic clock and a sleep function, injectable so the retry/acquire state
# machine is fully deterministic under test (no real wall-clock waiting).
NowFn = Callable[[], float]
SleepFn = Callable[[float], None]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CameraError(RuntimeError):
    """A camera-access failure surfaced to the user (Req 1.3).

    The default message identifies the device that could not be acquired. This
    is the *generic* acquisition failure; OS permission denial is signalled by
    the :class:`CameraPermissionError` subclass with a distinct message.
    """

    def __init__(self, device_index: int, message: Optional[str] = None) -> None:
        self.device_index = device_index
        super().__init__(
            message
            or f"Unable to access camera device {device_index}. "
            f"Check that a webcam is connected and not in use by another "
            f"application."
        )


class CameraPermissionError(CameraError):
    """OS denied camera access -- distinct "grant camera permission" message (Req 1.4).

    Raised/surfaced when the acquisition failure is attributable to an operating
    system permission denial rather than a missing/busy device, so the UI can
    prompt the user to grant permission specifically.
    """

    def __init__(self, device_index: int, message: Optional[str] = None) -> None:
        super().__init__(
            device_index,
            message
            or "Camera permission is required. Please grant camera permission "
            "to this application in your operating system privacy settings.",
        )


class DependencyUnavailableError(RuntimeError):
    """A required native dependency (``cv2``/``mediapipe``) is not installed.

    Carries the original :class:`ImportError` so the exact missing-wheel error
    can be surfaced. Only raised by the *default* backends when a real device is
    opened; injected fakes never trigger it, and importing this module never does.
    """

    def __init__(self, dependency: str, cause: BaseException) -> None:
        self.dependency = dependency
        self.__cause__ = cause
        super().__init__(
            f"The '{dependency}' library is required for real camera/vision "
            f"capture but is not available: {cause!r}"
        )


class AcquireResult(Enum):
    """The outcome of a single acquisition attempt."""

    ACQUIRED = "acquired"
    FAILED = "failed"
    PERMISSION_DENIED = "permission_denied"


class CaptureState(Enum):
    """The lifecycle state of the :class:`CameraCapture` state machine."""

    IDLE = "idle"  # never started / stopped
    CAPTURING = "capturing"  # device acquired and open
    WAITING_RETRY = "waiting_retry"  # prior failure; retrying at <=10s intervals


# ---------------------------------------------------------------------------
# Injected collaborator protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class CaptureBackend(Protocol):
    """An opened capture device (wraps ``cv2.VideoCapture`` in production).

    Only the small surface the engine needs is modelled, so a fake backend is
    trivial to write for tests without ``cv2``.
    """

    def is_opened(self) -> bool:
        """Return whether the underlying device is open and usable."""
        ...

    def read(self) -> tuple[bool, object]:
        """Return ``(ok, frame)``. ``ok`` is ``False`` when no frame is available.

        ``frame`` is an opaque image object (a ``numpy`` array for OpenCV); the
        engine passes it straight to the :class:`FaceMeshDetector` and never
        inspects it, so tests may use any sentinel.
        """
        ...

    def release(self) -> None:
        """Release the device and free its resources."""
        ...


# A capture factory opens the configured device index and returns a
# :class:`CaptureBackend`, or ``None`` / a backend whose ``is_opened()`` is
# ``False`` when the device cannot be opened. It may raise
# :class:`CameraPermissionError` to signal an OS permission denial (Req 1.4).
CaptureFactory = Callable[[int], Optional[CaptureBackend]]


@runtime_checkable
class FaceMeshDetector(Protocol):
    """A FaceMesh landmark detector (wraps MediaPipe FaceMesh in production)."""

    def detect(self, frame: object) -> Sequence[Sequence[Sequence[float]]]:
        """Return per-face landmark lists for ``frame``.

        The return shape is ``faces -> landmarks -> (x, y[, z])`` where ``x``/``y``
        are normalized to ``[0, 1]`` (MediaPipe convention) or already in pixels;
        :func:`face_from_landmarks` accepts either. An empty sequence means no
        face was detected.
        """
        ...

    def close(self) -> None:
        """Release detector resources (optional for fakes)."""
        ...


# A detector factory constructs a :class:`FaceMeshDetector` lazily (so the heavy
# MediaPipe import only happens when a real detector is actually needed).
FaceMeshDetectorFactory = Callable[[], FaceMeshDetector]


# ---------------------------------------------------------------------------
# Pure FrameMeta-building helpers (NO cv2 / mediapipe dependency)
# ---------------------------------------------------------------------------
def _landmark_component(landmark: object, index: int) -> Optional[float]:
    """Best-effort read of component ``index`` (0=x, 1=y) from a landmark.

    Accepts both attribute-style MediaPipe landmarks (``.x``/``.y``) and plain
    ``(x, y[, z])`` sequences. Returns ``None`` if it cannot be interpreted.
    """
    attr = ("x", "y", "z")[index] if index < 3 else None
    if attr is not None:
        value = getattr(landmark, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    try:
        return float(landmark[index])  # type: ignore[index]
    except (TypeError, IndexError, ValueError, KeyError):
        return None


def landmarks_to_bounding_box(
    landmarks: Sequence[object], frame_w: int, frame_h: int
) -> Optional[BoundingBox]:
    """Derive a pixel-space :class:`BoundingBox` from a face's landmark extents.

    The box is the axis-aligned extent (min/max x and y) of every usable
    landmark. Landmarks whose x/y lie within the unit interval are treated as
    MediaPipe-normalized and scaled by the frame dimensions; values outside
    ``[0, 1]`` are assumed to already be pixels. Pure and dependency-free.

    Returns ``None`` when there are no landmarks, none are interpretable, or the
    frame has non-positive area -- in those cases there is no meaningful box to
    build. A single-point / zero-extent face yields a zero-area box (``w``/``h``
    of ``0``), which the core's :func:`~medusa.vision.core.proximity_fraction`
    correctly treats as a degenerate box (Req 3.4).

    Args:
        landmarks: The face's landmark sequence (attribute- or tuple-style).
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        A pixel-space :class:`BoundingBox`, or ``None`` if one cannot be built.
    """
    if frame_w <= 0 or frame_h <= 0 or not landmarks:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for landmark in landmarks:
        x = _landmark_component(landmark, 0)
        y = _landmark_component(landmark, 1)
        if x is None or y is None:
            continue
        # Normalized -> pixels; out-of-unit-range values assumed already pixels.
        if 0.0 <= x <= 1.0:
            x *= frame_w
        if 0.0 <= y <= 1.0:
            y *= frame_h
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return BoundingBox(x=min_x, y=min_y, w=max_x - min_x, h=max_y - min_y)


def face_from_landmarks(
    landmarks: Sequence[object], frame_w: int, frame_h: int
) -> Optional[Face]:
    """Build a :class:`Face` (box + immutable landmark snapshot) from landmarks.

    The bounding box is computed via :func:`landmarks_to_bounding_box`; the raw
    landmarks are stored as an immutable tuple on the :class:`Face` so the pure
    core's gaze solver (:func:`~medusa.vision.core.compute_gaze`) can index into
    them later. Returns ``None`` when no box can be derived (so a junk face is
    dropped rather than fed forward). Pure and dependency-free.
    """
    box = landmarks_to_bounding_box(landmarks, frame_w, frame_h)
    if box is None:
        return None
    return Face(box=box, landmarks=tuple(landmarks))


def build_frame_meta(
    faces_landmarks: Sequence[Sequence[object]],
    frame_w: int,
    frame_h: int,
    timestamp: float,
) -> FrameMeta:
    """Assemble a :class:`FrameMeta` from per-face landmark lists.

    Each face's landmarks are converted to a :class:`Face` via
    :func:`face_from_landmarks`; faces that yield no usable box are skipped. The
    resulting faces are the *pre-ordering* input the pure core expects (ordering
    is applied by :func:`~medusa.vision.core.order_faces` inside
    :func:`~medusa.vision.core.classify_frame`). Pure and dependency-free, so the
    landmarks -> FrameMeta mapping is fully testable without a camera.

    Args:
        faces_landmarks: Sequence of per-face landmark sequences (the detector
            output). May be empty (no faces detected).
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        timestamp: The capture timestamp carried onto the :class:`FrameMeta`.

    Returns:
        A :class:`FrameMeta` with ``width``/``height``, the derived ``faces``
        tuple (possibly empty), and ``timestamp``.
    """
    faces: list[Face] = []
    for landmarks in faces_landmarks:
        face = face_from_landmarks(landmarks, frame_w, frame_h)
        if face is not None:
            faces.append(face)
    return FrameMeta(
        width=int(frame_w),
        height=int(frame_h),
        faces=tuple(faces),
        timestamp=float(timestamp),
    )


# ---------------------------------------------------------------------------
# CameraCapture: the acquire / retry state machine (Req 1.1, 1.3, 1.5, 1.6)
# ---------------------------------------------------------------------------
@dataclass
class _AcquireAttempt:
    """Internal record of one acquisition attempt's result."""

    result: AcquireResult
    error: Optional[CameraError] = None


class CameraCapture:
    """Acquire the configured webcam and manage the retry/resume lifecycle.

    This class owns *only* the device lifecycle -- opening the configured index,
    reading frames, and the failure/retry state machine. It performs no face
    detection (that is :class:`FaceMeshExtractor`) and, critically, never
    transmits frames anywhere (Req 1.2); the only egress-adjacent method is
    :meth:`read_jpeg`, which produces a *single* raw frame on demand for the
    Alert_Service and does not stream.

    The capture backend is created through an injected ``capture_factory`` so the
    whole state machine can be exercised with a fake that implements
    :class:`CaptureBackend` -- no ``cv2`` required (the default factory lazily
    imports OpenCV only when a real device is opened).

    State machine (Req 1.3 / 1.5 / 1.6)
    -----------------------------------
    * :meth:`acquire` tries to open the device within :data:`ACQUIRE_BUDGET_S`
      (Req 1.1). Success -> :attr:`CaptureState.CAPTURING`. Failure -> a
      camera-access error is surfaced (``on_error``), the config is retained, and
      the state becomes :attr:`CaptureState.WAITING_RETRY` (Req 1.3). A permission
      denial surfaces the distinct :class:`CameraPermissionError` (Req 1.4).
    * :meth:`maybe_retry` performs the next acquisition attempt only once at least
      :data:`RETRY_INTERVAL_S` has elapsed since the last attempt (Req 1.5), so
      callers can poll it on the vision loop without exceeding the 10s bound.
    * A successful (re)acquire sets ``CAPTURING`` immediately, so capture resumes
      well within the 2s budget (Req 1.6).

    Parameters
    ----------
    device_index:
        The configured OpenCV device index to open.
    capture_factory:
        Opens ``device_index`` and returns a :class:`CaptureBackend` (or ``None``
        / a not-opened backend on failure). May raise
        :class:`CameraPermissionError` for an OS permission denial. Defaults to
        :func:`default_capture_factory` (lazy OpenCV).
    on_error:
        Optional callback invoked with each :class:`CameraError` when an
        acquisition attempt fails, so the application can surface it to the user
        (Req 1.3 / 1.4). Errors are always logged regardless.
    target_fps:
        The target capture rate used to derive the inter-frame interval (Req 1.1).
    acquire_budget_s / retry_interval_s:
        The acquisition budget and the max retry interval (defaults match the
        requirement bounds). Overridable for tests.
    now / sleep:
        Injected monotonic clock and sleep so acquisition timing is deterministic
        under test. ``acquire`` performs *no* real sleeping by default; it relies
        on the factory to either succeed or fail promptly.
    """

    def __init__(
        self,
        device_index: int,
        capture_factory: Optional[CaptureFactory] = None,
        on_error: Optional[Callable[[CameraError], None]] = None,
        target_fps: int = TARGET_FPS,
        acquire_budget_s: float = ACQUIRE_BUDGET_S,
        retry_interval_s: float = RETRY_INTERVAL_S,
        now: NowFn = time.monotonic,
        sleep: SleepFn = time.sleep,
    ) -> None:
        self._device_index = int(device_index)
        self._factory: CaptureFactory = capture_factory or default_capture_factory
        self._on_error = on_error
        self._target_fps = max(1, int(target_fps))
        self._acquire_budget_s = float(acquire_budget_s)
        self._retry_interval_s = float(retry_interval_s)
        self._now = now
        self._sleep = sleep

        self._backend: Optional[CaptureBackend] = None
        self._state = CaptureState.IDLE
        # Timestamp of the last acquisition attempt (for the retry interval).
        self._last_attempt_at: Optional[float] = None
        self._last_error: Optional[CameraError] = None

    # ------------------------------------------------------------------ #
    # Observable state
    # ------------------------------------------------------------------ #
    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def is_capturing(self) -> bool:
        """``True`` iff a device is currently acquired and open (Req 1.1/1.6)."""
        return self._state == CaptureState.CAPTURING and self._backend is not None

    @property
    def frame_interval_s(self) -> float:
        """Target inter-frame interval in seconds (``1 / target_fps``)."""
        return 1.0 / float(self._target_fps)

    @property
    def last_error(self) -> Optional[CameraError]:
        """The most recent acquisition error, if any (retained per Req 1.3)."""
        return self._last_error

    # ------------------------------------------------------------------ #
    # Acquisition / retry state machine
    # ------------------------------------------------------------------ #
    def acquire(self) -> bool:
        """Attempt to acquire the configured device within the budget (Req 1.1).

        Returns ``True`` and transitions to :attr:`CaptureState.CAPTURING` on
        success. On failure the camera-access error is surfaced (``on_error`` +
        log), the config is retained, the state becomes
        :attr:`CaptureState.WAITING_RETRY`, and ``False`` is returned (Req 1.3 /
        1.4). Any already-open backend is released first so re-acquisition starts
        clean.
        """
        self._release_backend()
        attempt = self._attempt_open()
        self._last_attempt_at = self._now()

        if attempt.result is AcquireResult.ACQUIRED:
            self._state = CaptureState.CAPTURING
            self._last_error = None
            _log.info(
                "Camera device %d acquired; capturing at target %d fps.",
                self._device_index,
                self._target_fps,
            )
            return True

        # Failure path: retain config, keep running without frames, schedule
        # retry (Req 1.3 / 1.5). Permission denial carries a distinct message
        # (Req 1.4).
        self._state = CaptureState.WAITING_RETRY
        error = attempt.error or CameraError(self._device_index)
        self._last_error = error
        _log.error("Camera acquisition failed: %s", error)
        if self._on_error is not None:
            self._on_error(error)
        return False

    def maybe_retry(self) -> bool:
        """Re-attempt acquisition if the retry interval has elapsed (Req 1.5/1.6).

        Intended to be polled by the vision loop while not capturing. It only
        performs an actual attempt once at least :attr:`retry_interval_s` has
        passed since the previous attempt, guaranteeing re-acquisition is tried
        at intervals *not exceeding* 10s without busy-looping. Returns ``True``
        when a retry succeeded (capture resumed), ``False`` otherwise.

        Calling this while already capturing is a no-op returning ``False``.
        """
        if self._state == CaptureState.CAPTURING:
            return False

        now = self._now()
        if self._last_attempt_at is not None:
            elapsed = now - self._last_attempt_at
            if elapsed < self._retry_interval_s:
                # Not yet time for the next attempt.
                return False
        return self.acquire()

    def seconds_until_retry(self) -> float:
        """Seconds remaining before the next retry attempt is due (>= 0).

        ``0.0`` means a retry may be attempted now. Useful for a caller that
        wants to sleep exactly the right amount instead of polling.
        """
        if self._state == CaptureState.CAPTURING or self._last_attempt_at is None:
            return 0.0
        remaining = self._retry_interval_s - (self._now() - self._last_attempt_at)
        return max(0.0, remaining)

    def read(self) -> Optional[object]:
        """Return the latest raw frame, or ``None`` if not capturing / read failed.

        A read failure (device unplugged mid-session) drops the backend and moves
        to :attr:`CaptureState.WAITING_RETRY` so the retry machinery re-acquires
        it (Req 1.5). The returned frame object is opaque and never transmitted
        anywhere (Req 1.2).
        """
        if not self.is_capturing or self._backend is None:
            return None
        try:
            ok, frame = self._backend.read()
        except Exception as exc:  # noqa: BLE001 - backend errors must not crash loop
            _log.warning("Camera read raised %s; dropping device for retry.",
                         type(exc).__name__)
            ok, frame = False, None
        if not ok or frame is None:
            # Lost the device: surface for retry without crashing the loop.
            self._release_backend()
            self._state = CaptureState.WAITING_RETRY
            self._last_attempt_at = self._now()
            self._last_error = CameraError(self._device_index)
            if self._on_error is not None:
                self._on_error(self._last_error)
            return None
        return frame

    def read_jpeg(self, frame: Optional[object] = None) -> Optional[bytes]:
        """Encode a *single* frame as JPEG bytes for the Alert_Service (Req 1.2).

        This is the only method that produces transmittable bytes, and it does so
        for exactly one frame on demand -- there is no streaming path, satisfying
        the "single triggered frame, never continuous video" requirement. If
        ``frame`` is omitted the latest frame is read first. Returns ``None`` when
        no frame is available or encoding is unavailable (e.g. ``cv2`` missing).
        """
        if frame is None:
            frame = self.read()
        if frame is None:
            return None
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - env-dependent
            _log.debug("JPEG encode unavailable (cv2 missing): %r", exc)
            return None
        try:
            ok, buffer = cv2.imencode(".jpg", frame)
        except Exception as exc:  # noqa: BLE001
            _log.warning("JPEG encode failed: %s", type(exc).__name__)
            return None
        if not ok:
            return None
        return bytes(buffer)

    def stop(self) -> None:
        """Release the device and return to :attr:`CaptureState.IDLE`."""
        self._release_backend()
        self._state = CaptureState.IDLE

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _attempt_open(self) -> _AcquireAttempt:
        """Open the device via the factory, classifying the outcome.

        A :class:`CameraPermissionError` raised by the factory maps to
        ``PERMISSION_DENIED`` (Req 1.4); any other exception, a ``None`` backend,
        or a backend reporting ``is_opened() == False`` maps to ``FAILED``
        (Req 1.3).
        """
        try:
            backend = self._factory(self._device_index)
        except CameraPermissionError as exc:
            return _AcquireAttempt(AcquireResult.PERMISSION_DENIED, exc)
        except CameraError as exc:
            return _AcquireAttempt(AcquireResult.FAILED, exc)
        except Exception as exc:  # noqa: BLE001 - unknown backend failure
            return _AcquireAttempt(
                AcquireResult.FAILED,
                CameraError(
                    self._device_index,
                    f"Unexpected error opening camera device "
                    f"{self._device_index}: {type(exc).__name__}.",
                ),
            )

        if backend is None or not backend.is_opened():
            if backend is not None:
                try:
                    backend.release()
                except Exception:  # noqa: BLE001
                    pass
            return _AcquireAttempt(
                AcquireResult.FAILED, CameraError(self._device_index)
            )

        self._backend = backend
        return _AcquireAttempt(AcquireResult.ACQUIRED)

    def _release_backend(self) -> None:
        if self._backend is not None:
            try:
                self._backend.release()
            except Exception:  # noqa: BLE001
                pass
            self._backend = None


# ---------------------------------------------------------------------------
# FaceMeshExtractor: capture frame -> FaceMesh landmarks -> FrameMeta
# ---------------------------------------------------------------------------
class FaceMeshExtractor:
    """Run MediaPipe FaceMesh on captured frames and build :class:`FrameMeta`.

    This ties the :class:`CameraCapture` (frames) to the pure core (which needs a
    :class:`FrameMeta`). It runs the injected :class:`FaceMeshDetector` on each
    frame and converts the per-face landmarks into :class:`Face` objects via the
    pure :func:`build_frame_meta` helper, deriving each face's bounding box from
    the landmark extents.

    The detector is injected (or lazily constructed through a
    ``detector_factory``) so this class is testable with a fake detector and no
    ``mediapipe``. The MediaPipe FaceMesh default is created only on first use.

    It also caches the most recent raw frame so the Alert_Service can grab a
    single JPEG on a threat (:meth:`last_jpeg`) without any streaming (Req 1.2).

    Parameters
    ----------
    capture:
        The :class:`CameraCapture` providing frames.
    detector:
        A ready :class:`FaceMeshDetector`, or ``None`` to construct one lazily
        from ``detector_factory`` on first :meth:`process` call.
    detector_factory:
        Builds the detector lazily; defaults to
        :func:`default_facemesh_detector_factory` (lazy MediaPipe import).
    now:
        Monotonic clock used for frame timestamps.
    """

    def __init__(
        self,
        capture: CameraCapture,
        detector: Optional[FaceMeshDetector] = None,
        detector_factory: Optional[FaceMeshDetectorFactory] = None,
        now: NowFn = time.monotonic,
    ) -> None:
        self._capture = capture
        self._detector = detector
        self._detector_factory = (
            detector_factory or default_facemesh_detector_factory
        )
        self._now = now
        self._last_frame: Optional[object] = None

    @property
    def capture(self) -> CameraCapture:
        return self._capture

    def _ensure_detector(self) -> FaceMeshDetector:
        if self._detector is None:
            self._detector = self._detector_factory()
        return self._detector

    def process(self, frame: Optional[object] = None) -> Optional[FrameMeta]:
        """Detect faces in a frame and build a :class:`FrameMeta` for the core.

        If ``frame`` is omitted, the latest frame is read from the capture. The
        frame's dimensions drive landmark de-normalization. Returns ``None`` when
        there is no frame to process (e.g. not capturing). A frame with no
        detected faces still yields a valid (empty-faces) :class:`FrameMeta`,
        which the core classifies Safe (Req 2.4).

        Detector exceptions are caught and treated as "no faces" so a transient
        FaceMesh error never crashes the vision loop; the frame still produces a
        valid empty :class:`FrameMeta`.
        """
        if frame is None:
            frame = self._capture.read()
        if frame is None:
            return None

        self._last_frame = frame
        width, height = _frame_dimensions(frame)

        try:
            faces_landmarks = self._ensure_detector().detect(frame)
        except DependencyUnavailableError:
            # Propagate the explicit missing-dependency signal to the caller so
            # the app can surface it; this is not a per-frame transient error.
            raise
        except Exception as exc:  # noqa: BLE001 - transient detector error
            _log.warning(
                "FaceMesh detection raised %s; treating frame as no-faces.",
                type(exc).__name__,
            )
            faces_landmarks = []

        return build_frame_meta(
            faces_landmarks, width, height, self._now()
        )

    def last_jpeg(self) -> Optional[bytes]:
        """Encode the most recently processed frame as a single JPEG (Req 1.2).

        Used by the Alert_Service to transmit exactly one triggered frame -- never
        a continuous stream. Returns ``None`` if no frame has been processed yet
        or encoding is unavailable.
        """
        if self._last_frame is None:
            return None
        return self._capture.read_jpeg(self._last_frame)

    def close(self) -> None:
        """Release the detector (and leave capture lifecycle to its owner)."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:  # noqa: BLE001
                pass
            self._detector = None


def _frame_dimensions(frame: object) -> tuple[int, int]:
    """Best-effort ``(width, height)`` for a frame object.

    Supports numpy-style ``frame.shape == (h, w, c)`` and an explicit
    ``(width, height)`` attribute pair on fakes. Falls back to ``(0, 0)`` which
    the pure core treats as a zero-area frame (Safe).
    """
    # Explicit width/height attributes (convenient for test fakes).
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    if width is not None and height is not None:
        try:
            return int(width), int(height)
        except (TypeError, ValueError):
            pass

    # numpy ndarray style: shape is (height, width[, channels]).
    shape = getattr(frame, "shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2:
        try:
            return int(shape[1]), int(shape[0])
        except (TypeError, ValueError):
            pass
    return (0, 0)


# ---------------------------------------------------------------------------
# Default (real) backends -- lazy native imports (Req 1: OpenCV + MediaPipe)
# ---------------------------------------------------------------------------
class _OpenCvCaptureBackend:
    """A :class:`CaptureBackend` wrapping a real ``cv2.VideoCapture``."""

    def __init__(self, video_capture: object) -> None:
        self._cap = video_capture

    def is_opened(self) -> bool:
        try:
            return bool(self._cap.isOpened())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return False

    def read(self) -> tuple[bool, object]:
        ok, frame = self._cap.read()  # type: ignore[attr-defined]
        return bool(ok), frame

    def release(self) -> None:
        try:
            self._cap.release()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def default_capture_factory(device_index: int) -> Optional[CaptureBackend]:
    """Open ``device_index`` with OpenCV (lazy import of ``cv2``).

    Raises :class:`DependencyUnavailableError` (carrying the original
    :class:`ImportError`) if ``cv2`` is not installed, so the exact missing-wheel
    error is reported rather than a bare import failure. Returns a backend whose
    ``is_opened()`` reflects whether the device actually opened; the caller
    (:class:`CameraCapture`) maps a not-opened backend to an acquisition failure
    (Req 1.3).
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise DependencyUnavailableError("cv2 (opencv-python)", exc) from exc
    video_capture = cv2.VideoCapture(device_index)
    return _OpenCvCaptureBackend(video_capture)


class _MediaPipeFaceMeshDetector:
    """A :class:`FaceMeshDetector` wrapping MediaPipe FaceMesh."""

    def __init__(self, face_mesh: object) -> None:
        self._face_mesh = face_mesh

    def detect(self, frame: object) -> Sequence[Sequence[Sequence[float]]]:
        try:
            import cv2  # type: ignore
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:  # noqa: BLE001 - fall back to the raw frame
            rgb = frame
        results = self._face_mesh.process(rgb)  # type: ignore[attr-defined]
        multi = getattr(results, "multi_face_landmarks", None)
        if not multi:
            return []
        faces: list[list[tuple[float, float, float]]] = []
        for face_landmarks in multi:
            points: list[tuple[float, float, float]] = []
            for lm in face_landmarks.landmark:
                points.append((float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))))
            faces.append(points)
        return faces

    def close(self) -> None:
        close = getattr(self._face_mesh, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


def default_facemesh_detector_factory() -> FaceMeshDetector:
    """Construct a real MediaPipe FaceMesh detector (lazy import).

    Raises :class:`DependencyUnavailableError` (carrying the original
    :class:`ImportError`) if ``mediapipe`` is not installed. FaceMesh is created
    with a modest ``max_num_faces`` since only the first two ordered faces are
    ever evaluated (Req 2.5).
    """
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise DependencyUnavailableError("mediapipe", exc) from exc
    face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore[attr-defined]
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return _MediaPipeFaceMeshDetector(face_mesh)
