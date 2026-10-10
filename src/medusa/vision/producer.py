"""VisionProducer: the single-producer capture + classify thread (task 12.2).

This module implements the **Vision thread (producer)** from the design's
Threading / Concurrency Model. It owns the real-time capture-and-classify loop
and is the *only* producer feeding the :class:`~medusa.core.bus.ClassificationBus`.

Responsibilities (Req 1.2, 11.3)
--------------------------------
* **Capture + classify loop.** While the camera is capturing, read a frame, run
  :class:`~medusa.vision.camera.FaceMeshExtractor.process` to build a
  :class:`~medusa.core.models.FrameMeta`, run the pure
  :func:`~medusa.vision.core.classify_frame` pipeline against the configured
  :class:`~medusa.core.models.DetectionConfig`, and publish the resulting
  :class:`~medusa.core.models.Classification` to the bus.
* **No I/O beyond the bus (Req 1.2, 11.3).** The loop performs *no* network I/O,
  never awaits the overlay, and never blocks on client sends. Publishing to the
  bus is a non-blocking, in-memory call; the Qt GUI thread and the asyncio I/O
  thread consume the bus's effective state off this thread. This keeps the
  vision loop as the system's real-time heartbeat (>=10 fps) with nothing able
  to stall capture.
* **Retry while down.** When the camera is not capturing (startup failure or a
  mid-session loss), the loop polls :meth:`~medusa.vision.camera.CameraCapture.maybe_retry`
  which only re-acquires once the <=10s retry interval has elapsed (Req 1.5/1.6).
  No classification is published while there is no frame.

Pacing / fps (Req 1.1/1.6)
--------------------------
When capturing, each iteration sleeps for the camera's ``frame_interval_s``
(``1 / target_fps``) so the loop targets >=10 fps without busy-spinning. When
waiting to retry, it sleeps for the shorter of ``seconds_until_retry`` and a
small responsiveness cap so :meth:`stop` stays snappy.

Determinism / testability
--------------------------
``now`` and ``sleep`` are injected so timing is deterministic under test, and a
single loop iteration can be driven directly via :meth:`run_once` without
starting a real thread or waiting on wall-clock time. The real thread entry
point (:meth:`_run`) simply calls :meth:`run_once` in a loop until the stop
event is set. This module imports only the already import-safe
``camera`` / ``core`` / ``bus`` modules, so it is importable without
``cv2``/``mediapipe``/``PySide6``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from medusa.core.bus import ClassificationBus
from medusa.core.models import Classification, DetectionConfig
from medusa.vision.camera import CameraCapture, FaceMeshExtractor, NowFn, SleepFn
from medusa.vision.core import classify_frame

__all__ = ["VisionProducer", "RETRY_POLL_CAP_S"]

_log = logging.getLogger(__name__)

#: Upper bound on how long the loop sleeps while waiting to retry acquisition.
#: Capping the wait (rather than sleeping the full ``seconds_until_retry``) keeps
#: :meth:`VisionProducer.stop` responsive even when the retry interval is long.
RETRY_POLL_CAP_S = 0.5


class VisionProducer:
    """Run the capture + classify loop and publish to the :class:`ClassificationBus`.

    The producer wires together an already-constructed :class:`CameraCapture`
    (device lifecycle + retry) and :class:`FaceMeshExtractor` (frame -> FrameMeta)
    with the pure :func:`classify_frame` pipeline, publishing one
    :class:`Classification` per captured frame to the injected bus. It never
    performs network I/O, awaits the overlay, or blocks on client sends -- it only
    publishes to the bus, which is a non-blocking in-memory hand-off (Req 1.2,
    11.3).

    Parameters
    ----------
    camera:
        The :class:`CameraCapture` owning the device lifecycle. The producer uses
        :attr:`~medusa.vision.camera.CameraCapture.is_capturing`,
        :meth:`~medusa.vision.camera.CameraCapture.maybe_retry`,
        :meth:`~medusa.vision.camera.CameraCapture.seconds_until_retry`, and
        :attr:`~medusa.vision.camera.CameraCapture.frame_interval_s`.
    extractor:
        The :class:`FaceMeshExtractor` that reads a frame from ``camera`` and
        builds a :class:`~medusa.core.models.FrameMeta`. Its
        :meth:`~medusa.vision.camera.FaceMeshExtractor.process` returns ``None``
        when there is no frame to process (e.g. not capturing).
    bus:
        The :class:`ClassificationBus` each :class:`Classification` is published
        to. ``bus.publish`` is the only outward call the loop makes.
    detection_config:
        The :class:`DetectionConfig` thresholds passed to
        :func:`classify_frame`.
    now:
        Injected monotonic clock (defaults to :func:`time.monotonic`).
    sleep:
        Injected sleep used to pace the loop (defaults to :func:`time.sleep`).
        Receiving a seam here keeps iteration timing deterministic under test.
    retry_poll_cap_s:
        Max sleep while waiting to retry acquisition (defaults to
        :data:`RETRY_POLL_CAP_S`), bounding :meth:`stop` latency.
    """

    def __init__(
        self,
        camera: CameraCapture,
        extractor: FaceMeshExtractor,
        bus: ClassificationBus,
        detection_config: DetectionConfig,
        now: NowFn = time.monotonic,
        sleep: SleepFn = time.sleep,
        retry_poll_cap_s: float = RETRY_POLL_CAP_S,
    ) -> None:
        self._camera = camera
        self._extractor = extractor
        self._bus = bus
        self._config = detection_config
        self._now = now
        self._sleep = sleep
        self._retry_poll_cap_s = float(retry_poll_cap_s)

        # Graceful-shutdown flag checked by the run loop (threading.Event so it
        # is safe to set from another thread).
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Latest published classification, exposed for debugging/wiring. Guarded
        # by a lock since it is read from other threads.
        self._latest_lock = threading.Lock()
        self._latest: Optional[Classification] = None

    # ------------------------------------------------------------------ #
    # Observable state
    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        """``True`` while the producer thread is alive and not stopping."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def latest_classification(self) -> Optional[Classification]:
        """The most recently published :class:`Classification`, or ``None``.

        Exposed for debugging and for the application wiring; the authoritative
        stabilized state lives on the :class:`ClassificationBus`.
        """
        with self._latest_lock:
            return self._latest

    # ------------------------------------------------------------------ #
    # Thread lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the capture + classify loop on its own daemon thread.

        If the camera has not been acquired yet, acquisition is attempted first
        so a healthy device begins capturing immediately; a failed initial
        acquire is non-fatal (the loop will keep retrying per Req 1.5). Calling
        :meth:`start` while already running is a no-op.
        """
        if self.is_running:
            return
        self._stop_event.clear()

        # Kick off the initial acquisition so capture can begin on the first
        # iteration. A failure here is expected to be handled by the retry loop
        # (Req 1.3/1.5); acquire() already surfaces/logs the error.
        if not self._camera.is_capturing:
            try:
                self._camera.acquire()
            except Exception as exc:  # noqa: BLE001 - never fail start on acquire
                _log.warning(
                    "Initial camera acquire raised %s; loop will retry.",
                    type(exc).__name__,
                )

        self._thread = threading.Thread(
            target=self._run, name="medusa-vision-producer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = 2.0) -> None:
        """Signal the loop to stop and join the thread (graceful shutdown).

        Sets the stop :class:`threading.Event` the run loop checks each
        iteration, then joins the thread (bounded by ``timeout``). The device is
        released via :meth:`CameraCapture.stop` so no webcam is left open. Safe to
        call when not running.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None
        # Release the device and the detector; leave the bus untouched.
        try:
            self._camera.stop()
        except Exception as exc:  # noqa: BLE001
            _log.debug("camera.stop() raised during shutdown: %r", exc)
        try:
            self._extractor.close()
        except Exception as exc:  # noqa: BLE001
            _log.debug("extractor.close() raised during shutdown: %r", exc)

    # ------------------------------------------------------------------ #
    # Loop body
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        """Thread entry point: run iterations until the stop event is set.

        All exceptions are contained so a transient error never crashes the
        vision thread; the next iteration simply tries again (fail-safe posture).
        """
        _log.info("Vision producer thread started.")
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - loop must never crash
                _log.exception("Vision loop iteration failed: %r", exc)
                # Avoid a tight error spin.
                self._sleep(self._camera.frame_interval_s)
        _log.info("Vision producer thread stopping.")

    def run_once(self) -> Optional[Classification]:
        """Run a single capture + classify iteration (one loop step).

        This is the deterministic unit the run loop repeats, and it can be called
        directly in tests to drive the producer without a real thread or
        wall-clock timing.

        Behavior:

        * **Capturing:** read + process a frame into a
          :class:`~medusa.core.models.FrameMeta`, classify it, publish the
          :class:`Classification` to the bus, cache it as the latest, pace the
          loop by sleeping ``camera.frame_interval_s``, and return it. If the
          frame read yields ``None`` (a transient read failure that dropped the
          device for retry), nothing is published and ``None`` is returned.
        * **Not capturing:** call :meth:`CameraCapture.maybe_retry` (which only
          re-acquires once the <=10s interval has elapsed, Req 1.5), sleep a
          bounded amount, publish nothing, and return ``None``.

        The only outward call on the publish path is ``bus.publish`` -- a
        non-blocking in-memory hand-off. No network I/O, overlay await, or client
        send occurs here (Req 1.2, 11.3).

        Returns:
            The published :class:`Classification` when a frame was captured and
            classified this iteration, otherwise ``None``.
        """
        if not self._camera.is_capturing:
            # No device: attempt a bounded-interval retry and wait responsively.
            # Never publishes a classification while there is no frame.
            self._camera.maybe_retry()
            self._sleep_until_retry()
            return None

        frame_meta = self._extractor.process()
        if frame_meta is None:
            # A read failure dropped the device for retry; nothing to classify.
            return None

        classification = classify_frame(frame_meta, self._config)

        # Publish to the bus -- the single, non-blocking outward hand-off. The
        # bus/consumers handle overlay, broadcast, and alerting off this thread.
        self._bus.publish(classification)

        with self._latest_lock:
            self._latest = classification

        # Pace the loop to target >=10 fps while staying responsive to stop.
        self._sleep(self._camera.frame_interval_s)
        return classification

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sleep_until_retry(self) -> None:
        """Sleep a bounded amount while waiting to re-acquire the camera.

        Sleeps for the smaller of the camera's reported ``seconds_until_retry``
        and :attr:`_retry_poll_cap_s` so a long (up to 10s) retry interval never
        makes :meth:`stop` sluggish, while still not busy-spinning. A zero/near-
        zero wait still yields a minimal sleep to avoid a tight loop.
        """
        try:
            remaining = self._camera.seconds_until_retry()
        except Exception:  # noqa: BLE001 - defensive; fall back to the cap
            remaining = self._retry_poll_cap_s
        wait = min(max(0.0, float(remaining)), self._retry_poll_cap_s)
        if wait <= 0.0:
            wait = min(self._retry_poll_cap_s, self._camera.frame_interval_s)
        self._sleep(wait)
