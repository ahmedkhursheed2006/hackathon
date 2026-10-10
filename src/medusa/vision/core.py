"""Pure, testable vision core: deterministic geometry and classification.

This module holds the I/O-free geometry and classification helpers for the
Vision_Engine so they can be unit- and property-tested without a webcam. Every
function here is a pure function of its inputs (frame metadata + config), which
keeps per-frame classification deterministic (same frame -> same result).

Functions are intentionally small and composable:

* :func:`order_faces` -- the single deterministic face ordering rule applied to
  every frame; ``result[0]`` is the Primary_User and ``result[1]`` the Snooper
  (Req 2.1, 2.2, 2.5).
* :func:`proximity_fraction` -- a single Snooper box's area as a fraction of the
  frame area, clamped to ``[0.0, 1.0]`` (Req 3.1, 3.4).
* :func:`frame_proximity_fraction` -- the maximum valid proximity fraction over
  all candidate Snooper boxes in a frame (Req 3.1, 3.4).

Later vision-core tasks extend this same module (gaze geometry in task 4.5 and
the ``classify_frame`` pipeline in task 4.6) so all pure detection logic stays
co-located and testable together, cleanly separated from camera I/O.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Optional

from medusa.core.models import (
    Classification,
    DetectionConfig,
    Face,
    FrameMeta,
    FrameState,
    GazeAngles,
)

__all__ = [
    "order_faces",
    "proximity_fraction",
    "frame_proximity_fraction",
    "compute_gaze",
    "gaze_within_tolerance",
    "classify_frame",
]


def _ordering_key(face: Face) -> tuple[float, float, float]:
    """Return the deterministic sort key for a single face (Req 2.1).

    The ordering rule, applied identically to every frame, is:

    1. **Descending bounding-box area** -- the largest face comes first, so the
       nearest/most prominent person is the Primary_User (Faces[0]).
    2. **Ascending top-left x, then y** -- a stable, total tie-breaker so equal-
       area faces still have a single well-defined order regardless of the input
       ordering produced by the detector.

    Area is negated so Python's ascending ``sorted`` yields descending area while
    the ``x``/``y`` tie-breakers remain ascending.
    """
    box = face.box
    return (-box.area, box.x, box.y)


def order_faces(faces: list[Face]) -> list[Face]:
    """Order detected faces by the single deterministic rule (Req 2.1, 2.2, 2.5).

    Ordering key: descending bounding-box area, then ascending top-left ``(x, y)``
    as a stable, total tie-breaker. After ordering, ``result[0]`` is the
    Primary_User and ``result[1]`` is the Snooper; any faces beyond the first two
    are retained in the returned permutation but are ignored by the threat
    pipeline (Req 2.5), so they can never influence the designated roles.

    The function is:

    - **Deterministic** -- identical input always yields identical output, because
      the key is a total order over ``(-area, x, y)``.
    - **A permutation** -- the returned list contains exactly the input faces, with
      none added or lost (``sorted`` reorders without inserting/dropping).
    - **Idempotent** -- ordering an already-ordered list returns an equivalent
      ordering, since the key defines a total order over the faces.

    Args:
        faces: The pre-ordering list of detected faces (``FrameMeta.faces``).

    Returns:
        A new list containing the same ``Face`` objects, ordered by the rule
        above. The input list is not mutated.
    """
    return sorted(faces, key=_ordering_key)


def proximity_fraction(face: Face, frame_w: int, frame_h: int) -> Optional[float]:
    """Return a Snooper's bounding-box area as a fraction of the frame area.

    The fraction is ``box.area / (frame_w * frame_h)`` clamped to the closed
    interval ``[0.0, 1.0]`` (Req 3.1). Because :class:`~medusa.core.models.BoundingBox`
    clamps negative dimensions to zero area, a degenerate box yields a box area
    of ``0.0``.

    A frame or box with no usable area is *not* a valid proximity measurement
    and is signaled with ``None`` rather than ``0.0`` so callers can exclude it
    from the frame's proximity computation (Req 3.4):

    * ``None`` is returned if the frame area is zero or negative
      (``frame_w * frame_h <= 0``), and
    * ``None`` is returned if the box area is zero or negative
      (``box.area <= 0``).

    The clamp to ``[0.0, 1.0]`` guards against boxes that extend beyond the
    frame bounds producing a fraction greater than one.

    Args:
        face: The designated Snooper whose bounding box is measured.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        The clamped proximity fraction in ``[0.0, 1.0]``, or ``None`` when the
        frame area or box area is non-positive.
    """
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return None

    box_area = face.box.area
    if box_area <= 0:
        return None

    fraction = box_area / frame_area

    # Clamp to the closed interval [0.0, 1.0]; a box larger than the frame (or
    # extending past its edges) must not report a fraction above 1.0 (Req 3.1).
    if fraction < 0.0:
        return 0.0
    if fraction > 1.0:
        return 1.0
    return fraction


def frame_proximity_fraction(
    faces: Iterable[Face], frame_w: int, frame_h: int
) -> Optional[float]:
    """Return the frame's proximity fraction: the max over valid Snooper boxes.

    For each candidate Snooper box, :func:`proximity_fraction` is computed; boxes
    that are degenerate/absent (``None``) are excluded. The frame's proximity
    fraction is the largest remaining fraction (Req 3.1). When no valid box
    remains -- e.g. an empty iterable, a zero-area frame, or every box being
    degenerate -- the frame has no proximity measurement and ``None`` is
    returned so the caller can classify the frame Safe and retain prior
    proximity data (Req 3.4).

    Args:
        faces: The candidate Snooper faces to evaluate.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        The maximum valid proximity fraction in ``[0.0, 1.0]``, or ``None`` when
        no valid bounding box is available.
    """
    best: Optional[float] = None
    for face in faces:
        fraction = proximity_fraction(face, frame_w, frame_h)
        if fraction is None:
            continue
        if best is None or fraction > best:
            best = fraction
    return best


# ---------------------------------------------------------------------------
# Gaze geometry (yaw / pitch) via solvePnP + Rodrigues (task 4.5, Req 4.1/4.5)
# ---------------------------------------------------------------------------
#
# The established approach for head-pose/gaze from MediaPipe FaceMesh landmarks
# is to map a small set of canonical 3D face points to their observed 2D
# landmark positions and solve the perspective-n-point problem with
# ``cv2.solvePnP``, then convert the returned rotation vector to a rotation
# matrix with ``cv2.Rodrigues`` and decompose that matrix into Euler yaw/pitch
# angles (see design.md "Research Notes"). Content was rephrased for compliance
# with licensing restrictions.
#
# IMPORTANT (runtime note): ``cv2`` / ``numpy`` are heavy, native-wheel
# dependencies that may be unavailable on some interpreters (e.g. a Python
# build with no matching OpenCV/MediaPipe wheels). They are therefore imported
# *lazily inside* :func:`compute_gaze` only. The rest of this module -- and in
# particular the pure predicate :func:`gaze_within_tolerance` -- has NO
# dependency on ``cv2`` or ``numpy`` and remains importable and usable without
# them. When the solve cannot run (missing dependency, too few landmarks, or a
# failed/garbage solve), :func:`compute_gaze` returns ``None`` to signal "gaze
# not computable", which the classification pipeline treats as Safe_State with
# ``gaze_evaluated=False`` (Req 4.5).

# The MediaPipe FaceMesh landmark indices used as PnP correspondences. These are
# the widely-used, stable points for head-pose estimation: nose tip, chin, the
# outer eye corners, and the mouth corners.
_FACEMESH_NOSE_TIP = 1
_FACEMESH_CHIN = 152
_FACEMESH_LEFT_EYE_LEFT_CORNER = 33
_FACEMESH_RIGHT_EYE_RIGHT_CORNER = 263
_FACEMESH_LEFT_MOUTH_CORNER = 61
_FACEMESH_RIGHT_MOUTH_CORNER = 291

# PnP correspondence ordering: each entry pairs a FaceMesh landmark index with
# the matching canonical 3D model point (millimetres, nose tip at the origin).
# This is the classic generic 3D face model used for head-pose estimation.
_GAZE_LANDMARK_INDICES: tuple[int, ...] = (
    _FACEMESH_NOSE_TIP,
    _FACEMESH_CHIN,
    _FACEMESH_LEFT_EYE_LEFT_CORNER,
    _FACEMESH_RIGHT_EYE_RIGHT_CORNER,
    _FACEMESH_LEFT_MOUTH_CORNER,
    _FACEMESH_RIGHT_MOUTH_CORNER,
)

# Canonical 3D model points (x, y, z) in the same order as the indices above.
_CANONICAL_3D_MODEL_POINTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
)

# The minimum number of landmarks required to look up every PnP correspondence.
_MIN_LANDMARKS = max(_GAZE_LANDMARK_INDICES) + 1


def _landmark_xy(landmark: object, frame_w: int, frame_h: int) -> Optional[tuple[float, float]]:
    """Best-effort extraction of a pixel ``(x, y)`` from a FaceMesh landmark.

    MediaPipe landmarks expose normalized ``x``/``y`` attributes in ``[0, 1]``;
    they may also be represented as a plain ``(x, y[, z])`` sequence (normalized
    or already in pixels). This helper accepts either shape and returns pixel
    coordinates, or ``None`` if the landmark cannot be interpreted.

    Normalized coordinates (``0 <= x, y <= 1``) are scaled by the frame
    dimensions; values already outside that range are assumed to be pixels and
    used as-is.
    """
    x: Optional[float]
    y: Optional[float]

    # Attribute-style landmark (MediaPipe NormalizedLandmark).
    x_attr = getattr(landmark, "x", None)
    y_attr = getattr(landmark, "y", None)
    if x_attr is not None and y_attr is not None:
        x, y = float(x_attr), float(y_attr)
    else:
        # Sequence-style landmark: (x, y) or (x, y, z).
        try:
            x, y = float(landmark[0]), float(landmark[1])  # type: ignore[index]
        except (TypeError, IndexError, ValueError):
            return None

    # Treat values within the unit interval as normalized and scale to pixels;
    # otherwise assume they are already pixel coordinates.
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= frame_w
        y *= frame_h
    return (x, y)


def compute_gaze(face: Face, frame_w: int, frame_h: int) -> Optional[GazeAngles]:
    """Estimate a face's gaze yaw/pitch (degrees) via ``solvePnP``/``Rodrigues``.

    The canonical 3D face model points are mapped to the corresponding MediaPipe
    FaceMesh 2D landmarks and the perspective-n-point problem is solved with
    ``cv2.solvePnP``. The resulting rotation vector is converted to a rotation
    matrix via ``cv2.Rodrigues`` and decomposed into Euler angles; the yaw and
    pitch (in degrees) are returned as a :class:`~medusa.core.models.GazeAngles`.

    ``cv2`` and ``numpy`` are imported lazily *inside* this function so the rest
    of the vision core (including the pure :func:`gaze_within_tolerance`
    predicate) stays importable on interpreters where those native wheels are
    unavailable.

    Returns ``None`` to signal "gaze not computable" (Req 4.5) when:

    * ``numpy``/``cv2`` are not installed,
    * ``frame_w``/``frame_h`` are non-positive,
    * the face does not carry enough landmarks for the PnP correspondences,
    * any required landmark cannot be interpreted as a 2D point, or
    * ``cv2.solvePnP`` reports failure or returns a non-finite result.

    Args:
        face: The face whose gaze is estimated; ``face.landmarks`` must contain
            MediaPipe FaceMesh landmarks indexable at the canonical indices.
        frame_w: Frame width in pixels (used to de-normalize landmarks).
        frame_h: Frame height in pixels (used to de-normalize landmarks).

    Returns:
        A :class:`GazeAngles` with ``yaw_deg`` and ``pitch_deg``, or ``None`` if
        gaze cannot be computed for this frame.
    """
    # A frame with no area cannot define a sensible camera model.
    if frame_w <= 0 or frame_h <= 0:
        return None

    # Not enough landmarks to form every PnP correspondence -> not computable.
    landmarks = face.landmarks
    if landmarks is None or len(landmarks) < _MIN_LANDMARKS:
        return None

    # Lazy, guarded import of the heavy native dependencies. If either is
    # missing we simply report "not computable" rather than raising.
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None

    # Gather the observed 2D image points in the canonical correspondence order.
    image_points_list: list[tuple[float, float]] = []
    for index in _GAZE_LANDMARK_INDICES:
        xy = _landmark_xy(landmarks[index], frame_w, frame_h)
        if xy is None:
            return None
        image_points_list.append(xy)

    image_points = np.array(image_points_list, dtype=np.float64)
    model_points = np.array(_CANONICAL_3D_MODEL_POINTS, dtype=np.float64)

    # Approximate pinhole camera intrinsics: focal length ~ frame width, the
    # principal point at the frame centre, and no lens distortion. This is the
    # standard assumption when real calibration is unavailable.
    focal_length = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP. Any failure (geometric degeneracy, exception) -> not computable.
    try:
        success, rotation_vector, _translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not success:
        return None

    # Convert the rotation vector to a rotation matrix (Rodrigues) and decompose
    # it into Euler angles. ``cv2.RQDecomp3x3`` returns the Euler angles in
    # degrees as (pitch, yaw, roll) about the x, y, z axes respectively.
    try:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    except cv2.error:
        return None

    pitch_deg = float(euler_angles[0])
    yaw_deg = float(euler_angles[1])

    # Guard against NaN/inf leaking out of a numerically unstable solve.
    if not (math.isfinite(yaw_deg) and math.isfinite(pitch_deg)):
        return None

    return GazeAngles(yaw_deg=yaw_deg, pitch_deg=pitch_deg)


def gaze_within_tolerance(gaze: GazeAngles, tolerance_deg: float) -> bool:
    """Return whether a gaze intersects the monitor plane within ``tolerance_deg``.

    The monitor plane is modelled as a Snooper facing the camera head-on, i.e.
    ``yaw == 0`` and ``pitch == 0``. The gaze is considered directed at the
    screen when **both** the yaw and the pitch are within ``tolerance_deg`` of
    that head-on direction -- that is, when ``abs(yaw_deg) <= tolerance_deg``
    **and** ``abs(pitch_deg) <= tolerance_deg`` (Req 4.3/4.4).

    This is a pure, dependency-free predicate (no ``cv2``/``numpy``): it is the
    biconditional at the heart of Property 5, so the Threat decision is exactly
    "proximity exceeded AND gaze computable AND :func:`gaze_within_tolerance`".

    Args:
        gaze: The computed gaze angles in degrees.
        tolerance_deg: The Gaze_Tolerance (degrees); the inclusive half-width of
            the accepted yaw/pitch window about the head-on direction.

    Returns:
        ``True`` iff both ``abs(gaze.yaw_deg)`` and ``abs(gaze.pitch_deg)`` are
        less than or equal to ``tolerance_deg``.
    """
    return (
        abs(gaze.yaw_deg) <= tolerance_deg
        and abs(gaze.pitch_deg) <= tolerance_deg
    )


# ---------------------------------------------------------------------------
# Per-frame classification pipeline (task 4.6, Req 2-4 + 6.6 fail-safe)
# ---------------------------------------------------------------------------


def classify_frame(frame_meta: FrameMeta, config: DetectionConfig) -> Classification:
    """Classify a single frame as Threat / Safe / Unclassifiable (Req 2-4, 6.6).

    This is the deterministic threat-decision pipeline from design.md
    ("Threat-decision pipeline" flowchart). It is a pure function of
    ``frame_meta`` and ``config`` -- the same frame always yields the same
    :class:`~medusa.core.models.Classification` -- so it can be unit- and
    property-tested without any camera, ``cv2``, or ``numpy``.

    The pipeline, applied strictly in order:

    1. **Order faces** with :func:`order_faces`; ``ordered[0]`` is the
       Primary_User and ``ordered[1]`` is the Snooper (Req 2.1, 2.2).
    2. **Require >= 2 faces.** Zero or one face can never be a shoulder-surfing
       threat, so the frame is Safe_State with no proximity/gaze data
       (Req 2.3, 2.4).
    3. **Proximity of the Snooper** (``ordered[1]`` only; faces beyond the first
       two are ignored, Req 2.5) via :func:`proximity_fraction`.
    4. **Degenerate/absent box** -- when the Snooper box or the frame has
       non-positive area, :func:`proximity_fraction` returns ``None``; the frame
       is Safe_State and the prior proximity data is retained (reported as
       ``None`` here), with gaze not evaluated (Req 3.4).
    5. **At or below threshold** -- a proximity fraction ``<=`` the
       Proximity_Threshold is Safe_State *without* evaluating gaze; the measured
       fraction is carried on the result (Req 3.2, 4.2).
    6. **Above threshold** -- a fraction strictly greater than the threshold
       triggers :func:`compute_gaze` (Req 3.3, 4.1):

       * **Gaze not computable** (``None``) -> Safe_State with
         ``gaze_evaluated=False``, carrying the proximity fraction (Req 4.5).
       * **Gaze computable** -> ``gaze_evaluated=True``; the frame is
         Threat_State **iff** :func:`gaze_within_tolerance` holds for both yaw
         and pitch, else Safe_State. Both the gaze and the proximity fraction
         are carried (Req 4.3, 4.4).
    7. **Fail-safe:** any unexpected exception anywhere in the pipeline resolves
       to ``UNCLASSIFIABLE`` so the Desktop_Shield holds opaque (Req 6.6); the
       vision loop is never crashed by a classification error.

    The returned classification always carries ``frame_meta.timestamp``.

    Args:
        frame_meta: The per-frame metadata (dimensions, pre-ordering faces,
            timestamp).
        config: The detection thresholds (proximity threshold, gaze tolerance).

    Returns:
        A :class:`~medusa.core.models.Classification` with the decided ``state``,
        the ``proximity_fraction`` and ``gaze`` carried where applicable, the
        ``gaze_evaluated`` flag, and the frame timestamp.
    """
    timestamp = frame_meta.timestamp
    try:
        # Step 1: deterministic ordering -> Primary_User[0], Snooper[1].
        ordered = order_faces(list(frame_meta.faces))

        # Step 2: fewer than two faces can never be a threat (Req 2.3, 2.4).
        if len(ordered) < 2:
            return Classification(
                state=FrameState.SAFE,
                proximity_fraction=None,
                gaze=None,
                gaze_evaluated=False,
                timestamp=timestamp,
            )

        # Step 3: proximity of the single designated Snooper (Faces[1]); faces
        # beyond the first two are ignored (Req 2.5).
        snooper = ordered[1]
        fraction = proximity_fraction(snooper, frame_meta.width, frame_meta.height)

        # Step 4: degenerate/absent box -> Safe, retain prior proximity data
        # (reported as None), gaze not evaluated (Req 3.4).
        if fraction is None:
            return Classification(
                state=FrameState.SAFE,
                proximity_fraction=None,
                gaze=None,
                gaze_evaluated=False,
                timestamp=timestamp,
            )

        # Step 5: at or below the threshold -> Safe without evaluating gaze,
        # carrying the measured fraction (Req 3.2, 4.2).
        if fraction <= config.proximity_threshold:
            return Classification(
                state=FrameState.SAFE,
                proximity_fraction=fraction,
                gaze=None,
                gaze_evaluated=False,
                timestamp=timestamp,
            )

        # Step 6: strictly above the threshold -> evaluate gaze (Req 3.3, 4.1).
        gaze = compute_gaze(snooper, frame_meta.width, frame_meta.height)

        # Step 6a: gaze not computable -> Safe with gaze_evaluated=False,
        # carrying the proximity fraction (Req 4.5).
        if gaze is None:
            return Classification(
                state=FrameState.SAFE,
                proximity_fraction=fraction,
                gaze=None,
                gaze_evaluated=False,
                timestamp=timestamp,
            )

        # Step 6b: gaze computable -> Threat iff yaw AND pitch within tolerance,
        # else Safe (Req 4.3, 4.4). Carry both gaze and proximity fraction.
        is_threat = gaze_within_tolerance(gaze, config.gaze_tolerance_deg)
        state = FrameState.THREAT if is_threat else FrameState.SAFE
        return Classification(
            state=state,
            proximity_fraction=fraction,
            gaze=gaze,
            gaze_evaluated=True,
            timestamp=timestamp,
        )
    except Exception:
        # Step 7: fail-safe -- any unexpected error resolves to UNCLASSIFIABLE
        # so the shield stays opaque until a confidently-Safe frame (Req 6.6).
        return Classification(
            state=FrameState.UNCLASSIFIABLE,
            proximity_fraction=None,
            gaze=None,
            gaze_evaluated=False,
            timestamp=timestamp,
        )
