"""Unit tests for the pure gaze geometry helpers in ``medusa.vision.core``.

Covers the dependency-free :func:`gaze_within_tolerance` biconditional and the
"not computable" (``None``) paths of :func:`compute_gaze` that are exercisable
without the heavy native ``cv2``/``numpy`` dependencies (task 4.5, Req 4.3/4.4/
4.5). The full gaze-threat biconditional property test is task 4.8.
"""

from __future__ import annotations

import importlib.util

import pytest

from medusa.core.models import BoundingBox, Face, GazeAngles
from medusa.vision.core import compute_gaze, gaze_within_tolerance

# Whether the native solve dependencies are importable in this environment.
_HAS_CV2 = (
    importlib.util.find_spec("cv2") is not None
    and importlib.util.find_spec("numpy") is not None
)


class TestGazeWithinTolerance:
    """The pure monitor-plane predicate (Req 4.3/4.4)."""

    def test_head_on_gaze_is_within_tolerance(self) -> None:
        assert gaze_within_tolerance(GazeAngles(0.0, 0.0), 15.0) is True

    def test_boundary_values_are_inclusive(self) -> None:
        # abs(angle) == tolerance is accepted (<=), on both axes and both signs.
        assert gaze_within_tolerance(GazeAngles(15.0, -15.0), 15.0) is True
        assert gaze_within_tolerance(GazeAngles(-15.0, 15.0), 15.0) is True

    def test_yaw_outside_tolerance_fails(self) -> None:
        assert gaze_within_tolerance(GazeAngles(15.1, 0.0), 15.0) is False
        assert gaze_within_tolerance(GazeAngles(-16.0, 0.0), 15.0) is False

    def test_pitch_outside_tolerance_fails(self) -> None:
        assert gaze_within_tolerance(GazeAngles(0.0, 20.0), 15.0) is False
        assert gaze_within_tolerance(GazeAngles(0.0, -20.0), 15.0) is False

    def test_requires_both_axes_within_tolerance(self) -> None:
        # Within on one axis but not the other is still a miss (AND, not OR).
        assert gaze_within_tolerance(GazeAngles(5.0, 30.0), 15.0) is False
        assert gaze_within_tolerance(GazeAngles(30.0, 5.0), 15.0) is False

    def test_zero_tolerance_only_accepts_exact_head_on(self) -> None:
        assert gaze_within_tolerance(GazeAngles(0.0, 0.0), 0.0) is True
        assert gaze_within_tolerance(GazeAngles(0.1, 0.0), 0.0) is False


def _landmarks(count: int) -> tuple:
    """A tuple of ``count`` normalized ``(x, y)`` landmark points."""
    return tuple((0.5, 0.5) for _ in range(count))


class TestComputeGazeNotComputable:
    """``compute_gaze`` returns ``None`` ("not computable") on bad input (Req 4.5)."""

    def test_insufficient_landmarks_returns_none(self) -> None:
        face = Face(BoundingBox(0, 0, 10, 10), landmarks=_landmarks(10))
        assert compute_gaze(face, 640, 480) is None

    def test_empty_landmarks_returns_none(self) -> None:
        face = Face(BoundingBox(0, 0, 10, 10), landmarks=())
        assert compute_gaze(face, 640, 480) is None

    def test_non_positive_frame_dimensions_return_none(self) -> None:
        face = Face(BoundingBox(0, 0, 10, 10), landmarks=_landmarks(500))
        assert compute_gaze(face, 0, 480) is None
        assert compute_gaze(face, 640, 0) is None
        assert compute_gaze(face, -640, 480) is None


@pytest.mark.skipif(not _HAS_CV2, reason="cv2/numpy not installed in this environment")
class TestComputeGazeWithCv2:
    """When the native solve is available, a plausible face yields GazeAngles."""

    def test_plausible_landmarks_produce_gaze_angles(self) -> None:
        frame_w, frame_h = 640, 480
        # Place the six PnP correspondence landmarks at plausible, roughly
        # face-shaped pixel positions near the frame centre; all other indices
        # are filler. Indices mirror the module's canonical correspondences.
        pts = {
            1: (320, 250),  # nose tip
            152: (320, 360),  # chin
            33: (250, 220),  # left eye left corner
            263: (390, 220),  # right eye right corner
            61: (280, 310),  # left mouth corner
            291: (360, 310),  # right mouth corner
        }
        landmarks = tuple(
            pts.get(i, (320, 240)) for i in range(max(pts) + 1)
        )
        face = Face(BoundingBox(240, 200, 160, 180), landmarks=landmarks)

        gaze = compute_gaze(face, frame_w, frame_h)

        assert gaze is not None
        assert isinstance(gaze, GazeAngles)
        # Angles are finite, real degrees.
        assert -180.0 <= gaze.yaw_deg <= 180.0
        assert -180.0 <= gaze.pitch_deg <= 180.0
