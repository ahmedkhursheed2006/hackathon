"""On-device vision pipeline: capture, face detection, proximity, and gaze.

The pure, I/O-free detection logic lives in :mod:`medusa.vision.core` (face
ordering, proximity, gaze geometry, and frame classification) so it can be
property-tested without a webcam. Camera acquisition and the MediaPipe FaceMesh
runtime are layered on top of that core in :mod:`medusa.vision.camera`.

The ``camera`` runtime is import-safe without ``cv2``/``mediapipe`` installed:
the native libraries are imported lazily only when a real device is opened, and
all collaborators are injectable so the acquire/retry state machine and the
landmarks -> :class:`~medusa.core.models.FrameMeta` mapping can be tested with
fakes. For that reason the camera symbols are *not* eagerly re-exported here
(to avoid pulling the runtime into every ``import medusa.vision``); import them
directly from :mod:`medusa.vision.camera`.
"""

from medusa.vision.core import (
    frame_proximity_fraction,
    order_faces,
    proximity_fraction,
)

__all__ = ["order_faces", "proximity_fraction", "frame_proximity_fraction"]
