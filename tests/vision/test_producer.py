"""Integration-style tests for the VisionProducer capture + classify thread (task 12.2).

These tests drive the producer deterministically via :meth:`VisionProducer.run_once`
(no real thread / wall-clock timing) with fake camera + extractor collaborators,
and assert the loop's contract from the design's Threading / Concurrency Model
(Req 1.2, 11.3):

* a single step while capturing publishes a :class:`Classification` to the bus;
* a step while not capturing triggers ``maybe_retry`` and publishes nothing;
* the loop performs no network I/O / overlay await / client send -- it only
  publishes to the bus.

They import only the already import-safe ``camera`` / ``core`` / ``bus`` /
``producer`` modules, so they run without ``cv2``/``mediapipe``/``PySide6``.
"""

from __future__ import annotations

from typing import Optional

from medusa.core.bus import ClassificationBus
from medusa.core.models import (
    BoundingBox,
    Classification,
    DetectionConfig,
    Face,
    FrameMeta,
    FrameState,
)
from medusa.vision.producer import VisionProducer


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeClock:
    """A deterministic monotonic clock + sleep that records sleep durations."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class _RecordingBus(ClassificationBus):
    """A real bus that also records every published classification."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[Classification] = []

    def publish(self, classification: Classification) -> None:  # type: ignore[override]
        self.published.append(classification)
        super().publish(classification)


class _FakeExtractor:
    """A stand-in :class:`FaceMeshExtractor` returning a scripted FrameMeta.

    Tracks whether :meth:`process` was called and whether any transmit-adjacent
    method (there are none on the producer's path) was ever reached.
    """

    def __init__(self, frame_meta: Optional[FrameMeta]) -> None:
        self._frame_meta = frame_meta
        self.process_calls = 0
        self.closed = False

    def process(self, frame: object = None) -> Optional[FrameMeta]:
        self.process_calls += 1
        return self._frame_meta

    def close(self) -> None:
        self.closed = True


class _CapturingCamera:
    """A minimal capturing camera exposing the surface the producer uses."""

    def __init__(self) -> None:
        self.maybe_retry_calls = 0
        self.seconds_until_retry_calls = 0
        self.stopped = False

    @property
    def is_capturing(self) -> bool:
        return True

    @property
    def frame_interval_s(self) -> float:
        return 0.1  # 10 fps

    def maybe_retry(self) -> bool:
        self.maybe_retry_calls += 1
        return False

    def seconds_until_retry(self) -> float:
        self.seconds_until_retry_calls += 1
        return 0.0

    def stop(self) -> None:
        self.stopped = True


class _DownCamera(_CapturingCamera):
    """A camera that is *not* capturing (down / awaiting retry)."""

    def __init__(self, seconds_until_retry: float = 3.0) -> None:
        super().__init__()
        self._until = seconds_until_retry

    @property
    def is_capturing(self) -> bool:
        return False

    def seconds_until_retry(self) -> float:
        self.seconds_until_retry_calls += 1
        return self._until


def _threat_frame_meta() -> FrameMeta:
    """A two-face frame whose Snooper box is well above the proximity threshold.

    The gaze solve is "not computable" (no cv2 and too few landmarks), so this
    classifies Safe-with-gaze-not-evaluated rather than Threat; that is fine --
    the test only asserts that a Classification is produced and published.
    """
    big = Face(BoundingBox(0, 0, 100, 100))  # area 10000
    small = Face(BoundingBox(0, 0, 60, 60))  # area 3600 -> fraction 0.3 > 0.12
    return FrameMeta(width=100, height=100, faces=(big, small), timestamp=1.0)


DEFAULT_CONFIG = DetectionConfig()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_single_step_while_capturing_publishes_classification() -> None:
    """A capturing step reads+classifies a frame and publishes to the bus."""
    clock = _FakeClock()
    bus = _RecordingBus()
    camera = _CapturingCamera()
    extractor = _FakeExtractor(_threat_frame_meta())

    producer = VisionProducer(
        camera=camera,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        bus=bus,
        detection_config=DEFAULT_CONFIG,
        now=clock.now,
        sleep=clock.sleep,
    )

    result = producer.run_once()

    assert extractor.process_calls == 1
    assert isinstance(result, Classification)
    # Exactly one classification was published to the bus this step.
    assert len(bus.published) == 1
    assert bus.published[0] is result
    # Exposed for debugging/wiring.
    assert producer.latest_classification is result
    # Loop paced itself by one frame interval (targets >=10 fps).
    assert clock.sleeps == [camera.frame_interval_s]
    # It never fell back to the retry path while capturing.
    assert camera.maybe_retry_calls == 0


def test_step_while_not_capturing_triggers_retry_and_publishes_nothing() -> None:
    """A non-capturing step calls maybe_retry and publishes no classification."""
    clock = _FakeClock()
    bus = _RecordingBus()
    camera = _DownCamera(seconds_until_retry=3.0)
    extractor = _FakeExtractor(_threat_frame_meta())

    producer = VisionProducer(
        camera=camera,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        bus=bus,
        detection_config=DEFAULT_CONFIG,
        now=clock.now,
        sleep=clock.sleep,
    )

    result = producer.run_once()

    assert result is None
    assert camera.maybe_retry_calls == 1
    # Nothing captured -> nothing classified -> nothing published.
    assert extractor.process_calls == 0
    assert bus.published == []
    assert producer.latest_classification is None
    # Slept, but bounded by the retry poll cap (not the full 3s interval).
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] <= producer_retry_cap(producer)


def test_no_frame_from_extractor_publishes_nothing() -> None:
    """If a capturing read yields no frame, nothing is published."""
    clock = _FakeClock()
    bus = _RecordingBus()
    camera = _CapturingCamera()
    extractor = _FakeExtractor(None)  # read failure dropped the device

    producer = VisionProducer(
        camera=camera,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        bus=bus,
        detection_config=DEFAULT_CONFIG,
        now=clock.now,
        sleep=clock.sleep,
    )

    result = producer.run_once()

    assert result is None
    assert extractor.process_calls == 1
    assert bus.published == []
    assert producer.latest_classification is None


def test_loop_only_touches_the_bus_no_network_or_overlay_calls() -> None:
    """The producer's only collaborators are the camera, extractor, and bus.

    This guards Req 1.2 / 11.3 structurally: the fake camera/extractor expose no
    network/overlay/client-send surface, and the only publish-path call is
    ``bus.publish``. A step completes using exactly those, confirming the loop
    performs no network I/O, overlay await, or client send.
    """
    clock = _FakeClock()
    bus = _RecordingBus()
    camera = _CapturingCamera()
    extractor = _FakeExtractor(_threat_frame_meta())

    producer = VisionProducer(
        camera=camera,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        bus=bus,
        detection_config=DEFAULT_CONFIG,
        now=clock.now,
        sleep=clock.sleep,
    )
    producer.run_once()

    # The bus is the sole outward hand-off; its effective state reflects the
    # published classification (a non-blocking in-memory call).
    assert len(bus.published) == 1
    assert bus.effective_state in (FrameState.SAFE, FrameState.THREAT)


def test_start_stop_runs_real_thread_and_shuts_down_gracefully() -> None:
    """start() runs the loop on its own thread; stop() joins + releases cleanly."""
    import threading
    import time

    bus = _RecordingBus()
    camera = _CapturingCamera()
    extractor = _FakeExtractor(_threat_frame_meta())

    # Use the real (tiny) sleep so the daemon thread actually paces + yields.
    producer = VisionProducer(
        camera=camera,  # type: ignore[arg-type]
        extractor=extractor,  # type: ignore[arg-type]
        bus=bus,
        detection_config=DEFAULT_CONFIG,
    )

    producer.start()
    try:
        # Give the thread a brief moment to publish at least one classification.
        deadline = time.monotonic() + 2.0
        while not bus.published and time.monotonic() < deadline:
            time.sleep(0.01)
        assert producer.is_running
        assert bus.published, "expected the running loop to publish a classification"
    finally:
        producer.stop(timeout=2.0)

    assert not producer.is_running
    # Graceful shutdown released the device and detector.
    assert camera.stopped is True
    assert extractor.closed is True
    # No producer threads should be left alive.
    assert not any(
        t.name == "medusa-vision-producer" and t.is_alive()
        for t in threading.enumerate()
    )


def producer_retry_cap(producer: VisionProducer) -> float:
    """Read the producer's retry poll cap for assertions."""
    from medusa.vision.producer import RETRY_POLL_CAP_S

    return RETRY_POLL_CAP_S
