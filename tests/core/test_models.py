"""Unit tests for the core data models and enums (task 1.2).

These cover the concrete behaviors specified in design.md's Data Models section:
enum values, ``Status.message()`` payloads, ``BoundingBox.area``, frozen/
immutable semantics, and - most importantly - that API keys never leak through
``repr``/``str`` of a ``Secret`` or an ``AppConfig`` (Requirement 10.6).
"""

from __future__ import annotations

import dataclasses

import pytest

from medusa.core.models import (
    AppConfig,
    BoundingBox,
    Classification,
    DetectionConfig,
    Face,
    FrameMeta,
    FrameState,
    GazeAngles,
    Secret,
    Status,
)


def _app_config(vlm_key: str | None = "vlm-SECRET-123", tts_key: str | None = "tts-SECRET-456") -> AppConfig:
    return AppConfig(
        detection=DetectionConfig(),
        vlm_provider="example-vlm",
        tts_provider="example-tts",
        vlm_api_key=Secret(vlm_key) if vlm_key is not None else None,
        tts_api_key=Secret(tts_key) if tts_key is not None else None,
    )


class TestEnums:
    def test_frame_state_values(self):
        assert FrameState.THREAT.value == "THREAT"
        assert FrameState.SAFE.value == "SAFE"
        assert FrameState.UNCLASSIFIABLE.value == "UNCLASSIFIABLE"

    def test_status_message_threat(self):
        assert Status.THREAT_DETECTED.message() == {"status": "THREAT_DETECTED"}

    def test_status_message_safe(self):
        assert Status.SAFE.message() == {"status": "SAFE"}

    def test_status_message_has_no_extra_fields(self):
        for status in Status:
            assert set(status.message().keys()) == {"status"}


class TestBoundingBox:
    def test_area_positive(self):
        assert BoundingBox(0, 0, 4, 5).area == 20.0

    def test_area_zero_width(self):
        assert BoundingBox(0, 0, 0, 5).area == 0.0

    def test_area_negative_dims_clamp_to_zero(self):
        assert BoundingBox(0, 0, -4, 5).area == 0.0
        assert BoundingBox(0, 0, 4, -5).area == 0.0
        assert BoundingBox(0, 0, -4, -5).area == 0.0


class TestFrozenImmutability:
    def test_bounding_box_is_frozen(self):
        box = BoundingBox(0, 0, 1, 1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            box.x = 5  # type: ignore[misc]

    def test_app_config_is_frozen(self):
        cfg = _app_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.server_port = 9000  # type: ignore[misc]

    def test_classification_is_frozen(self):
        c = Classification(
            state=FrameState.SAFE,
            proximity_fraction=None,
            gaze=None,
            gaze_evaluated=False,
            timestamp=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.state = FrameState.THREAT  # type: ignore[misc]


class TestDefaults:
    def test_detection_config_defaults(self):
        d = DetectionConfig()
        assert d.proximity_threshold == pytest.approx(0.12)
        assert d.gaze_tolerance_deg == pytest.approx(15.0)

    def test_app_config_defaults(self):
        cfg = _app_config()
        assert cfg.cooldown_period_s == pytest.approx(2.0)
        assert cfg.server_port == 8080
        assert cfg.request_timeout_s == pytest.approx(10.0)
        assert cfg.consent_granted is False


class TestSecretRedaction:
    """Requirement 10.6: API key values must never leak via repr/str."""

    def test_secret_repr_hides_value(self):
        s = Secret("super-secret-key")
        assert "super-secret-key" not in repr(s)

    def test_secret_str_hides_value(self):
        s = Secret("super-secret-key")
        assert "super-secret-key" not in str(s)
        assert "super-secret-key" not in f"{s}"

    def test_secret_reveal_returns_raw_value(self):
        s = Secret("super-secret-key")
        assert s.reveal() == "super-secret-key"

    def test_app_config_repr_does_not_leak_keys(self):
        cfg = _app_config(vlm_key="vlm-LEAK-ME", tts_key="tts-LEAK-ME")
        rendered = repr(cfg)
        assert "vlm-LEAK-ME" not in rendered
        assert "tts-LEAK-ME" not in rendered

    def test_app_config_str_does_not_leak_keys(self):
        cfg = _app_config(vlm_key="vlm-LEAK-ME", tts_key="tts-LEAK-ME")
        rendered = str(cfg)
        assert "vlm-LEAK-ME" not in rendered
        assert "tts-LEAK-ME" not in rendered

    def test_app_config_keys_accessible_via_reveal(self):
        cfg = _app_config(vlm_key="vlm-ok", tts_key="tts-ok")
        assert cfg.vlm_api_key is not None and cfg.vlm_api_key.reveal() == "vlm-ok"
        assert cfg.tts_api_key is not None and cfg.tts_api_key.reveal() == "tts-ok"

    def test_app_config_allows_absent_keys(self):
        cfg = _app_config(vlm_key=None, tts_key=None)
        assert cfg.vlm_api_key is None
        assert cfg.tts_api_key is None


class TestCompositeConstruction:
    def test_frame_meta_and_faces(self):
        face = Face(box=BoundingBox(0, 0, 10, 10), landmarks=())
        meta = FrameMeta(width=640, height=480, faces=(face,), timestamp=1.5)
        assert meta.faces[0].box.area == 100.0
        assert meta.width == 640

    def test_gaze_angles(self):
        g = GazeAngles(yaw_deg=5.0, pitch_deg=-3.0)
        assert g.yaw_deg == 5.0
        assert g.pitch_deg == -3.0
