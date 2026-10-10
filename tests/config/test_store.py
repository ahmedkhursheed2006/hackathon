"""Unit tests for the ConfigurationStore load/save with range validation (task 2.3).

Covers the concrete behaviors specified by Requirements 10.1-10.4 and 10.6:
- absent values collapse to documented defaults (Req 10.2),
- out-of-range values collapse to defaults with a per-setting error (Req 10.3),
- a whole-file read failure yields all defaults plus a single global error (Req 10.4),
- a missing file (first run) yields all defaults with *no* error,
- save() round-trips config including keys, and keys are registered as secrets
  so they are never emitted to logs (Req 10.6).
"""

from __future__ import annotations

import json

import pytest

from medusa.config.logging_setup import clear_secrets, redact_text, registered_secrets
from medusa.config.store import (
    DEFAULT_COOLDOWN_PERIOD_S,
    DEFAULT_GAZE_TOLERANCE_DEG,
    DEFAULT_PROXIMITY_THRESHOLD,
    DEFAULT_SERVER_PORT,
    ConfigurationStore,
    validate_in_range,
)
from medusa.core.models import AppConfig, DetectionConfig, Secret


@pytest.fixture(autouse=True)
def _clean_secret_registry():
    """Keep the module-level secret registry isolated between tests."""
    clear_secrets()
    yield
    clear_secrets()


def _store(tmp_path):
    return ConfigurationStore(path=tmp_path / "config.json")


class TestLoadDefaults:
    def test_missing_file_is_first_run_no_error(self, tmp_path):
        result = _store(tmp_path).load()
        assert result.ok
        assert result.errors == []
        cfg = result.config
        assert cfg.detection.proximity_threshold == DEFAULT_PROXIMITY_THRESHOLD
        assert cfg.detection.gaze_tolerance_deg == DEFAULT_GAZE_TOLERANCE_DEG
        assert cfg.cooldown_period_s == DEFAULT_COOLDOWN_PERIOD_S
        assert cfg.server_port == DEFAULT_SERVER_PORT
        assert cfg.consent_granted is False
        assert cfg.vlm_provider is None and cfg.vlm_api_key is None

    def test_absent_values_apply_defaults_without_error(self, tmp_path):
        path = tmp_path / "config.json"
        # Present only some keys; the rest are absent and must default silently.
        path.write_text(json.dumps({"vlm_provider": "acme"}), encoding="utf-8")
        result = ConfigurationStore(path=path).load()
        assert result.ok
        assert result.config.vlm_provider == "acme"
        assert result.config.detection.proximity_threshold == DEFAULT_PROXIMITY_THRESHOLD
        assert result.config.server_port == DEFAULT_SERVER_PORT


class TestLoadRangeValidation:
    def test_out_of_range_values_fall_back_to_defaults_with_errors(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "proximity_threshold": 5.0,  # > 1.0
                    "gaze_tolerance_deg": -1.0,  # < 0.0
                    "cooldown_period_s": 0.1,  # < 0.5
                    "server_port": 70000,  # > 65535
                }
            ),
            encoding="utf-8",
        )
        result = ConfigurationStore(path=path).load()
        assert not result.ok
        # One per-setting error for each invalid value.
        assert len(result.errors) == 4
        cfg = result.config
        assert cfg.detection.proximity_threshold == DEFAULT_PROXIMITY_THRESHOLD
        assert cfg.detection.gaze_tolerance_deg == DEFAULT_GAZE_TOLERANCE_DEG
        assert cfg.cooldown_period_s == DEFAULT_COOLDOWN_PERIOD_S
        assert cfg.server_port == DEFAULT_SERVER_PORT

    def test_non_numeric_value_falls_back(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"server_port": "not-a-port"}), encoding="utf-8")
        result = ConfigurationStore(path=path).load()
        assert not result.ok
        assert result.config.server_port == DEFAULT_SERVER_PORT

    def test_boundary_values_are_accepted(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "proximity_threshold": 0.0,
                    "gaze_tolerance_deg": 90.0,
                    "cooldown_period_s": 10.0,
                    "server_port": 1,
                }
            ),
            encoding="utf-8",
        )
        result = ConfigurationStore(path=path).load()
        assert result.ok
        cfg = result.config
        assert cfg.detection.proximity_threshold == 0.0
        assert cfg.detection.gaze_tolerance_deg == 90.0
        assert cfg.cooldown_period_s == 10.0
        assert cfg.server_port == 1

    def test_non_integer_port_falls_back(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"server_port": 8080.5}), encoding="utf-8")
        result = ConfigurationStore(path=path).load()
        assert not result.ok
        assert result.config.server_port == DEFAULT_SERVER_PORT


class TestLoadReadFailure:
    def test_corrupt_file_yields_all_defaults_with_global_error(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{ this is : not valid json", encoding="utf-8")
        result = ConfigurationStore(path=path).load()
        assert not result.ok
        assert len(result.errors) == 1  # a single global error (Req 10.4)
        cfg = result.config
        assert cfg.detection.proximity_threshold == DEFAULT_PROXIMITY_THRESHOLD
        assert cfg.server_port == DEFAULT_SERVER_PORT

    def test_non_object_json_yields_all_defaults_with_global_error(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = ConfigurationStore(path=path).load()
        assert not result.ok
        assert len(result.errors) == 1


class TestSaveRoundTrip:
    def test_save_then_load_round_trips(self, tmp_path):
        store = _store(tmp_path)
        cfg = AppConfig(
            detection=DetectionConfig(proximity_threshold=0.25, gaze_tolerance_deg=20.0),
            vlm_provider="acme-vlm",
            tts_provider="acme-tts",
            vlm_api_key=Secret("vlm-key-123"),
            tts_api_key=Secret("tts-key-456"),
            cooldown_period_s=3.5,
            server_port=9090,
            request_timeout_s=12.0,
            consent_granted=True,
        )
        assert store.save(cfg) is True

        loaded = store.load()
        assert loaded.ok
        got = loaded.config
        assert got.detection.proximity_threshold == 0.25
        assert got.detection.gaze_tolerance_deg == 20.0
        assert got.vlm_provider == "acme-vlm"
        assert got.tts_provider == "acme-tts"
        assert got.cooldown_period_s == 3.5
        assert got.server_port == 9090
        assert got.request_timeout_s == 12.0
        assert got.consent_granted is True
        # Keys round-trip but remain Secret-wrapped.
        assert got.vlm_api_key is not None and got.vlm_api_key.reveal() == "vlm-key-123"
        assert got.tts_api_key is not None and got.tts_api_key.reveal() == "tts-key-456"

    def test_save_registers_keys_as_secrets_for_redaction(self, tmp_path):
        store = _store(tmp_path)
        cfg = AppConfig(
            detection=DetectionConfig(),
            vlm_provider="acme-vlm",
            tts_provider=None,
            vlm_api_key=Secret("super-secret-key"),
            tts_api_key=None,
        )
        store.save(cfg)
        assert "super-secret-key" in registered_secrets()
        # A log line that accidentally included the key would be redacted.
        assert "super-secret-key" not in redact_text("key=super-secret-key")

    def test_load_registers_keys_as_secrets(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"vlm_api_key": "loaded-secret"}), encoding="utf-8"
        )
        result = ConfigurationStore(path=path).load()
        assert result.config.vlm_api_key is not None
        assert "loaded-secret" in registered_secrets()

    def test_save_failure_returns_false(self, tmp_path):
        # Point the store at a path whose parent is an existing *file*, so the
        # directory creation / write fails and save() reports failure.
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        store = ConfigurationStore(path=blocker / "nested" / "config.json")
        cfg = AppConfig(
            detection=DetectionConfig(),
            vlm_provider=None,
            tts_provider=None,
            vlm_api_key=None,
            tts_api_key=None,
        )
        assert store.save(cfg) is False


class TestValidateInRangeHelper:
    def test_none_defaults_without_error(self):
        value, err = validate_in_range(
            None, name="x", low=0.0, high=1.0, default=0.5
        )
        assert value == 0.5 and err is None

    def test_bool_rejected_as_non_numeric(self):
        value, err = validate_in_range(
            True, name="x", low=0.0, high=1.0, default=0.5
        )
        assert value == 0.5 and err is not None
