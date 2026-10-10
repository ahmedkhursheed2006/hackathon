"""Configuration_Store: load/save application config with range validation.

This module implements the cross-cutting Configuration_Store described in the
design (section "Cross-cutting: Configuration_Store"). It is responsible for:

- Reading the Proximity_Threshold, Gaze_Tolerance, VLM/TTS provider names and
  keys, the Status_Server port, the Cooldown_Period, the request timeout, and
  the persisted consent decision on startup (Req 10.1).
- Applying documented defaults for any absent value (Req 10.2).
- Range-validating every value; an out-of-range value collapses to its
  documented default and is reported with a per-setting error (Req 10.3).
- Falling back to all defaults with a single global error if the backing store
  cannot be read at all (corrupt / unreadable file) (Req 10.4).
- Persisting the config via :meth:`ConfigurationStore.save`, including API keys,
  which are registered as secrets so they are never emitted to logs (Req 10.6).

Persistence format and location
--------------------------------
The config is stored as a JSON document in a per-user config directory. The path
is injectable via the constructor so tests (and the forthcoming Property 11
test, task 2.4) can point at a temporary file. A missing file is a normal
first-run condition and simply yields all defaults with *no* error; only a file
that exists but cannot be parsed/read is treated as a read failure (Req 10.4).

Purity note (supports Property 11)
----------------------------------
:meth:`ConfigurationStore.load` is a deterministic function of the backing
store's raw contents: for identical raw input it always returns an
``AppConfig`` whose every setting lies within its documented range. Validation
and defaulting are factored into small pure helpers (:func:`validate_in_range`)
so the load logic can be exercised directly over arbitrary raw inputs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from medusa.config.logging_setup import register_secret
from medusa.core.models import AppConfig, DetectionConfig, Secret

__all__ = [
    "DEFAULT_PROXIMITY_THRESHOLD",
    "DEFAULT_GAZE_TOLERANCE_DEG",
    "DEFAULT_COOLDOWN_PERIOD_S",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_CONSENT_GRANTED",
    "PROXIMITY_THRESHOLD_RANGE",
    "GAZE_TOLERANCE_RANGE",
    "COOLDOWN_PERIOD_RANGE",
    "SERVER_PORT_RANGE",
    "ConfigLoadResult",
    "ConfigurationStore",
    "default_config",
    "validate_in_range",
]

_log = logging.getLogger(__name__)

# --- Documented default values (Req 10.2) ---
DEFAULT_PROXIMITY_THRESHOLD = 0.12
DEFAULT_GAZE_TOLERANCE_DEG = 15.0
DEFAULT_COOLDOWN_PERIOD_S = 2.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
DEFAULT_SERVER_PORT = 8080
DEFAULT_CONSENT_GRANTED = False

# --- Documented valid ranges, inclusive on both ends (Req 10.3) ---
PROXIMITY_THRESHOLD_RANGE = (0.0, 1.0)
GAZE_TOLERANCE_RANGE = (0.0, 90.0)
COOLDOWN_PERIOD_RANGE = (0.5, 10.0)
SERVER_PORT_RANGE = (1, 65535)

# JSON keys used in the persisted document.
_KEY_PROXIMITY = "proximity_threshold"
_KEY_GAZE = "gaze_tolerance_deg"
_KEY_VLM_PROVIDER = "vlm_provider"
_KEY_TTS_PROVIDER = "tts_provider"
_KEY_VLM_KEY = "vlm_api_key"
_KEY_TTS_KEY = "tts_api_key"
_KEY_COOLDOWN = "cooldown_period_s"
_KEY_PORT = "server_port"
_KEY_TIMEOUT = "request_timeout_s"
_KEY_CONSENT = "consent_granted"


def _default_config_path() -> Path:
    """Return the default per-user config file location.

    Uses the standard per-user config directory for the host OS. On Windows this
    is ``%APPDATA%``; elsewhere ``$XDG_CONFIG_HOME`` or ``~/.config`` is used.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "medusa-desktop-core" / "config.json"


def default_config() -> AppConfig:
    """Return an ``AppConfig`` populated entirely with documented defaults.

    Providers and keys are absent (``None``) by default, which disables the
    corresponding Alert_Service capability (Req 10.5); consent is deny-by-default
    (Req 9.1).
    """
    return AppConfig(
        detection=DetectionConfig(
            proximity_threshold=DEFAULT_PROXIMITY_THRESHOLD,
            gaze_tolerance_deg=DEFAULT_GAZE_TOLERANCE_DEG,
        ),
        vlm_provider=None,
        tts_provider=None,
        vlm_api_key=None,
        tts_api_key=None,
        cooldown_period_s=DEFAULT_COOLDOWN_PERIOD_S,
        server_port=DEFAULT_SERVER_PORT,
        request_timeout_s=DEFAULT_REQUEST_TIMEOUT_S,
        consent_granted=DEFAULT_CONSENT_GRANTED,
    )


def validate_in_range(
    raw: Any,
    *,
    name: str,
    low: float,
    high: float,
    default: float,
    integer: bool = False,
) -> tuple[float, Optional[str]]:
    """Validate ``raw`` against ``[low, high]``, falling back to ``default``.

    Returns a ``(value, error)`` tuple. ``error`` is ``None`` when the value was
    present and in range; otherwise it is a per-setting message and ``value`` is
    ``default``.

    - An absent value (``None``) returns the default with *no* error: absence is
      handled by defaulting (Req 10.2), not by a validation error.
    - A non-numeric or out-of-range value returns the default *with* a
      per-setting error (Req 10.3). Booleans are rejected as non-numeric so a
      stray ``true`` does not masquerade as ``1``.
    """
    if raw is None:
        return default, None

    # Reject booleans explicitly: ``bool`` is a subclass of ``int`` in Python,
    # and a boolean is never a valid numeric threshold/port.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return (
            default,
            f"Configuration value for '{name}' is not a number ({raw!r}); "
            f"applying default {default}.",
        )

    value = float(raw)
    if value < low or value > high:
        return (
            default,
            f"Configuration value for '{name}' ({value}) is outside the valid "
            f"range [{low}, {high}]; applying default {default}.",
        )

    if integer:
        if float(raw) != int(raw):
            return (
                default,
                f"Configuration value for '{name}' ({raw}) must be an integer; "
                f"applying default {default}.",
            )
        return int(value), None
    return value, None


def _coerce_provider(raw: Any) -> Optional[str]:
    """Normalize a provider name: a non-empty string or ``None`` (disabled)."""
    if raw is None:
        return None
    text = raw if isinstance(raw, str) else str(raw)
    text = text.strip()
    return text or None


def _coerce_key(raw: Any) -> Optional[Secret]:
    """Wrap a raw key value in a :class:`Secret`, or ``None`` when absent/empty."""
    if raw is None:
        return None
    text = raw if isinstance(raw, str) else str(raw)
    if text == "":
        return None
    return Secret(text)


def _coerce_consent(raw: Any) -> bool:
    """Coerce the persisted consent decision; anything non-boolean -> denied."""
    return raw is True


@dataclass(frozen=True)
class ConfigLoadResult:
    """The outcome of a :meth:`ConfigurationStore.load` call.

    ``config`` is always a fully-valid ``AppConfig`` (every setting within range).
    ``errors`` collects per-setting validation messages (Req 10.3) and, on a
    whole-file read failure, a single global message (Req 10.4). An empty
    ``errors`` list means the load was clean.
    """

    config: AppConfig
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when the config loaded without any errors."""
        return not self.errors


class ConfigurationStore:
    """Loads and persists the application configuration as a JSON document.

    The backing file path is injectable so tests can target a temporary file.
    """

    def __init__(self, path: Optional[os.PathLike[str] | str] = None) -> None:
        self._path = Path(path) if path is not None else _default_config_path()

    @property
    def path(self) -> Path:
        """The backing config file path."""
        return self._path

    # -- Loading -----------------------------------------------------------

    def load(self) -> ConfigLoadResult:
        """Read, validate, and return the configuration (Req 10.1-10.4).

        A missing file is a normal first-run condition: all defaults are applied
        with no error. A file that exists but cannot be read or parsed is a read
        failure: all defaults are applied with a single global error (Req 10.4).
        Present-but-out-of-range values collapse to their defaults with a
        per-setting error (Req 10.3); absent values default silently (Req 10.2).

        API keys found in the store are registered with the redacting log filter
        so they can never leak into log output (Req 10.6).
        """
        raw, read_error = self._read_raw()
        if read_error is not None:
            cfg = default_config()
            self._register_keys(cfg)
            return ConfigLoadResult(config=cfg, errors=[read_error])
        return self._build_from_raw(raw)

    def _read_raw(self) -> tuple[dict[str, Any], Optional[str]]:
        """Return ``(raw_dict, error)``.

        ``error`` is ``None`` on success or when the file is simply absent
        (first run). It is a global message only when the file exists but is
        unreadable or malformed (Req 10.4).
        """
        if not self._path.exists():
            # First run: no persisted config yet. Not an error.
            return {}, None
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, ValueError) as exc:
            return (
                {},
                f"Configuration could not be read from '{self._path}' "
                f"({type(exc).__name__}); applying all defaults.",
            )
        if not isinstance(data, dict):
            return (
                {},
                f"Configuration in '{self._path}' is not a valid object; "
                f"applying all defaults.",
            )
        return data, None

    def has_persisted_consent(self) -> bool:
        """Return ``True`` if a consent decision has ever been persisted.

        This distinguishes a *made* decision (the backing file exists and holds
        a boolean ``consent_granted`` key) from the *first-run* condition (file
        absent, or present but with no consent key). It supports first-run
        detection for the Consent model (Req 9.1, 9.3) without changing the
        semantics of :meth:`load` or :meth:`save`, both of which continue to
        treat a missing file as a clean first run that defaults consent to
        denied.

        A file that exists but cannot be read or parsed is treated as *no*
        persisted decision, so the system falls back to the privacy-preserving
        first-run (deny + prompt) path rather than silently suppressing the
        prompt.
        """
        raw, read_error = self._read_raw()
        if read_error is not None:
            return False
        return isinstance(raw.get(_KEY_CONSENT), bool)

    def _build_from_raw(self, raw: dict[str, Any]) -> ConfigLoadResult:
        """Build a validated ``AppConfig`` from an already-parsed raw mapping.

        This is the pure core backing Property 11: for any raw mapping, the
        returned config has every setting within its documented range.
        """
        errors: list[str] = []

        proximity, err = validate_in_range(
            raw.get(_KEY_PROXIMITY),
            name=_KEY_PROXIMITY,
            low=PROXIMITY_THRESHOLD_RANGE[0],
            high=PROXIMITY_THRESHOLD_RANGE[1],
            default=DEFAULT_PROXIMITY_THRESHOLD,
        )
        if err:
            errors.append(err)

        gaze, err = validate_in_range(
            raw.get(_KEY_GAZE),
            name=_KEY_GAZE,
            low=GAZE_TOLERANCE_RANGE[0],
            high=GAZE_TOLERANCE_RANGE[1],
            default=DEFAULT_GAZE_TOLERANCE_DEG,
        )
        if err:
            errors.append(err)

        cooldown, err = validate_in_range(
            raw.get(_KEY_COOLDOWN),
            name=_KEY_COOLDOWN,
            low=COOLDOWN_PERIOD_RANGE[0],
            high=COOLDOWN_PERIOD_RANGE[1],
            default=DEFAULT_COOLDOWN_PERIOD_S,
        )
        if err:
            errors.append(err)

        port, err = validate_in_range(
            raw.get(_KEY_PORT),
            name=_KEY_PORT,
            low=SERVER_PORT_RANGE[0],
            high=SERVER_PORT_RANGE[1],
            default=DEFAULT_SERVER_PORT,
            integer=True,
        )
        if err:
            errors.append(err)

        # request_timeout_s has no documented validation range in Req 10.3; apply
        # the default when absent or non-positive, otherwise accept as given.
        timeout = self._coerce_timeout(raw.get(_KEY_TIMEOUT))

        config = AppConfig(
            detection=DetectionConfig(
                proximity_threshold=float(proximity),
                gaze_tolerance_deg=float(gaze),
            ),
            vlm_provider=_coerce_provider(raw.get(_KEY_VLM_PROVIDER)),
            tts_provider=_coerce_provider(raw.get(_KEY_TTS_PROVIDER)),
            vlm_api_key=_coerce_key(raw.get(_KEY_VLM_KEY)),
            tts_api_key=_coerce_key(raw.get(_KEY_TTS_KEY)),
            cooldown_period_s=float(cooldown),
            server_port=int(port),
            request_timeout_s=timeout,
            consent_granted=_coerce_consent(raw.get(_KEY_CONSENT)),
        )

        self._register_keys(config)

        # Log validation issues through the (redacting) logging machinery. Key
        # values are never part of these messages.
        for message in errors:
            _log.error(message)

        return ConfigLoadResult(config=config, errors=errors)

    @staticmethod
    def _coerce_timeout(raw: Any) -> float:
        if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return DEFAULT_REQUEST_TIMEOUT_S
        value = float(raw)
        return value if value > 0 else DEFAULT_REQUEST_TIMEOUT_S

    @staticmethod
    def _register_keys(config: AppConfig) -> None:
        """Register any present API keys as secrets for log redaction (Req 10.6)."""
        for key in (config.vlm_api_key, config.tts_api_key):
            if key is not None:
                register_secret(key.reveal())

    # -- Saving ------------------------------------------------------------

    def save(self, config: AppConfig) -> bool:
        """Persist ``config`` to the backing JSON file (Req 10.6-compatible).

        API keys are written to the store but are registered as secrets first so
        that nothing logged during or after save can leak them. Returns ``True``
        on success, ``False`` if the file could not be written.
        """
        self._register_keys(config)

        payload: dict[str, Any] = {
            _KEY_PROXIMITY: config.detection.proximity_threshold,
            _KEY_GAZE: config.detection.gaze_tolerance_deg,
            _KEY_VLM_PROVIDER: config.vlm_provider,
            _KEY_TTS_PROVIDER: config.tts_provider,
            _KEY_VLM_KEY: config.vlm_api_key.reveal() if config.vlm_api_key else None,
            _KEY_TTS_KEY: config.tts_api_key.reveal() if config.tts_api_key else None,
            _KEY_COOLDOWN: config.cooldown_period_s,
            _KEY_PORT: config.server_port,
            _KEY_TIMEOUT: config.request_timeout_s,
            _KEY_CONSENT: config.consent_granted,
        }

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Serialize first so a serialization error does not truncate the file.
            text = json.dumps(payload, indent=2, sort_keys=True)
            self._path.write_text(text, encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            # Never include the payload (which holds key values) in the log.
            _log.error(
                "Failed to persist configuration to '%s' (%s).",
                self._path,
                type(exc).__name__,
            )
            return False
        return True
