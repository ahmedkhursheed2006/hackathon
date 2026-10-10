"""Core data models and enums for Screen Privacy Guardian (Medusa_Desktop_Core).

This module defines the pure, immutable data structures and enumerations shared
across the vision, server, shield, alert, and config subsystems. All structures
are frozen dataclasses so they are hashable, comparable, and safe to pass across
the vision / Qt / asyncio thread boundaries without accidental mutation.

Secret handling
---------------
``AppConfig.vlm_api_key`` and ``AppConfig.tts_api_key`` hold third-party API
credentials. Requirement 10.6 mandates that key values never appear in log
output. To make leakage hard by construction, those fields are typed as
``Secret`` wrappers whose ``__repr__`` / ``__str__`` never expose the underlying
value. The raw value is only accessible via the explicit
:meth:`Secret.reveal` accessor, which forces call sites to opt in deliberately.

The redacting ``logging.Filter`` (implemented alongside the Configuration_Store)
provides defense in depth for any value that is revealed and then logged, but
the ``Secret`` wrapper ensures that merely ``repr``-ing or interpolating an
``AppConfig`` can never leak a key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

__all__ = [
    "Secret",
    "FrameState",
    "Status",
    "BoundingBox",
    "Face",
    "GazeAngles",
    "FrameMeta",
    "Classification",
    "DetectionConfig",
    "AppConfig",
]


# The placeholder shown anywhere a secret would otherwise be rendered.
_SECRET_PLACEHOLDER = "***REDACTED***"


@dataclass(frozen=True)
class Secret:
    """An immutable wrapper around a sensitive string (e.g. an API key).

    The wrapped value is deliberately kept out of ``__repr__`` and ``__str__`` so
    that logging, f-strings, exception messages, and ``repr(AppConfig(...))`` can
    never leak the credential verbatim (Requirement 10.6). Access the raw value
    explicitly via :meth:`reveal`.
    """

    # Stored under a private field; access is funneled through ``reveal``.
    _value: str = field(repr=False)

    def reveal(self) -> str:
        """Return the raw secret value. Use only at the point of transmission."""
        return self._value

    def __repr__(self) -> str:  # pragma: no cover - trivial, but security-critical
        return f"Secret({_SECRET_PLACEHOLDER})"

    def __str__(self) -> str:  # pragma: no cover - trivial, but security-critical
        return _SECRET_PLACEHOLDER


class FrameState(Enum):
    """The classification outcome for a single analyzed frame."""

    THREAT = "THREAT"
    SAFE = "SAFE"
    UNCLASSIFIABLE = "UNCLASSIFIABLE"


class Status(Enum):
    """The privacy status broadcast to Status_Clients over the WebSocket."""

    THREAT_DETECTED = "THREAT_DETECTED"
    SAFE = "SAFE"

    def message(self) -> dict:
        """Return the exact broadcast payload for this status (Req 5.2/5.3)."""
        return {"status": self.value}


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned face bounding box using a top-left origin."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        """Non-negative area; degenerate (negative) dimensions clamp to 0.0."""
        return max(0.0, self.w) * max(0.0, self.h)


@dataclass(frozen=True)
class Face:
    """A detected face: its bounding box plus an immutable landmark snapshot."""

    box: BoundingBox
    # MediaPipe FaceMesh landmarks captured as an immutable tuple.
    landmarks: tuple = ()


@dataclass(frozen=True)
class GazeAngles:
    """Head-pose gaze angles, in degrees, derived via solvePnP/Rodrigues."""

    yaw_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class FrameMeta:
    """Per-frame metadata handed to the pure classification core.

    ``faces`` is the pre-ordering input; :func:`order_faces` applies the
    deterministic ordering rule before role assignment.
    """

    width: int
    height: int
    faces: tuple[Face, ...]
    timestamp: float


@dataclass(frozen=True)
class Classification:
    """The deterministic result of classifying a single frame."""

    state: FrameState
    proximity_fraction: Optional[float]  # in [0.0, 1.0] or None
    gaze: Optional[GazeAngles]
    gaze_evaluated: bool
    timestamp: float


@dataclass(frozen=True)
class DetectionConfig:
    """Tunable detection thresholds (range-validated on config load)."""

    proximity_threshold: float = 0.12  # [0.0, 1.0]
    gaze_tolerance_deg: float = 15.0  # [0.0, 90.0]


@dataclass(frozen=True)
class AppConfig:
    """The complete, validated application configuration.

    ``vlm_api_key`` and ``tts_api_key`` are :class:`Secret`-wrapped so their
    values never leak via ``repr``/``str`` (Requirement 10.6). They are optional
    because a provider/key may be absent, which disables the corresponding Alert_
    Service capability (Requirement 10.5).
    """

    detection: DetectionConfig
    vlm_provider: Optional[str]
    tts_provider: Optional[str]
    vlm_api_key: Optional[Secret]  # secret: never logged
    tts_api_key: Optional[Secret]  # secret: never logged
    cooldown_period_s: float = 2.0  # [0.5, 10.0]
    server_port: int = 8080  # [1, 65535]
    request_timeout_s: float = 10.0
    consent_granted: bool = False  # deny-by-default
