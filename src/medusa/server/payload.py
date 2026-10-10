"""Broadcast payload construction for the Status_Server.

This module holds the pure, I/O-free logic that turns a :class:`Status` into the
exact message broadcast to Status_Clients over the WebSocket. Keeping it separate
from the FastAPI/uvicorn wiring (task 7.5) lets the broadcast-validity property
(Property 9, task 7.2) be exercised without a running socket.

Message contract (Requirements 5.2 / 5.3)
-----------------------------------------
The payload is *exactly*:

- ``{"status": "THREAT_DETECTED"}`` for a threat, and
- ``{"status": "SAFE"}`` for a safe state,

with no additional fields and no other values. :func:`build_payload` constructs
a fresh ``dict`` with a single ``"status"`` key whose value is drawn solely from
the :class:`Status` enum, so no other key or value can ever appear.
:func:`serialize_payload` renders that dict to the exact JSON string sent on the
wire using a canonical, compact, deterministic encoding.
"""

from __future__ import annotations

import json

from medusa.core.models import Status

__all__ = [
    "STATUS_KEY",
    "build_payload",
    "serialize_payload",
]


# The one and only key permitted in a broadcast payload.
STATUS_KEY = "status"


def build_payload(status: Status) -> dict:
    """Return the exact broadcast payload for ``status``.

    The result is always a single-key mapping ``{"status": <value>}`` where
    ``<value>`` is ``"THREAT_DETECTED"`` for :attr:`Status.THREAT_DETECTED` and
    ``"SAFE"`` for :attr:`Status.SAFE` (Requirements 5.2 / 5.3). A fresh dict is
    built on each call so callers can never mutate shared state, and the only key
    is :data:`STATUS_KEY`, so no additional fields can leak in.

    :param status: The classified privacy status to encode.
    :raises TypeError: If ``status`` is not a :class:`Status` member.
    """
    if not isinstance(status, Status):
        raise TypeError(f"build_payload expects a Status, got {type(status)!r}")
    # Value is sourced directly from the enum, so it is always one of the two
    # permitted strings and never an arbitrary value.
    return {STATUS_KEY: status.value}


def serialize_payload(status: Status) -> str:
    """Return the exact JSON string broadcast over the WebSocket for ``status``.

    Uses a compact, key-sorted encoding so the wire format is deterministic and
    free of incidental whitespace: ``{"status": "THREAT_DETECTED"}`` or
    ``{"status": "SAFE"}``.

    :param status: The classified privacy status to encode.
    """
    return json.dumps(build_payload(status), separators=(",", ":"), sort_keys=True)
