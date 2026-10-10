"""Client registry with per-client failure isolation for the Status_Server.

This module holds the FastAPI/uvicorn-independent core of the Status_Server: the
set of connected Status_Clients, the broadcast fan-out, and the retained latest
status for late joiners. Keeping it decoupled from FastAPI (task 7.5 wires the
real server lifecycle on top) lets the per-client failure-isolation property
(Property 10, task 7.4) be exercised with fake clients and no running socket.

Responsibilities (Requirements 5.6 / 5.7)
-----------------------------------------
- Track the connected clients (:meth:`ClientRegistry.add` /
  :meth:`ClientRegistry.remove`).
- Broadcast the serialized payload for a :class:`Status` to every connected
  client, iterating over a *snapshot* of the client set so the set may be
  mutated (clients removed) during the fan-out without corrupting iteration.
- **Per-client isolation (Req 5.7):** if an individual client's send raises, that
  client — and only that client — is removed from the connected set; every other
  client still receives the message and the broadcast continues uninterrupted.
- **Late-joiner retention (Req 5.6):** the latest broadcast status is retained so
  a newly connected client can be sent the current status immediately via
  :meth:`ClientRegistry.send_current`. Before anything has been broadcast the
  retained status defaults to :attr:`Status.SAFE` (the privacy-neutral default).

Concurrency
-----------
All mutating/reading operations are guarded by an :class:`asyncio.Lock` so the
registry is safe to use from the single asyncio I/O loop even under concurrent
``add`` / ``remove`` / ``broadcast`` coroutines. Sends themselves are performed
*outside* the lock (against a snapshot) so a slow or hanging client cannot block
registry bookkeeping for the others.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from medusa.core.models import Status
from medusa.server.payload import serialize_payload

__all__ = [
    "SupportsSend",
    "ClientRegistry",
]


@runtime_checkable
class SupportsSend(Protocol):
    """A minimal sendable abstraction for a connected Status_Client.

    Any object exposing an awaitable ``send_text(data: str)`` satisfies this
    protocol. The real FastAPI ``WebSocket`` has exactly this method, and tests
    can supply lightweight fakes (including ones that raise on send) without a
    running socket.
    """

    async def send_text(self, data: str) -> None:  # pragma: no cover - protocol
        ...


class ClientRegistry:
    """Thread-safe-on-one-loop registry of connected clients with isolated sends.

    The registry owns the connected-client set and the retained latest status. It
    never imports or depends on FastAPI, so it is fully unit- and property-
    testable in isolation.
    """

    def __init__(self, initial_status: Status = Status.SAFE) -> None:
        """Create an empty registry.

        :param initial_status: The status reported to late joiners before any
            broadcast has occurred. Defaults to :attr:`Status.SAFE`, the
            privacy-neutral default (Req 5.6).
        """
        self._clients: set[SupportsSend] = set()
        self._latest_status: Status = initial_status
        self._lock = asyncio.Lock()

    async def add(self, client: SupportsSend) -> None:
        """Register ``client`` as connected."""
        async with self._lock:
            self._clients.add(client)

    async def remove(self, client: SupportsSend) -> None:
        """Deregister ``client`` if present; a no-op if it was already removed."""
        async with self._lock:
            self._clients.discard(client)

    async def client_count(self) -> int:
        """Return the number of currently connected clients."""
        async with self._lock:
            return len(self._clients)

    def current_status(self) -> Status:
        """Return the latest broadcast status (or the initial default).

        Read without acquiring the lock: a :class:`Status` reference assignment is
        atomic for the single-loop usage model, and callers only need the most
        recent value.
        """
        return self._latest_status

    async def send_current(self, client: SupportsSend) -> bool:
        """Send the retained current status to a single ``client`` (Req 5.6).

        Used immediately after a client connects so a late joiner is consistent
        with everyone else. If the send fails the client is *not* retained (it is
        never added here; callers add on successful connect), and ``False`` is
        returned so the caller can abandon the connection.

        :returns: ``True`` if the status was sent successfully, ``False`` if the
            send raised.
        """
        data = serialize_payload(self._latest_status)
        try:
            await client.send_text(data)
        except Exception:
            return False
        return True

    async def broadcast(self, status: Status) -> list[SupportsSend]:
        """Broadcast ``status`` to every connected client with per-client isolation.

        The latest status is retained first (so a concurrent late joiner observes
        the new value), then the serialized payload is sent to a *snapshot* of the
        current client set. Iterating over a snapshot means removing failed clients
        does not disturb the fan-out, and a client connecting mid-broadcast is
        simply picked up on the next broadcast (it already got the current status
        via :meth:`send_current` on connect).

        **Per-client isolation (Req 5.7):** if a client's ``send_text`` raises, that
        client is collected as failed and removed from the connected set after the
        fan-out completes; every non-failing client still receives the message.

        :returns: The list of clients whose send failed (and were removed). Empty
            when every send succeeded or when no clients are connected (the
            no-clients case is a harmless no-op, Req 5.5).
        """
        async with self._lock:
            self._latest_status = status
            snapshot = tuple(self._clients)

        if not snapshot:
            return []

        data = serialize_payload(status)
        failed: list[SupportsSend] = []
        for client in snapshot:
            try:
                await client.send_text(data)
            except Exception:
                # Isolate the failure: record it and keep broadcasting to the rest.
                failed.append(client)

        if failed:
            async with self._lock:
                for client in failed:
                    self._clients.discard(client)

        return failed
