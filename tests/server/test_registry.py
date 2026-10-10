"""Verification tests for the Status_Server client registry (task 7.3).

These exercise the per-client failure isolation and late-joiner retention
behavior of :class:`medusa.server.registry.ClientRegistry` using fake clients
(no running socket). The formal Property 10 test lives in task 7.4; these are
targeted example/unit tests confirming the implementation (Req 5.6 / 5.7).
"""

from __future__ import annotations

import pytest

from medusa.core.models import Status
from medusa.server.payload import serialize_payload
from medusa.server.registry import ClientRegistry


class FakeClient:
    """A fake Status_Client that records sent payloads."""

    def __init__(self) -> None:
        self.received: list[str] = []

    async def send_text(self, data: str) -> None:
        self.received.append(data)


class FailingClient:
    """A fake client whose ``send_text`` always raises."""

    def __init__(self) -> None:
        self.attempts = 0

    async def send_text(self, data: str) -> None:
        self.attempts += 1
        raise ConnectionError("send failed")


async def test_broadcast_delivers_to_all_connected_clients():
    reg = ClientRegistry()
    a, b, c = FakeClient(), FakeClient(), FakeClient()
    for client in (a, b, c):
        await reg.add(client)

    failed = await reg.broadcast(Status.THREAT_DETECTED)

    expected = serialize_payload(Status.THREAT_DETECTED)
    assert failed == []
    assert a.received == [expected]
    assert b.received == [expected]
    assert c.received == [expected]


async def test_failed_send_removes_only_that_client_others_still_receive():
    reg = ClientRegistry()
    good1, bad, good2 = FakeClient(), FailingClient(), FakeClient()
    await reg.add(good1)
    await reg.add(bad)
    await reg.add(good2)

    failed = await reg.broadcast(Status.THREAT_DETECTED)

    expected = serialize_payload(Status.THREAT_DETECTED)
    # Only the failing client is reported and removed (Req 5.7).
    assert failed == [bad]
    assert await reg.client_count() == 2
    # Non-failing clients still received the message.
    assert good1.received == [expected]
    assert good2.received == [expected]

    # A subsequent broadcast no longer targets the removed client.
    bad.attempts = 0
    await reg.broadcast(Status.SAFE)
    assert bad.attempts == 0
    assert good1.received[-1] == serialize_payload(Status.SAFE)
    assert good2.received[-1] == serialize_payload(Status.SAFE)


async def test_current_status_defaults_to_safe_before_any_broadcast():
    reg = ClientRegistry()
    assert reg.current_status() is Status.SAFE


async def test_broadcast_retains_latest_status_for_late_joiners():
    reg = ClientRegistry()
    await reg.broadcast(Status.THREAT_DETECTED)
    assert reg.current_status() is Status.THREAT_DETECTED

    # A late joiner is sent the retained current status (Req 5.6).
    late = FakeClient()
    ok = await reg.send_current(late)
    assert ok is True
    assert late.received == [serialize_payload(Status.THREAT_DETECTED)]


async def test_send_current_reports_failure_without_raising():
    reg = ClientRegistry()
    bad = FailingClient()
    ok = await reg.send_current(bad)
    assert ok is False


async def test_no_clients_broadcast_is_noop():
    reg = ClientRegistry()
    failed = await reg.broadcast(Status.SAFE)
    assert failed == []
    assert reg.current_status() is Status.SAFE


async def test_remove_is_idempotent():
    reg = ClientRegistry()
    a = FakeClient()
    await reg.add(a)
    await reg.remove(a)
    await reg.remove(a)  # no error on second removal
    assert await reg.client_count() == 0


async def test_all_clients_failing_removes_all():
    reg = ClientRegistry()
    bad1, bad2 = FailingClient(), FailingClient()
    await reg.add(bad1)
    await reg.add(bad2)

    failed = await reg.broadcast(Status.THREAT_DETECTED)

    assert set(failed) == {bad1, bad2}
    assert await reg.client_count() == 0
