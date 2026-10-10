"""Unit tests for the alert-event cooldown gate on ClassificationBus (task 6.4).

These cover the *separate* alert-event cooldown accounting (Req 8.3 / 8.5):
a new spoken alert event waits at least the Cooldown_Period after the previous
event completes, and no new alert may start while one is in-flight. The gate is
timestamp-driven (the caller passes ``now``) so it is deterministic.

The effective-state transition cooldown FSM (task 6.1) is intentionally not
re-tested here; this file is scoped to the alert-event gate added by task 6.4.
"""

from __future__ import annotations

from medusa.core.bus import ClassificationBus
from medusa.core.models import FrameState


def _bus(cooldown: float = 2.0) -> ClassificationBus:
    return ClassificationBus(cooldown_period_s=cooldown, initial_state=FrameState.SAFE)


def test_fresh_bus_allows_alert() -> None:
    """On a fresh bus (no prior alert) a new alert may start (Req 8.3)."""
    bus = _bus()
    assert bus.may_start_alert(now=0.0) is True
    assert bus.may_start_alert(now=100.0) is True


def test_in_flight_blocks_new_alert() -> None:
    """While an alert is in-flight, may_start_alert is False (busy guard)."""
    bus = _bus()
    bus.note_alert_started(now=1.0)
    assert bus.may_start_alert(now=1.0) is False
    # Even far in the future, an in-flight alert blocks a new start.
    assert bus.may_start_alert(now=1_000.0) is False


def test_blocked_until_cooldown_elapses_after_completion() -> None:
    """After completion, must wait the full Cooldown_Period (Req 8.3 / 8.5)."""
    bus = _bus(cooldown=2.0)
    bus.note_alert_started(now=1.0)
    bus.note_alert_completed(now=3.0)

    # Immediately after completion: still cooling down.
    assert bus.may_start_alert(now=3.0) is False
    # Partway through the cooldown window: still blocked.
    assert bus.may_start_alert(now=4.5) is False
    # Just before the boundary: still blocked.
    assert bus.may_start_alert(now=4.999) is False


def test_allowed_at_and_after_cooldown_boundary() -> None:
    """At exactly completion + cooldown (and after), a new alert is allowed."""
    bus = _bus(cooldown=2.0)
    bus.note_alert_started(now=1.0)
    bus.note_alert_completed(now=3.0)

    # Boundary is inclusive: elapsed >= cooldown allows starting.
    assert bus.may_start_alert(now=5.0) is True
    assert bus.may_start_alert(now=10.0) is True


def test_second_event_cooldown_measured_from_its_own_completion() -> None:
    """Cooldown is measured from the completion of the most recent event."""
    bus = _bus(cooldown=2.0)

    # First event.
    bus.note_alert_started(now=0.0)
    bus.note_alert_completed(now=1.0)
    assert bus.may_start_alert(now=3.0) is True

    # Second event starts and completes later; cooldown re-arms from its end.
    bus.note_alert_started(now=3.0)
    assert bus.may_start_alert(now=3.5) is False  # in-flight
    bus.note_alert_completed(now=4.0)
    assert bus.may_start_alert(now=5.0) is False  # within cooldown of 2nd end
    assert bus.may_start_alert(now=6.0) is True   # cooldown elapsed


def test_separate_alert_cooldown_period_param() -> None:
    """A distinct alert_cooldown_period_s overrides the state cooldown."""
    bus = ClassificationBus(
        cooldown_period_s=2.0,
        initial_state=FrameState.SAFE,
        alert_cooldown_period_s=5.0,
    )
    bus.note_alert_started(now=0.0)
    bus.note_alert_completed(now=0.0)
    assert bus.may_start_alert(now=2.0) is False   # would pass under 2s gate
    assert bus.may_start_alert(now=4.999) is False
    assert bus.may_start_alert(now=5.0) is True


def test_alert_gate_independent_of_state_fsm() -> None:
    """Driving the effective-state FSM does not affect the alert gate."""
    bus = _bus(cooldown=2.0)
    # Publish state updates; alert gate should still permit the first alert.
    bus.update(FrameState.THREAT, timestamp=1.0)
    bus.update(FrameState.SAFE, timestamp=10.0)
    assert bus.may_start_alert(now=10.0) is True
