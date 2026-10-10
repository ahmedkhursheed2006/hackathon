"""The ClassificationBus: a debounce / cooldown finite state machine.

The :class:`ClassificationBus` is the single arbiter of *effective* privacy
state for Screen Privacy Guardian (Medusa_Desktop_Core). Per-frame
:class:`~medusa.core.models.Classification` events are produced by the vision
loop and fed to the bus; the bus applies a debounce/cooldown FSM so that every
downstream consumer (Desktop_Shield opacity, Status_Server broadcast, and the
Alert_Service trigger) observes the *same* stabilized state and timing.

Design reference: "Debounce / Cooldown State Machine (ClassificationBus)" in
``design.md``.

Semantics
---------
Internal effective state is either ``FrameState.SAFE`` or ``FrameState.THREAT``.
On entering a state the bus arms a cooldown of ``cooldown_period_s`` seconds.

* An *opposite* transition (SAFE <-> THREAT) is **suppressed** if it arrives
  within ``cooldown_period_s`` of the last transition (Req 8.1, 8.2). Same-state
  inputs simply refresh / stay.
* ``FrameState.UNCLASSIFIABLE`` never changes the effective state and never
  settles to safe; instead it latches a "hold opaque" flag that keeps the
  effective opacity at 100% until a subsequent *settled* Safe_State clears it
  (Req 6.6).
* After a Threat_State, the overlay remains 100% opaque for the duration of the
  post-threat cooldown window (Req 8.4); opacity reaches 0% only once the state
  has settled to Safe (threat cooldown elapsed and no opaque hold is active)
  (Req 6.4, 6.5, 8.4).

Determinism / testability
-------------------------
The bus is a pure function of the sequence of inputs: each input carries its own
timestamp (``classification.timestamp``) and the FSM derives all timing from
those timestamps alone. No wall-clock is read. This makes the FSM deterministic
and directly property-testable (design Properties 7 and 8).

Alert-event cooldown accounting (Req 8.3 / 8.5) is a *separate* gate added by
task 6.4; this module intentionally leaves room for those methods and keeps the
effective-state FSM self-contained.
"""

from __future__ import annotations

from typing import Optional

from medusa.core.models import Classification, FrameState

__all__ = ["ClassificationBus"]

# Default debounce/cooldown window in seconds (Req 8.1; configurable 0.5-10s).
DEFAULT_COOLDOWN_PERIOD_S = 2.0

# The two possible *effective* (stabilized) states. UNCLASSIFIABLE is never an
# effective state; it only latches the opaque hold.
_SETTLED_STATES = (FrameState.SAFE, FrameState.THREAT)


class ClassificationBus:
    """Debounce/cooldown FSM that stabilizes per-frame classifications.

    Parameters
    ----------
    cooldown_period_s:
        Minimum time a state must be held before an *opposite* transition is
        applied, in seconds (Req 8.1, default 2.0). Range 0.5-10s is enforced by
        the Configuration_Store on load; the bus accepts whatever it is given.
    initial_state:
        The effective state the bus starts in. Defaults to ``SAFE`` (the FSM's
        start state in the design state diagram).
    alert_cooldown_period_s:
        Minimum time that must elapse after a spoken alert *event* completes
        before a new alert event may start (Req 8.3 / 8.5). Defaults to
        ``cooldown_period_s`` -- the design uses the same Cooldown_Period for the
        alert-event gate -- unless a distinct value is explicitly supplied.
    """

    def __init__(
        self,
        cooldown_period_s: float = DEFAULT_COOLDOWN_PERIOD_S,
        initial_state: FrameState = FrameState.SAFE,
        alert_cooldown_period_s: Optional[float] = None,
    ) -> None:
        if initial_state not in _SETTLED_STATES:
            raise ValueError(
                "initial_state must be FrameState.SAFE or FrameState.THREAT"
            )
        self._cooldown = float(cooldown_period_s)
        self._state: FrameState = initial_state

        # Timestamp of the most recent *effective* transition. ``None`` means no
        # transition has occurred yet (fresh bus / initial state).
        self._last_transition_ts: Optional[float] = None

        # Timestamp of the most recent input observed. Used as the reference
        # "now" for evaluating the post-threat cooldown window in a way that is
        # a pure function of the input sequence.
        self._last_input_ts: Optional[float] = None

        # Timestamp at which the effective state last entered SAFE *from* THREAT.
        # Drives the post-threat cooldown opacity window (Req 8.4). ``None`` when
        # the current SAFE state was not reached from THREAT (e.g. initial SAFE).
        self._safe_entered_from_threat_ts: Optional[float] = None

        # Latches when an UNCLASSIFIABLE frame is observed; keeps opacity 100%
        # until a subsequent settled Safe_State clears it (Req 6.6).
        self._holding_opaque: bool = False

        # --- Alert-event cooldown gate (Req 8.3 / 8.5), task 6.4 -------------
        # This is a SEPARATE cooldown accounting, independent of the effective
        # state-transition cooldown above. It gates spoken alert *events*: a new
        # alert waits at least ``cooldown_period_s`` after the previous event
        # *completes* (Req 8.3), suppressing repeats within that window (Req 8.5).
        self._alert_cooldown = (
            self._cooldown
            if alert_cooldown_period_s is None
            else float(alert_cooldown_period_s)
        )

        # True while a spoken alert event is in-flight (started, not yet
        # completed). While in-flight, no new alert may start (busy-guard
        # accounting; the Alert_Service also enforces its own busy guard).
        self._alert_in_flight: bool = False

        # Timestamp at which the most recent alert event *completed*. ``None``
        # means no alert has completed yet (fresh bus allows starting). The
        # cooldown is measured from completion of the previous event (Req 8.3).
        self._last_alert_completed_ts: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    def publish(self, classification: Classification) -> None:
        """Feed a per-frame classification into the FSM.

        Latest-wins: consumers read :attr:`effective_state` /
        :meth:`effective_opacity` after one or more publishes; raw per-frame
        classifications are not buffered.
        """
        self.update(classification.state, classification.timestamp)

    def update(self, state: FrameState, timestamp: float) -> None:
        """Feed a classification outcome and its timestamp into the FSM.

        This is the timestamp-driven core; :meth:`publish` delegates here so the
        bus can be driven either from full :class:`Classification` objects or
        from a bare ``(state, timestamp)`` pair in tests.
        """
        ts = float(timestamp)
        self._last_input_ts = ts

        if state is FrameState.UNCLASSIFIABLE:
            # Hold opaque; does not change effective state nor settle to safe.
            self._holding_opaque = True
            return

        if state is self._state:
            # Same-state input: stay. A Safe input while already SAFE is a
            # settled Safe_State and clears any opaque hold (Req 6.6).
            if state is FrameState.SAFE:
                self._holding_opaque = False
            return

        # Opposite transition requested (SAFE<->THREAT). Suppress it if we are
        # still within the cooldown window of the last transition (Req 8.1/8.2).
        if self._within_cooldown(ts):
            return

        # Apply the transition.
        previous = self._state
        self._state = state
        self._last_transition_ts = ts

        if state is FrameState.SAFE:
            # Entered SAFE from THREAT: arm the post-threat opacity window and
            # clear any opaque hold (we have reached a settled Safe_State).
            self._safe_entered_from_threat_ts = ts if previous is FrameState.THREAT else None
            self._holding_opaque = False
        else:
            # Entered THREAT: no post-threat window applies while in THREAT.
            self._safe_entered_from_threat_ts = None

    # ------------------------------------------------------------------ #
    # Consumer-facing effective state
    # ------------------------------------------------------------------ #
    @property
    def effective_state(self) -> FrameState:
        """The current stabilized state (latest-wins coalescing for consumers).

        Always ``FrameState.SAFE`` or ``FrameState.THREAT`` -- never
        ``UNCLASSIFIABLE`` (which only latches the opaque hold).
        """
        return self._state

    def effective_opacity(self) -> int:
        """Return the overlay opacity the shield must apply: 100 or 0.

        Opacity is 100% when (Req 6.4, 6.5, 6.6, 8.4):

        * the effective state is Threat_State, OR
        * an UNCLASSIFIABLE frame is being held opaque (not yet cleared by a
          settled Safe_State), OR
        * the system is still within the post-threat cooldown window.

        Opacity is 0% exactly when the effective state is a *settled* Safe_State
        (threat cooldown elapsed and no opaque hold active).
        """
        if self._state is FrameState.THREAT:
            return 100
        if self._holding_opaque:
            return 100
        if self._in_post_threat_cooldown():
            return 100
        return 0

    # ------------------------------------------------------------------ #
    # Alert-event cooldown gate (Req 8.3 / 8.5)
    # ------------------------------------------------------------------ #
    def may_start_alert(self, now: float) -> bool:
        """Return True iff a new spoken alert event may start at ``now``.

        This is the decision point for "may start alert now" (task 6.4). It is a
        *separate* gate from the effective-state transition cooldown and is a
        pure function of the alert lifecycle timestamps -- the caller supplies
        ``now`` so the result is deterministic and property-testable (Property 6,
        tested in task 10.5).

        A new alert may start only when **both** hold:

        * no alert event is currently in-flight (busy-guard accounting; a new
          event does not start while a prior one is playing), **and**
        * at least ``alert_cooldown_period_s`` has elapsed since the previous
          alert event *completed* (Req 8.3 / 8.5).

        On a fresh bus (no prior alert event has completed and none is in
        flight) this returns ``True`` -- the first alert is always allowed.
        """
        if self._alert_in_flight:
            return False
        if self._last_alert_completed_ts is None:
            # No prior event has completed: nothing to wait on.
            return True
        return (float(now) - self._last_alert_completed_ts) >= self._alert_cooldown

    def note_alert_started(self, now: float) -> None:
        """Record that a spoken alert event has started at ``now``.

        Marks an alert as in-flight so that :meth:`may_start_alert` returns
        ``False`` until the event completes. ``now`` is accepted for a uniform,
        timestamp-driven interface and future accounting even though the in-flight
        flag alone suppresses overlapping starts.
        """
        self._alert_in_flight = True

    def note_alert_completed(self, now: float) -> None:
        """Record that the in-flight spoken alert event completed at ``now``.

        Clears the in-flight flag and stamps the completion time so the
        alert-event cooldown is measured from the *completion* of this event
        (Req 8.3). After this, :meth:`may_start_alert` stays ``False`` until
        ``alert_cooldown_period_s`` has elapsed past ``now`` (Req 8.5).
        """
        self._alert_in_flight = False
        self._last_alert_completed_ts = float(now)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _within_cooldown(self, ts: float) -> bool:
        """True if ``ts`` falls within the cooldown window of the last transition."""
        if self._last_transition_ts is None:
            return False
        return (ts - self._last_transition_ts) < self._cooldown

    def _in_post_threat_cooldown(self) -> bool:
        """True while within the post-threat cooldown window after leaving THREAT.

        Evaluated against the most recent input timestamp so the result is a
        pure function of the observed input sequence (Req 8.4).
        """
        if self._safe_entered_from_threat_ts is None:
            return False
        if self._last_input_ts is None:
            return False
        return (self._last_input_ts - self._safe_entered_from_threat_ts) < self._cooldown
