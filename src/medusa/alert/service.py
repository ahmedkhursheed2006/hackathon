"""Alert_Service: the async VLM -> TTS spoken-alert pipeline.

This module implements the Alert_Service component from the design (section
"4. Alert_Service"). It turns a *consented* threat into a brief, neutral spoken
alert by sending the single triggered frame to a hosted vision-language model
(VLM) and the returned description to a hosted text-to-speech (TTS) service,
then playing the synthesized audio.

The whole pipeline runs on the asyncio event loop and must never block the
vision loop (Req 7.9, 11.3). It is driven by a single coroutine,
:meth:`AlertService.on_threat`, which the application wiring (task 13.1)
schedules on the I/O loop whenever the ClassificationBus observes a threat.

Gating pipeline (:meth:`AlertService.on_threat`)
------------------------------------------------
Each threat is subjected, in order, to a sequence of gates. If any gate rejects
the event, the pipeline returns early *without* ever blocking overlay/broadcast:

1. **Busy guard (Req 7.8):** if an alert is already playing/in-flight, the new
   event is *discarded* (never queued) and the call returns immediately.
2. **Cooldown gate (Req 8.3, 8.5, 11.4):** if the alert gate reports that a new
   alert may not start yet (previous event too recent), the event is skipped.
   This enforces at-most-once per completed event plus the inter-event cooldown.
3. **Consent gate (Req 7.5, 9.2, 9.5):** consent is re-checked *immediately*
   before transmission so an in-flight revocation prevents a send that had not
   yet begun. If consent is not granted, the VLM/TTS calls are skipped entirely.
4. **Credential gate (Req 7.1, 10.5):** if the VLM provider is absent (``None``)
   the event logs a "credential missing" notice (never the key value) and skips.

Once every gate passes, the event is marked started on the gate and ``busy`` is
set, then:

5. **VLM (Req 7.1, 7.4, 11.1):** the single triggered frame plus the neutral
   prompt are sent to the VLM under a per-request timeout (default 10s). Only
   the one frame is ever sent -- never a continuous stream. On timeout/failure
   the failure is logged (no key value), the event is marked failed, no retry is
   attempted within the cooldown, and the pipeline aborts (Req 7.6, 11.2).
6. **TTS (Req 7.7, 11.1):** if a TTS provider is present, the description is
   synthesized under the same per-request timeout. On timeout/failure the event
   completes *without* audio. If no TTS provider is configured, audio is skipped.
7. **Playback:** the synthesized audio is played via the injected async player.

Regardless of outcome, the event is *always* marked completed on the gate and
``busy`` is cleared in a ``finally`` block so cooldown accounting stays correct
even on failure (Req 8.3, 11.4).

Decoupling / testability
------------------------
The service depends only on small injected collaborators -- the provider
protocols, a consent check, an *alert gate* abstraction, an async audio player,
and a monotonic ``now`` clock -- so it can be unit- and property-tested with
fakes without a camera, network, or audio device. The alert gate is defined as a
structural :class:`AlertGate` protocol (``may_start_alert`` /
``note_alert_started`` / ``note_alert_completed``); the application wiring passes
the :class:`~medusa.core.bus.ClassificationBus`, which implements it.

Secret handling
---------------
This module never logs API key values. The credential gate observes only the
*presence/absence* of a provider (``None`` means the capability is disabled by a
missing name or key, per :mod:`medusa.alert.providers`).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, Protocol, runtime_checkable

from medusa.alert.providers import NEUTRAL_PROMPT, TTSProvider, VLMProvider

__all__ = [
    "AlertGate",
    "ConsentCheck",
    "AudioPlayer",
    "NowFn",
    "AlertOutcome",
    "AlertService",
    "DEFAULT_REQUEST_TIMEOUT_S",
]

_log = logging.getLogger(__name__)

# Default per-request timeout for both the VLM and TTS calls (Req 11.1).
DEFAULT_REQUEST_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Injected collaborator types
# ---------------------------------------------------------------------------
@runtime_checkable
class AlertGate(Protocol):
    """Alert-event cooldown / accounting gate (Req 8.3, 8.5, 11.4).

    This is the small abstraction the Alert_Service depends on for cooldown and
    at-most-once accounting. The :class:`~medusa.core.bus.ClassificationBus`
    implements this surface (task 6.4); tests pass a fake. Keeping it structural
    decouples the service from the bus module entirely.

    * :meth:`may_start_alert` -- may a new alert begin at ``now``? ``False`` while
      within the inter-event cooldown or while an event is already in flight.
    * :meth:`note_alert_started` -- record that an event has begun at ``now``.
    * :meth:`note_alert_completed` -- record that an event finished at ``now``;
      the inter-event cooldown is measured from this moment (Req 8.3).
    """

    def may_start_alert(self, now: float) -> bool: ...

    def note_alert_started(self, now: float) -> None: ...

    def note_alert_completed(self, now: float) -> None: ...


# A consent check is any zero-arg callable returning the *current* consent state
# (re-read freshly each call to honor revocation, Req 9.5). ConsentModel.is_granted
# satisfies this directly.
ConsentCheck = Callable[[], bool]

# An async audio player consumes synthesized audio bytes and completes when
# playback finishes. Injected so tests can use a fake.
AudioPlayer = Callable[[bytes], Awaitable[None]]

# A monotonic clock used for cooldown timestamps (default ``time.monotonic``).
NowFn = Callable[[], float]


class AlertOutcome:
    """String constants describing why :meth:`AlertService.on_threat` returned.

    Returned (via the service's internal bookkeeping and surfaced in logs) to
    make the gating decisions observable and testable without inspecting logs.
    """

    DISCARDED_BUSY = "discarded_busy"
    SKIPPED_COOLDOWN = "skipped_cooldown"
    SKIPPED_NO_CONSENT = "skipped_no_consent"
    SKIPPED_NO_CREDENTIAL = "skipped_no_credential"
    FAILED_VLM = "failed_vlm"
    COMPLETED_NO_AUDIO = "completed_no_audio"
    COMPLETED = "completed"


class AlertService:
    """Async pipeline turning a consented threat into a neutral spoken alert.

    See the module docstring for the full gating pipeline and requirement
    mapping. All collaborators are injected so the service is testable with
    fakes.

    Parameters
    ----------
    vlm:
        The resolved VLM provider, or ``None`` when the capability is disabled
        (missing provider name or key). ``None`` makes every threat skip at the
        credential gate (Req 7.1, 10.5).
    tts:
        The resolved TTS provider, or ``None`` to complete events without audio
        (Req 7.7 fallback / disabled capability).
    consent:
        Either an object exposing ``is_granted() -> bool`` (e.g.
        :class:`~medusa.config.consent.ConsentModel`) or a zero-arg callable
        returning the current consent state. Re-checked immediately before
        transmission to honor revocation (Req 9.2, 9.5).
    gate:
        The :class:`AlertGate` (typically the ClassificationBus) used for
        cooldown and at-most-once accounting (Req 8.3, 8.5, 11.4).
    audio_player:
        Async callable that plays audio bytes; injected for testing.
    request_timeout_s:
        Per-request timeout applied to *both* the VLM and TTS calls (Req 11.1,
        default 10.0 seconds).
    now:
        Monotonic clock callable used for cooldown timestamps (default
        ``time.monotonic``). ``on_threat`` also accepts an explicit ``now`` to
        make tests deterministic.
    """

    def __init__(
        self,
        vlm: Optional[VLMProvider],
        tts: Optional[TTSProvider],
        consent: "object | ConsentCheck",
        gate: AlertGate,
        audio_player: AudioPlayer,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        now: NowFn = time.monotonic,
    ) -> None:
        self._vlm = vlm
        self._tts = tts
        self._is_granted = self._coerce_consent(consent)
        self._gate = gate
        self._play = audio_player
        self._timeout = float(request_timeout_s)
        self._now = now

        # Busy guard (Req 7.8): True while an event is playing/in-flight. New
        # calls are *discarded* (not queued) while this is set.
        self._busy = False

    # ------------------------------------------------------------------ #
    # Public state
    # ------------------------------------------------------------------ #
    @property
    def busy(self) -> bool:
        """``True`` while an alert event is in flight (busy guard, Req 7.8)."""
        return self._busy

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    async def on_threat(
        self, frame: bytes, event_id: str, now: Optional[float] = None
    ) -> str:
        """Run the gated VLM -> TTS -> playback pipeline for one threat event.

        Non-blocking and fire-and-forget: the vision loop is never awaited here.
        Returns an :class:`AlertOutcome` string describing the decision/result,
        which is handy for tests and logging; callers may ignore it.

        See the module docstring for the ordered gating semantics.
        """
        ts = self._now() if now is None else float(now)

        # 1. Busy guard (Req 7.8): discard, do not queue.
        if self._busy:
            _log.debug("Alert event %s discarded: already playing/in-flight.", event_id)
            return AlertOutcome.DISCARDED_BUSY

        # 2. Cooldown gate (Req 8.3, 8.5, 11.4): previous event too recent.
        if not self._gate.may_start_alert(ts):
            _log.debug("Alert event %s skipped: within alert cooldown.", event_id)
            return AlertOutcome.SKIPPED_COOLDOWN

        # 3. Consent gate (Req 7.5, 9.2, 9.5): re-check immediately before send.
        if not self._is_granted():
            _log.info(
                "Alert event %s skipped: consent for frame transmission not granted.",
                event_id,
            )
            return AlertOutcome.SKIPPED_NO_CONSENT

        # 4. Credential gate (Req 7.1, 10.5): no VLM provider -> skip, no key logged.
        if self._vlm is None:
            _log.info(
                "Alert event %s skipped: VLM credential missing; capability disabled.",
                event_id,
            )
            return AlertOutcome.SKIPPED_NO_CREDENTIAL

        # All gates passed: mark started and set busy. The ``finally`` block
        # guarantees completion accounting even on failure (Req 8.3, 11.4).
        self._gate.note_alert_started(ts)
        self._busy = True
        try:
            # 5. VLM: single triggered frame + neutral prompt, per-request
            #    timeout (Req 7.1, 7.4, 11.1). Never a stream.
            try:
                description = await asyncio.wait_for(
                    self._vlm.describe(frame, NEUTRAL_PROMPT, self._timeout),
                    self._timeout,
                )
            except Exception as exc:  # noqa: BLE001  (TimeoutError is an Exception)
                # Timeout or failure: log (no key), abort, mark failed, no retry
                # within cooldown; overlay/broadcast continue (Req 7.6, 11.2).
                _log.warning(
                    "Alert event %s: VLM request failed or timed out (%s); "
                    "skipping TTS and marking event failed.",
                    event_id,
                    type(exc).__name__,
                )
                return AlertOutcome.FAILED_VLM

            # 6. TTS: optional. If no provider, complete without audio.
            if self._tts is None:
                _log.info(
                    "Alert event %s: no TTS provider configured; "
                    "completing without audio.",
                    event_id,
                )
                return AlertOutcome.COMPLETED_NO_AUDIO

            try:
                audio = await asyncio.wait_for(
                    self._tts.synthesize(description, self._timeout),
                    self._timeout,
                )
            except Exception as exc:  # noqa: BLE001  (TimeoutError is an Exception)
                # TTS timeout/failure: log and complete event without audio
                # (Req 7.7); overlay/broadcast continue.
                _log.warning(
                    "Alert event %s: TTS request failed or timed out (%s); "
                    "completing without audio.",
                    event_id,
                    type(exc).__name__,
                )
                return AlertOutcome.COMPLETED_NO_AUDIO

            # 7. Playback via the injected player.
            await self._play(audio)
            return AlertOutcome.COMPLETED
        finally:
            # Always record completion and clear busy so cooldown accounting is
            # correct even on failure/exception (Req 8.3, 11.4).
            self._gate.note_alert_completed(self._now() if now is None else float(now))
            self._busy = False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_consent(consent: "object | ConsentCheck") -> ConsentCheck:
        """Return a zero-arg consent callable from either form of ``consent``.

        Accepts an object exposing ``is_granted()`` (e.g. ``ConsentModel``) or a
        plain callable. The returned callable is invoked fresh on every gate
        check so revocation is honored just-in-time (Req 9.5).
        """
        is_granted = getattr(consent, "is_granted", None)
        if callable(is_granted):
            return is_granted  # type: ignore[return-value]
        if callable(consent):
            return consent  # type: ignore[return-value]
        raise TypeError(
            "consent must expose is_granted() or be a zero-arg callable "
            "returning bool"
        )
