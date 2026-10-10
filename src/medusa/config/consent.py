"""Consent model: deny-by-default, persisted consent for frame transmission.

This module implements the cross-cutting Consent model described in the design
(section "Cross-cutting: Consent Model"). It governs whether the single
triggered frame may be transmitted to the configured third-party VLM_API.

Responsibilities (Requirement 9)
--------------------------------
- **Deny-by-default (Req 9.1):** until the user explicitly grants consent, frame
  transmission is treated as denied.
- **First-run detection (Req 9.1, 9.3):** when *no* consent decision has ever
  been persisted, a prompt is required; once a decision exists, the prompt is
  suppressed on subsequent runs.
- **Persistence (Req 9.3):** a submitted decision is persisted to the
  Configuration_Store so later runs can read it back.
- **Persist-failure fallback (Req 9.4):** if persisting the decision fails, the
  decision is treated as denied and a "decision not saved" indication is
  surfaced to the caller.
- **Just-in-time re-check / revocation (Req 9.2, 9.5):** :meth:`ConsentModel.is_granted`
  reads the current decision freshly from the store so that an in-flight
  revocation is honored at the moment transmission is about to begin.

Design rationale
----------------
The backing :class:`~medusa.config.store.ConfigurationStore` persists
``consent_granted`` as a boolean but, by itself, cannot distinguish "no decision
made yet" (first run) from "decision = denied" because both surface as
``False`` after :meth:`~medusa.config.store.ConfigurationStore.load`. This model
layers first-run detection on top via
:meth:`~medusa.config.store.ConfigurationStore.has_persisted_consent`, which
checks whether a boolean consent key is actually present in the backing store.

The model never caches the decision: ``is_granted`` and ``needs_prompt`` always
consult the store so a concurrent revocation (written by another part of the
system) is reflected immediately. This directly supports the just-in-time
re-check required for revocation (Req 9.5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from medusa.config.store import ConfigurationStore

__all__ = [
    "ConsentDecision",
    "ConsentModel",
]

_log = logging.getLogger(__name__)

# User-facing text surfaced when persisting the consent decision fails (Req 9.4).
DECISION_NOT_SAVED_MESSAGE = (
    "Your consent decision could not be saved. Frame transmission will remain "
    "disabled, and you may be asked again."
)


@dataclass(frozen=True)
class ConsentDecision:
    """The outcome of a :meth:`ConsentModel.set_consent` call.

    - ``granted`` is the *effective* consent state after the call. On a persist
      failure this is forced to ``False`` regardless of the requested value,
      because an unsaved decision is treated as denied (Req 9.4).
    - ``saved`` is ``True`` when the decision was persisted successfully.
    - ``message`` is a user-facing indication; it is populated with a
      "decision not saved" notice when ``saved`` is ``False`` (Req 9.4), and is
      ``None`` on success.
    """

    granted: bool
    saved: bool
    message: str | None = None

    def __bool__(self) -> bool:
        """Truthy iff the decision was persisted successfully.

        Lets callers treat the result like the ``bool`` returned by
        :meth:`ConsentModel.set_consent` (``False`` signals a persist failure).
        """
        return self.saved


class ConsentModel:
    """Deny-by-default, persisted consent gate for frame transmission (Req 9).

    Wraps a :class:`~medusa.config.store.ConfigurationStore`. All queries read
    freshly from the store so revocation is honored just-in-time (Req 9.5).
    """

    def __init__(self, store: ConfigurationStore) -> None:
        self._store = store

    # -- First-run / prompt gating (Req 9.1, 9.3) --------------------------

    def is_first_run(self) -> bool:
        """``True`` when no consent decision has ever been persisted (Req 9.1, 9.3).

        A first run (no persisted decision, or an unreadable store) means the
        consent request must be presented and transmission stays denied until
        the user explicitly grants it.
        """
        return not self._store.has_persisted_consent()

    def needs_prompt(self) -> bool:
        """``True`` iff the consent request should be presented (Req 9.1, 9.3).

        The prompt is required on the first run (no persisted decision) and is
        suppressed on every subsequent run once any decision exists. This is an
        alias of :meth:`is_first_run` expressed in prompt-centric terms for
        call-site clarity.
        """
        return self.is_first_run()

    # -- Current state / just-in-time re-check (Req 9.2, 9.5) --------------

    def is_granted(self) -> bool:
        """Return the current consent state, deny-by-default (Req 9.2, 9.5).

        Reads freshly from the Configuration_Store on every call so that an
        in-flight revocation is honored at the moment of this check — the Alert_
        Service calls this immediately before transmission begins. Any ambiguity
        (unreadable store, absent decision) resolves to denied.
        """
        result = self._store.load()
        return bool(result.config.consent_granted)

    # -- Recording a decision (Req 9.3, 9.4) -------------------------------

    def set_consent(self, granted: bool) -> ConsentDecision:
        """Persist the user's consent decision (Req 9.3, 9.4).

        Loads the current config, updates ``consent_granted``, and persists it.
        On a persist failure the decision is treated as denied and a
        "decision not saved" indication is surfaced (Req 9.4).

        Returns a :class:`ConsentDecision`. The returned object is falsy on a
        persist failure (``bool(decision) is False``), so callers that only care
        about success can use it directly in a boolean context.
        """
        result = self._store.load()
        updated = replace(result.config, consent_granted=granted)

        saved = self._store.save(updated)
        if not saved:
            # Persist failure: treat as denied and surface the "not saved"
            # indication. We do not log the requested value as sensitive, but a
            # consent decision is not a secret, so a plain warning is fine.
            _log.warning(
                "Failed to persist consent decision; treating frame "
                "transmission as denied."
            )
            return ConsentDecision(
                granted=False,
                saved=False,
                message=DECISION_NOT_SAVED_MESSAGE,
            )

        return ConsentDecision(granted=granted, saved=True, message=None)
