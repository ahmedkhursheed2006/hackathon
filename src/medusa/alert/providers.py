"""Alert_Service provider abstraction: VLM/TTS protocols, factories, registry.

This module implements the "Provider Abstraction Layer" from the design's
Alert_Service section. The Alert_Service turns a *consented* threat frame into a
brief, neutral spoken alert by sending the single triggered frame to a hosted
vision-language model (VLM) and the returned description to a hosted
text-to-speech (TTS) service. Both services are **hosted and pluggable**: no
provider is hard-coded as a default (Open Question 4, Req 10.5).

What this module provides
-------------------------
* :data:`NEUTRAL_PROMPT` -- the single, informational, non-identity-targeting
  prompt sent to the VLM (Req 7.2). It asks for a brief, factual, one-sentence
  description and explicitly forbids demeaning, harassing, or identity-targeting
  content.
* :class:`VLMProvider` / :class:`TTSProvider` -- structural ``Protocol`` types
  describing the async adapter surface the Alert_Service depends on. Any object
  with a matching ``describe`` / ``synthesize`` coroutine satisfies them; there
  is no required base class to inherit.
* :func:`register_vlm_provider` / :func:`register_tts_provider` -- register an
  adapter *factory* under a provider name so new hosted providers can be plugged
  in without modifying this module.
* :func:`make_vlm_provider` / :func:`make_tts_provider` -- resolve a provider by
  name from configuration. They return ``None`` -- disabling the capability --
  when the name is absent/unknown **or** the key is absent (Req 10.5). No
  provider is baked in, so the registry ships empty by default.

Capability-disabled semantics (Req 10.5)
-----------------------------------------
A factory returns ``None`` (capability disabled) when *any* of the following
holds, so the Alert_Service can simply skip the corresponding API call:

* the provider ``name`` is ``None`` or empty/whitespace,
* no adapter is registered under ``name``, or
* the ``api_key`` is ``None`` or empty/whitespace.

The caller (Alert_Service, task 10.3) is responsible for logging a
"credential missing" message *without* the key value when a key is absent; this
module never logs key values.

Secret handling
---------------
``api_key`` is typed ``Optional[str]`` per the task contract. Callers that hold
a :class:`~medusa.core.models.Secret` must reveal it (``secret.reveal()``) right
before invoking the factory; the Alert_Service does this at the point of use.
This module treats the key only as an opaque presence/absence signal and passes
it to the registered adapter factory -- it does not log or otherwise expose it.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol, runtime_checkable

__all__ = [
    "NEUTRAL_PROMPT",
    "VLMProvider",
    "TTSProvider",
    "VLMProviderFactory",
    "TTSProviderFactory",
    "register_vlm_provider",
    "register_tts_provider",
    "unregister_vlm_provider",
    "unregister_tts_provider",
    "available_vlm_providers",
    "available_tts_providers",
    "make_vlm_provider",
    "make_tts_provider",
]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neutral VLM prompt (Req 7.2)
# ---------------------------------------------------------------------------
#
# A single, informational, non-identity-targeting instruction. It requests a
# brief, factual, one-sentence description and explicitly forbids demeaning,
# harassing, or identity-targeting content. This is the exact prompt from the
# design's Alert_Service section.
NEUTRAL_PROMPT = (
    "In one short, neutral, factual sentence, describe the person who has appeared "
    "behind the primary user (e.g., approximate position and that they appear to be "
    "looking toward the screen). Do not speculate about identity, demographics, or "
    "intent, and do not use demeaning or judgmental language."
)


# ---------------------------------------------------------------------------
# Provider protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class VLMProvider(Protocol):
    """A hosted vision-language model adapter.

    Implementations send ``image`` to the hosted VLM together with ``prompt``
    (the Alert_Service always passes :data:`NEUTRAL_PROMPT`) and return a short
    textual description. The call must honor ``timeout_s`` as a per-request
    timeout (Req 11.1). The protocol is structural: any object exposing a
    matching ``describe`` coroutine satisfies it without inheritance.
    """

    async def describe(self, image: bytes, prompt: str, timeout_s: float) -> str:
        """Return a brief, neutral description of ``image`` for ``prompt``."""
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """A hosted text-to-speech adapter.

    Implementations synthesize ``text`` into audio bytes via the hosted TTS
    service, honoring ``timeout_s`` as a per-request timeout (Req 11.1). The
    protocol is structural: any object exposing a matching ``synthesize``
    coroutine satisfies it without inheritance.
    """

    async def synthesize(self, text: str, timeout_s: float) -> bytes:
        """Return synthesized audio bytes for ``text``."""
        ...


# A provider *factory* builds an adapter instance from the resolved API key. The
# key is guaranteed non-empty by the time the factory is invoked.
VLMProviderFactory = Callable[[str], VLMProvider]
TTSProviderFactory = Callable[[str], TTSProvider]


# ---------------------------------------------------------------------------
# Pluggable registries
# ---------------------------------------------------------------------------
#
# The registries ship *empty*: no provider is hard-coded (Req 10.5, Open
# Question 4). Hosted provider adapters register themselves (or are registered
# by the app wiring) under a stable name, keeping this module free of any
# baked-in default.
_vlm_registry: dict[str, VLMProviderFactory] = {}
_tts_registry: dict[str, TTSProviderFactory] = {}


def _normalize(name: Optional[str]) -> Optional[str]:
    """Return a trimmed provider name, or ``None`` if absent/blank.

    Registry lookups are performed on the normalized, case-folded name so
    provider names are matched case-insensitively and whitespace-insensitively.
    """
    if name is None:
        return None
    trimmed = name.strip()
    return trimmed or None


def register_vlm_provider(name: str, factory: VLMProviderFactory) -> None:
    """Register a VLM adapter ``factory`` under ``name`` (case-insensitive).

    Registering the same name again replaces the previous factory. Raises
    ``ValueError`` if ``name`` is empty/blank.
    """
    key = _normalize(name)
    if key is None:
        raise ValueError("provider name must be a non-empty string")
    _vlm_registry[key.casefold()] = factory


def register_tts_provider(name: str, factory: TTSProviderFactory) -> None:
    """Register a TTS adapter ``factory`` under ``name`` (case-insensitive).

    Registering the same name again replaces the previous factory. Raises
    ``ValueError`` if ``name`` is empty/blank.
    """
    key = _normalize(name)
    if key is None:
        raise ValueError("provider name must be a non-empty string")
    _tts_registry[key.casefold()] = factory


def unregister_vlm_provider(name: str) -> None:
    """Remove a registered VLM factory; a no-op if ``name`` is not registered."""
    key = _normalize(name)
    if key is not None:
        _vlm_registry.pop(key.casefold(), None)


def unregister_tts_provider(name: str) -> None:
    """Remove a registered TTS factory; a no-op if ``name`` is not registered."""
    key = _normalize(name)
    if key is not None:
        _tts_registry.pop(key.casefold(), None)


def available_vlm_providers() -> tuple[str, ...]:
    """Return the registered VLM provider names (normalized, sorted)."""
    return tuple(sorted(_vlm_registry))


def available_tts_providers() -> tuple[str, ...]:
    """Return the registered TTS provider names (normalized, sorted)."""
    return tuple(sorted(_tts_registry))


# ---------------------------------------------------------------------------
# Factories (Req 10.5): resolve by name + key, else disable the capability
# ---------------------------------------------------------------------------
def make_vlm_provider(name: Optional[str], api_key: Optional[str]) -> Optional[VLMProvider]:
    """Resolve a VLM provider by ``name`` using ``api_key`` (Req 10.5).

    Returns ``None`` -- meaning the VLM capability is disabled -- when the
    ``name`` is absent/blank, the ``api_key`` is absent/blank, or no adapter is
    registered under ``name``. Otherwise returns the adapter produced by the
    registered factory. No provider is hard-coded, so an empty registry always
    yields ``None``.

    This function never logs the key value; the Alert_Service is responsible for
    emitting a "credential missing" message (without the value) when it observes
    a disabled capability due to a missing key.
    """
    return _make(name, api_key, _vlm_registry, kind="VLM")


def make_tts_provider(name: Optional[str], api_key: Optional[str]) -> Optional[TTSProvider]:
    """Resolve a TTS provider by ``name`` using ``api_key`` (Req 10.5).

    Returns ``None`` -- meaning the TTS capability is disabled -- when the
    ``name`` is absent/blank, the ``api_key`` is absent/blank, or no adapter is
    registered under ``name``. Otherwise returns the adapter produced by the
    registered factory. No provider is hard-coded, so an empty registry always
    yields ``None``.
    """
    return _make(name, api_key, _tts_registry, kind="TTS")


def _make(
    name: Optional[str],
    api_key: Optional[str],
    registry: dict[str, Callable[[str], object]],
    *,
    kind: str,
) -> Optional[object]:
    """Shared resolution logic for both factories.

    Order of checks (all paths disable the capability by returning ``None``):

    1. name absent/blank -> disabled (no provider selected),
    2. key absent/blank  -> disabled (credential missing, Req 10.5),
    3. name not registered -> disabled (unknown provider).
    """
    normalized = _normalize(name)
    if normalized is None:
        # No provider selected: capability disabled by configuration.
        return None

    # A key that is None or only whitespace is treated as absent (Req 10.5).
    if api_key is None or api_key.strip() == "":
        _log.info(
            "%s provider '%s' is configured but its credential is missing; "
            "the capability is disabled.",
            kind,
            normalized,
        )
        return None

    factory = registry.get(normalized.casefold())
    if factory is None:
        _log.info(
            "%s provider '%s' is not registered; the capability is disabled.",
            kind,
            normalized,
        )
        return None

    return factory(api_key)
