"""Unit tests for the Alert_Service provider abstraction layer.

Covers the provider protocols, the pluggable registry, the resolution factories
(:func:`make_vlm_provider` / :func:`make_tts_provider`), and the neutral prompt
constant from ``medusa.alert.providers``.

These tests verify the task 10.1 contract:

* factories return ``None`` when the provider name *or* key is absent
  (capability disabled, Req 10.5),
* no provider is hard-coded (an empty registry yields ``None``),
* a registered dummy provider resolves to an adapter instance, and
* ``NEUTRAL_PROMPT`` exists and is neutral/non-identity-targeting (Req 7.2).
"""

from __future__ import annotations

import asyncio

import pytest

from medusa.alert.providers import (
    NEUTRAL_PROMPT,
    TTSProvider,
    VLMProvider,
    available_tts_providers,
    available_vlm_providers,
    make_tts_provider,
    make_vlm_provider,
    register_tts_provider,
    register_vlm_provider,
    unregister_tts_provider,
    unregister_vlm_provider,
)


# --- Test doubles ---------------------------------------------------------
class DummyVLM:
    """A minimal structural ``VLMProvider`` used to verify resolution."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def describe(self, image: bytes, prompt: str, timeout_s: float) -> str:
        return f"saw {len(image)} bytes with prompt len {len(prompt)}"


class DummyTTS:
    """A minimal structural ``TTSProvider`` used to verify resolution."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def synthesize(self, text: str, timeout_s: float) -> bytes:
        return text.encode("utf-8")


@pytest.fixture
def clean_registry():
    """Register dummy providers for the test, then remove them afterward."""
    register_vlm_provider("dummy", lambda key: DummyVLM(key))
    register_tts_provider("dummy", lambda key: DummyTTS(key))
    try:
        yield
    finally:
        unregister_vlm_provider("dummy")
        unregister_tts_provider("dummy")


# --- NEUTRAL_PROMPT (Req 7.2) --------------------------------------------
def test_neutral_prompt_exists_and_is_a_nonempty_string():
    assert isinstance(NEUTRAL_PROMPT, str)
    assert NEUTRAL_PROMPT.strip() != ""


def test_neutral_prompt_requests_neutral_factual_content():
    lowered = NEUTRAL_PROMPT.lower()
    # It explicitly asks for neutral/factual content...
    assert "neutral" in lowered
    assert "factual" in lowered
    # ...and explicitly forbids identity-targeting / demeaning content (Req 7.2).
    assert "identity" in lowered
    assert "demeaning" in lowered


def test_neutral_prompt_has_no_demeaning_or_identity_targeting_terms():
    lowered = NEUTRAL_PROMPT.lower()
    forbidden = ["race", "gender", "ugly", "suspicious", "criminal", "threatening"]
    for term in forbidden:
        assert term not in lowered, f"prompt should not contain demeaning term {term!r}"


# --- No provider hard-coded ----------------------------------------------
def test_no_provider_hardcoded_unknown_name_disables_capability():
    # Even with a key present, an unregistered name yields None (nothing baked in).
    assert make_vlm_provider("nonexistent-provider", "key-123") is None
    assert make_tts_provider("nonexistent-provider", "key-123") is None


# --- Factories return None when name is absent (Req 10.5) -----------------
@pytest.mark.parametrize("name", [None, "", "   "])
def test_make_vlm_provider_returns_none_when_name_absent(name, clean_registry):
    assert make_vlm_provider(name, "key-123") is None


@pytest.mark.parametrize("name", [None, "", "   "])
def test_make_tts_provider_returns_none_when_name_absent(name, clean_registry):
    assert make_tts_provider(name, "key-123") is None


# --- Factories return None when key is absent (Req 10.5) ------------------
@pytest.mark.parametrize("key", [None, "", "   "])
def test_make_vlm_provider_returns_none_when_key_absent(key, clean_registry):
    assert make_vlm_provider("dummy", key) is None


@pytest.mark.parametrize("key", [None, "", "   "])
def test_make_tts_provider_returns_none_when_key_absent(key, clean_registry):
    assert make_tts_provider("dummy", key) is None


# --- A registered dummy provider resolves --------------------------------
def test_registered_vlm_provider_resolves(clean_registry):
    provider = make_vlm_provider("dummy", "key-abc")
    assert provider is not None
    assert isinstance(provider, DummyVLM)
    assert provider.api_key == "key-abc"
    # Structural protocol conformance.
    assert isinstance(provider, VLMProvider)


def test_registered_tts_provider_resolves(clean_registry):
    provider = make_tts_provider("dummy", "key-abc")
    assert provider is not None
    assert isinstance(provider, DummyTTS)
    assert provider.api_key == "key-abc"
    assert isinstance(provider, TTSProvider)


def test_provider_name_resolution_is_case_insensitive(clean_registry):
    assert make_vlm_provider("DUMMY", "k") is not None
    assert make_vlm_provider("  Dummy  ", "k") is not None


def test_available_providers_lists_registered_names(clean_registry):
    assert "dummy" in available_vlm_providers()
    assert "dummy" in available_tts_providers()


# --- Resolved adapters are usable coroutines ------------------------------
def test_resolved_vlm_adapter_describe_is_awaitable(clean_registry):
    provider = make_vlm_provider("dummy", "k")
    assert provider is not None
    result = asyncio.run(provider.describe(b"abc", NEUTRAL_PROMPT, 10.0))
    assert "saw 3 bytes" in result


def test_resolved_tts_adapter_synthesize_is_awaitable(clean_registry):
    provider = make_tts_provider("dummy", "k")
    assert provider is not None
    audio = asyncio.run(provider.synthesize("hello", 10.0))
    assert audio == b"hello"
