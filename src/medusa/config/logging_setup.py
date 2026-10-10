"""Secret-redacting logging support for the Configuration_Store.

This module provides a redacting :class:`logging.Filter` together with a small
module-level registry of secret values (e.g. VLM/TTS API keys). Any value
registered as a secret is masked verbatim in all log records that pass through
a logger carrying the filter.

Design notes (supports Property 12 - "API keys never appear in logs", Req 10.6):

- Secrets are held in a module-level registry so they can be registered once at
  config load time and apply to every logger the filter is installed on.
- :class:`RedactingFilter` rewrites both the pre-formatted ``record.msg`` and
  the positional ``record.args`` so that a secret cannot leak through either the
  format string or an interpolated argument.
- Redaction is performed on a snapshot of the registry taken per record, so the
  filter is safe to use while secrets are being registered/unregistered.
- Empty or ``None`` secrets are never registered: masking the empty string would
  corrupt every message, and there is nothing sensitive to hide.

The functions here are deliberately pure/testable: :func:`redact_text` performs
the actual substitution and can be exercised directly, while the filter and
setup helpers wire it into the standard logging machinery.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

__all__ = [
    "REDACTION_PLACEHOLDER",
    "RedactingFilter",
    "register_secret",
    "unregister_secret",
    "clear_secrets",
    "registered_secrets",
    "redact_text",
    "install_redacting_filter",
    "setup_logging",
]

#: The literal text that replaces any registered secret in a log record.
REDACTION_PLACEHOLDER = "***REDACTED***"

# Module-level registry of secret values. Guarded by ``_lock`` so that
# registration from config load and redaction from a logging thread are safe to
# interleave.
_secrets: set[str] = set()
_lock = threading.RLock()


def register_secret(value: object) -> bool:
    """Register ``value`` as a secret to be redacted from log output.

    ``None`` and empty strings are ignored (masking the empty string would
    corrupt every message). Non-string values are coerced to ``str``.

    Returns ``True`` if a secret was registered, ``False`` otherwise.
    """
    if value is None:
        return False
    text = value if isinstance(value, str) else str(value)
    if text == "":
        return False
    with _lock:
        _secrets.add(text)
    return True


def unregister_secret(value: object) -> bool:
    """Remove ``value`` from the secret registry.

    Returns ``True`` if the value was present and removed, ``False`` otherwise.
    """
    if value is None:
        return False
    text = value if isinstance(value, str) else str(value)
    if text == "":
        return False
    with _lock:
        if text in _secrets:
            _secrets.discard(text)
            return True
    return False


def clear_secrets() -> None:
    """Remove all registered secrets. Primarily useful for tests/teardown."""
    with _lock:
        _secrets.clear()


def registered_secrets() -> frozenset[str]:
    """Return an immutable snapshot of the currently registered secrets."""
    with _lock:
        return frozenset(_secrets)


def redact_text(text: str, secrets: Iterable[str] | None = None) -> str:
    """Replace every registered secret found in ``text`` with the placeholder.

    If ``secrets`` is ``None`` the current registry snapshot is used. Longer
    secrets are replaced first so that a secret which contains a shorter secret
    as a substring is still fully masked.
    """
    if not isinstance(text, str) or text == "":
        return text
    values = registered_secrets() if secrets is None else set(secrets)
    if not values:
        return text
    # Replace longest-first to avoid a shorter secret partially consuming a
    # longer one before it can be matched.
    for secret in sorted(values, key=len, reverse=True):
        if secret and secret in text:
            text = text.replace(secret, REDACTION_PLACEHOLDER)
    return text


class RedactingFilter(logging.Filter):
    """A :class:`logging.Filter` that masks registered secrets in log records.

    The filter never drops records; it only rewrites them. Both ``record.msg``
    and ``record.args`` are sanitized so a secret cannot leak via the format
    string or any interpolated argument.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - stdlib name
        secrets = registered_secrets()
        if not secrets:
            return True

        # Redact the message/format string itself.
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg, secrets)

        # Redact positional args, which are interpolated into the final message.
        args = record.args
        if args:
            if isinstance(args, dict):
                record.args = {
                    key: self._redact_value(value, secrets)
                    for key, value in args.items()
                }
            elif isinstance(args, tuple):
                record.args = tuple(
                    self._redact_value(value, secrets) for value in args
                )
        return True

    @staticmethod
    def _redact_value(value: object, secrets: Iterable[str]) -> object:
        """Redact a single arg value, preserving non-string values as-is unless
        their string form contains a secret."""
        if isinstance(value, str):
            return redact_text(value, secrets)
        # Coerce to check embedded secrets (e.g. a secret inside a container's
        # repr). Only substitute when a secret is actually present to avoid
        # changing the type of ordinary values.
        text = str(value)
        redacted = redact_text(text, secrets)
        return redacted if redacted != text else value


def install_redacting_filter(logger: logging.Logger | None = None) -> RedactingFilter:
    """Install a :class:`RedactingFilter` on ``logger`` (root logger by default).

    The filter is attached to the logger and to each of its handlers so that
    records are redacted regardless of where formatting happens. A filter is not
    duplicated if one is already present. Returns the installed (or existing)
    filter.
    """
    target = logger if logger is not None else logging.getLogger()

    existing = _find_filter(target.filters)
    redactor = existing if existing is not None else RedactingFilter()
    if existing is None:
        target.addFilter(redactor)

    # Also attach to handlers: handler-level filtering guarantees masking even
    # when records propagate from child loggers directly to handlers.
    for handler in target.handlers:
        if _find_filter(handler.filters) is None:
            handler.addFilter(redactor)

    return redactor


def _find_filter(filters: Iterable[logging.Filter]) -> RedactingFilter | None:
    for flt in filters:
        if isinstance(flt, RedactingFilter):
            return flt
    return None


def setup_logging(
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
) -> RedactingFilter:
    """Configure application logging with secret redaction enabled.

    Ensures the target logger (root by default) has at least one handler, sets
    the level, and installs the redacting filter on the logger and its handlers.
    Returns the installed :class:`RedactingFilter` for convenience.
    """
    target = logger if logger is not None else logging.getLogger()
    target.setLevel(level)
    if not target.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        target.addHandler(handler)
    return install_redacting_filter(target)
