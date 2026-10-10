"""PlatformWindow adapter: isolates OS-specific overlay quirks.

The Desktop_Shield needs a borderless, always-on-top, click-through overlay on
every connected monitor. The *mechanics* of "always-on-top" and "click-through"
differ by operating system, so this module hides those differences behind a
small :class:`PlatformWindow` adapter abstraction (design: "A ``PlatformWindow``
adapter isolates OS-specific always-on-top/click-through quirks").

Design decision #1 in the design document selects **Windows 10/11 as the primary
target**. The cross-platform behaviour is summarised there and repeated in the
adapter docstrings below so the limitations travel with the code:

* **Windows (primary):** Qt's ``WindowStaysOnTopHint`` +
  ``WindowTransparentForInput`` reliably yields an always-on-top, output-only
  (click-through) overlay. No extra native work is required.
* **macOS (stub):** reliable click-through + always-on-top typically needs the
  native ``NSWindow`` to set ``ignoresMouseEvents`` and a "screen saver" window
  level, and the overlay can still be constrained by Spaces / full-screen apps.
  The stub documents this and applies only the portable Qt hints.
* **Linux (stub):** always-on-top and click-through depend on the compositor
  (X11 vs Wayland). Wayland in particular restricts always-on-top and global
  overlay placement. The stub documents this and applies only the portable Qt
  hints.

Importability without Qt
-------------------------
PySide6 may be unavailable in a given environment (e.g. a Python version without
a published wheel). This module therefore performs **no** top-level PySide6
import. The exact set of window flags and the translucent-background attribute
are expressed as *names* (strings) in :data:`OVERLAY_WINDOW_FLAG_NAMES` and
:data:`TRANSLUCENT_ATTRIBUTE_NAME`, and resolved against Qt lazily only when a
real window is actually configured. This lets the smoke tests (task 9.3) assert
the intended flag/attribute composition even where PySide6 cannot be imported,
while the real adapter still applies genuine ``Qt`` enum values at runtime.
"""

from __future__ import annotations

import platform
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "OVERLAY_WINDOW_FLAG_NAMES",
    "TRANSLUCENT_ATTRIBUTE_NAME",
    "PlatformWindow",
    "WindowsPlatformWindow",
    "MacOSPlatformWindow",
    "LinuxPlatformWindow",
    "make_platform_window",
    "resolve_window_flags",
    "resolve_translucent_attribute",
]


# The window flags that compose a borderless, always-on-top, output-only overlay
# (Req 6.1, 6.2). Named here (not imported) so the composition is assertable
# without PySide6 present. Resolved against ``Qt.WindowType`` at runtime.
#
# - FramelessWindowHint      -> borderless (no title bar / frame)
# - WindowStaysOnTopHint     -> always-on-top
# - WindowTransparentForInput-> output-only; all input passes through (Req 6.2)
OVERLAY_WINDOW_FLAG_NAMES: tuple[str, ...] = (
    "FramelessWindowHint",
    "WindowStaysOnTopHint",
    "WindowTransparentForInput",
)

# The widget attribute that makes the window background translucent so the
# overlay composites over the desktop and so mouse input is not captured by an
# opaque surface (design: "plus a translucent background attribute"). Resolved
# against ``Qt.WidgetAttribute`` at runtime.
TRANSLUCENT_ATTRIBUTE_NAME: str = "WA_TranslucentBackground"


def resolve_window_flags(qt_namespace: Any) -> Any:
    """OR together the overlay window flags from a ``Qt`` namespace.

    ``qt_namespace`` is normally ``PySide6.QtCore.Qt``. Each name in
    :data:`OVERLAY_WINDOW_FLAG_NAMES` is looked up as either a ``Qt.WindowType``
    member or a direct ``Qt`` attribute (PySide6 exposes both styles across
    versions) and bitwise-OR'd into a single combined flag value.

    Kept as a free function so it can be unit-tested with a lightweight stand-in
    namespace (an object exposing the same attribute names) when PySide6 is not
    installed.
    """
    window_type = getattr(qt_namespace, "WindowType", qt_namespace)
    combined: Any = None
    for name in OVERLAY_WINDOW_FLAG_NAMES:
        flag = getattr(window_type, name, None)
        if flag is None:
            flag = getattr(qt_namespace, name)
        combined = flag if combined is None else (combined | flag)
    return combined


def resolve_translucent_attribute(qt_namespace: Any) -> Any:
    """Resolve the translucent-background widget attribute from a ``Qt`` namespace.

    Looks up :data:`TRANSLUCENT_ATTRIBUTE_NAME` as a ``Qt.WidgetAttribute``
    member or a direct ``Qt`` attribute, matching PySide6's dual exposure.
    """
    widget_attribute = getattr(qt_namespace, "WidgetAttribute", qt_namespace)
    attr = getattr(widget_attribute, TRANSLUCENT_ATTRIBUTE_NAME, None)
    if attr is None:
        attr = getattr(qt_namespace, TRANSLUCENT_ATTRIBUTE_NAME)
    return attr


@runtime_checkable
class PlatformWindow(Protocol):
    """Adapter that applies OS-appropriate overlay behaviour to a Qt widget.

    Implementations isolate the per-OS mechanics of achieving an always-on-top,
    click-through, borderless overlay. The portable Qt hints are applied by all
    implementations; platform-specific implementations may additionally perform
    native work (documented per subclass).
    """

    #: Human-readable platform name, for diagnostics/logging.
    name: str

    def configure(self, widget: Any) -> None:
        """Apply window flags, the translucent attribute, and any native quirks.

        ``widget`` is a ``QWidget`` (typed as ``Any`` so this module imports
        cleanly without PySide6). Implementations must leave the widget
        borderless, always-on-top, and click-through.
        """
        ...

    def supports_click_through(self) -> bool:
        """Whether reliable click-through is expected on this platform."""
        ...

    def limitations(self) -> str:
        """A short, human-readable description of known platform limitations."""
        ...


class _BaseQtPlatformWindow:
    """Shared implementation applying the portable Qt hints to a widget.

    All concrete adapters build on this: it resolves and applies the overlay
    window flags and the translucent-background attribute using lazy imports so
    the module stays importable without PySide6. Subclasses override
    :meth:`_apply_native` to add OS-specific behaviour and :meth:`limitations`
    to document platform caveats.
    """

    name: str = "generic"

    def configure(self, widget: Any) -> None:
        # Lazy import: only required when actually configuring a real window.
        from PySide6.QtCore import Qt  # noqa: PLC0415 (intentional lazy import)

        widget.setWindowFlags(resolve_window_flags(Qt))
        widget.setAttribute(resolve_translucent_attribute(Qt), True)
        self._apply_native(widget)

    def _apply_native(self, widget: Any) -> None:
        """Hook for OS-specific native configuration. Default: no-op."""

    def supports_click_through(self) -> bool:
        return True

    def limitations(self) -> str:
        return "No known limitations recorded for this platform."


class WindowsPlatformWindow(_BaseQtPlatformWindow):
    """Primary-target adapter for Windows 10/11.

    On Windows, the Qt hints ``WindowStaysOnTopHint`` and
    ``WindowTransparentForInput`` reliably produce an always-on-top, output-only
    overlay with no extra native calls, so :meth:`_apply_native` is a no-op.

    Limitations: a window owned by another process running *as administrator*
    (higher integrity level) can render above a non-elevated overlay; this is a
    Windows security boundary and is accepted for v1.
    """

    name = "windows"

    def supports_click_through(self) -> bool:
        return True

    def limitations(self) -> str:
        return (
            "Windows (primary target): always-on-top + click-through are "
            "reliable via Qt hints. A higher-integrity (elevated) window from "
            "another process can still paint above a non-elevated overlay."
        )


class MacOSPlatformWindow(_BaseQtPlatformWindow):
    """Stub adapter for macOS, with documented limitations.

    The portable Qt hints are applied, but fully reliable click-through +
    always-on-top on macOS generally requires native ``NSWindow`` work that this
    stub does **not** perform: setting ``ignoresMouseEvents = YES`` and raising
    the window to the "screen saver" window level. Even then, the overlay can be
    constrained by Spaces and by other applications' full-screen (native
    full-screen) spaces, which may obscure or bypass the overlay.
    """

    name = "macos"

    def supports_click_through(self) -> bool:
        # The portable hint is applied, but native reinforcement is absent.
        return False

    def limitations(self) -> str:
        return (
            "macOS (stub): reliable click-through/always-on-top needs native "
            "NSWindow ignoresMouseEvents + screen-saver window level, which is "
            "not implemented. Spaces and native full-screen apps can constrain "
            "or bypass the overlay."
        )


class LinuxPlatformWindow(_BaseQtPlatformWindow):
    """Stub adapter for Linux, with documented limitations.

    The portable Qt hints are applied, but actual always-on-top and
    click-through behaviour is compositor-dependent. On X11 the hints usually
    work; on Wayland, always-on-top and global overlay placement are commonly
    restricted by the compositor's security model, so the overlay may not stay
    above all windows or span every output as intended. No compositor-specific
    native work is implemented in this stub.
    """

    name = "linux"

    def supports_click_through(self) -> bool:
        return False

    def limitations(self) -> str:
        return (
            "Linux (stub): behaviour depends on the compositor. X11 generally "
            "honours the Qt hints; Wayland commonly restricts always-on-top and "
            "global overlay placement. No compositor-specific work is done."
        )


def make_platform_window(system: str | None = None) -> PlatformWindow:
    """Return the :class:`PlatformWindow` adapter for the given OS.

    Args:
        system: An OS name as returned by :func:`platform.system`
            (``"Windows"``, ``"Darwin"``, ``"Linux"``). When ``None``, the
            current host OS is detected. Unknown systems fall back to the Linux
            stub (the most conservative portable-hints-only behaviour).

    Returns:
        The matching platform adapter. Windows is the primary implementation;
        macOS and Linux are documented stubs.
    """
    name = (system or platform.system()).strip().lower()
    if name in ("windows", "win32", "nt"):
        return WindowsPlatformWindow()
    if name in ("darwin", "macos", "mac"):
        return MacOSPlatformWindow()
    # Default to the Linux stub for Linux and any unrecognised platform.
    return LinuxPlatformWindow()
