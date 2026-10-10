"""Desktop_Shield: Qt click-through privacy overlay.

Exposes the overlay windows (:class:`DesktopShield`) and the OS-isolating
:class:`PlatformWindow` adapter. These import cleanly even when PySide6 is not
installed; Qt is only touched when overlays are actually built/shown.
"""

from medusa.shield.overlay import (
    OVERLAY_WINDOW_FLAG_NAMES,
    TRANSLUCENT_ATTRIBUTE_NAME,
    DesktopShield,
    GridRenderer,
    opacity_pct_to_float,
)
from medusa.shield.platform_window import (
    LinuxPlatformWindow,
    MacOSPlatformWindow,
    PlatformWindow,
    WindowsPlatformWindow,
    make_platform_window,
    resolve_translucent_attribute,
    resolve_window_flags,
)

__all__ = [
    "DesktopShield",
    "GridRenderer",
    "opacity_pct_to_float",
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
