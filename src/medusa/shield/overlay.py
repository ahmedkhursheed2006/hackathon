"""Desktop_Shield overlay windows (Qt click-through privacy overlay).

This module implements the overlay-window side of the Desktop_Shield: one
borderless, always-on-top, click-through overlay per connected monitor, created
at 0% opacity (Req 6.1, 6.2). The OS-specific mechanics of always-on-top and
click-through are delegated to a :class:`~medusa.shield.platform_window.PlatformWindow`
adapter so this module stays free of per-OS branching.

Scope (task 9.1)
----------------
This task covers window construction, flags/attributes, per-monitor placement,
and opacity plumbing. The moiré interference grid rendering and the thread-safe,
coalescing opacity control arrive in task 9.2; this module deliberately leaves
seams for them:

* A :class:`GridRenderer` protocol is defined and an :class:`OverlayWindow`
  accepts an optional renderer, which its ``paintEvent`` invokes. Task 9.2
  supplies the concrete 1px moiré :class:`MoireGridRenderer` (geometry delegated
  to the pure :func:`moire_grid_lines` helper) and attaches it by default.
* :meth:`OverlayWindow.set_opacity` provides the per-window 0/100 opacity
  plumbing; :meth:`DesktopShield.set_opacity` / :meth:`DesktopShield.set_failsafe_opaque`
  layer the thread-safe, latest-wins :class:`CoalescingOpacityController` on top
  (Req 6.4/6.5/6.6), draining onto the Qt GUI thread within 100ms.

Importability without Qt
-------------------------
PySide6 may be unavailable (e.g. a Python version lacking a wheel). To keep the
module importable and unit-testable regardless, PySide6 is **not** imported at
module top level. The ``QWidget`` subclass is produced lazily by
:func:`_overlay_window_class` on first use, and :class:`DesktopShield` only
touches Qt when actually asked to build/show windows. Pure helpers
(:func:`opacity_pct_to_float`, screen geometry application) are importable and
testable without a display.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from medusa.shield.platform_window import (
    OVERLAY_WINDOW_FLAG_NAMES,
    TRANSLUCENT_ATTRIBUTE_NAME,
    PlatformWindow,
    make_platform_window,
)

__all__ = [
    "GridRenderer",
    "MoireGridRenderer",
    "CoalescingOpacityController",
    "DesktopShield",
    "opacity_pct_to_float",
    "moire_grid_lines",
    "DEFAULT_GRID_SPACING_PX",
    "DEFAULT_GRID_OFFSET_PX",
    "DEFAULT_GRID_COLOR_RGBA",
    "FAILSAFE_OPACITY_PCT",
    "OPAQUE_OPACITY_PCT",
    "TRANSPARENT_OPACITY_PCT",
    "OVERLAY_WINDOW_FLAG_NAMES",
    "TRANSLUCENT_ATTRIBUTE_NAME",
]

_log = logging.getLogger(__name__)

# Overlays are created fully transparent (Req 6.1): 0% opacity.
INITIAL_OPACITY_PCT = 0

# The only two meaningful opacity values the shield drives (Req 6.4/6.5): 0%
# (settled Safe) and 100% (Threat / within cooldown / UNCLASSIFIABLE fail-safe).
TRANSPARENT_OPACITY_PCT = 0
OPAQUE_OPACITY_PCT = 100

# UNCLASSIFIABLE frames fail safe to a fully-opaque overlay (Req 6.6).
FAILSAFE_OPACITY_PCT = OPAQUE_OPACITY_PCT

# Moiré grid defaults. The interference effect comes from drawing two very fine
# 1px line families whose spacing is offset by a small amount; viewed off-angle
# the two near-coincident families beat against each other and the sampling of
# the display, rendering content unreadable from the side (Req 6.3). A tight
# spacing keeps the pattern dense; the small per-family offset creates the beat.
DEFAULT_GRID_SPACING_PX = 3
DEFAULT_GRID_OFFSET_PX = 1
# Semi-transparent dark lines (R, G, B, A). Alpha < 255 so the pattern layers
# over the opaque fill without being a solid blackout, preserving the moiré look.
DEFAULT_GRID_COLOR_RGBA: tuple[int, int, int, int] = (0, 0, 0, 160)


@runtime_checkable
class GridRenderer(Protocol):
    """Renders the moiré interference grid onto an overlay (task 9.2 fills this).

    Defined here so :class:`OverlayWindow` can accept and invoke a renderer
    without depending on its concrete implementation. The concrete 1px moiré
    QPainter renderer (Req 6.3) is implemented in task 9.2.
    """

    def paint(self, painter: Any, width: int, height: int) -> None:
        """Paint the grid onto ``painter`` for a ``width`` x ``height`` surface."""
        ...


def opacity_pct_to_float(pct: int) -> float:
    """Convert an integer opacity percentage to a Qt window-opacity float.

    Qt's ``QWidget.setWindowOpacity`` takes a float in ``[0.0, 1.0]``. The
    Desktop_Shield only ever uses 0% (transparent/settled-safe) or 100%
    (opaque/threat), but any percentage is clamped to the valid range so callers
    cannot drive the window out of bounds.

    This is a pure function, independent of Qt, so it is unit-testable without a
    display.
    """
    if pct <= 0:
        return 0.0
    if pct >= 100:
        return 1.0
    return pct / 100.0


def moire_grid_lines(
    width: int,
    height: int,
    spacing: int = DEFAULT_GRID_SPACING_PX,
    offset: int = DEFAULT_GRID_OFFSET_PX,
) -> dict[str, list[int]]:
    """Compute the 1px moiré grid line positions for a surface (pure helper).

    Returns the integer pixel coordinates of the vertical and horizontal lines
    that make up the two interfering 1px line families, with no dependency on Qt
    or a display. Task 9.3 snapshot-tests the grid against this function.

    Two families are produced per axis:

    * a *base* family on a regular ``spacing`` grid, and
    * an *offset* family on the same ``spacing`` but shifted by ``offset`` px.

    The two near-coincident families are what beat against each other (and the
    display's own pixel sampling) to create the moiré interference that renders
    the screen unreadable from an angle (Req 6.3).

    Args:
        width: Surface width in pixels.
        height: Surface height in pixels.
        spacing: Pixel spacing between lines within a family. Values ``<= 0`` are
            treated as ``1`` so the function never divides by zero or loops
            forever; the smallest useful spacing is 1px.
        offset: Pixel shift of the second (offset) family relative to the base.

    Returns:
        A mapping with four keys -- ``"vertical"`` / ``"horizontal"`` for the
        base families (x-coordinates / y-coordinates respectively) and
        ``"vertical_offset"`` / ``"horizontal_offset"`` for the offset families.
        Each value is a sorted list of in-bounds integer coordinates.
    """
    if spacing <= 0:
        spacing = 1
    w = max(0, int(width))
    h = max(0, int(height))
    off = int(offset)

    def _positions(extent: int, start: int) -> list[int]:
        # Lines at start, start+spacing, ... strictly within [0, extent).
        first = start % spacing
        if first < 0:
            first += spacing
        return [p for p in range(first, extent, spacing) if 0 <= p < extent]

    return {
        "vertical": _positions(w, 0),
        "horizontal": _positions(h, 0),
        "vertical_offset": _positions(w, off),
        "horizontal_offset": _positions(h, off),
    }


class MoireGridRenderer:
    """Concrete :class:`GridRenderer` drawing a 1px moiré interference grid.

    Implements the ``GridRenderer`` protocol (``paint(painter, width, height)``)
    using QPainter. The drawing is parameterised by line ``spacing``, per-family
    ``offset`` and line ``color`` so the pattern can be tuned, and the geometry
    is delegated to the pure :func:`moire_grid_lines` helper so the exact lines
    drawn are computable (and snapshot-testable) without a display.

    PySide6 is imported lazily inside :meth:`paint`, so this class can be
    instantiated and its geometry inspected on a host without PySide6 present.
    """

    def __init__(
        self,
        spacing: int = DEFAULT_GRID_SPACING_PX,
        offset: int = DEFAULT_GRID_OFFSET_PX,
        color: tuple[int, int, int, int] = DEFAULT_GRID_COLOR_RGBA,
    ) -> None:
        self._spacing = max(1, int(spacing))
        self._offset = int(offset)
        self._color = color

    @property
    def spacing(self) -> int:
        """Pixel spacing between lines within a family."""
        return self._spacing

    @property
    def offset(self) -> int:
        """Pixel offset of the second line family relative to the base."""
        return self._offset

    @property
    def color(self) -> tuple[int, int, int, int]:
        """The line colour as an ``(r, g, b, a)`` tuple."""
        return self._color

    def line_positions(self, width: int, height: int) -> dict[str, list[int]]:
        """Return the exact line coordinates this renderer would draw.

        Pure (Qt-free) accessor used by snapshot tests (task 9.3) to assert the
        rendered geometry without needing a display.
        """
        return moire_grid_lines(width, height, self._spacing, self._offset)

    def paint(self, painter: Any, width: int, height: int) -> None:
        """Draw the 1px moiré grid onto ``painter`` for ``width`` x ``height``.

        Uses a 1px-wide pen in the configured colour and strokes every vertical
        and horizontal line from both the base and offset families. PySide6 is
        imported here (lazily) so the module stays importable without Qt.
        """
        if width <= 0 or height <= 0:
            return

        from PySide6.QtCore import Qt  # noqa: PLC0415 (intentional lazy import)
        from PySide6.QtGui import QColor, QPen  # noqa: PLC0415 (lazy)

        r, g, b, a = self._color
        pen = QPen(QColor(r, g, b, a))
        pen.setWidth(1)
        # Cosmetic + square caps keep each line exactly 1px regardless of any
        # device-pixel scaling, which is essential to the interference effect.
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.SquareCap)
        painter.setPen(pen)

        lines = self.line_positions(width, height)
        for x in lines["vertical"]:
            painter.drawLine(x, 0, x, height)
        for x in lines["vertical_offset"]:
            painter.drawLine(x, 0, x, height)
        for y in lines["horizontal"]:
            painter.drawLine(0, y, width, y)
        for y in lines["horizontal_offset"]:
            painter.drawLine(0, y, width, y)


class CoalescingOpacityController:
    """Thread-safe, latest-wins opacity coalescer (Qt-independent core).

    The vision producer thread can request opacity changes faster than the Qt
    GUI thread repaints. To avoid queuing stale repaints, this controller stores
    only the *latest requested* opacity and marks whether a change is pending.
    The GUI side later drains the pending value exactly once (:meth:`take`),
    always observing the most recent request and never a backlog (design:
    "latest-wins coalescing ... never queues up stale repaints").

    This class contains **no** Qt dependency: it is the small, testable core of
    the coalescing mechanism. :class:`DesktopShield` wraps it with a Qt signal /
    timer when PySide6 is present so the drain happens on the GUI thread within
    100ms (Req 6.4/6.5); without Qt the same logic is still fully exercisable.

    An optional ``notify`` callback is invoked (outside the lock) whenever a new
    distinct request is coalesced, letting a Qt wrapper schedule a GUI-thread
    drain. The callback must itself be cheap and thread-safe.
    """

    def __init__(self, notify: Optional[Callable[[], None]] = None) -> None:
        self._lock = threading.Lock()
        self._notify = notify
        self._requested_pct: Optional[int] = None
        self._pending = False

    @staticmethod
    def _clamp(pct: int) -> int:
        return max(0, min(100, int(pct)))

    def request(self, pct: int) -> None:
        """Record the latest desired opacity (coalescing, latest wins).

        Overwrites any previously-pending value so only the most recent request
        survives. Thread-safe; may be called from any thread.
        """
        clamped = self._clamp(pct)
        with self._lock:
            self._requested_pct = clamped
            self._pending = True
        if self._notify is not None:
            # Fire outside the lock to avoid re-entrancy deadlocks.
            self._notify()

    def request_failsafe_opaque(self) -> None:
        """Request the fail-safe fully-opaque opacity for UNCLASSIFIABLE (Req 6.6)."""
        self.request(FAILSAFE_OPACITY_PCT)

    @property
    def pending(self) -> bool:
        """Whether a coalesced request is waiting to be applied."""
        with self._lock:
            return self._pending

    def peek(self) -> Optional[int]:
        """Return the latest requested opacity without clearing the pending flag."""
        with self._lock:
            return self._requested_pct

    def take(self) -> Optional[int]:
        """Atomically consume the latest requested opacity, clearing pending.

        Returns the most recent requested percentage and marks the controller as
        drained, or ``None`` if nothing is pending. Only the latest value is ever
        returned; intermediate requests between drains are discarded (coalesced).
        """
        with self._lock:
            if not self._pending:
                return None
            self._pending = False
            return self._requested_pct


# The concrete QWidget subclass is cached after first construction so repeated
# calls return the same type (and so the lazy PySide6 import happens once).
_OVERLAY_WINDOW_CLASS: Optional[type] = None


def _overlay_window_class() -> type:
    """Build (once) and return the ``OverlayWindow`` ``QWidget`` subclass.

    Defined lazily inside a function so that importing this module never imports
    PySide6. The class is created on first use and memoised. Raises
    ``ImportError`` (via the underlying import) if PySide6 is unavailable, which
    callers that need a real window must handle.
    """
    global _OVERLAY_WINDOW_CLASS
    if _OVERLAY_WINDOW_CLASS is not None:
        return _OVERLAY_WINDOW_CLASS

    from PySide6.QtWidgets import QWidget  # noqa: PLC0415 (intentional lazy import)

    class OverlayWindow(QWidget):
        """A single borderless, click-through overlay for one monitor.

        Construction applies the overlay window flags and translucent-background
        attribute via the supplied :class:`PlatformWindow` adapter, starts at 0%
        opacity (Req 6.1), and stores an optional :class:`GridRenderer` that the
        paint event will delegate to once task 9.2 provides it.
        """

        def __init__(
            self,
            platform_window: PlatformWindow,
            grid_renderer: Optional[GridRenderer] = None,
            parent: Any = None,
        ) -> None:
            super().__init__(parent)
            self._platform_window = platform_window
            self._grid_renderer = grid_renderer

            # Apply flags + translucent attribute + any native quirks (Req 6.2).
            platform_window.configure(self)

            # Start fully transparent / click-through (Req 6.1).
            self.setWindowOpacity(opacity_pct_to_float(INITIAL_OPACITY_PCT))
            self._opacity_pct = INITIAL_OPACITY_PCT

        # -- Opacity -------------------------------------------------------

        def set_opacity(self, pct: int) -> None:
            """Set window opacity from an integer percentage (0 or 100).

            Task 9.2 wraps this with a thread-safe coalescing signal and the
            100ms timing guarantee (Req 6.4/6.5); here it is the direct plumbing.
            """
            self._opacity_pct = max(0, min(100, int(pct)))
            self.setWindowOpacity(opacity_pct_to_float(self._opacity_pct))

        def opacity_pct(self) -> int:
            """Return the last-applied opacity percentage."""
            return self._opacity_pct

        def set_grid_renderer(self, renderer: Optional[GridRenderer]) -> None:
            """Attach/replace the moiré grid renderer (populated in task 9.2)."""
            self._grid_renderer = renderer

        # -- Painting ------------------------------------------------------

        def paintEvent(self, event: Any) -> None:  # noqa: N802 (Qt override name)
            """Delegate drawing to the grid renderer when one is present.

            Until task 9.2 supplies a :class:`GridRenderer`, this is effectively
            a no-op over a translucent surface (nothing is drawn), which is the
            correct appearance for a 0% overlay.
            """
            if self._grid_renderer is None:
                return
            from PySide6.QtGui import QPainter  # noqa: PLC0415 (lazy)

            painter = QPainter(self)
            try:
                self._grid_renderer.paint(painter, self.width(), self.height())
            finally:
                painter.end()

    _OVERLAY_WINDOW_CLASS = OverlayWindow
    return _OVERLAY_WINDOW_CLASS


class DesktopShield:
    """Manages one click-through overlay per connected monitor (Req 6.1).

    The shield enumerates the connected screens via ``QGuiApplication.screens()``
    and creates an :class:`OverlayWindow` sized to each screen's geometry, all at
    0% opacity. OS quirks are isolated behind the injected
    :class:`PlatformWindow` adapter (Windows primary; macOS/Linux stubs).

    Qt is only touched when :meth:`show_all_monitors` runs, so a ``DesktopShield``
    can be constructed in environments without PySide6; attempting to show
    overlays there raises the underlying ``ImportError``.
    """

    def __init__(
        self,
        platform_window: Optional[PlatformWindow] = None,
        grid_renderer: Optional[GridRenderer] = None,
    ) -> None:
        self._platform_window = platform_window or make_platform_window()
        # Attach a concrete 1px moiré renderer by default (Req 6.3) so every
        # overlay draws the interference grid; callers may inject an alternative
        # (e.g. a GL/shader renderer) via ``grid_renderer``.
        self._grid_renderer = grid_renderer or MoireGridRenderer()
        self._windows: list[Any] = []
        # Latest-wins opacity coalescer (Req 6.4/6.5). The notify callback drains
        # the controller onto the Qt GUI thread when PySide6 is available; the
        # coalescing logic itself is Qt-independent and testable.
        self._opacity_controller = CoalescingOpacityController(
            notify=self._schedule_opacity_apply
        )

    @property
    def grid_renderer(self) -> Optional[GridRenderer]:
        """The grid renderer attached to overlays (default: moiré 1px grid)."""
        return self._grid_renderer

    @property
    def opacity_controller(self) -> CoalescingOpacityController:
        """The latest-wins opacity coalescer backing :meth:`set_opacity`."""
        return self._opacity_controller

    @property
    def platform_window(self) -> PlatformWindow:
        """The platform adapter in use (exposes its documented limitations)."""
        return self._platform_window

    @property
    def windows(self) -> list[Any]:
        """The overlay windows currently managed (one per monitor)."""
        return list(self._windows)

    def _screens(self) -> list[Any]:
        """Return the connected screens via ``QGuiApplication`` (lazy import)."""
        from PySide6.QtGui import QGuiApplication  # noqa: PLC0415 (lazy)

        return list(QGuiApplication.screens())

    def show_all_monitors(self) -> None:
        """Create and show one overlay per connected monitor at 0% opacity.

        Each overlay is positioned and sized to its screen's full geometry so it
        covers the entire visible area of every connected monitor (Req 6.1).
        Existing overlays are torn down first so repeated calls (e.g. after a
        monitor hot-plug) rebuild cleanly.
        """
        self.close_all()

        window_cls = _overlay_window_class()
        screens = self._screens()
        if not screens:
            _log.warning("No screens reported by QGuiApplication; no overlay shown.")
            return

        for screen in screens:
            window = window_cls(self._platform_window, self._grid_renderer)
            self._place_on_screen(window, screen)
            # Show without stealing focus/activation (keeps it output-only).
            if hasattr(window, "showFullScreen"):
                window.setScreen(screen)
                window.showFullScreen()
            else:  # pragma: no cover - defensive
                window.show()
            self._windows.append(window)

    @staticmethod
    def _place_on_screen(window: Any, screen: Any) -> None:
        """Size and move ``window`` to cover ``screen``'s full geometry.

        Separated out so the geometry math can be exercised with a fake screen
        object (one exposing ``geometry()``) without a real display.
        """
        geometry = screen.geometry()
        window.setGeometry(geometry)

    def set_opacity(self, pct: int) -> None:
        """Request an overlay opacity (0 or 100) via the coalescing controller.

        Thread-safe and latest-wins (Req 6.4/6.5): callable from the vision
        producer thread, it records the most recent desired opacity and schedules
        a single GUI-thread apply. A burst of calls collapses to one repaint with
        the final value, so stale repaints are never queued. The apply is marshalled
        onto the Qt GUI thread (via a queued invocation / short timer when PySide6
        is present) so it lands within 100ms of the classification.
        """
        self._opacity_controller.request(pct)

    def set_failsafe_opaque(self) -> None:
        """Force overlays fully opaque for UNCLASSIFIABLE frames (Req 6.6).

        Routes through the same coalescing controller so it participates in the
        latest-wins ordering: an UNCLASSIFIABLE frame holds the shield opaque
        until a subsequent settled-Safe request lowers it again.
        """
        self._opacity_controller.request_failsafe_opaque()

    def _schedule_opacity_apply(self) -> None:
        """Marshal a coalesced opacity drain onto the Qt GUI thread.

        Invoked by the controller whenever a new request is coalesced. When
        PySide6 is available the apply is posted to the GUI thread via a 0ms
        single-shot ``QTimer`` on the application's main thread, guaranteeing the
        window-opacity call happens on the GUI thread and within 100ms. Without
        PySide6 (or before an app/windows exist) this degrades to a best-effort
        direct drain so the coalescing logic stays exercisable headless.
        """
        try:
            from PySide6.QtCore import QTimer  # noqa: PLC0415 (lazy)
        except Exception:
            # No Qt available: apply directly (used by headless tests).
            self._apply_pending_opacity()
            return

        # Post to the GUI thread's event loop; coalescing means multiple posts
        # still resolve to the single latest value when drained.
        try:
            QTimer.singleShot(0, self._apply_pending_opacity)
        except Exception:  # pragma: no cover - no running Qt event loop yet
            self._apply_pending_opacity()

    def _apply_pending_opacity(self) -> None:
        """Drain the controller once and apply the latest opacity to all windows.

        Consumes exactly the most recent coalesced value (``None`` if already
        drained by a prior scheduled apply) and pushes it to every overlay.
        """
        pct = self._opacity_controller.take()
        if pct is None:
            return
        for window in self._windows:
            window.set_opacity(pct)

    def close_all(self) -> None:
        """Close and release every managed overlay window."""
        for window in self._windows:
            try:
                window.close()
            except Exception:  # pragma: no cover - defensive teardown
                _log.exception("Failed to close an overlay window during teardown.")
        self._windows.clear()
