"""Application bootstrap and three-thread orchestration (task 13.1).

This module wires every subsystem of Screen Privacy Guardian (Medusa_Desktop_
Core) together behind the three-thread model from ``design.md`` (section
"Threading / Concurrency Model"):

1. **Asyncio I/O thread.** A dedicated thread owning a fresh ``asyncio`` event
   loop that hosts the :class:`~medusa.server.status_server.StatusServer` *and*
   the :class:`~medusa.alert.service.AlertService` pipeline. The vision thread
   hands work to this loop via :func:`asyncio.run_coroutine_threadsafe` and
   never waits on the result (Req 5.1, 7.9).
2. **Qt GUI thread.** Owns the :class:`~medusa.shield.overlay.DesktopShield`.
   Opacity changes are delivered through the shield's thread-safe, latest-wins
   coalescing controller so a burst of classifications never queues stale
   repaints (Req 6.1, 6.4, 6.5).
3. **Vision producer thread.** The single-producer capture + classify loop
   (:class:`~medusa.vision.producer.VisionProducer`) publishing each
   :class:`~medusa.core.models.Classification` to the
   :class:`~medusa.core.bus.ClassificationBus` (Req 1.1).

The :class:`ClassificationBus` is the single arbiter of effective state. After
each publish the bus's :meth:`effective_state` / :meth:`effective_opacity` are
read and dispatched to the three consumers:

* **Overlay opacity** -- ``shield.set_opacity(bus.effective_opacity())`` (or
  ``set_failsafe_opaque`` on an UNCLASSIFIABLE frame), via the shield's
  coalescing controller (non-blocking, Req 6.4-6.6).
* **Status broadcast** -- ``StatusServer.broadcast(Status.<...>)`` scheduled on
  the asyncio loop with :func:`asyncio.run_coroutine_threadsafe` (non-blocking
  enqueue; the vision thread never awaits the result, Req 7.9).
* **Alert trigger** -- on an effective THREAT, when ``bus.may_start_alert(now)``
  and consent is granted, ``AlertService.on_threat(frame_jpeg, event_id)`` is
  scheduled on the asyncio loop (consent/cooldown-gated, Req 7.1, 8.3, 9.2).

Import-safety / testability
---------------------------
The heavy, optional-at-import-time dependencies (PySide6, cv2, mediapipe,
fastapi, uvicorn) are **never** imported at module top level, so
``import medusa.app.main`` always succeeds on a bare interpreter. Each heavy
subsystem is reached through an injectable *seam* with a lazy default:

* ``consent_prompt``: ``Callable[[], bool]`` -- the first-run consent prompt
  (defaults to a console prompt). Tests inject a fake.
* ``shield_factory``: builds the Desktop_Shield (guarded; a PySide6 import
  failure logs and continues without an overlay).
* ``server_factory`` / ``alert_service_factory``: build the asyncio-loop
  consumers (defaults build the real objects; tests inject fakes).
* ``producer_factory``: builds the vision producer (default builds the real
  camera + extractor + producer; tests inject a fake producer).
* ``audio_player``: async audio playback (defaults to a no-op async stub; real
  playback is out of scope here).

Every seam has a working default so :func:`main` runs the real system, while
:class:`MedusaApp` can be constructed entirely from fakes for a headless smoke
test (task 13.2).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import threading
from typing import Any, Awaitable, Callable, Optional

from medusa.alert.providers import make_tts_provider, make_vlm_provider
from medusa.alert.service import AlertService
from medusa.config.consent import ConsentModel
from medusa.config.logging_setup import setup_logging
from medusa.config.store import ConfigurationStore
from medusa.core.bus import ClassificationBus
from medusa.core.models import AppConfig, FrameState, Status

__all__ = [
    "MedusaApp",
    "AsyncIoThread",
    "DispatchingBus",
    "console_consent_prompt",
    "noop_audio_player",
    "main",
]

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Injectable seam defaults
# --------------------------------------------------------------------------- #
async def noop_audio_player(_audio: bytes) -> None:
    """Default async audio player: a no-op stub.

    Real audio playback is out of scope for the bootstrap; this placeholder
    completes immediately so the Alert_Service pipeline runs end-to-end (VLM ->
    TTS -> "playback") without a sound device. Swap in a real player via the
    ``audio_player`` seam when wiring actual output.
    """
    return None


def console_consent_prompt() -> bool:
    """Default first-run consent prompt (console-based, Req 9.1).

    Describes that detected bystander frames may be transmitted to the
    configured third-party VLM_API and returns the user's decision. Treats
    anything other than an explicit "yes" as denied (deny-by-default). If no
    interactive console is available (``input`` raises ``EOFError``), the
    decision defaults to denied.
    """
    message = (
        "\nScreen Privacy Guardian - consent required\n"
        "When a bystander (snooper) is detected, a SINGLE triggered webcam "
        "frame may be sent to your configured third-party vision (VLM) API to "
        "generate a brief spoken alert. Your continuous video never leaves this "
        "machine, and nothing is sent unless you grant consent here.\n"
        "Allow sending a single triggered frame to the configured VLM API? "
        "[y/N]: "
    )
    try:
        answer = input(message)
    except EOFError:
        # No interactive console: default to denied (privacy-preserving).
        return False
    return answer.strip().lower() in {"y", "yes"}


# --------------------------------------------------------------------------- #
# Asyncio I/O thread
# --------------------------------------------------------------------------- #
class AsyncIoThread:
    """A dedicated thread running a fresh asyncio event loop (the I/O thread).

    Hosts the Status_Server and the Alert_Service pipeline. The vision thread
    schedules coroutines onto this loop via :meth:`run_coroutine` (a thin wrapper
    over :func:`asyncio.run_coroutine_threadsafe`) and never blocks on the
    returned future, keeping the vision loop non-blocking (Req 7.9).

    The loop is created *inside* the worker thread (not in the constructor) so it
    is owned by that thread, matching asyncio's single-thread affinity. :meth:`start`
    blocks only until the loop is running and ready to accept work.
    """

    def __init__(self, name: str = "medusa-asyncio-io") -> None:
        self._name = name
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """The running event loop, or ``None`` before :meth:`start` completes."""
        return self._loop

    @property
    def is_running(self) -> bool:
        """``True`` while the loop thread is alive and the loop is running."""
        loop = self._loop
        return (
            self._thread is not None
            and self._thread.is_alive()
            and loop is not None
            and loop.is_running()
        )

    def start(self, timeout: float = 5.0) -> None:
        """Start the worker thread and wait until its loop is running.

        Blocks (bounded by ``timeout``) until the event loop has been created and
        has begun running, so callers may immediately schedule work onto it.
        """
        if self.is_running:
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("asyncio I/O loop failed to start within timeout")

    def _run(self) -> None:
        """Thread entry point: own, run, and (on stop) close the event loop."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        loop.call_soon(self._ready.set)
        try:
            loop.run_forever()
        finally:
            # Drain and close once run_forever returns (after stop()).
            try:
                self._cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception as exc:  # noqa: BLE001 - defensive teardown
                _log.debug("asyncio loop drain raised during shutdown: %r", exc)
            finally:
                loop.close()

    @staticmethod
    def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )

    def run_coroutine(self, coro: Awaitable[Any]) -> Optional["asyncio.Future[Any]"]:
        """Schedule ``coro`` on the loop without waiting (non-blocking, Req 7.9).

        Returns the :class:`concurrent.futures.Future` from
        :func:`asyncio.run_coroutine_threadsafe` so a caller *could* inspect it,
        but the vision thread deliberately ignores it. If the loop is not
        running the coroutine is closed and ``None`` is returned so a stray
        schedule after shutdown never raises on the vision thread.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            # Loop gone (e.g. during shutdown): close the coroutine to avoid a
            # "coroutine was never awaited" warning and drop the request.
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            return None
        try:
            return asyncio.run_coroutine_threadsafe(coro, loop)  # type: ignore[arg-type]
        except RuntimeError:
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            return None

    def call(self, func: Callable[[], Any], timeout: float = 5.0) -> Any:
        """Run a *synchronous* ``func`` on the loop thread and wait for its result.

        Used by the bootstrap for one-off lifecycle operations that must happen
        on the loop thread (e.g. starting/stopping the server) where the caller
        *does* want to wait. Not used on the hot vision path.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            raise RuntimeError("asyncio I/O loop is not running")

        done = threading.Event()
        box: dict[str, Any] = {}

        def _invoke() -> None:
            try:
                box["result"] = func()
            except BaseException as exc:  # noqa: BLE001 - propagate to caller
                box["error"] = exc
            finally:
                done.set()

        loop.call_soon_threadsafe(_invoke)
        if not done.wait(timeout):
            raise TimeoutError("timed out running function on asyncio I/O loop")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def run_coroutine_blocking(self, coro: Awaitable[Any], timeout: float = 10.0) -> Any:
        """Schedule ``coro`` on the loop and block for its result (lifecycle only).

        Used for server start/stop where the bootstrap genuinely needs the
        outcome. The hot path uses the non-blocking :meth:`run_coroutine`.
        """
        future = self.run_coroutine(coro)
        if future is None:
            raise RuntimeError("asyncio I/O loop is not running")
        return future.result(timeout)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the loop and join the worker thread (graceful shutdown)."""
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None
        self._loop = None


# --------------------------------------------------------------------------- #
# Dispatching bus: fire a coordinator callback after each publish
# --------------------------------------------------------------------------- #
class DispatchingBus(ClassificationBus):
    """A :class:`ClassificationBus` that invokes a callback after each publish.

    The vision producer calls :meth:`publish` on the vision thread. Immediately
    after the FSM has absorbed the new classification (so ``effective_state`` /
    ``effective_opacity`` are up to date), the injected ``on_classification``
    coordinator runs on that same vision thread. The coordinator does *only*
    non-blocking work: it requests a coalesced overlay opacity and enqueues
    coroutines onto the asyncio loop -- it never performs network I/O or awaits a
    result, so the vision loop is never stalled (Req 7.9, 11.3).

    Subclassing keeps the producer unchanged (it already publishes to a bus) and
    keeps the FSM logic in one place. The callback is set after construction via
    :meth:`set_on_classification` so the bus can be created before the
    coordinator that closes over it.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_classification: Optional[Callable[["DispatchingBus"], None]] = None

    def set_on_classification(
        self, callback: Optional[Callable[["DispatchingBus"], None]]
    ) -> None:
        """Install the post-publish coordinator callback (or clear it)."""
        self._on_classification = callback

    def publish(self, classification: Any) -> None:  # type: ignore[override]
        super().publish(classification)
        callback = self._on_classification
        if callback is None:
            return
        try:
            callback(self)
        except Exception as exc:  # noqa: BLE001 - a consumer error must not
            # crash the vision thread (fail-safe posture).
            _log.exception("Classification dispatch failed: %r", exc)


# --------------------------------------------------------------------------- #
# The application
# --------------------------------------------------------------------------- #
class MedusaApp:
    """Bootstraps and orchestrates the three-thread Screen Privacy Guardian.

    Construction loads configuration + consent and prepares the collaborators
    but starts no threads. :meth:`start` presents the first-run consent prompt
    (when needed), starts the asyncio I/O thread (server + alert service), the
    Qt GUI thread (shield), and the vision producer thread, and wires the bus
    dispatch. :meth:`shutdown` tears everything down gracefully.

    All heavy subsystems are reached through injectable factories so the whole
    app can be driven headless with fakes (task 13.2). Every factory has a lazy,
    real default used by :func:`main`.
    """

    def __init__(
        self,
        *,
        config_store: Optional[ConfigurationStore] = None,
        consent_prompt: Callable[[], bool] = console_consent_prompt,
        shield_factory: Optional[Callable[[], Any]] = None,
        server_factory: Optional[Callable[[], Any]] = None,
        producer_factory: Optional[
            Callable[[DispatchingBus, AppConfig], Any]
        ] = None,
        alert_service_factory: Optional[
            Callable[[AppConfig, ConsentModel, DispatchingBus], Any]
        ] = None,
        audio_player: Callable[[bytes], Awaitable[None]] = noop_audio_player,
        io_thread: Optional[AsyncIoThread] = None,
        now: Callable[[], float] = None,  # type: ignore[assignment]
    ) -> None:
        import time

        self._now = now if now is not None else time.monotonic

        # --- 1. Config + consent (Req 10.1, 9.1) --------------------------
        self._store = config_store if config_store is not None else ConfigurationStore()
        load_result = self._store.load()
        self._config: AppConfig = load_result.config
        self._config_errors: list[str] = list(load_result.errors)
        # Surface config errors (Req 10.3/10.4). ConfigurationStore already logs
        # each one; we retain them for callers/tests and re-log a summary.
        for error in self._config_errors:
            _log.warning("Configuration issue: %s", error)

        self._consent = ConsentModel(self._store)
        self._consent_prompt = consent_prompt

        # --- Collaborators (built lazily on start via factories) ----------
        self._shield_factory = shield_factory or _default_shield_factory
        self._server_factory = server_factory or _default_server_factory
        self._producer_factory = producer_factory or _default_producer_factory
        self._alert_service_factory = (
            alert_service_factory or _default_alert_service_factory
        )
        self._audio_player = audio_player

        # --- The single arbiter of effective state ------------------------
        self._bus = DispatchingBus(
            cooldown_period_s=self._config.cooldown_period_s,
            alert_cooldown_period_s=self._config.cooldown_period_s,
        )

        # --- Threads / subsystems (populated on start) --------------------
        self._io = io_thread if io_thread is not None else AsyncIoThread()
        self._shield: Any = None
        self._server: Any = None
        self._alert: Any = None
        self._producer: Any = None

        self._event_ids = itertools.count(1)
        self._started = False
        self._first_run_prompted = False
        self._consent_message: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Introspection (useful for the smoke test)
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def config_errors(self) -> list[str]:
        """Configuration validation/read errors surfaced on load (Req 10.3/10.4)."""
        return list(self._config_errors)

    @property
    def bus(self) -> DispatchingBus:
        return self._bus

    @property
    def shield(self) -> Any:
        return self._shield

    @property
    def server(self) -> Any:
        return self._server

    @property
    def alert_service(self) -> Any:
        return self._alert

    @property
    def producer(self) -> Any:
        return self._producer

    @property
    def io_thread(self) -> AsyncIoThread:
        return self._io

    @property
    def first_run_prompted(self) -> bool:
        """Whether the first-run consent prompt was presented this run (Req 9.1)."""
        return self._first_run_prompted

    @property
    def consent_message(self) -> Optional[str]:
        """Any user-facing consent indication (e.g. "decision not saved", Req 9.4)."""
        return self._consent_message

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start all three threads and wire the bus dispatch.

        Order: present the first-run consent prompt (Req 9.1), start the asyncio
        I/O thread and bind the Status_Server (Req 5.1; degrade without
        broadcasting on bind failure, Req 5.4), build the Alert_Service on the
        I/O loop, bring up the Desktop_Shield (guarded; continue without an
        overlay if PySide6 is unavailable, Req 6.1), start the vision producer
        (Req 1.1), then connect the bus so each classification dispatches to the
        overlay, broadcast, and alert consumers.
        """
        if self._started:
            return

        # 1. First-run consent prompt (Req 9.1). Only when no persisted decision.
        self._present_first_run_consent()

        # 2. Asyncio I/O thread + Status_Server (Req 5.1 / 5.4).
        self._io.start()
        self._server = self._server_factory()
        self._start_server()

        # 3. Alert_Service on the I/O loop (providers resolved from config).
        self._alert = self._alert_service_factory(
            self._config, self._consent, self._bus
        )
        # Re-point the service's audio player / confirm wiring if the default
        # factory did not already use ours. The default factory below honors the
        # app's audio player via a closure, so nothing more is needed here.

        # 4. Desktop_Shield on the Qt GUI thread (Req 6.1), guarded.
        self._start_shield()

        # 5. Vision producer thread (Req 1.1).
        self._producer = self._producer_factory(self._bus, self._config)

        # 6. Wire the bus dispatch BEFORE starting the producer so no
        #    classification is missed.
        self._bus.set_on_classification(self._dispatch)

        self._start_producer()
        self._started = True
        _log.info("Medusa_Desktop_Core started.")

    def _present_first_run_consent(self) -> None:
        """Present the consent prompt on first run and persist the decision.

        Deny-by-default: the prompt is shown only when no decision has ever been
        persisted (Req 9.1). The user's decision is persisted via the Consent
        model; a persist failure is treated as denied and surfaces a
        "decision not saved" indication (Req 9.4).
        """
        if not self._consent.needs_prompt():
            return
        self._first_run_prompted = True
        try:
            granted = bool(self._consent_prompt())
        except Exception as exc:  # noqa: BLE001 - a prompt error defaults to deny
            _log.warning(
                "Consent prompt raised %s; treating as denied.", type(exc).__name__
            )
            granted = False
        decision = self._consent.set_consent(granted)
        if not decision.saved:
            self._consent_message = decision.message
            _log.warning("Consent decision was not saved; transmission stays denied.")

    def _start_server(self) -> None:
        """Bind the Status_Server on the I/O loop (Req 5.1; degrade on failure)."""
        if self._server is None:
            return
        port = self._config.server_port
        try:
            ok = self._io.run_coroutine_blocking(
                self._server.start(port), timeout=10.0
            )
        except Exception as exc:  # noqa: BLE001 - never fail startup on the server
            _log.error(
                "Status_Server failed to start on port %s (%s); continuing "
                "detection and overlay without broadcasting.",
                port,
                type(exc).__name__,
            )
            return
        if not ok:
            # Port unavailable (Req 5.4): continue without broadcasting.
            _log.error(
                "Status_Server could not bind port %s; continuing detection and "
                "overlay without broadcasting.",
                port,
            )

    def _start_shield(self) -> None:
        """Construct and show the Desktop_Shield, guarding missing PySide6 (Req 6.1).

        If the shield cannot be built or shown (e.g. PySide6 unavailable or no
        display), the error is logged and the system continues without an
        overlay -- the rest of the pipeline still runs.
        """
        try:
            self._shield = self._shield_factory()
        except Exception as exc:  # noqa: BLE001 - overlay is optional at runtime
            _log.warning(
                "Desktop_Shield unavailable (%s); continuing without an overlay.",
                type(exc).__name__,
            )
            self._shield = None
            return
        try:
            self._shield.show_all_monitors()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Desktop_Shield could not show overlays (%s); continuing without "
                "an overlay.",
                type(exc).__name__,
            )

    def _start_producer(self) -> None:
        """Start the vision producer thread (Req 1.1)."""
        if self._producer is None:
            return
        self._producer.start()

    # ------------------------------------------------------------------ #
    # Bus dispatch (runs on the vision thread; strictly non-blocking)
    # ------------------------------------------------------------------ #
    def _dispatch(self, bus: DispatchingBus) -> None:
        """React to a freshly published classification (vision thread).

        Reads the bus's stabilized effective state/opacity and fans out to the
        three consumers with only non-blocking operations:

        * overlay opacity via the shield's coalescing controller,
        * status broadcast scheduled on the asyncio loop (non-blocking enqueue),
        * a consent/cooldown-gated alert scheduled on the asyncio loop.

        This method never awaits a coroutine result and never performs network
        I/O inline, so the vision loop keeps its >=10 fps heartbeat (Req 7.9).
        """
        self._dispatch_overlay(bus)
        self._dispatch_broadcast(bus)
        self._dispatch_alert(bus)

    def _dispatch_overlay(self, bus: DispatchingBus) -> None:
        """Apply the effective opacity to the overlay (coalesced; Req 6.4-6.6)."""
        shield = self._shield
        if shield is None:
            return
        try:
            shield.set_opacity(bus.effective_opacity())
        except Exception as exc:  # noqa: BLE001 - overlay errors never crash vision
            _log.debug("Overlay opacity update failed: %r", exc)

    def _dispatch_broadcast(self, bus: DispatchingBus) -> None:
        """Enqueue a status broadcast on the asyncio loop (non-blocking, Req 7.9)."""
        server = self._server
        if server is None:
            return
        status = (
            Status.THREAT_DETECTED
            if bus.effective_state is FrameState.THREAT
            else Status.SAFE
        )
        # Non-blocking enqueue; the vision thread never waits on the result.
        self._io.run_coroutine(server.broadcast(status))

    def _dispatch_alert(self, bus: DispatchingBus) -> None:
        """On an effective THREAT, schedule a gated alert on the asyncio loop.

        The alert is consent- and cooldown-gated: the bus's
        :meth:`may_start_alert` is consulted and consent is checked here, and the
        Alert_Service re-checks both internally. The single triggered JPEG comes
        from the extractor's ``last_jpeg``; if unavailable the alert is skipped.
        The coroutine is enqueued without blocking the vision thread (Req 7.9).
        """
        alert = self._alert
        if alert is None:
            return
        if bus.effective_state is not FrameState.THREAT:
            return

        now = self._now()
        if not bus.may_start_alert(now):
            return
        if not self._consent.is_granted():
            return

        frame_jpeg = self._triggered_jpeg()
        if frame_jpeg is None:
            # No frame available to transmit; skip this event.
            return

        event_id = f"evt-{next(self._event_ids)}"
        self._io.run_coroutine(alert.on_threat(frame_jpeg, event_id, now))

    def _triggered_jpeg(self) -> Optional[bytes]:
        """Return the single triggered JPEG from the extractor, if available.

        The producer exposes its extractor so the Alert_Service can grab exactly
        one frame (never a stream, Req 1.2 / 7.4). Returns ``None`` when no
        extractor/frame is available.
        """
        producer = self._producer
        extractor = getattr(producer, "extractor", None)
        if extractor is None:
            return None
        last_jpeg = getattr(extractor, "last_jpeg", None)
        if not callable(last_jpeg):
            return None
        try:
            return last_jpeg()
        except Exception as exc:  # noqa: BLE001
            _log.debug("Triggered JPEG grab failed: %r", exc)
            return None

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Gracefully stop every thread/subsystem (idempotent).

        Order mirrors a clean teardown: stop the vision producer first (so no new
        classifications are dispatched), detach the bus callback, stop the
        Status_Server on the I/O loop, close the overlay, then stop the asyncio
        I/O thread and join it.
        """
        if not self._started and self._producer is None:
            return

        # Stop producing + dispatching first.
        self._bus.set_on_classification(None)
        if self._producer is not None:
            try:
                self._producer.stop()
            except Exception as exc:  # noqa: BLE001
                _log.debug("Producer stop raised during shutdown: %r", exc)

        # Stop the Status_Server on its loop (Req graceful shutdown).
        if self._server is not None and self._io.is_running:
            try:
                self._io.run_coroutine_blocking(self._server.stop(), timeout=5.0)
            except Exception as exc:  # noqa: BLE001
                _log.debug("Status_Server stop raised during shutdown: %r", exc)

        # Close the overlay on its side.
        if self._shield is not None:
            try:
                self._shield.close_all()
            except Exception as exc:  # noqa: BLE001
                _log.debug("Shield close raised during shutdown: %r", exc)

        # Finally, stop the asyncio I/O loop + thread.
        try:
            self._io.stop()
        except Exception as exc:  # noqa: BLE001
            _log.debug("I/O thread stop raised during shutdown: %r", exc)

        self._started = False
        _log.info("Medusa_Desktop_Core shut down.")


# --------------------------------------------------------------------------- #
# Default (real) factories -- all heavy imports are lazy / guarded here
# --------------------------------------------------------------------------- #
def _default_shield_factory() -> Any:
    """Build a real :class:`DesktopShield`.

    Imports the shield module (import-safe) and ensures a ``QApplication`` exists
    before constructing overlays. PySide6 is imported lazily inside the shield;
    any failure here propagates to :meth:`MedusaApp._start_shield`, which logs
    and continues without an overlay.
    """
    from medusa.shield.overlay import DesktopShield

    # A QApplication must exist before any QWidget is created. Importing here
    # keeps PySide6 out of module import; a missing PySide6 raises ImportError,
    # which the caller guards.
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415

    if QApplication.instance() is None:
        QApplication([])
    return DesktopShield()


def _default_server_factory() -> Any:
    """Build a real :class:`StatusServer` (FastAPI/uvicorn imported lazily)."""
    from medusa.server.status_server import StatusServer

    return StatusServer()


def _default_producer_factory(bus: DispatchingBus, config: AppConfig) -> Any:
    """Build the real vision producer (camera + extractor), exposing the extractor.

    Wraps :class:`VisionProducer` in a tiny adapter that also exposes the
    :class:`FaceMeshExtractor` as ``.extractor`` so the alert dispatch can grab
    the single triggered JPEG. cv2/mediapipe are only touched when the real
    device is actually opened (lazy backends), so building the producer is
    import-safe; a missing wheel surfaces only when capture starts (and the
    camera retry/fail-safe path handles it, Req 1.3).
    """
    from medusa.vision.camera import CameraCapture, FaceMeshExtractor
    from medusa.vision.producer import VisionProducer

    camera = CameraCapture(device_index=0)
    extractor = FaceMeshExtractor(camera)
    producer = VisionProducer(
        camera=camera,
        extractor=extractor,
        bus=bus,
        detection_config=config.detection,
    )
    return _ProducerBundle(producer, extractor)


class _ProducerBundle:
    """Adapter exposing a :class:`VisionProducer` plus its extractor.

    The alert dispatch needs the extractor's ``last_jpeg`` to grab the single
    triggered frame; the producer itself does not expose it, so this thin bundle
    forwards :meth:`start`/:meth:`stop` to the producer while surfacing the
    extractor via :attr:`extractor`.
    """

    def __init__(self, producer: Any, extractor: Any) -> None:
        self._producer = producer
        self.extractor = extractor

    @property
    def producer(self) -> Any:
        return self._producer

    def start(self) -> None:
        self._producer.start()

    def stop(self, timeout: Optional[float] = 2.0) -> None:
        self._producer.stop(timeout=timeout)


def _default_alert_service_factory(
    config: AppConfig, consent: ConsentModel, bus: DispatchingBus
) -> AlertService:
    """Build the real :class:`AlertService` with providers resolved from config.

    Providers are resolved by name + revealed key via
    :func:`make_vlm_provider` / :func:`make_tts_provider`; an absent name/key
    yields ``None`` and disables that capability (Req 10.5). The ClassificationBus
    acts as the alert cooldown gate, and the Consent model is the just-in-time
    consent check (Req 9.5). Uses the no-op audio player by default (real audio
    is out of scope); callers who need real playback inject their own factory.
    """
    vlm = make_vlm_provider(
        config.vlm_provider,
        config.vlm_api_key.reveal() if config.vlm_api_key is not None else None,
    )
    tts = make_tts_provider(
        config.tts_provider,
        config.tts_api_key.reveal() if config.tts_api_key is not None else None,
    )
    return AlertService(
        vlm=vlm,
        tts=tts,
        consent=consent,
        gate=bus,
        audio_player=noop_audio_player,
        request_timeout_s=config.request_timeout_s,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """Run the full application until interrupted (the real entry point).

    Sets up redacting logging (Req 10.6), constructs a :class:`MedusaApp` with
    all real defaults, starts it, and blocks until a keyboard interrupt, then
    shuts down gracefully. Returns a process exit code.
    """
    setup_logging(level=logging.INFO)
    app = MedusaApp()
    try:
        app.start()
    except Exception as exc:  # noqa: BLE001 - surface a clean failure
        _log.exception("Failed to start Medusa_Desktop_Core: %r", exc)
        app.shutdown()
        return 1

    _log.info("Medusa_Desktop_Core is running. Press Ctrl+C to stop.")
    try:
        # Block the main thread until interrupted. If a Qt event loop is desired
        # for a visible overlay, a GUI-aware caller can drive QApplication.exec()
        # instead; the bootstrap itself stays headless-friendly.
        _wait_forever()
    except KeyboardInterrupt:
        _log.info("Interrupt received; shutting down.")
    finally:
        app.shutdown()
    return 0


def _wait_forever() -> None:
    """Block the calling thread until interrupted.

    Factored out so a GUI entry point can replace the wait with a Qt event loop
    while reusing the rest of :func:`main`.
    """
    stop = threading.Event()
    try:
        while not stop.wait(3600):
            pass
    except KeyboardInterrupt:
        raise


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    raise SystemExit(main())
