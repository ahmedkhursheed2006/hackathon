"""FastAPI/uvicorn WebSocket server lifecycle for the Status_Server (task 7.5).

This module wires the real network lifecycle on top of the FastAPI/uvicorn-free
core (:mod:`medusa.server.registry` and :mod:`medusa.server.payload`). It owns
the FastAPI application, the ``/ws`` WebSocket endpoint, and the programmatic
uvicorn server, all driven on the single asyncio I/O loop described in the
design's threading model.

Responsibilities (Requirements 5.1 / 5.4 / 5.5 / 5.6)
-----------------------------------------------------
- **Bind within 5s (Req 5.1):** :meth:`StatusServer.start` binds uvicorn to
  ``127.0.0.1:port`` and waits (bounded by a configurable timeout, default 5s)
  for the socket to come up before returning.
- **Current status on connect (Req 5.6):** when a client connects to ``/ws`` the
  endpoint registers it in the :class:`ClientRegistry` and immediately sends the
  retained current status, so a late joiner is consistent within 500ms.
- **Per-client lifecycle (Req 5.7):** on disconnect or send error the client is
  removed from the registry; broadcast fan-out and per-client isolation are
  delegated to the registry.
- **Port unavailable (Req 5.4):** if the bind fails, the error is caught and
  surfaced/logged, :attr:`StatusServer.broadcasting` becomes ``False``, and the
  server does **not** crash — the caller continues detection + overlay without
  broadcasting.
- **No clients (Req 5.5):** :meth:`StatusServer.broadcast` delegates to the
  registry, whose fan-out over an empty snapshot is a harmless no-op.

Lazy FastAPI/uvicorn imports
----------------------------
FastAPI and uvicorn are heavy, optional-at-import-time dependencies (they may not
be installed in every environment, e.g. a minimal CI running only the pure-core
tests). They are therefore imported **lazily inside methods**, never at module
top level, so ``import medusa.server.status_server`` always succeeds. The
registry-delegating methods (:meth:`broadcast`, :meth:`current_status`) and the
no-broadcast degradation path work with or without FastAPI/uvicorn present.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from medusa.core.models import Status
from medusa.server.payload import serialize_payload
from medusa.server.registry import ClientRegistry

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from fastapi import FastAPI, WebSocket

__all__ = [
    "StatusServer",
    "PortBindError",
]

logger = logging.getLogger(__name__)

# Loopback host so only local processes can connect (design decision #2).
_HOST = "127.0.0.1"
# Default WebSocket route for Status_Clients.
_WS_PATH = "/ws"
# Req 5.1: the socket must be listening within 5 seconds of start.
_DEFAULT_BIND_TIMEOUT_S = 5.0

# A tiny self-contained live dashboard served at ``/`` so a browser can
# visualize the privacy status without any external tooling. It opens the
# ``/ws`` WebSocket and flips a big SAFE / THREAT indicator in real time.
_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Screen Privacy Guardian - Live Status</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 1.5rem;
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0d1117; color: #e6edf3; transition: background .25s ease;
  }
  body.safe   { background: #0b2e1a; }
  body.threat { background: #3a0d12; }
  h1 { font-size: 1.1rem; font-weight: 500; letter-spacing: .04em;
       opacity: .75; margin: 0; text-transform: uppercase; }
  #status {
    font-size: 4.5rem; font-weight: 800; letter-spacing: .02em;
    padding: 1.2rem 2.8rem; border-radius: 1rem; border: 2px solid currentColor;
    transition: all .2s ease; text-align: center;
  }
  .safe   #status { color: #3fb950; box-shadow: 0 0 40px #3fb95055; }
  .threat #status { color: #ff6b6b; box-shadow: 0 0 60px #ff6b6b66;
                    animation: pulse .9s ease-in-out infinite; }
  .connecting #status, .closed #status { color: #8b949e; }
  @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.04)} }
  #meta { font-size: .85rem; opacity: .55; font-variant-numeric: tabular-nums; }
  #dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%;
         background:#8b949e; margin-right:.4rem; vertical-align:middle; }
  .safe #dot { background:#3fb950; } .threat #dot { background:#ff6b6b; }
</style>
</head>
<body class="connecting">
  <h1>Screen Privacy Guardian</h1>
  <div id="status">CONNECTING...</div>
  <div id="meta"><span id="dot"></span><span id="conn">connecting to ws://%HOST%:%PORT%/ws</span>
    &nbsp;&middot;&nbsp; last update: <span id="ts">--:--:--</span></div>
<script>
  const body = document.body;
  const statusEl = document.getElementById('status');
  const connEl = document.getElementById('conn');
  const tsEl = document.getElementById('ts');
  const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://')
    + location.host + '/ws';

  function setState(cls, text) {
    body.className = cls;
    statusEl.textContent = text;
  }

  function connect() {
    setState('connecting', 'CONNECTING...');
    const ws = new WebSocket(WS_URL);
    ws.onopen = () => { connEl.textContent = 'connected: ' + WS_URL; };
    ws.onmessage = (ev) => {
      let s;
      try { s = JSON.parse(ev.data).status; } catch { s = ev.data; }
      if (s === 'THREAT_DETECTED') setState('threat', 'THREAT DETECTED');
      else if (s === 'SAFE') setState('safe', 'SAFE');
      else statusEl.textContent = String(s);
      tsEl.textContent = new Date().toLocaleTimeString();
    };
    ws.onclose = () => {
      setState('closed', 'DISCONNECTED');
      connEl.textContent = 'connection closed - retrying in 1s';
      setTimeout(connect, 1000);
    };
    ws.onerror = () => { ws.close(); };
  }
  connect();
</script>
</body>
</html>"""


class PortBindError(RuntimeError):
    """Raised internally when the configured port cannot be bound.

    Surfaced via logging and the :attr:`StatusServer.port_error` property rather
    than propagated to the caller, so detection + overlay continue without
    broadcasting (Req 5.4).
    """


class StatusServer:
    """Local FastAPI WebSocket server broadcasting the current privacy status.

    The server runs programmatically via ``uvicorn.Server`` on the asyncio loop
    (so it shares the single I/O loop with the Alert_Service). All connected-client
    bookkeeping, broadcast fan-out, per-client failure isolation, and late-joiner
    retention are delegated to the injected :class:`ClientRegistry`.

    Degradation (Req 5.4): if the port cannot be bound, :meth:`start` returns
    without raising; :attr:`broadcasting` is then ``False`` and :attr:`port_error`
    holds the surfaced error. Subsequent :meth:`broadcast` calls remain safe
    no-ops in that state.
    """

    def __init__(
        self,
        registry: Optional[ClientRegistry] = None,
        *,
        bind_timeout_s: float = _DEFAULT_BIND_TIMEOUT_S,
    ) -> None:
        """Create a server bound to a :class:`ClientRegistry`.

        :param registry: The client registry to delegate to. A fresh one is
            created when omitted; tests may inject a fake/real registry.
        :param bind_timeout_s: Maximum seconds to wait for the socket to come up
            before treating the bind as failed (Req 5.1, default 5s).
        """
        self._registry = registry if registry is not None else ClientRegistry()
        self._bind_timeout_s = bind_timeout_s

        self._app: Any = None  # FastAPI app, built lazily on start.
        self._server: Any = None  # uvicorn.Server instance.
        self._serve_task: Optional[asyncio.Task[Any]] = None
        self._port: Optional[int] = None
        self._broadcasting = False
        self._port_error: Optional[BaseException] = None

    # -- Introspection -----------------------------------------------------

    @property
    def broadcasting(self) -> bool:
        """Whether the server bound successfully and can broadcast (Req 5.4).

        ``True`` only after a successful :meth:`start`; ``False`` before start,
        after a bind failure, and after :meth:`stop`.
        """
        return self._broadcasting

    @property
    def port_error(self) -> Optional[BaseException]:
        """The surfaced bind error when the port was unavailable, else ``None``."""
        return self._port_error

    @property
    def port(self) -> Optional[int]:
        """The port the server was asked to bind, once :meth:`start` has run."""
        return self._port

    @property
    def registry(self) -> ClientRegistry:
        """The underlying client registry (for wiring/tests)."""
        return self._registry

    # -- App construction --------------------------------------------------

    def build_app(self) -> "FastAPI":
        """Construct the FastAPI app with the ``/ws`` endpoint (lazy import).

        Importing FastAPI here (rather than at module top level) keeps the module
        importable without FastAPI installed. The WebSocket endpoint registers the
        client, immediately sends the current status (Req 5.6), then loops until
        the client disconnects or a send fails, removing the client on exit (so a
        failed/closed connection is cleaned up, consistent with Req 5.7).
        """
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse

        app = FastAPI()
        registry = self._registry
        port = self._port

        @app.get("/", response_class=HTMLResponse)
        async def dashboard() -> str:  # noqa: WPS430
            """Serve the live status dashboard so a browser can visualize status."""
            html = _DASHBOARD_HTML.replace("%HOST%", _HOST)
            html = html.replace("%PORT%", str(port if port is not None else ""))
            return html

        async def status_ws(websocket) -> None:  # noqa: WPS430
            await websocket.accept()
            # Req 5.6: send the retained current status immediately on connect.
            sent = await registry.send_current(websocket)
            if not sent:
                # Could not deliver the initial status; abandon the connection.
                return
            await registry.add(websocket)
            try:
                # The server pushes status via broadcast(); we only need to keep
                # the connection open and notice disconnects. Any inbound frames
                # are ignored (clients are receive-only consumers).
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception:  # noqa: BLE001 - isolate any per-connection error
                logger.debug("Status_Client connection ended with error", exc_info=True)
            finally:
                await registry.remove(websocket)

        # Annotate the parameter with the REAL WebSocket class (an object, not a
        # string). ``from __future__ import annotations`` stringifies inline
        # annotations, and FastAPI resolves string annotations against the
        # module globals -- where ``WebSocket`` is not present (it is a lazy,
        # function-local import). An unresolved annotation makes FastAPI treat
        # ``websocket`` as a required query parameter and reject the handshake
        # with HTTP 403. Setting the annotation to the class object directly
        # lets FastAPI recognise and inject the WebSocket. 
        status_ws.__annotations__["websocket"] = WebSocket
        app.add_api_websocket_route(_WS_PATH, status_ws)

        self._app = app
        return app

    # -- Lifecycle ---------------------------------------------------------

    async def start(self, port: int) -> bool:
        """Start the server on ``127.0.0.1:port`` (Req 5.1 / 5.4).

        Builds the app, configures ``uvicorn.Server`` programmatically, launches
        it as a background task on the current event loop, and waits up to
        :attr:`bind_timeout_s` for the socket to come up.

        On success :attr:`broadcasting` becomes ``True`` and the method returns
        ``True``. On bind failure (port in use / permission denied / timeout) the
        error is caught and surfaced via :attr:`port_error`, :attr:`broadcasting`
        stays ``False``, and the method returns ``False`` **without raising** so
        the caller continues detection + overlay without broadcasting (Req 5.4).

        :param port: TCP port to bind on the loopback interface.
        :returns: ``True`` if bound and serving, ``False`` on bind failure.
        """
        import uvicorn

        self._port = port
        self._port_error = None
        self._broadcasting = False

        app = self._app if self._app is not None else self.build_app()
        config = uvicorn.Config(
            app,
            host=_HOST,
            port=port,
            log_level="warning",
            # Pin to the websockets backend; websockets is pinned to 12.x which
            # uvicorn 0.54 fully supports (newer 17.x rejected the handshake with
            # HTTP 403).
            ws="websockets",
            # We manage startup/shutdown ourselves and install no signal handlers
            # since the server shares a loop in a worker thread.
            lifespan="off",
        )
        server = uvicorn.Server(config)
        # Prevent uvicorn from hijacking process-wide signal handlers; it runs on
        # a dedicated worker thread, not the main thread.
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._server = server

        loop = asyncio.get_event_loop()
        self._serve_task = loop.create_task(server.serve())

        try:
            await self._await_started(server)
        except PortBindError as exc:
            self._port_error = exc
            self._broadcasting = False
            logger.error(
                "Status_Server could not bind %s:%s - continuing detection and "
                "overlay without broadcasting: %s",
                _HOST,
                port,
                exc,
            )
            await self._cancel_serve_task()
            return False

        self._broadcasting = True
        logger.info("Status_Server listening on ws://%s:%s%s", _HOST, port, _WS_PATH)
        return True

    async def _await_started(self, server: Any) -> None:
        """Wait until uvicorn reports ``started`` or fail within the timeout.

        uvicorn sets ``server.started = True`` once the socket is bound and
        listening. If the serve task dies first (bind error) or the timeout
        elapses, raise :class:`PortBindError` so :meth:`start` can degrade.
        """
        deadline = asyncio.get_event_loop().time() + self._bind_timeout_s
        while True:
            if getattr(server, "started", False):
                return
            serve_task = self._serve_task
            if serve_task is not None and serve_task.done():
                # The serve coroutine exited before signalling started: the bind
                # almost certainly failed. Surface its exception as a bind error.
                exc = serve_task.exception()
                raise PortBindError(
                    f"server stopped before binding {_HOST}:{self._port}"
                ) from exc
            if asyncio.get_event_loop().time() >= deadline:
                raise PortBindError(
                    f"timed out after {self._bind_timeout_s}s binding "
                    f"{_HOST}:{self._port}"
                )
            await asyncio.sleep(0.02)

    async def _cancel_serve_task(self) -> None:
        """Cancel the background serve task, swallowing cancellation errors."""
        task = self._serve_task
        self._serve_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    async def stop(self) -> None:
        """Gracefully shut down the server and stop broadcasting.

        Signals uvicorn to exit, awaits the serve task, and clears broadcasting
        state. Safe to call when the server never started or already stopped.
        """
        self._broadcasting = False
        server = self._server
        if server is not None:
            # uvicorn watches ``should_exit`` to unwind serve() cleanly.
            server.should_exit = True
        await self._cancel_serve_task()
        self._server = None

    # -- Broadcasting / status (registry delegation) -----------------------

    async def broadcast(self, status: Status) -> list[Any]:
        """Broadcast ``status`` to all connected clients (Req 5.2 / 5.3 / 5.5).

        Delegates to the registry, which fans out over a snapshot with per-client
        failure isolation and retains the latest status for late joiners. When no
        clients are connected the fan-out is a harmless no-op (Req 5.5).

        When the server failed to bind (:attr:`broadcasting` is ``False``) the
        status is still retained in the registry (so a future client would be
        consistent), but there are no clients to receive it; the call stays a safe
        no-op and never raises (Req 5.4).

        :returns: The list of clients whose send failed and were removed.
        """
        return await self._registry.broadcast(status)

    def current_status(self) -> Status:
        """Return the current retained privacy status (delegates to registry)."""
        return self._registry.current_status()
