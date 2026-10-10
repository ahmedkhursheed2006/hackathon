"""Verify the /ws route now accepts the handshake."""
import asyncio
import uvicorn
import websockets
from medusa.server.status_server import StatusServer

srv = StatusServer()
srv._port = 8099
app = srv.build_app()

for r in app.routes:
    if getattr(r, "path", None) == "/ws":
        print("query_params:", [p.name for p in r.dependant.query_params])


async def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=8099, log_level="warning",
                            lifespan="off")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if getattr(server, "started", False):
            break
        await asyncio.sleep(0.05)
    try:
        async with websockets.connect("ws://127.0.0.1:8099/ws") as c:
            print("OURS CONNECTED; first:", await asyncio.wait_for(c.recv(), 5))
    except Exception as e:
        print("OURS CONNECT ERROR:", type(e).__name__, e)
    server.should_exit = True
    await task


asyncio.run(main())
