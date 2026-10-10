"""GUI launcher for the Screen Privacy Guardian MVP (visual demo).

``medusa.app.main.main()`` is intentionally headless-friendly: it starts the
asyncio I/O thread, the vision producer thread, and constructs the Desktop_Shield,
but then blocks on a plain thread event. A Qt overlay, however, only paints and
stays responsive while a ``QApplication`` event loop runs **on the main thread**.

This launcher wires the real :class:`MedusaApp` and then drives
``QApplication.exec()`` on the main thread so the privacy overlay is actually
visible. Everything else (config, consent, server, vision, alert) is unchanged.

Run it with the Python 3.11 environment that has the native wheels installed:

    .\.venv311\Scripts\python.exe run_mvp.py

What you should see:
- The app starts; a first-run consent prompt appears in this console (answer y/N).
- Your webcam is acquired (an LED may light up). Nothing leaves your machine.
- When a SECOND face appears close behind you and looks toward the screen, the
  full-screen moiré privacy overlay snaps to opaque. When the snooper leaves /
  looks away (after the cooldown) it fades back to transparent.
- A local WebSocket status server runs on ws://127.0.0.1:<port>/ws broadcasting
  {"status": "THREAT_DETECTED"} / {"status": "SAFE"}.

Press Ctrl+C in this console (or close via the tray/task manager) to stop.
"""

from __future__ import annotations

import logging
import signal
import sys

from medusa.app.main import MedusaApp
from medusa.config.logging_setup import setup_logging

_log = logging.getLogger("medusa.run_mvp")


def main() -> int:
    setup_logging(level=logging.INFO)

    # A QApplication must exist on the main thread before any overlay is built
    # and before exec() is called. Import PySide6 here so a missing wheel gives
    # a clear message rather than an obscure failure deep in startup.
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        _log.error(
            "PySide6 is required to visualize the overlay but is not installed "
            "in this interpreter (%r). Run this with the 3.11 venv that has the "
            "native wheels: .\\.venv311\\Scripts\\python.exe run_mvp.py",
            exc,
        )
        return 1

    app_qt = QApplication.instance() or QApplication(sys.argv)

    # Build and start the full system. MedusaApp's default shield factory reuses
    # the existing QApplication instance we just created.
    medusa = MedusaApp()
    medusa.start()

    _log.info(
        "Medusa_Desktop_Core is running on port %s. Move a second face close "
        "behind you and look at the screen to trigger the privacy shield. "
        "Press Ctrl+C to stop.",
        medusa.config.server_port,
    )

    # Let Ctrl+C interrupt the Qt event loop: a periodic no-op timer returns
    # control to the Python interpreter so the SIGINT handler can fire.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    keepalive = QTimer()
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)

    try:
        exit_code = app_qt.exec()
    finally:
        _log.info("Shutting down Medusa_Desktop_Core...")
        medusa.shutdown()
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
