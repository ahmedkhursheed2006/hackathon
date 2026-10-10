"""``python -m medusa`` entry point.

Delegates to :func:`medusa.app.main.main` so the application can be launched as
a module. Kept trivial and import-safe: the heavy subsystems are only touched
once :func:`main` builds and starts the real :class:`~medusa.app.main.MedusaApp`.
"""

from __future__ import annotations

from medusa.app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
