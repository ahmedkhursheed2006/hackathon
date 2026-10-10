"""Shared pytest fixtures and Hypothesis profile configuration.

Registers Hypothesis profiles for the Screen Privacy Guardian test suite. Every
property-based test in this project runs a minimum of 100 examples, as mandated
by the design (``@settings(max_examples=100)`` or higher).
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, Phase, settings

# Default/CI profile: the project baseline of at least 100 examples per property.
settings.register_profile(
    "default",
    max_examples=100,
    deadline=None,
)

# A more thorough profile for local deep runs.
settings.register_profile(
    "thorough",
    max_examples=500,
    deadline=None,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target, Phase.shrink),
)

# A faster profile for quick iteration; still meets the >=100 example minimum.
settings.register_profile(
    "dev",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Activate the profile named by the HYPOTHESIS_PROFILE env var, defaulting to
# the project baseline which enforces max_examples >= 100.
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
