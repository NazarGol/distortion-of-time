"""
Optional post-processing effects, applied AFTER the core weave (see weave.py).

Import the effect modules for their @register side effects; the registry in
effects.base then knows every available effect.
"""
from effects.base import (  # noqa: F401
    Effect, available, build, get, register, to_u8,
)

# Import effect modules for their @register side effects.
from effects import feedback  # noqa: F401,E402
from effects import blur      # noqa: F401,E402
