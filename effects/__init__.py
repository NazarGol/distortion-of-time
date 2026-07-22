"""Effects package. Importing it registers every built-in effect.

Extension point: add a module here, decorate your Effect subclass with
@register, import it below, and it appears in the UI automatically.
"""
from effects.base import (  # noqa: F401
    Effect,
    available,
    build,
    get,
    register,
)

# Import effect modules for their @register side effects.
from effects import interlace     # noqa: F401,E402
from effects import superimpose   # noqa: F401,E402
from effects import blur          # noqa: F401,E402
