_BACKEND = ""

try:
    from .jax_impl import *

    _BACKEND = "jax"
except ImportError:
    from .numpy_impl import *

    _BACKEND = "numpy"
