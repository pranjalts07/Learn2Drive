"""autonomy_stack: Production-grade autonomous driving ML engineering portfolio."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

try:
    __version__ = version("autonomy_stack")
except Exception:
    __version__ = "0.1.0"

__all__ = ["__version__"]
