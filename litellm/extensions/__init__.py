"""Client-side adapters for the external Python extension host."""

from .runtime import configure_extension_runtime, get_extension_runtime

__all__ = ("configure_extension_runtime", "get_extension_runtime")
