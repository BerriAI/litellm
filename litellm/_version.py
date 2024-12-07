import importlib_metadata

try:
    __version__ = importlib_metadata.version("litellm")
except Exception:
    __version__ = "unknown"
