import importlib_metadata

try:
    version = importlib_metadata.version("litellm")
except Exception:
    pass
