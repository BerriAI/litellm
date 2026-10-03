from importlib import metadata

try:
    version = metadata.version("litellm")
except Exception:
    version = "unknown"
