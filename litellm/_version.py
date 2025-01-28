import importlib.metadata

try:
    version = importlib.metadata.version("litellm")
except Exception:
    pass
