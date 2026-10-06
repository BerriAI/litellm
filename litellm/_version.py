import importlib_metadata

try:
    try:
        version = importlib_metadata.version("litellm")
    except importlib_metadata.PackageNotFoundError:
        version = importlib_metadata.version("litellm-core")
except Exception:
    version = "unknown"
