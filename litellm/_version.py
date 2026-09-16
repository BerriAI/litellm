from importlib.metadata import PackageNotFoundError, version as _version

try:
    version = _version("litellm")
except PackageNotFoundError:
    version = "unknown"
