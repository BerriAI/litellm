import importlib_metadata

# Stably fork identifier - helps identify this is a Stably-customized build
STABLY_FORK = True
STABLY_FORK_VERSION = "stably-2"  # Increment this when making Stably-specific changes

try:
    _base_version = importlib_metadata.version("litellm")
    version = f"{_base_version}-{STABLY_FORK_VERSION}"
except Exception:
    version = f"unknown-{STABLY_FORK_VERSION}"
