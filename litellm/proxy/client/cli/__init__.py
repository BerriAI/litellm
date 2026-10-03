"""CLI package for LiteLLM Proxy Client."""

try:
    from .main import cli, litellm_proxy_cli
except ModuleNotFoundError as error:
    if error.name not in ("click", "filelock"):
        raise
    raise ImportError('Install the client CLI with pip install "litellm[cli]"') from error

__all__ = ["cli", "litellm_proxy_cli"]
