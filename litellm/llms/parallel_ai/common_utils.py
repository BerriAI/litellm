"""
Shared credential and endpoint resolution for the Parallel AI provider.
"""

from typing import Final
from urllib.parse import urlsplit

from litellm.secret_managers.main import get_secret_str

PARALLEL_AI_API_BASE: Final = "https://api.parallel.ai"


def _origin(url: str) -> tuple[str, str]:
    """Scheme and host of a base URL; a scheme-less base is read as https."""
    normalized: Final = url if "://" in url else f"https://{url}"
    split: Final = urlsplit(normalized)
    return split.scheme.lower(), split.netloc.lower()


def _is_trusted_api_base(caller_api_base: str, env_api_base: str | None) -> bool:
    """Whether a caller-supplied base names an origin the server key may be sent to.

    Compares scheme as well as host: matching the host alone would accept
    ``http://api.parallel.ai`` and put the server key on the wire in plaintext.
    """
    trusted: Final = frozenset(_origin(base) for base in (PARALLEL_AI_API_BASE, env_api_base) if base)
    scheme, host = _origin(caller_api_base)
    return bool(host) and (scheme, host) in trusted


def resolve_parallel_ai_credentials(api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
    """
    Resolve the effective (api_base, api_key) pair for a Parallel AI LLM request.

    A server-managed key (from env) is only used when the request targets the
    provider default or the operator's own PARALLEL_AI_API_BASE override; a
    caller-supplied base must bring its own key, so the server credential is
    never forwarded to a caller-chosen host.
    """
    env_api_base: Final = get_secret_str("PARALLEL_AI_API_BASE")
    resolved_api_base: Final = api_base or env_api_base or PARALLEL_AI_API_BASE

    if api_key:
        return resolved_api_base, api_key

    server_api_key: Final = get_secret_str("PARALLEL_AI_API_KEY") or get_secret_str("PARALLEL_API_KEY")
    if server_api_key and api_base and not _is_trusted_api_base(api_base, env_api_base):
        raise ValueError(
            f"Refusing to send the server-configured Parallel AI key to the caller-supplied "
            f"api_base '{api_base}'. Pass an explicit api_key when overriding api_base."
        )
    return resolved_api_base, server_api_key
