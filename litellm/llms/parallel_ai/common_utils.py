"""
Shared credential and endpoint resolution for the Parallel AI provider.
"""

from typing import Final
from urllib.parse import urlsplit

from litellm.secret_managers.main import get_secret_str

PARALLEL_AI_API_BASE: Final = "https://api.parallel.ai"


def _host(url: str) -> str:
    normalized: Final = url if "://" in url else f"https://{url}"
    return urlsplit(normalized).netloc.lower()


def _is_trusted_api_base(caller_api_base: str, env_api_base: str | None) -> bool:
    trusted: Final = frozenset(_host(base) for base in (PARALLEL_AI_API_BASE, env_api_base) if base)
    candidate: Final = _host(caller_api_base)
    return bool(candidate) and candidate in trusted


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
