import os
import re
from dataclasses import dataclass
from typing import Final

from litellm.constants import GITHUB_COPILOT_DEFAULT_HOST

from .authenticator import (
    DEFAULT_GITHUB_ACCESS_TOKEN_URL,
    DEFAULT_GITHUB_API_KEY_URL,
    DEFAULT_GITHUB_CLIENT_ID,
    DEFAULT_GITHUB_DEVICE_CODE_URL,
)

_GHE_SUBDOMAIN_PATTERN: Final = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


@dataclass(frozen=True, slots=True)
class GithubHostEndpoints:
    host: str
    device_code_url: str
    access_token_url: str
    api_key_url: str
    user_url: str
    client_id: str
    copilot_api_hosts: frozenset[str]
    copilot_api_host_suffix: str | None
    default_copilot_api_base: str


def normalized_github_host(raw: object) -> str | None:
    if raw is None or raw == "":
        return GITHUB_COPILOT_DEFAULT_HOST
    if not isinstance(raw, str):
        return None

    host: Final = raw.lower()
    if host == GITHUB_COPILOT_DEFAULT_HOST:
        return host
    if not host.endswith(".ghe.com"):
        return None

    subdomain: Final = host.removesuffix(".ghe.com")
    return host if _GHE_SUBDOMAIN_PATTERN.fullmatch(subdomain) is not None else None


def github_host_endpoints(host: str) -> GithubHostEndpoints:
    normalized: Final = normalized_github_host(host)
    if normalized is None:
        raise ValueError("host must be github.com or a single-subdomain GHE.com hostname")
    client_id: Final = os.getenv("GITHUB_COPILOT_CLIENT_ID", DEFAULT_GITHUB_CLIENT_ID)
    if normalized == GITHUB_COPILOT_DEFAULT_HOST:
        return GithubHostEndpoints(
            host=normalized,
            device_code_url=os.getenv("GITHUB_COPILOT_DEVICE_CODE_URL", DEFAULT_GITHUB_DEVICE_CODE_URL),
            access_token_url=os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_URL", DEFAULT_GITHUB_ACCESS_TOKEN_URL),
            api_key_url=os.getenv("GITHUB_COPILOT_API_KEY_URL", DEFAULT_GITHUB_API_KEY_URL),
            user_url="https://api.github.com/user",
            client_id=client_id,
            copilot_api_hosts=frozenset({"githubcopilot.com"}),
            copilot_api_host_suffix=".githubcopilot.com",
            default_copilot_api_base="https://api.githubcopilot.com",
        )

    subdomain: Final = normalized.removesuffix(".ghe.com")
    api_host: Final = f"api.{subdomain}.ghe.com"
    copilot_api_host: Final = f"copilot-api.{subdomain}.ghe.com"
    return GithubHostEndpoints(
        host=normalized,
        device_code_url=f"https://{normalized}/login/device/code",
        access_token_url=f"https://{normalized}/login/oauth/access_token",
        api_key_url=f"https://{api_host}/copilot_internal/v2/token",
        user_url=f"https://{api_host}/user",
        client_id=client_id,
        copilot_api_hosts=frozenset({copilot_api_host}),
        copilot_api_host_suffix=None,
        default_copilot_api_base=f"https://{copilot_api_host}",
    )
