"""Shared by the Anthropic Managed Agents configs (beta managed-agents-2026-04-01)."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.constants import ANTHROPIC_MANAGED_AGENTS_BETA_VERSION
from litellm.llms.anthropic.common_utils import AnthropicError, AnthropicModelInfo
from litellm.types.utils import LlmProviders

ANTHROPIC_VERSION: Final = "2023-06-01"
_DEFAULT_API_BASE: Final = "https://api.anthropic.com"


class _BetaHeader(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str | tuple[str, ...]


def _requested_betas(requested: object) -> frozenset[str]:
    try:
        listed: Final = _BetaHeader.model_validate(MappingProxyType({"value": requested})).value
    except ValidationError:
        return frozenset()
    values: Final = listed.split(",") if isinstance(listed, str) else listed
    return frozenset(beta.strip() for beta in values if beta.strip())


def with_managed_agents_beta(requested: object) -> str:
    return ",".join(sorted(_requested_betas(requested) | frozenset((ANTHROPIC_MANAGED_AGENTS_BETA_VERSION,))))


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def managed_agents_api_base(api_base: str | None) -> str:
    return AnthropicModelInfo.get_api_base(api_base) or _DEFAULT_API_BASE


def managed_agents_headers(
    headers: Mapping[str, str],
    api_key: str | None,
    api_base: str | None,
) -> dict[str, str]:  # mutable-ok: the http handlers pass this straight to httpx as headers
    if api_base and not api_key:
        raise ValueError(
            "When overriding api_base for Anthropic managed agents, you must also supply an explicit api_key. "
            "Falling back to ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN with a custom api_base is refused "
            "to prevent leaking the shared provider key to arbitrary hosts."
        )
    auth_header: Final = AnthropicModelInfo.get_auth_header(api_key, api_base)
    if auth_header is None:
        raise ValueError(
            "Anthropic API key is required. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN, or pass api_key."
        )
    return {  # mutable-ok: the http handlers pass this straight to httpx as headers
        **headers,
        **auth_header,
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": with_managed_agents_beta(headers.get("anthropic-beta")),
        "content-type": "application/json",
    }


def raise_for_status(raw_response: httpx.Response) -> None:
    if 200 <= raw_response.status_code < 300:
        return
    raise AnthropicError(status_code=raw_response.status_code, message=raw_response.text, headers=raw_response.headers)


def invalid_request(message: str) -> litellm.BadRequestError:
    return litellm.BadRequestError(message=message, model="", llm_provider=LlmProviders.ANTHROPIC.value)
