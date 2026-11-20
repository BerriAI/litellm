"""
Pieces shared by every deAPI endpoint: credentials, the exception class, and the per-request cost deAPI
reports as `usage.cost` in JSON bodies and in the `X-DeAPI-Cost` header, the only place for binary answers.
"""

from collections.abc import Container, Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import ConfigDict, Field, ValidationError

import litellm
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import LlmProviders

DEAPI_API_BASE: Final = "https://oai.deapi.ai/v1"
DEAPI_COST_HEADER: Final = "x-deapi-cost"


class DeapiException(BaseLLMException):
    pass


class _DeapiUsage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    cost: float | None = None


class _DeapiBody(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    usage: _DeapiUsage | None = None


class _DeapiCostHeader(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    cost: float | None = Field(default=None, validation_alias=DEAPI_COST_HEADER)


def resolve_api_base(api_base: str | None) -> str:
    return api_base or get_secret_str("DEAPI_API_BASE") or DEAPI_API_BASE


def resolve_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("DEAPI_API_KEY")


def require_api_key(api_key: str | None, model: str) -> str:
    resolved: Final = resolve_api_key(api_key or litellm.api_key)
    if resolved is None:
        raise AuthenticationError(
            message="Missing deAPI API key: set DEAPI_API_KEY or pass api_key",
            llm_provider=LlmProviders.DEAPI.value,
            model=model,
        )
    return resolved


def body_cost(payload: object) -> float | None:
    try:
        body: Final = (
            _DeapiBody.model_validate_json(payload)
            if isinstance(payload, bytes)
            else _DeapiBody.model_validate(payload)
        )
    except ValidationError:
        return None
    return body.usage.cost if body.usage is not None else None


def header_cost(headers: httpx.Headers) -> float | None:
    try:
        return _DeapiCostHeader.model_validate(dict(headers)).cost
    except ValidationError:
        return None


def reported_cost(response: httpx.Response) -> float | None:
    from_header: Final = header_cost(response.headers)
    return from_header if from_header is not None else body_cost(response.content)


def authorized_headers(
    headers: Mapping[str, object], api_key: str | None, model: str
) -> dict[str, object]:  # mutable-ok: header contract
    return {**headers, "Authorization": f"Bearer {require_api_key(api_key, model)}"}


def json_headers(
    headers: Mapping[str, object], api_key: str | None, model: str
) -> dict[str, object]:  # mutable-ok: header contract
    """The shared HTTP handler sends some JSON bodies as raw content, so the type must be set here."""
    authorized: Final = authorized_headers(headers, api_key, model)
    return {**authorized, "Content-Type": "application/json"}


def endpoint_url(api_base: str | None, path: str) -> str:
    return join_api_base(resolve_api_base(api_base), path)


def pick(params: Mapping[str, object], keys: Container[str]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in params.items() if key in keys})
