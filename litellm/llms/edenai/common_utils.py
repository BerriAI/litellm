"""
Pieces shared by every Eden AI endpoint: credentials, the exception class, and the per-request
`cost` Eden reports at the top level of each response body, or in a header when the body is binary.
"""

from collections.abc import Container, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import AliasChoices, BaseModel, Field, ValidationError

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

EDENAI_API_BASE: Final = "https://api.edenai.run/v3"
EDENAI_COST_HEADER: Final = "x-edenai-cost"


class EdenAIException(BaseLLMException):
    pass


class _EdenAIExtras(BaseModel):
    cost: float | None = Field(default=None, validation_alias=AliasChoices("cost", EDENAI_COST_HEADER))


def resolve_api_base(api_base: str | None) -> str:
    return api_base or get_secret_str("EDENAI_API_BASE") or EDENAI_API_BASE


def resolve_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("EDENAI_API_KEY")


def require_api_key(api_key: str | None, model: str) -> str:
    resolved: Final = resolve_api_key(api_key or litellm.api_key)
    if resolved is None:
        raise AuthenticationError(
            message="Missing Eden AI API key: set EDENAI_API_KEY or pass api_key",
            llm_provider=LlmProviders.EDENAI.value,
            model=model,
        )
    return resolved


def reported_cost(payload: object) -> float | None:
    try:
        extras: Final = (
            _EdenAIExtras.model_validate_json(payload)
            if isinstance(payload, bytes)
            else _EdenAIExtras.model_validate(payload)
        )
    except ValidationError:
        return None
    return extras.cost


def authorized_headers(
    headers: Mapping[str, object], api_key: str | None, model: str
) -> dict[str, object]:  # mutable-ok: header contract
    return {**headers, "Authorization": f"Bearer {require_api_key(api_key, model)}"}  # mutable-ok: header contract


def json_headers(
    headers: Mapping[str, object], api_key: str | None, model: str
) -> dict[str, object]:  # mutable-ok: header contract
    """The shared HTTP handler sends some JSON bodies as raw content, so the type must be set here."""
    authorized: Final = authorized_headers(headers, api_key, model)
    return {**authorized, "Content-Type": "application/json"}  # mutable-ok: header contract


def endpoint_url(api_base: str | None, path: str) -> str:
    return f"{resolve_api_base(api_base).rstrip('/')}/{path}"


def pick(params: Mapping[str, object], keys: Container[str]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in params.items() if key in keys})
