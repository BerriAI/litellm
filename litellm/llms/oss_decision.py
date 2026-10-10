from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from litellm.secret_managers.main import get_secret_str

OssDecisionProvider: TypeAlias = Literal["laya", "bespoke"]
OSS_DECISION_MODELS: Final = MappingProxyType(
    {
        "laya": ("english", "multilingual", "typed-decisions"),
        "bespoke": ("nimble-latest", "nimble", "bespokelabs/Bespoke-Nimble-9B"),
    }
)


def validate_oss_model(provider: OssDecisionProvider, value: object) -> str:
    if not isinstance(value, str) or value not in OSS_DECISION_MODELS[provider]:
        raise ValueError(f"{provider} model must be one of {', '.join(OSS_DECISION_MODELS[provider])}")
    return value


def validate_oss_request(provider: OssDecisionProvider, body: Mapping[str, object]) -> str:
    if "custom_body" in body:
        raise ValueError(f"custom_body is not supported for {provider} requests")
    if body.get("stream"):
        raise ValueError(f"Streaming is not supported for {provider} requests")
    return validate_oss_model(provider, body.get("model"))


@dataclass(frozen=True, slots=True)
class OssDecisionConnection:
    api_base: str
    api_key: str | None = field(repr=False)


def validate_oss_api_base(provider: OssDecisionProvider, value: str) -> str:
    try:
        url: Final = TypeAdapter(AnyHttpUrl).validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"{provider} api_base must be an HTTP or HTTPS server URL") from exc
    if url.username or url.password or url.query or url.fragment:
        raise ValueError(f"{provider} api_base must not contain credentials, a query, or a fragment")
    return str(url).rstrip("/")


def oss_connection(
    provider: OssDecisionProvider, api_base: str | None = None, api_key: str | None = None
) -> OssDecisionConnection:
    base: Final = api_base if api_base is not None else get_secret_str(f"{provider.upper()}_API_BASE")
    if not base:
        raise ValueError(f"{provider} requires api_base or {provider.upper()}_API_BASE pointing to its server")
    key: Final = api_key if api_base is not None else api_key or get_secret_str(f"{provider.upper()}_API_KEY")
    return OssDecisionConnection(api_base=validate_oss_api_base(provider, base), api_key=key)
