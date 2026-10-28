from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm.anthropic_interface.exceptions.exception_mapping_utils import AnthropicExceptionMapping
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.proxy._types import ProxyException
from litellm.types.token_count import TokenCountProvider, TokenCountSourceFields, TokenCountTokenizer
from litellm.types.utils import TokenCountResponse

CANNOT_COUNT_MODEL_STATUSES: Final = frozenset({400, 404})
TOKENIZER_LOAD_WAIT_SECONDS: Final = 5.0
TOKENIZER_LOAD_REUSE_SECONDS: Final = 60.0
TOKENIZER_LOAD_MAX_RUNNING: Final = 4


class _UpstreamErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    message: str


class _UpstreamErrorEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    type: Literal["error"]
    error: _UpstreamErrorDetail


def _joined(message: str, hint: str) -> str:
    return f"{message.rstrip().rstrip('.')}. {hint}"


def _with_hint(message: str, hint: str) -> str:
    try:
        envelope: Final = _UpstreamErrorEnvelope.model_validate_json(message)
    except ValidationError:
        return _joined(AnthropicExceptionMapping.extract_error_message(message), hint)
    detail: Final = envelope.error.model_copy(update={"message": _joined(envelope.error.message, hint)})
    return envelope.model_copy(update={"error": detail}).model_dump_json()


@dataclass(frozen=True, slots=True)
class TokenCountModel:
    model: str
    region: str | None
    litellm_provider: TokenCountProvider | None = None


@dataclass(frozen=True, slots=True)
class TokenCountSettings:
    framing: int | None
    count_model: TokenCountModel | None
    tokenizer: TokenCountTokenizer | None = None


@dataclass(frozen=True, slots=True)
class ProviderCountFailure:
    status_code: int
    message: str
    rejected_before_provider_call: bool = False

    @property
    def cannot_count_model(self) -> bool:
        return not self.rejected_before_provider_call and self.status_code in CANNOT_COUNT_MODEL_STATUSES

    def with_hint(self, hint: str) -> "ProviderCountFailure":
        return replace(self, message=_with_hint(self.message, hint))

    def to_proxy_exception(self) -> ProxyException:
        return ProxyException(
            message=self.message,
            type="token_counting_error",
            param="model",
            code=self.status_code,
        )


def cost_map_entry(model: str, custom_llm_provider: str | None = None) -> Mapping[str, object]:
    try:
        return litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        return MappingProxyType({})


def _framing(value: object) -> int | None:
    match value:
        case bool():
            return None
        case int() if value >= 0:
            return value
        case float() if value.is_integer() and value >= 0:
            return int(value)
        case _:
            return None


def _non_blank(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _count_provider(value: object) -> TokenCountProvider | None:
    match value:
        case "bedrock":
            return "bedrock"
        case "bedrock_mantle":
            return "bedrock_mantle"
        case _:
            return None


def _count_model(value: Mapping[str, object]) -> TokenCountModel | None:
    model: Final = _non_blank(value.get("model"))
    return (
        None
        if model is None
        else TokenCountModel(
            model=model,
            region=_non_blank(value.get("region")),
            litellm_provider=_count_provider(value.get("litellm_provider")),
        )
    )


def _geography(region: str) -> str:
    return region.rsplit("-", 2)[0]


def _within_deployment_geography(
    count_model: TokenCountModel | None, deployment_region: Callable[[], str]
) -> TokenCountModel | None:
    if count_model is None or count_model.region is None:
        return count_model
    model: Final = count_model.model.removeprefix("bedrock/").removeprefix("bedrock_mantle/")
    if model.startswith("global."):
        return count_model
    if _geography(deployment_region()) == _geography(count_model.region):
        return count_model
    return replace(count_model, region=None)


def _resolved_count_model(
    configured: object, default: object, deployment_region: Callable[[], str]
) -> TokenCountModel | None:
    configured_mapping: Final = as_str_mapping(configured)
    if configured_mapping is not None:
        return _count_model(configured_mapping)
    default_mapping: Final = as_str_mapping(default)
    return (
        None
        if default_mapping is None
        else _within_deployment_geography(_count_model(default_mapping), deployment_region)
    )


def _tokenizer(value: object) -> TokenCountTokenizer | None:
    match value:
        case "kimi_k3":
            return "kimi_k3"
        case _:
            return None


def resolve_token_count_settings(
    configured: Mapping[str, object], default: Mapping[str, object], deployment_region: Callable[[], str]
) -> TokenCountSettings:
    framing: Final = _framing(configured.get("token_count_framing"))
    return TokenCountSettings(
        framing=framing if framing is not None else _framing(default.get("token_count_framing")),
        count_model=_resolved_count_model(
            configured.get("token_count_model"), default.get("token_count_model"), deployment_region
        ),
        tokenizer=_tokenizer(default.get("token_count_tokenizer")),
    )


def count_source_fields(response: TokenCountResponse) -> TokenCountSourceFields:
    match response.count_source:
        case None:
            return {}
        case "token_count_model":
            return {"litellm_count_source": "token_count_model", "litellm_count_model": response.model_used}
        case source:
            return {"litellm_count_source": source}
