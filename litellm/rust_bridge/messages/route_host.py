from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final, cast  # noqa: TID251  # narrows the normalized native payload to the public TypedDict

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import normalize_drop_params
from litellm.llms.anthropic.experimental_pass_through.utils import is_reasoning_auto_summary_enabled
from litellm.rust_bridge import failures
from litellm.rust_bridge.messages.entrypoints import LiteLLMMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

_DROP_PATHS: Final = TypeAdapter(list[object])


@dataclass(frozen=True, slots=True)
class EffortTiers:
    minimal: bool
    low: bool
    medium: bool
    high: bool
    xhigh: bool
    max: bool


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    supports_reasoning: bool
    supports_adaptive_thinking: bool
    thinking_always_on: bool
    supports_legacy_thinking: bool
    supports_output_config: bool
    supports_sampling_params: bool
    supports_speed: bool
    effort_tiers: EffortTiers


@dataclass(frozen=True, slots=True)
class MessagesShaping:
    capabilities: ModelCapabilities
    drop_params: bool
    reasoning_auto_summary: bool
    additional_drop_params: Sequence[str]


def response(value: Mapping[str, object]) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict over the normalized native payload
        AnthropicMessagesResponse,
        dict(value),  # mutable-ok: the public Messages response is a TypedDict the caller may annotate in place
    )


def arguments(request: LiteLLMMessagesRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMMessagesRequest, request_provider: str) -> Exception:
    if getattr(error, "messages_request_error", False):
        return litellm.BadRequestError(
            message=str(error),
            model=request.model.removeprefix(f"{request_provider}/"),
            llm_provider=request_provider,
        )
    return failures.map_native_failure(error, request.model, request_provider, arguments(request), request.api_base)


def _resolved_provider(model: str, custom_llm_provider: str | None) -> tuple[str, str]:
    try:
        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # an unroutable model still shapes as a bare Anthropic id
        return model, custom_llm_provider or "anthropic"
    return resolved_model, provider


def model_capabilities(model: str, custom_llm_provider: str | None) -> ModelCapabilities:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    resolved_model, provider = _resolved_provider(model, custom_llm_provider)

    def supports(flag: str) -> bool:
        return AnthropicModelInfo._supports_model_capability(model, flag, provider)  # pyright: ignore[reportPrivateUsage]  # same probes the Python transform runs; forking them would drift

    def tier(level: str) -> bool:
        return AnthropicConfig._supports_effort_level(model, level, provider)  # pyright: ignore[reportPrivateUsage]  # same probe the Python transform runs

    return ModelCapabilities(
        supports_reasoning=supports("supports_reasoning"),
        supports_adaptive_thinking=supports("supports_adaptive_thinking"),
        thinking_always_on=supports("thinking_always_on"),
        supports_legacy_thinking=supports("supports_legacy_thinking"),
        supports_output_config=supports("supports_output_config"),
        supports_sampling_params=AnthropicModelInfo._supports_sampling_params(resolved_model),  # pyright: ignore[reportPrivateUsage]  # same gate the handler applies
        supports_speed=AnthropicConfig._model_supports_speed_param(resolved_model, provider),  # pyright: ignore[reportPrivateUsage]  # same gate the handler applies
        effort_tiers=EffortTiers(
            minimal=tier("minimal"),
            low=tier("low"),
            medium=tier("medium"),
            high=tier("high"),
            xhigh=tier("xhigh"),
            max=tier("max"),
        ),
    )


def _drop_params(kwargs: Mapping[str, object]) -> bool:
    return bool(litellm.drop_params) or normalize_drop_params(kwargs.get("drop_params")) is True


def _additional_drop_params(kwargs: Mapping[str, object]) -> tuple[str, ...]:
    try:
        configured: Final = _DROP_PATHS.validate_python(kwargs.get("additional_drop_params"))
    except ValidationError:
        return ()
    return tuple(path for path in configured if isinstance(path, str))


def shaping(model: str, custom_llm_provider: str | None, kwargs: Mapping[str, object]) -> dict[str, object]:
    return asdict(
        MessagesShaping(
            capabilities=model_capabilities(model, custom_llm_provider),
            drop_params=_drop_params(kwargs),
            reasoning_auto_summary=is_reasoning_auto_summary_enabled(),
            additional_drop_params=_additional_drop_params(kwargs),
        )
    )
