"""
Check if prompt caching is valid for a given deployment

Route to previously cached model id, if valid
"""

import secrets
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final, NamedTuple, Protocol, TypedDict, cast

from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly

from litellm import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.integrations.anthropic_cache_control_hook import (
    AllToolParamValues,
    AnthropicCacheControlHook,
)
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.routing_common_utils import deployment_ids_failed_in_request
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.llms.bedrock.common_utils import bedrock_cache_point_ttl_supported, bedrock_capability_model
from litellm.responses.litellm_completion_transformation.transformation import (
    RESPONSES_TOOL_TYPES_WITHOUT_CHAT_FORM,
    RESPONSES_WEB_SEARCH_TOOL_TYPES,
    LiteLLMCompletionResponsesConfig,
    ResponsesToolChatForm,
    ResponseTools,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam, ResponseInputParam
from litellm.types.utils import CallTypes, StandardLoggingPayload
from litellm.utils import (
    get_base_model_from_metadata,
    get_prompt_cache_min_tokens,
    is_prompt_caching_valid_prompt,
)

from ..prompt_caching_cache import PROMPT_CACHE_PIN_TTL_SECONDS, PromptCachingCache


class _PromptCachingSuccessEventKwargs(TypedDict, total=False):
    standard_logging_object: ReadOnly[StandardLoggingPayload | None]
    litellm_params: ReadOnly[Mapping[str, object] | None]


class _ResponsesPin(NamedTuple):
    cache_key: str
    marked_ttl_seconds: int
    cacheable_deployment_ids: tuple[str, ...]


class ResponsesPinsInFlight(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object) -> None: ...

    def delete_cache(self, key: str) -> None: ...


class _ResponsesPrompt(NamedTuple):
    messages: list[AllMessageValues]  # mutable-ok: the prefix helpers take chat messages as a list
    tools: list[AllToolParamValues] | None  # mutable-ok: the prefix helpers take chat tools as a list
    key_tools: list[ChatCompletionToolParam] | None  # mutable-ok: the prefix helpers take chat tools as a list


_RESPONSES_PIN_METADATA_KEY: Final = "responses_prompt_cache_pin"
_RESPONSES_PINS_IN_FLIGHT_LIMIT: Final = 10_000
_RESPONSES_PIN_LIFETIME_SECONDS: Final = 3600
_RETRY_RECORDS: Final = TypeAdapter(tuple[object, ...])
_RESPONSES_CALL_TYPES: Final = frozenset((CallTypes.responses.value, CallTypes.aresponses.value))
_RESPONSES_TOOL_TYPES_KEPT_AS_SENT: Final = RESPONSES_TOOL_TYPES_WITHOUT_CHAT_FORM | RESPONSES_WEB_SEARCH_TOOL_TYPES


def new_responses_pins_in_flight() -> ResponsesPinsInFlight:
    return InMemoryCache(
        max_size_in_memory=_RESPONSES_PINS_IN_FLIGHT_LIMIT, default_ttl=_RESPONSES_PIN_LIFETIME_SECONDS
    )


def _key_chat_tool(source: Mapping[str, object], chat_tool: Mapping[str, object]) -> Mapping[str, object]:
    function: Final = as_str_mapping(chat_tool.get("function"))
    if source.get("type") != "function" or function is None:
        return chat_tool
    return {**chat_tool, "function": {key: source[key] for key in function if key in source}}


def _key_chat_tools(
    sources: Sequence[Mapping[str, object]], forms: Sequence[ResponsesToolChatForm]
) -> Iterator[Mapping[str, object]]:
    for source, form in zip(sources, forms, strict=True):
        yield from (_key_chat_tool(source, chat_tool) for chat_tool in form.chat_tools)


def _responses_prompt(request_data: Mapping[str, object]) -> _ResponsesPrompt | None:
    request_input: Final = request_data.get("input")
    request_tools: Final = request_data.get("tools")
    if request_data.get("previous_response_id") is not None or not isinstance(request_input, str | list):
        return None
    listed_tools: Final = (
        cast(ResponseTools, request_tools)  # cast-ok: a bad tool raises inside the guard below and is caught
        if isinstance(request_tools, list)
        else None
    )
    try:
        tools_kept_as_sent: Final = tuple(
            tool for tool in listed_tools or () if tool.get("type") in _RESPONSES_TOOL_TYPES_KEPT_AS_SENT
        )
        convertible_tools: Final = (
            [tool for tool in listed_tools if tool.get("type") not in _RESPONSES_TOOL_TYPES_KEPT_AS_SENT]
            if listed_tools is not None
            else None
        )
        messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(  # pyright: ignore[reportUnknownMemberType]  # its input type is untyped
            input=cast(str | ResponseInputParam, request_input),  # cast-ok: a bad item raises and is caught here
            responses_api_request={"instructions": request_data.get("instructions")},
            item_cache_control=True,
            only_since_compaction=True,
        )
        forms: Final = LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(convertible_tools)
    except Exception as error:  # noqa: BLE001  # the provider judges the input, a routing preference never does
        verbose_logger.debug("Prompt Caching: skipping an input with no chat form: %s", type(error).__name__)
        return None
    tools: Final = list(chain.from_iterable(form.chat_tools for form in forms))
    key_tools: Final = [*_key_chat_tools(convertible_tools or (), forms), *tools_kept_as_sent]
    return _ResponsesPrompt(
        messages=cast(list[AllMessageValues], messages),  # cast-ok: the bridge's chat messages, read as plain JSON
        tools=cast(list[AllToolParamValues], tools) or None,  # cast-ok: the bridge's chat tools, read as plain JSON
        key_tools=cast(list[ChatCompletionToolParam], key_tools) or None,  # cast-ok: tools are hashed as plain JSON
    )


def _retry_records(metadata: Mapping[str, object]) -> tuple[object, ...]:
    try:
        return _RETRY_RECORDS.validate_python(metadata.get("previous_models", ()))
    except ValidationError:
        return ()


def _record_deployment_ids(records: tuple[object, ...]) -> Iterator[str]:
    deployment_ids: Final = ((as_str_mapping(record) or {}).get("deployment_id") for record in records)
    return (deployment_id for deployment_id in deployment_ids if isinstance(deployment_id, str))


def _retried_deployment_ids(metadata: Mapping[str, object]) -> frozenset[str] | None:
    records: Final = _retry_records(metadata)
    retry_count: Final = metadata.get("request_retry_count")
    if type(retry_count) is int and retry_count > len(records):
        return None
    return frozenset(_record_deployment_ids(records))


def _deployment_ids_tried_in_request(request_data: Mapping[str, object]) -> frozenset[str] | None:
    retried: Final = tuple(
        _retried_deployment_ids(as_str_mapping(request_data.get(bucket)) or {})
        for bucket in ("litellm_metadata", "metadata")
    )
    known: Final = tuple(ids for ids in retried if ids is not None)
    if len(known) < len(retried):
        return None
    return deployment_ids_failed_in_request(request_data).union(*known)


def _previous_response_model_id(request_data: Mapping[str, object]) -> str | None:
    previous_response_id: Final = request_data.get("previous_response_id")
    model_id: Final = (
        ResponsesAPIRequestUtils.get_model_id_from_response_id(previous_response_id)
        if isinstance(previous_response_id, str)
        else None
    )
    tried: Final = _deployment_ids_tried_in_request(request_data) if model_id is not None else None
    return model_id if model_id is not None and tried is not None and model_id not in tried else None


def _deployment_min_tokens(deployment: Mapping[str, Mapping[str, object]], fallback: int) -> int:
    deployment_model: Final = deployment.get("litellm_params", {}).get("model")
    return get_prompt_cache_min_tokens(model=deployment_model) if isinstance(deployment_model, str) else fallback


async def _cacheable_deployment_ids(
    healthy_deployments: Sequence[Mapping[str, Mapping[str, object]]],
    reached_minimum: int,
    reaches: Callable[[int], Awaitable[bool]],
) -> tuple[str, ...]:
    minimums: Final = tuple(
        (str(deployment["model_info"]["id"]), _deployment_min_tokens(deployment, reached_minimum))
        for deployment in healthy_deployments
    )
    reached: Final = MappingProxyType(
        {
            minimum: minimum <= reached_minimum or await reaches(minimum)
            for minimum in frozenset(minimum for _, minimum in minimums)
        }
    )
    return tuple(sorted(deployment_id for deployment_id, minimum in minimums if reached[minimum]))


def _deployment_models(healthy_deployments: Sequence[Mapping[str, object]]) -> Iterator[str]:
    models: Final = (
        (as_str_mapping(deployment.get("litellm_params")) or {}).get("model") for deployment in healthy_deployments
    )
    return (model for model in models if isinstance(model, str) and model)


def _get_min_token_count_for_deployments(healthy_deployments: Sequence[Mapping[str, object]]) -> int:
    """
    Returns the lowest minimum cacheable prefix across a model group.

    This gate only decides whether the cache lookup is worth doing. It cannot cause a wrong pin,
    because a deployment is only pinned when the cache already holds an entry for the prefix, and
    entries are written by `async_log_success_event` against the deployment's real model. A model
    that will not cache a prefix never records one, so there is nothing to pin it to.

    That makes the lowest minimum in the group the correct threshold rather than the highest.
    `model` here is the model-group alias the operator chose, not a model name, so the threshold
    has to come from the deployments themselves, and a group may mix models whose minimums differ.
    Taking the highest would skip the lookup for a prefix a lower-minimum member genuinely cached,
    losing a cache hit it had earned. The lowest can only cost a lookup that finds nothing.
    """
    return min(
        (get_prompt_cache_min_tokens(model=model) for model in _deployment_models(healthy_deployments)),
        default=DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT,
    )


def _honors_one_hour_cache_ttl(
    standard_logging_object: StandardLoggingPayload, litellm_params: Mapping[str, object]
) -> bool:
    if standard_logging_object["custom_llm_provider"] == "bedrock":
        deployment_model: Final = standard_logging_object["model"]
        capability_model: Final = bedrock_capability_model(
            deployment_model, get_base_model_from_metadata({"litellm_params": litellm_params})
        )
        return bedrock_cache_point_ttl_supported(capability_model, deployment_model)
    model_map_value: Final = standard_logging_object["model_map_information"]["model_map_value"]
    return model_map_value is not None and (model_map_value.get("cache_creation_input_token_cost_above_1hr") or 0) > 0


class PromptCachingDeploymentCheck(CustomLogger):
    def __init__(
        self,
        cache: DualCache,
        is_priority_group: Callable[[str], bool] | None = None,
        responses_pins_in_flight: ResponsesPinsInFlight | None = None,
        group_has_alternatives: Callable[[str], bool] | None = None,
    ):
        super().__init__()
        self.cache = cache
        self.is_priority_group = is_priority_group
        self.group_has_alternatives: Final = group_has_alternatives
        self.responses_pins_in_flight: Final = (
            responses_pins_in_flight if responses_pins_in_flight is not None else new_responses_pins_in_flight()
        )

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        preferred_id: Final = await self.async_preferred_deployment_id(
            model, healthy_deployments, messages, request_kwargs
        )
        preferred: Final = [
            deployment for deployment in healthy_deployments if deployment["model_info"]["id"] == preferred_id
        ]
        return preferred[:1] or healthy_deployments

    async def async_preferred_deployment_id(
        self,
        model: str,
        healthy_deployments: Sequence[Mapping[str, Mapping[str, object]]],
        messages: list[AllMessageValues] | None,  # mutable-ok: the request's messages list, read as is
        request_data: Mapping[str, object] | None,
    ) -> str | None:
        if self.is_priority_group is not None and self.is_priority_group(model):
            return None
        if request_data is not None and request_data.get("_target_order") is not None:
            return None
        pin_metadata: Final = (
            request_data.get("litellm_metadata") if messages is None and request_data is not None else None
        )
        single_healthy: Final = not healthy_deployments[1:]
        if single_healthy and not (
            isinstance(pin_metadata, dict)
            and self.group_has_alternatives is not None
            and self.group_has_alternatives(model)
        ):
            return None
        previous_attempt_token: Final = (as_str_mapping(pin_metadata) or {}).get(_RESPONSES_PIN_METADATA_KEY)
        previous_response_model_id: Final = (
            _previous_response_model_id(request_data)
            if request_data is not None and isinstance(pin_metadata, dict)
            else None
        )
        if previous_response_model_id is not None:
            return previous_response_model_id
        responses_prompt: Final = (
            await offload_token_count(_responses_prompt)(request_data)
            if request_data is not None and isinstance(pin_metadata, dict)
            else None
        )
        request_tools: Final = (
            responses_prompt.tools
            if responses_prompt is not None
            else cast(  # cast-ok: request_kwargs is untyped; the stand-down scan duck-types every tool it reads
                list[AllToolParamValues] | None, request_data.get("tools")
            )
            if request_data is not None and messages is not None
            else None
        )
        request_messages: Final = responses_prompt.messages if responses_prompt is not None else messages
        prompt_messages: Final = (
            PromptCachingCache.prepend_system_prompt(
                request_messages, request_data.get("system") if request_data is not None else None
            )
            if request_messages is not None
            else None
        )

        async def reaches(minimum: int) -> bool:
            return await offload_token_count(
                is_prompt_caching_valid_prompt
            )(
                messages=prompt_messages,
                model=model,
                min_token_count=minimum,
                tools=cast(  # cast-ok: AllToolParamValues widens the union for Anthropic tool shapes the counter duck-types
                    list[ChatCompletionToolParam] | None,
                    request_tools,
                ),
            )

        group_minimum: Final = _get_min_token_count_for_deployments(healthy_deployments)
        if prompt_messages is not None and await reaches(group_minimum):
            prompt_cache: Final = PromptCachingCache(
                cache=self.cache,
            )

            ## AUTO PROMPT CACHING - the breakpoints this request will carry are injected inside
            ## `litellm.acompletion`, after a deployment has been picked, so the affinity key has to
            ## be derived from the messages as they will be sent, not as they arrive here.
            affinity_messages: Final = AnthropicCacheControlHook.messages_with_default_injections(
                messages=prompt_messages,
                models=_deployment_models(healthy_deployments),
                tools=request_tools,
                enable_prompt_caching=(
                    request_data.get("enable_prompt_caching") is True if request_data is not None else None
                ),
                request_kwargs=request_data,
            )
            prefix_tools: Final = (
                responses_prompt.key_tools
                if responses_prompt is not None
                else cast(  # cast-ok: AllToolParamValues widens the union for Anthropic tool shapes
                    Sequence[ChatCompletionToolParam] | None, request_tools
                )
            )
            positions: Final = await PromptCachingCache.async_prefix_positions(affinity_messages, prefix_tools)
            if responses_prompt is not None and isinstance(pin_metadata, dict) and positions:
                if isinstance(previous_attempt_token, str):
                    self.responses_pins_in_flight.delete_cache(previous_attempt_token)
                pin_token: Final = secrets.token_hex(16)
                self.responses_pins_in_flight.set_cache(
                    pin_token,
                    _ResponsesPin(
                        cache_key=positions[-1].cache_key,
                        marked_ttl_seconds=await offload_token_count(PromptCachingCache.pin_ttl_seconds)(
                            affinity_messages, prefix_tools
                        ),
                        cacheable_deployment_ids=await _cacheable_deployment_ids(
                            healthy_deployments,
                            group_minimum,
                            reaches,
                        ),
                    ),
                )
                pin_metadata[_RESPONSES_PIN_METADATA_KEY] = pin_token
            if single_healthy:
                return None

            model_id_dict: Final = await prompt_cache.async_get_model_id_at(positions)
            tried: Final = _deployment_ids_tried_in_request(request_data or {}) if model_id_dict is not None else None
            if model_id_dict is not None and tried is not None and model_id_dict["model_id"] not in tried:
                return model_id_dict["model_id"]

        return None

    def exempt_from_message_redaction(self) -> bool:
        return True

    async def async_log_success_event(
        self,
        kwargs: _PromptCachingSuccessEventKwargs,
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        recorded_pin: Final = self._take_responses_pin(kwargs)
        standard_logging_object: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)

        if standard_logging_object is None:
            return

        raw_litellm_params: Final = kwargs.get("litellm_params")
        litellm_params: Final[Mapping[str, object]] = (
            raw_litellm_params if isinstance(raw_litellm_params, Mapping) else {}
        )
        metadata: Final = (
            as_str_mapping(litellm_params.get("litellm_metadata"))
            or as_str_mapping(litellm_params.get("metadata"))
            or {}
        )
        model_group_size: Final = metadata.get("model_group_size")
        if type(model_group_size) is int and model_group_size == 1:
            return

        call_type: Final = standard_logging_object["call_type"]

        if call_type in _RESPONSES_CALL_TYPES:
            await self._pin_responses_request(recorded_pin, kwargs, standard_logging_object)
            return

        if (
            call_type != CallTypes.completion.value
            and call_type != CallTypes.acompletion.value
            and call_type != CallTypes.anthropic_messages.value
        ):  # only use prompt caching for completion calls
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, CALL TYPE IS NOT COMPLETION or ANTHROPIC MESSAGE"
            )
            return

        model: Final = standard_logging_object["model"]
        messages: Final = standard_logging_object["messages"]
        model_id: Final = standard_logging_object["model_id"]

        if messages is None or not isinstance(messages, list):
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MESSAGES IS NOT A LIST"
            )
            return
        if model_id is None:
            verbose_logger.debug(
                "litellm.router_utils.pre_call_checks.prompt_caching_deployment_check: skipping adding model id to prompt caching cache, MODEL ID IS NONE"
            )
            return

        event_data: Final = cast(  # cast-ok: success-event kwargs are an untyped dict
            Mapping[str, object], kwargs
        )
        tools: Final = cast(list[AllToolParamValues] | None, event_data.get("tools"))
        prompt_messages: Final = PromptCachingCache.prepend_system_prompt(
            cast(list[AllMessageValues], messages),  # cast-ok: isinstance(list) checked above
            event_data.get("system"),
        )

        ## PROMPT CACHING - cache model id, if prompt caching valid prompt + provider
        if await offload_token_count(is_prompt_caching_valid_prompt)(
            model=model,
            messages=prompt_messages,
            tools=cast(  # cast-ok: AllToolParamValues widens the union for Anthropic tool shapes the counter duck-types
                list[ChatCompletionToolParam] | None,
                tools,
            ),
        ):
            cache: Final = PromptCachingCache(
                cache=self.cache,
            )
            await cache.async_add_model_id(
                model_id=model_id,
                messages=prompt_messages,
                tools=cast(  # cast-ok: AllToolParamValues widens the union for Anthropic tool shapes
                    Sequence[ChatCompletionToolParam] | None,
                    tools,
                ),
                honors_one_hour_ttl=_honors_one_hour_cache_ttl(standard_logging_object, litellm_params),
            )

        return

    def _take_responses_pin(self, kwargs: _PromptCachingSuccessEventKwargs) -> object:
        litellm_params: Final = kwargs.get("litellm_params") or {}
        metadata: Final = litellm_params.get("litellm_metadata")
        pin_token: Final = (as_str_mapping(metadata) or {}).get(_RESPONSES_PIN_METADATA_KEY)
        if not isinstance(pin_token, str):
            return None
        if isinstance(metadata, dict):
            metadata.pop(_RESPONSES_PIN_METADATA_KEY, None)  # pyright: ignore[reportUnknownMemberType]  # the caller's metadata dict
        recorded_pin: Final = self.responses_pins_in_flight.get_cache(pin_token)
        self.responses_pins_in_flight.delete_cache(pin_token)
        return recorded_pin

    async def _pin_responses_request(
        self,
        recorded_pin: object,
        kwargs: _PromptCachingSuccessEventKwargs,
        standard_logging_object: StandardLoggingPayload,
    ) -> None:
        litellm_params: Final = kwargs.get("litellm_params") or {}
        model_id: Final = standard_logging_object["model_id"]
        if not isinstance(recorded_pin, _ResponsesPin) or model_id not in recorded_pin.cacheable_deployment_ids:
            return
        await PromptCachingCache(cache=self.cache).async_set_pin(
            model_id,
            recorded_pin.cache_key,
            recorded_pin.marked_ttl_seconds
            if _honors_one_hour_cache_ttl(standard_logging_object, litellm_params)
            else PROMPT_CACHE_PIN_TTL_SECONDS,
        )
