"""
Dynamic configuration class generator for JSON-based providers.
"""

from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn, TypeAlias, overload

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    handle_messages_with_text_only_content_lists_as_str,
    migrate_file_to_image_url,
)
from litellm.litellm_core_utils.reasoning_effort_capability import declared_reasoning_efforts
from litellm.litellm_core_utils.reasoning_effort_strength import nearest_declared_reasoning_effort
from litellm.litellm_core_utils.url_utils import ensure_api_route, join_api_base
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionUserMessage,
    OpenAIMessageContentListBlock,
)

from .json_loader import SimpleProviderConfig

_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_SCHEMA_MAP_KEYWORDS: Final = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SCHEMA_LITERAL_KEYWORDS: Final = frozenset({"const", "default", "enum", "examples"})


@dataclass(frozen=True, slots=True)
class _FilePartsSentByFileId:
    provider_slug: str


@dataclass(frozen=True, slots=True)
class _MapFieldInResponseFormat:
    provider_slug: str
    model: str


_UnsupportedRequest: TypeAlias = _FilePartsSentByFileId | _MapFieldInResponseFormat


def _raise_public(failure: _UnsupportedRequest) -> NoReturn:
    match failure:
        case _FilePartsSentByFileId(provider_slug=provider_slug):
            raise UnsupportedParamsError(
                message=(
                    f"{provider_slug} cannot read file parts sent by file_id. "
                    "Send the file as file_data, or set `drop_params=True` to drop them"
                ),
                llm_provider=provider_slug,
            )
        case _MapFieldInResponseFormat(provider_slug=provider_slug, model=model):
            raise UnsupportedParamsError(
                message=(
                    f"{provider_slug} does not support schema-valued additionalProperties (map fields) in "
                    "response_format. To drop them, set `drop_params=True`"
                ),
                llm_provider=provider_slug,
                model=model,
            )
        case _:
            assert_never(failure)


if TYPE_CHECKING:
    from litellm.llms.openai_like.responses.transformation import OpenAILikeResponsesConfig

_PAID_SEARCH_TOOLS: Final = frozenset({"web_search", "web_search_image", "image_search"})
_PAYLOAD_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_TOOL_USAGE_ADAPTER: Final[TypeAdapter[Mapping[str, int]]] = TypeAdapter(Mapping[str, int])


def web_search_requests_from_tool_usage(payload: object) -> int | None:
    """
    Billable search calls in a parsed OpenAI-compatible chat response whose usage carries a
    tool_usage {tool_name: count} map, as EmpirioLabs reports built-in tool invocations.
    Free tools (code_interpreter, web_extractor) are not counted.
    """
    try:
        body: Final = _PAYLOAD_ADAPTER.validate_python(payload)
        usage: Final = _PAYLOAD_ADAPTER.validate_python(body.get("usage"))
        tool_usage: Final = _TOOL_USAGE_ADAPTER.validate_python(usage.get("tool_usage"), strict=True)
    except ValidationError:
        return None
    calls: Final = sum(count for tool, count in tool_usage.items() if tool in _PAID_SEARCH_TOOLS and count > 0)
    return calls if calls > 0 else None


def _with_file_parts_as_image_url(message: AllMessageValues) -> AllMessageValues:
    if message["role"] != "user" or not isinstance(message["content"], list):
        return message
    content: Final = [
        migrate_file_to_image_url(part) if part["type"] == "file" else part for part in message["content"]
    ]
    updated: Final[ChatCompletionUserMessage] = {**message, "content": content}
    return updated


def _is_unreadable_file_part(part: OpenAIMessageContentListBlock) -> bool:
    if part["type"] != "file":
        return False
    file: Final = part.get("file")
    return (
        file is not None  # pyright: ignore[reportUnnecessaryComparison]  # untrusted request JSON
        and not file.get("file_data")
    )


def _without_unreadable_file_parts(message: AllMessageValues) -> AllMessageValues:
    if message["role"] != "user" or not isinstance(message["content"], list):
        return message
    content: Final = [part for part in message["content"] if not _is_unreadable_file_part(part)]
    updated: Final[ChatCompletionUserMessage] = {**message, "content": content}
    return updated


def _messages_with_readable_file_parts(
    messages: Sequence[AllMessageValues], provider_slug: str, drop_params: bool
) -> tuple[AllMessageValues, ...] | _FilePartsSentByFileId:
    readable: Final = tuple(_without_unreadable_file_parts(message) for message in messages)
    if readable == tuple(messages) or drop_params:
        return readable
    return _FilePartsSentByFileId(provider_slug=provider_slug)


def _without_additional_properties(
    node: JsonValue, keep: Callable[[JsonValue], bool], is_schema_map: bool = False
) -> JsonValue:
    if isinstance(node, list):
        return [_without_additional_properties(item, keep) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: _schema_child_without_additional_properties(key, item, keep, is_schema_map)
        for key, item in node.items()
        if is_schema_map or key != "additionalProperties" or keep(item)
    }


def _schema_child_without_additional_properties(
    key: str, item: JsonValue, keep: Callable[[JsonValue], bool], parent_is_schema_map: bool
) -> JsonValue:
    if parent_is_schema_map:
        return _without_additional_properties(item, keep)
    if key in _SCHEMA_LITERAL_KEYWORDS:
        return item
    return _without_additional_properties(item, keep, key in _SCHEMA_MAP_KEYWORDS)


def _response_format_without_additional_properties(
    response_format: JsonValue, provider_slug: str, model: str, drop_params: bool
) -> JsonValue | _MapFieldInResponseFormat:
    without_flags: Final = _without_additional_properties(response_format, lambda item: not isinstance(item, bool))
    without_any: Final = _without_additional_properties(response_format, lambda _: False)
    if without_flags == without_any or drop_params:
        return without_any
    return _MapFieldInResponseFormat(provider_slug=provider_slug, model=model)


def _flattened_messages(
    messages: Sequence[AllMessageValues], handling: Mapping[str, bool]
) -> Sequence[AllMessageValues]:
    if handling.get("convert_content_list_to_string"):
        return handle_messages_with_content_list_to_str_conversion(list(messages))
    if handling.get("convert_text_content_list_to_string"):
        return handle_messages_with_text_only_content_lists_as_str(messages)
    return messages


def _catalog_reasoning_efforts(model: str, provider_slug: str) -> tuple[str, ...] | None:
    entry: Final = GetModelCostMap.loaded_model_cost_map().get(f"{provider_slug}/{model}")
    return declared_reasoning_efforts(entry) if entry is not None else None


def _with_declared_reasoning_effort(
    params: Mapping[str, object], *, model: str, provider_slug: str, drop_params: bool
) -> Mapping[str, object]:
    requested: Final = params.get("reasoning_effort")
    declared: Final = _catalog_reasoning_efforts(model, provider_slug)
    if declared is None or not isinstance(requested, str):
        return params
    accepted: Final = nearest_declared_reasoning_effort(requested, declared)
    if drop_params and accepted not in declared:
        return MappingProxyType({key: value for key, value in params.items() if key != "reasoning_effort"})
    return MappingProxyType({**params, "reasoning_effort": accepted})


def create_config_class(provider: SimpleProviderConfig):
    """Generate config class dynamically from JSON configuration"""

    # Choose base class
    base_class: Final[type] = OpenAIGPTConfig if provider.base_class == "openai_gpt" else OpenAILikeChatConfig

    class JSONProviderConfig(base_class):
        def get_attribution_headers(self) -> Mapping[str, str]:
            return provider.get_attribution_headers()

        def _flattens_content_lists(self) -> bool:
            return bool(provider.special_handling.get("convert_content_list_to_string"))

        @overload
        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
        ) -> Coroutine[object, object, list[AllMessageValues]]: ...

        @overload
        def _transform_messages(
            self,
            messages: list[AllMessageValues],
            model: str,
            is_async: Literal[False] = False,
        ) -> list[AllMessageValues]: ...

        def _transform_messages(
            self, messages: list[AllMessageValues], model: str, is_async: bool = False
        ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
            """Transform messages based on special_handling config"""
            flattened: Final = _flattened_messages(messages, provider.special_handling_for(model))
            transformed: Final = (
                [_with_file_parts_as_image_url(message) for message in flattened]
                if provider.special_handling.get("convert_file_parts_to_image_url")
                else list(flattened)
            )
            if is_async:
                return super()._transform_messages(messages=transformed, model=model, is_async=True)
            return super()._transform_messages(messages=transformed, model=model, is_async=False)

        def _prompt_cache_ordered_messages(
            self,
            messages: list[AllMessageValues],  # mutable-ok: overrides OpenAIGPTConfig's list signature
            litellm_params: Mapping[str, object],
        ) -> list[AllMessageValues]:  # mutable-ok: overrides OpenAIGPTConfig's list signature
            readable: Final = self._readable_messages(messages, litellm_params)
            return super()._prompt_cache_ordered_messages(list(readable), litellm_params)

        def _readable_messages(
            self, messages: Sequence[AllMessageValues], litellm_params: Mapping[str, object]
        ) -> Sequence[AllMessageValues]:
            if not provider.special_handling.get("convert_file_parts_to_image_url"):
                return messages
            readable: Final = _messages_with_readable_file_parts(
                messages,
                provider_slug=provider.slug,
                drop_params=drops_unsupported_params(litellm_params.get("drop_params")),
            )
            if isinstance(readable, _FilePartsSentByFileId):
                _raise_public(readable)
            return readable

        def _should_preserve_cache_control_for_endpoint(
            self,
            custom_llm_provider: str | None,
            api_base: str | None,
        ) -> bool:
            return provider.special_handling.get(
                "preserve_cache_control", False
            ) or super()._should_preserve_cache_control_for_endpoint(custom_llm_provider, api_base)

        def _get_openai_compatible_provider_info(
            self, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            return self.get_model_provider_info(None, api_base, api_key)

        def get_openai_compatible_provider_info(
            self,
            api_base: str | None,
            api_key: str | None,
        ) -> tuple[str | None, str | None]:
            return self._get_openai_compatible_provider_info(api_base, api_key)

        def get_model_provider_info(
            self, model: str | None, api_base: str | None, api_key: str | None
        ) -> tuple[str | None, str | None]:
            resolved_base: Final = (
                api_base
                or (get_secret_str(provider.api_base_env) if provider.api_base_env else None)
                or provider.default_base_url(model)
            )
            resolved_key: Final = api_key or get_secret_str(provider.api_key_env)
            return resolved_base, resolved_key

        def get_complete_url(
            self,
            api_base: str | None,
            api_key: str | None,
            model: str,
            optional_params: dict,
            litellm_params: dict,
            stream: bool | None = None,
        ) -> str:
            """Build complete URL for the API endpoint"""
            return ensure_api_route(api_base or provider.default_base_url(model), "chat/completions")

        def get_supported_openai_params(self, model: str) -> list:
            """Get supported OpenAI params, excluding tool-related params for models
            that don't support function calling."""
            from litellm.utils import supports_function_calling, supports_reasoning

            supported_params: Final = super().get_supported_openai_params(model=model)

            _supports_fc: Final = supports_function_calling(model=model, custom_llm_provider=provider.slug)

            if not _supports_fc:
                tool_params: Final = [
                    "tools",
                    "tool_choice",
                    "function_call",
                    "functions",
                    "parallel_tool_calls",
                ]
                for param in tool_params:
                    if param in supported_params:
                        supported_params.remove(param)
                verbose_logger.debug(
                    "Model %s on provider %s does not support function calling — removed tool-related params from supported params.",
                    model,
                    provider.slug,
                )

            _supports_reasoning: Final = supports_reasoning(model=model, custom_llm_provider=provider.slug)
            if _supports_reasoning and "reasoning_effort" not in supported_params:
                supported_params.append("reasoning_effort")

            return supported_params

        def _readable_response_format(self, response_format: JsonValue, model: str, drop_params: bool) -> JsonValue:
            stripped: Final = _response_format_without_additional_properties(
                response_format,
                provider_slug=provider.slug,
                model=model,
                drop_params=drop_params or litellm.drop_params is True,
            )
            if isinstance(stripped, _MapFieldInResponseFormat):
                _raise_public(stripped)
            return stripped

        def map_openai_params(
            self,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            """Apply parameter mappings and constraints"""

            supported_params: Final = self.get_supported_openai_params(model)
            caller_params: Final = _PAYLOAD_ADAPTER.validate_python(non_default_params)
            requested_params: Final = (
                _with_declared_reasoning_effort(
                    caller_params,
                    model=model,
                    provider_slug=provider.slug,
                    drop_params=drop_params or litellm.drop_params is True,
                )
                if provider.special_handling_for(model).get("round_reasoning_effort_to_declared_levels")
                else caller_params
            )

            # Apply supported params
            for param, value in requested_params.items():
                # Check parameter mappings first
                if param in provider.param_mappings:
                    optional_params[provider.param_mappings[param]] = value
                elif param in supported_params:
                    optional_params[param] = (
                        self._readable_response_format(_JSON_VALUE_ADAPTER.validate_python(value), model, drop_params)
                        if param == "response_format"
                        and provider.special_handling.get("remove_additional_properties_from_response_format")
                        else value
                    )

            # Apply temperature constraints if present
            if "temperature" in optional_params:
                temp = optional_params["temperature"]
                constraints: Final = provider.constraints

                # Clamp to max
                if "temperature_max" in constraints:
                    temp = min(temp, constraints["temperature_max"])

                # Clamp to min
                if "temperature_min" in constraints:
                    temp = max(temp, constraints["temperature_min"])

                # Special case: temperature_min_with_n_gt_1
                if "temperature_min_with_n_gt_1" in constraints:
                    n: Final = optional_params.get("n", 1)
                    if n > 1 and temp < constraints["temperature_min_with_n_gt_1"]:
                        temp = constraints["temperature_min_with_n_gt_1"]

                optional_params["temperature"] = temp

            return optional_params

        def transform_parsed_response_dict(
            self,
            parsed_response: dict[str, object],  # mutable-ok: BaseConfig signature
        ) -> dict[str, object]:  # mutable-ok: BaseConfig signature
            """
            Carry the provider's usage.tool_usage search count as usage.server_tool_use.web_search_requests,
            the field the built-in tool cost path bills per call.
            """
            repaired: Final = _PAYLOAD_ADAPTER.validate_python(
                super().transform_parsed_response_dict(parsed_response)  # pyright: ignore[reportUnknownMemberType]  # base_class is picked at runtime
            )
            if not provider.special_handling.get("web_search_requests_from_tool_usage"):
                return {**repaired}
            web_search_requests: Final = web_search_requests_from_tool_usage(repaired)
            if web_search_requests is None:
                return {**repaired}
            usage: Final = _PAYLOAD_ADAPTER.validate_python(repaired["usage"])
            return {**repaired, "usage": {**usage, "server_tool_use": {"web_search_requests": web_search_requests}}}

        @property
        def custom_llm_provider(self) -> str | None:
            return provider.slug

    return JSONProviderConfig


_responses_config_cache: Final[dict[str, "type[OpenAILikeResponsesConfig]"]] = {}


def create_responses_config_class(provider: SimpleProviderConfig) -> "type[OpenAILikeResponsesConfig]":
    """Generate a Responses API config class dynamically from JSON configuration.

    Parallel to create_config_class() but for /v1/responses endpoints.
    Classes are cached per provider slug to avoid regeneration on every request.
    """
    if provider.slug in _responses_config_cache:
        return _responses_config_cache[provider.slug]

    from litellm.llms.openai_like.responses.transformation import (
        OpenAILikeResponsesConfig,
    )
    from litellm.types.llms.openai import ResponseInputParam
    from litellm.types.router import GenericLiteLLMParams

    class JSONProviderResponsesConfig(OpenAILikeResponsesConfig):
        @property
        def custom_llm_provider(self):
            return provider.slug

        def validate_environment(
            self,
            headers: dict,
            model: str,
            litellm_params: GenericLiteLLMParams | None,
        ) -> dict:
            litellm_params = litellm_params or GenericLiteLLMParams()
            api_key: Final = litellm_params.api_key or get_secret_str(provider.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            return headers

        def get_complete_url(
            self,
            api_base: str | None,
            litellm_params: dict,
        ) -> str:
            resolved_base: Final = (
                api_base
                or (get_secret_str(provider.api_base_env) if provider.api_base_env else None)
                or provider.base_url
            )
            return join_api_base(resolved_base, "responses")

        def transform_responses_api_request(
            self,
            model: str,
            input: str | ResponseInputParam,
            response_api_optional_request_params: dict,
            litellm_params: GenericLiteLLMParams,
            headers: dict,
        ) -> dict:
            if provider.special_handling.get("force_store_false"):
                response_api_optional_request_params["store"] = False
            return super().transform_responses_api_request(
                model=model,
                input=input,
                response_api_optional_request_params=response_api_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

    _responses_config_cache[provider.slug] = JSONProviderResponsesConfig
    return JSONProviderResponsesConfig
