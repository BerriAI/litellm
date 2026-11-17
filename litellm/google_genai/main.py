import asyncio
import base64
import contextvars
from collections.abc import Awaitable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Final

import httpx
from pydantic import ConfigDict, JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python

import litellm
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH, request_timeout

# Import the adapter for fallback to completion format
from litellm.google_genai.adapters.handler import CUSTOM_PRICING_PARAMS, GenerateContentToCompletionHandler
from litellm.google_genai.adapters.transformation import adapter_untranslated_generation_config
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.per_call_pricing import per_call_model_info
from litellm.llms.base_llm.google_genai.transformation import (
    GENERATE_CONTENT_REQUEST_TOP_LEVEL_FIELDS,
    BaseGoogleGenAIGenerateContentConfig,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.custom_llm import CustomLLM
from litellm.llms.vertex_ai.gemini.transformation import snake_to_camel
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import CallTypes, LlmProviders
from litellm.utils import ProviderConfigManager, client

if TYPE_CHECKING:
    from litellm.types.google_genai.main import (
        GenerateContentConfigDict,
        GenerateContentContentListUnionDict,
        GenerateContentResponse,
        ToolConfigDict,
    )
else:
    GenerateContentConfigDict = Any
    GenerateContentContentListUnionDict = Any
    GenerateContentResponse = Any
    ToolConfigDict = Any


####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
_GENERATE_CONTENT_RESULT: Final = TypeAdapter(object)
_CUSTOM_GENERATE_CONTENT_BODY: Final = TypeAdapter(dict[str, object])
_CUSTOM_NATIVE_REQUEST_FIELDS: Final = (
    *GENERATE_CONTENT_REQUEST_TOP_LEVEL_FIELDS,
    "systemInstruction",
    "system_instruction",
)
_BUILT_IN_PROVIDERS: Final = frozenset(provider.value for provider in LlmProviders)
_CALLER_DATA_FIELDS: Final = frozenset({"args", "response"})
_SDK_ONLY_RESPONSE_FIELDS: Final = frozenset({"sdkHttpResponse", "automaticFunctionCallingHistory", "parsed"})
_STR_KEYED_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT_SEQUENCE: Final = TypeAdapter(Sequence[object])
_JSON_VALUE: Final = TypeAdapter[JsonValue](JsonValue)
_GENERATE_CONTENT_PAYLOAD: Final = TypeAdapter(object)
#################################################


def _custom_generate_content_handler(provider: str) -> CustomLLM | None:
    if provider in _BUILT_IN_PROVIDERS:
        return None
    return ProviderConfigManager.get_custom_llm_handler(provider)


def _mark_async_entrypoint(logging_obj: LiteLLMLoggingObj | None, marker: str, is_async: bool) -> None:
    if logging_obj is not None:
        logging_obj.model_call_details.setdefault("litellm_params", {})[marker] = is_async


class GenerateContentSetupResult(LiteLLMBaseModel):
    """Internal Type - Result of setting up a generate content call"""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    model: str
    request_body: dict[str, object]
    custom_llm_provider: str
    generate_content_provider_config: BaseGoogleGenAIGenerateContentConfig | None
    generate_content_config_dict: dict[str, object]
    native_request_fields: dict[str, object]
    litellm_params: GenericLiteLLMParams
    litellm_logging_obj: LiteLLMLoggingObj
    litellm_call_id: str | None


class GenerateContentHelper:
    """Helper class for Google GenAI generate content operations"""

    @staticmethod
    def mock_generate_content_response(
        mock_response: str = "This is a mock response from Google GenAI generate_content.",
    ) -> dict[str, object]:
        """Mock response for generate_content for testing purposes"""
        return {
            "text": mock_response,
            "candidates": [
                {
                    "content": {"parts": [{"text": mock_response}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        }

    @staticmethod
    def setup_generate_content_call(
        model: str,
        contents: GenerateContentContentListUnionDict,
        config: GenerateContentConfigDict | None = None,
        custom_llm_provider: str | None = None,
        tools: ToolConfigDict | None = None,
        **kwargs,
    ) -> GenerateContentSetupResult:
        """
        Common setup logic for generate_content calls

        Args:
            model: The model name
            contents: The content to generate from
            config: Optional configuration
            custom_llm_provider: Optional custom LLM provider
            tools: Optional tools
            **kwargs: Additional keyword arguments

        Returns:
            GenerateContentSetupResult containing all setup information
        """
        litellm_logging_obj: Final[LiteLLMLoggingObj | None] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Final[str | None] = kwargs.get("litellm_call_id", None)

        # get llm provider logic
        litellm_params: Final = GenericLiteLLMParams(**kwargs)

        ## MOCK RESPONSE LOGIC (only for non-streaming)
        if (
            not kwargs.get("stream", False)
            and litellm_params.mock_response
            and isinstance(litellm_params.mock_response, str)
        ):
            raise ValueError("Mock response should be handled by caller")

        (
            model,
            custom_llm_provider,
            dynamic_api_key,
            dynamic_api_base,
        ) = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=litellm_params.api_base,
            api_key=litellm_params.api_key,
        )

        if litellm_params.custom_llm_provider is None:
            litellm_params.custom_llm_provider = custom_llm_provider

        custom_handler: Final = _custom_generate_content_handler(custom_llm_provider)
        # get provider config
        generate_content_provider_config: Final[BaseGoogleGenAIGenerateContentConfig | None] = (
            ProviderConfigManager.get_provider_google_genai_generate_content_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
                litellm_params=litellm_params.model_dump(include={"base_model", "api_base"}),
            )
            if custom_handler is None
            else None
        )

        if generate_content_provider_config is None:
            # Use adapter to transform to completion format when provider config is None
            # Signal that we should use the adapter by returning special result
            if litellm_logging_obj is None:
                raise ValueError("litellm_logging_obj is required, but got None")
            if custom_handler is not None:
                request_kwargs: Final = _STR_KEYED_MAPPING.validate_python(kwargs)
                litellm_logging_obj.update_from_kwargs(  # pyright: ignore[reportUnknownMemberType]  # legacy logging interface is untyped
                    kwargs=kwargs,
                    model=model,
                    optional_params={"generationConfig": _CUSTOM_GENERATE_CONTENT_BODY.validate_python(config or {})},
                    litellm_params={
                        "litellm_call_id": litellm_call_id,
                        **{key: value for key, value in request_kwargs.items() if key in CUSTOM_PRICING_PARAMS},
                    },
                    custom_llm_provider=custom_llm_provider,
                )
                litellm_logging_obj.set_per_call_pricing(  # pyright: ignore[reportUnknownMemberType]  # ModelInfo has untyped fields
                    per_call_model_info(model, custom_llm_provider, request_kwargs, "chat")
                )
            return GenerateContentSetupResult(
                model=model,
                custom_llm_provider=custom_llm_provider,
                request_body={},  # Will be handled by adapter
                generate_content_provider_config=None,
                generate_content_config_dict=dict(config or {}),
                native_request_fields={},
                litellm_params=litellm_params,
                litellm_logging_obj=litellm_logging_obj,
                litellm_call_id=litellm_call_id,
            )

        #########################################################################################
        # Construct request body
        #########################################################################################
        # Create Google Optional Params Config
        generate_content_config_dict: Final = generate_content_provider_config.map_generate_content_optional_params(
            generate_content_config_dict=config or {},
            model=model,
        )
        # Extract systemInstruction from kwargs to pass to transform
        system_instruction: Final = kwargs.get("systemInstruction") or kwargs.get("system_instruction")
        # Native top-level REST fields arrive as loose kwargs and are otherwise dropped.
        native_request_fields: Final[dict[str, object]] = {
            field: kwargs[field]
            for field in generate_content_provider_config.get_generate_content_request_top_level_fields()
            if field in kwargs
        }
        request_body: Final = generate_content_provider_config.transform_generate_content_request(
            model=model,
            contents=contents,
            tools=tools,
            generate_content_config_dict=generate_content_config_dict,
            system_instruction=system_instruction,
        )

        # Pre Call logging
        if litellm_logging_obj is None:
            raise ValueError("litellm_logging_obj is required, but got None")

        vertex_location: Final = VertexBase.explicit_vertex_ai_location(dict(litellm_params))
        litellm_logging_obj.update_from_kwargs(
            kwargs=kwargs,
            model=model,
            optional_params=dict(generate_content_config_dict),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **({"vertex_location": vertex_location} if vertex_location else {}),
            },
            custom_llm_provider=custom_llm_provider,
        )

        return GenerateContentSetupResult(
            model=model,
            custom_llm_provider=custom_llm_provider,
            request_body=request_body,
            generate_content_provider_config=generate_content_provider_config,
            generate_content_config_dict=generate_content_config_dict,
            native_request_fields=native_request_fields,
            litellm_params=litellm_params,
            litellm_logging_obj=litellm_logging_obj,
            litellm_call_id=litellm_call_id,
        )


def _merge_native_request_fields(
    native_request_fields: dict[str, object],
    extra_body: dict[str, object] | None,
) -> dict[str, object] | None:
    """
    Merge native top-level request fields into ``extra_body`` so the HTTP handler
    forwards them verbatim onto the outgoing request body. An explicit ``extra_body``
    value wins on conflict. Returns ``None`` only when there is genuinely nothing to
    forward (no native fields and no caller-supplied ``extra_body``), preserving the
    prior behavior without discarding an explicit ``extra_body={}``.
    """
    if not native_request_fields and extra_body is None:
        return None
    return {**native_request_fields, **(extra_body or {})}


def _custom_adapter_logging_obj(setup_result: GenerateContentSetupResult) -> LiteLLMLoggingObj | None:
    if _custom_generate_content_handler(setup_result.custom_llm_provider) is None:
        return None
    return setup_result.litellm_logging_obj


def _require_custom_stream_hook(setup_result: GenerateContentSetupResult, is_async: bool) -> None:
    custom_handler: Final = _custom_generate_content_handler(setup_result.custom_llm_provider)
    if custom_handler is None:
        return
    hook_name: Final = "astreaming" if is_async else "streaming"
    if getattr(type(custom_handler), hook_name) is getattr(CustomLLM, hook_name):
        raise litellm.APIConnectionError(
            message=f"{setup_result.custom_llm_provider} does not implement {hook_name}, so it cannot stream generateContent",
            llm_provider=setup_result.custom_llm_provider,
            model=setup_result.model,
        )


def _require_adapter_generation_config(setup_result: GenerateContentSetupResult) -> None:
    if _custom_generate_content_handler(setup_result.custom_llm_provider) is None:
        return
    if setup_result.litellm_params.drop_params or litellm.drop_params:
        return
    untranslated: Final = adapter_untranslated_generation_config(setup_result.generate_content_config_dict)
    if untranslated:
        raise litellm.UnsupportedParamsError(
            message=(
                f"{setup_result.custom_llm_provider} has no generateContent hook, and the completion adapter cannot "
                f"send generationConfig {list(untranslated)}. Set drop_params=True to drop them."
            ),
            llm_provider=setup_result.custom_llm_provider,
            model=setup_result.model,
        )


def _plain_json_value(value: object) -> JsonValue:
    return _JSON_VALUE.validate_python(to_jsonable_python(value, fallback=str))


def _gemini_wire_value(value: object, camel_case_keys: bool, depth: int = 0) -> JsonValue:
    if depth > DEFAULT_MAX_RECURSE_DEPTH:
        return _plain_json_value(value)
    match value:
        case bytes():
            return base64.b64encode(value).decode()
        case str():
            return value
        case Mapping():
            return {
                (snake_to_camel(key) if camel_case_keys else key): _gemini_wire_value(
                    item, camel_case_keys and key not in _CALLER_DATA_FIELDS, depth + 1
                )
                for key, item in _STR_KEYED_MAPPING.validate_python(value).items()
                if item is not None or not camel_case_keys
            }
        case Sequence():
            return [
                _gemini_wire_value(item, camel_case_keys, depth + 1) for item in _OBJECT_SEQUENCE.validate_python(value)
            ]
        case _:
            return _plain_json_value(value)


def _custom_generate_content_wire_body(response: object) -> Mapping[str, JsonValue]:
    wire_body: Final = _gemini_wire_value(
        _GENERATE_CONTENT_RESULT.validate_python(
            _GENERATE_CONTENT_RESULT.dump_python(response, by_alias=True, exclude_none=True)
        ),
        camel_case_keys=True,
    )
    if not isinstance(wire_body, dict):
        return {}
    return {key: value for key, value in wire_body.items() if key not in _SDK_ONLY_RESPONSE_FIELDS}


def _log_custom_generate_content_response(
    response: object, logging_obj: LiteLLMLoggingObj, contents: object, api_key: str | None
) -> Mapping[str, JsonValue]:
    wire_body: Final = _custom_generate_content_wire_body(response)
    raw_response: Final = httpx.Response(status_code=200, json={"usageMetadata": {}, **wire_body})
    logging_obj.model_call_details["httpx_response"] = raw_response  # rebind-ok: success logging needs raw response
    logging_obj.post_call(input=contents, api_key=api_key, original_response=raw_response.text)  # pyright: ignore[reportUnknownMemberType]  # legacy logging interface is untyped
    return wire_body


async def _await_custom_generate_content_response(
    response: Awaitable[object], logging_obj: LiteLLMLoggingObj, contents: object, api_key: str | None
) -> Mapping[str, JsonValue]:
    result: Final = await response
    return _log_custom_generate_content_response(result, logging_obj, contents, api_key)


def _finish_custom_generate_content_response(
    response: object, logging_obj: LiteLLMLoggingObj, contents: object, api_key: str | None
) -> object:
    if asyncio.iscoroutine(response):
        return _await_custom_generate_content_response(response, logging_obj, contents, api_key)
    return _log_custom_generate_content_response(response, logging_obj, contents, api_key)


@client
async def agenerate_content(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: GenerateContentConfigDict | None = None,
    tools: ToolConfigDict | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> object:
    """
    Async: Generate content using Google GenAI
    """
    local_vars: Final = locals()
    try:
        loop: Final = asyncio.get_event_loop()
        kwargs["agenerate_content"] = True

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
            )

        func: Final = partial(
            generate_content,
            model=model,
            contents=contents,
            config=config,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            **kwargs,
        )

        ctx: Final = contextvars.copy_context()
        func_with_context: Final = partial(ctx.run, func)
        init_response: Final = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = _GENERATE_CONTENT_RESULT.validate_python(await init_response)
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def generate_content(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: GenerateContentConfigDict | None = None,
    tools: ToolConfigDict | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> object:
    """
    Generate content using Google GenAI
    """
    local_vars: Final = locals()
    try:
        _is_async: Final = kwargs.pop("agenerate_content", False)

        _mark_async_entrypoint(kwargs.get("litellm_logging_obj"), CallTypes.agenerate_content.value, _is_async)

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # Check for mock response first
        litellm_params: Final = GenericLiteLLMParams(**kwargs)
        if litellm_params.mock_response and isinstance(litellm_params.mock_response, str):
            return GenerateContentHelper.mock_generate_content_response(mock_response=litellm_params.mock_response)

        # Setup the call
        setup_result: Final = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            **kwargs,
        )

        custom_handler: Final = _custom_generate_content_handler(setup_result.custom_llm_provider)
        native_hook_overridden: Final = custom_handler is not None and (
            type(custom_handler).agenerate_content is not CustomLLM.agenerate_content
            if _is_async
            else type(custom_handler).generate_content is not CustomLLM.generate_content
        )
        if custom_handler is not None and native_hook_overridden:
            logging_obj: Final = setup_result.litellm_logging_obj
            native_contents: Final = _GENERATE_CONTENT_RESULT.validate_python(contents)
            native_tools: Final = _GENERATE_CONTENT_RESULT.validate_python(tools)
            native_optional_params: Final[Mapping[str, object]] = _CUSTOM_GENERATE_CONTENT_BODY.validate_python(
                {
                    **{field: kwargs[field] for field in _CUSTOM_NATIVE_REQUEST_FIELDS if field in kwargs},
                    "extra_headers": extra_headers,
                    "extra_query": extra_query,
                    "extra_body": extra_body,
                }
            )
            logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # legacy logging interface is untyped
                input=native_contents,
                api_key=setup_result.litellm_params.api_key,
                additional_args={
                    "complete_input_dict": {"contents": contents, "generationConfig": config, "tools": tools}
                },
            )
            hook: Final = custom_handler.agenerate_content if _is_async else custom_handler.generate_content
            native_response: Final = hook(
                model=setup_result.model,
                contents=native_contents,
                config=setup_result.generate_content_config_dict if config is not None else None,
                tools=native_tools,
                litellm_params=setup_result.litellm_params,
                logging_obj=logging_obj,
                optional_params=native_optional_params,
                api_key=setup_result.litellm_params.api_key,
                api_base=setup_result.litellm_params.api_base,
                timeout=timeout or request_timeout,
            )
            return _finish_custom_generate_content_response(
                native_response, logging_obj, native_contents, setup_result.litellm_params.api_key
            )

        # Extract systemInstruction from kwargs to pass to handler
        system_instruction: Final = kwargs.get("systemInstruction") or kwargs.get("system_instruction")

        # Check if we should use the adapter (when provider config is None)
        if setup_result.generate_content_provider_config is None:
            _require_adapter_generation_config(setup_result)
            # Use the adapter to convert to completion format
            adapter_response: Final = GenerateContentToCompletionHandler.generate_content_handler(
                model=model,
                contents=contents,
                config=setup_result.generate_content_config_dict,
                tools=tools,
                _is_async=_is_async,
                litellm_params=setup_result.litellm_params,
                extra_headers=extra_headers,
                custom_logging_obj=setup_result.litellm_logging_obj if custom_handler is not None else None,
                **kwargs,
            )
            if custom_handler is not None:
                return _finish_custom_generate_content_response(
                    adapter_response,
                    setup_result.litellm_logging_obj,
                    _GENERATE_CONTENT_RESULT.validate_python(contents),
                    setup_result.litellm_params.api_key,
                )
            return adapter_response

        # Call the standard handler
        response: Final = _GENERATE_CONTENT_RESULT.validate_python(
            base_llm_http_handler.generate_content_handler(
                model=setup_result.model,
                contents=_GENERATE_CONTENT_PAYLOAD.validate_python(contents),
                tools=_GENERATE_CONTENT_PAYLOAD.validate_python(tools),
                generate_content_provider_config=setup_result.generate_content_provider_config,
                generate_content_config_dict=setup_result.generate_content_config_dict,
                custom_llm_provider=setup_result.custom_llm_provider,
                litellm_params=setup_result.litellm_params,
                logging_obj=setup_result.litellm_logging_obj,
                extra_headers=extra_headers,
                extra_body=_merge_native_request_fields(setup_result.native_request_fields, extra_body),
                timeout=timeout or request_timeout,
                _is_async=_is_async,
                client=kwargs.get("client"),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                system_instruction=system_instruction,
            )
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def agenerate_content_stream(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: GenerateContentConfigDict | None = None,
    tools: ToolConfigDict | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> object:
    """
    Async: Generate content using Google GenAI with streaming response
    """
    local_vars: Final = locals()
    try:
        kwargs["agenerate_content_stream"] = True

        _mark_async_entrypoint(kwargs.get("litellm_logging_obj"), CallTypes.agenerate_content_stream.value, True)

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=local_vars.get("base_url", None)
            )

        # Setup the call
        setup_result: Final = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            **kwargs,
        )

        # Extract systemInstruction from kwargs to pass to handler
        system_instruction: Final = kwargs.get("systemInstruction") or kwargs.get("system_instruction")

        # Check if we should use the adapter (when provider config is None)
        if setup_result.generate_content_provider_config is None:
            if "stream" in kwargs:
                kwargs.pop("stream", None)
            _require_custom_stream_hook(setup_result, is_async=True)
            _require_adapter_generation_config(setup_result)

            # Use the adapter to convert to completion format
            return await GenerateContentToCompletionHandler.async_generate_content_handler(
                model=model,
                contents=contents,
                config=setup_result.generate_content_config_dict,
                litellm_params=setup_result.litellm_params,
                tools=tools,
                stream=True,
                extra_headers=extra_headers,
                custom_logging_obj=_custom_adapter_logging_obj(setup_result),
                **kwargs,
            )

        # Call the handler with async enabled and streaming
        # Return the coroutine directly for the router to handle
        return _GENERATE_CONTENT_RESULT.validate_python(
            await base_llm_http_handler.generate_content_handler(
                model=setup_result.model,
                contents=_GENERATE_CONTENT_PAYLOAD.validate_python(contents),
                generate_content_provider_config=setup_result.generate_content_provider_config,
                generate_content_config_dict=setup_result.generate_content_config_dict,
                tools=_GENERATE_CONTENT_PAYLOAD.validate_python(tools),
                custom_llm_provider=setup_result.custom_llm_provider,
                litellm_params=setup_result.litellm_params,
                logging_obj=setup_result.litellm_logging_obj,
                extra_headers=extra_headers,
                extra_body=_merge_native_request_fields(setup_result.native_request_fields, extra_body),
                timeout=timeout or request_timeout,
                _is_async=True,
                client=kwargs.get("client"),
                stream=True,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                system_instruction=system_instruction,
            )
        )

    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def generate_content_stream(
    model: str,
    contents: GenerateContentContentListUnionDict,
    config: GenerateContentConfigDict | None = None,
    tools: ToolConfigDict | None = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: dict[str, object] | None = None,
    extra_query: dict[str, object] | None = None,
    extra_body: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = None,
    # LiteLLM specific params,
    custom_llm_provider: str | None = None,
    **kwargs,
) -> object:
    """
    Generate content using Google GenAI with streaming response
    """
    local_vars: Final = locals()
    try:
        # Remove any async-related flags since this is the sync function
        _is_async: Final = kwargs.pop("agenerate_content_stream", False)

        _mark_async_entrypoint(kwargs.get("litellm_logging_obj"), CallTypes.agenerate_content_stream.value, _is_async)

        # Handle generationConfig parameter from kwargs for backward compatibility
        if "generationConfig" in kwargs and config is None:
            config = kwargs.pop("generationConfig")
        # Setup the call
        setup_result: Final = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            **kwargs,
        )

        # Extract systemInstruction from kwargs to pass to handler
        system_instruction: Final = kwargs.get("systemInstruction") or kwargs.get("system_instruction")

        # Check if we should use the adapter (when provider config is None)
        if setup_result.generate_content_provider_config is None:
            if "stream" in kwargs:
                kwargs.pop("stream", None)
            _require_custom_stream_hook(setup_result, is_async=_is_async is True)
            _require_adapter_generation_config(setup_result)

            # Use the adapter to convert to completion format
            return GenerateContentToCompletionHandler.generate_content_handler(
                model=model,
                contents=contents,
                config=setup_result.generate_content_config_dict,
                _is_async=_is_async,
                litellm_params=setup_result.litellm_params,
                stream=True,
                extra_headers=extra_headers,
                custom_logging_obj=_custom_adapter_logging_obj(setup_result),
                **kwargs,
            )

        # Call the handler with streaming enabled (sync version)
        return _GENERATE_CONTENT_RESULT.validate_python(
            base_llm_http_handler.generate_content_handler(
                model=setup_result.model,
                contents=_GENERATE_CONTENT_PAYLOAD.validate_python(contents),
                generate_content_provider_config=setup_result.generate_content_provider_config,
                generate_content_config_dict=setup_result.generate_content_config_dict,
                tools=_GENERATE_CONTENT_PAYLOAD.validate_python(tools),
                custom_llm_provider=setup_result.custom_llm_provider,
                litellm_params=setup_result.litellm_params,
                logging_obj=setup_result.litellm_logging_obj,
                extra_headers=extra_headers,
                extra_body=_merge_native_request_fields(setup_result.native_request_fields, extra_body),
                timeout=timeout or request_timeout,
                _is_async=_is_async,
                client=kwargs.get("client"),
                stream=True,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                system_instruction=system_instruction,
            )
        )

    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
