from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast  # noqa: TID251  # SDK adapters validate or narrow each cast at its boundary

import litellm
from litellm.litellm_core_utils.asyncify import (
    run_async_function,  # pyright: ignore[reportUnknownVariableType]  # shared sync bridge is intentionally untyped
)
from litellm.llms.litellm.base import (
    BaseLiteLLMModel,
    get_litellm_model,
    stamp_litellm_model_response,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper


async def adispatch_completion(
    *,
    model: str,
    messages: list[AllMessageValues],  # mutable-ok: public SDK boundary
    stream: bool,
    request_kwargs: Mapping[str, object],
    litellm_model: BaseLiteLLMModel | None = None,
) -> ModelResponse | CustomStreamWrapper:
    model_implementation: Final = litellm_model or get_litellm_model(model)
    response: Final = await model_implementation.acompletion(
        messages=messages,
        stream=stream,
        request_kwargs=request_kwargs,
    )
    return stamp_litellm_model_response(response, model)


def dispatch_completion(
    *,
    model: str,
    messages: list[AllMessageValues],  # mutable-ok: public SDK boundary
    stream: bool,
    request_kwargs: Mapping[str, object],
    litellm_model: BaseLiteLLMModel | None = None,
) -> ModelResponse | CustomStreamWrapper:
    return run_async_function(
        adispatch_completion,
        model=model,
        messages=messages,
        stream=stream,
        request_kwargs=request_kwargs,
        litellm_model=litellm_model,
    )


async def adispatch_responses(
    *,
    model: str,
    input: object,
    stream: bool,
    request_kwargs: Mapping[str, object],
    litellm_model: BaseLiteLLMModel | None = None,
) -> object:
    if request_kwargs.get("background") is True:
        raise litellm.BadRequestError(
            message="Background Responses are not supported for LiteLLM models",
            model=model,
            llm_provider="litellm",
        )

    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )
    from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams

    response_input: Final = cast(str | ResponseInputParam, input)  # cast-ok: Responses transformer validates input
    responses_request: Final = cast(  # cast-ok: canonical transformer validates the typed request
        ResponsesAPIOptionalRequestParams, request_kwargs
    )
    transform_kwargs: Final = {  # mutable-ok: canonical transformer requires keyword arguments
        key: value for key, value in request_kwargs.items() if key != "extra_headers"
    }
    initial_completion_request: Final = cast(  # cast-ok: canonical transformer owns this response schema
        dict[str, object],
        LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(  # pyright: ignore[reportUnknownMemberType]  # upstream transformer lacks parameterized dict typing
            model=model,
            input=response_input,
            responses_api_request=responses_request,
            stream=stream,
            extra_headers=cast(  # cast-ok: transformer validates forwarded header values
                Mapping[str, object] | None, request_kwargs.get("extra_headers")
            ),
            **transform_kwargs,  # pyright: ignore[reportArgumentType]  # transformer validates extension keywords
        ),
    )
    previous_response_id: Final[str | None] = responses_request.get(  # pyright: ignore[reportUnknownMemberType]  # TypedDict overload contains unrelated unknown fields
        "previous_response_id"
    )
    completion_request: Final[dict[str, object]] = (  # mutable-ok: canonical transformer returns a native mapping
        cast(  # cast-ok: canonical session handler owns this response schema
            dict[str, object],
            await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(  # pyright: ignore[reportUnknownMemberType]  # upstream handler lacks parameterized dict typing
                previous_response_id=previous_response_id,
                litellm_completion_request=initial_completion_request,
            ),
        )
        if previous_response_id
        else initial_completion_request
    )
    completion_response: Final = await adispatch_completion(
        model=model,
        messages=cast(  # cast-ok: canonical Responses transformer produced these chat messages
            list[AllMessageValues], completion_request["messages"]
        ),
        stream=stream,
        request_kwargs={  # mutable-ok: public SDK adapter boundary
            **request_kwargs,
            **completion_request,
            "_skip_responses_api_bridge": True,
        },
        litellm_model=litellm_model,
    )
    if isinstance(completion_response, ModelResponse):
        response: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(  # pyright: ignore[reportUnknownMemberType]  # upstream transformer lacks complete annotations
            chat_completion_response=completion_response,
            request_input=response_input,
            responses_api_request=responses_request,
        )
        return stamp_litellm_model_response(response, model)
    raw_litellm_metadata: Final = request_kwargs.get("litellm_metadata")
    response_stream: Final = LiteLLMCompletionStreamingIterator(
        model=model,
        litellm_custom_stream_wrapper=completion_response,
        request_input=response_input,
        responses_api_request=responses_request,
        litellm_metadata=(
            dict(  # mutable-ok: streaming adapter requires a native mapping
                cast(  # cast-ok: guarded by the Mapping check below
                    Mapping[str, object], raw_litellm_metadata
                )
            )  # mutable-ok: adapter requires a native mapping
            if isinstance(raw_litellm_metadata, Mapping)
            else {}  # mutable-ok: streaming adapter requires a native mapping
        ),
    )
    return stamp_litellm_model_response(response_stream, model)


def dispatch_responses(
    *,
    model: str,
    input: object,
    stream: bool,
    request_kwargs: Mapping[str, object],
    litellm_model: BaseLiteLLMModel | None = None,
) -> object:
    if stream:
        raise litellm.BadRequestError(
            message="Synchronous Responses streaming is not supported for LiteLLM models; use aresponses",
            model=model,
            llm_provider="litellm",
        )
    return run_async_function(
        adispatch_responses,
        model=model,
        input=input,
        stream=False,
        request_kwargs=request_kwargs,
        litellm_model=litellm_model,
    )


async def adispatch_anthropic_messages(
    *,
    model: str,
    messages: list[dict[str, object]],  # mutable-ok: public Anthropic SDK boundary
    max_tokens: int,
    metadata: Mapping[str, object] | None,
    stop_sequences: list[str] | None,  # mutable-ok: public Anthropic SDK boundary
    stream: bool,
    system: str | list[dict[str, object]] | None,  # mutable-ok: public Anthropic SDK boundary
    temperature: float | None,
    thinking: dict[str, object] | None,  # mutable-ok: public Anthropic SDK boundary
    tool_choice: dict[str, object] | None,  # mutable-ok: public Anthropic SDK boundary
    tools: list[dict[str, object]] | None,  # mutable-ok: public Anthropic SDK boundary
    top_k: int | None,
    top_p: float | None,
    request_kwargs: Mapping[str, object],
    litellm_model: BaseLiteLLMModel | None = None,
) -> object:
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        ANTHROPIC_ADAPTER,
        LiteLLMMessagesToCompletionTransformationHandler,
    )
    from litellm.llms.anthropic.experimental_pass_through.utils import local_model_name

    completion_kwargs, tool_name_mapping = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(  # pyright: ignore[reportPrivateUsage]  # canonical Anthropic translation
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        metadata=(
            dict(metadata)  # mutable-ok: adapter requires a native mapping
            if metadata is not None
            else None
        ),
        stop_sequences=stop_sequences,
        stream=stream,
        system=system,
        temperature=temperature,
        thinking=thinking,
        tool_choice=tool_choice,
        tools=tools,
        top_k=top_k,
        top_p=top_p,
        output_format=cast(  # cast-ok: Anthropic transformer validates the output format
            dict[str, object] | None, request_kwargs.get("output_format")
        ),
        extra_kwargs=dict(request_kwargs),  # mutable-ok: adapter requires a native keyword mapping
    )
    completion_response: Final = await adispatch_completion(
        model=model,
        messages=cast(  # cast-ok: canonical Anthropic transformer produced these chat messages
            list[AllMessageValues], completion_kwargs["messages"]
        ),
        stream=stream,
        request_kwargs={  # mutable-ok: public SDK adapter boundary
            **request_kwargs,
            **completion_kwargs,
        },
        litellm_model=litellm_model,
    )
    if stream:
        transformed_stream: Final = ANTHROPIC_ADAPTER.translate_completion_output_params_streaming(
            completion_response,
            model=local_model_name(
                model,
                cast(  # cast-ok: provider name is validated by the public SDK boundary
                    str | None, request_kwargs.get("custom_llm_provider")
                ),
            ),
            tool_name_mapping=tool_name_mapping,
            polyfill_result=None,
            is_async=True,
        )
        if transformed_stream is None:
            raise ValueError("Failed to transform LiteLLM model stream to Anthropic format")
        return stamp_litellm_model_response(
            transformed_stream,
            model,
            source_response=completion_response,
        )

    anthropic_response: Final = ANTHROPIC_ADAPTER.translate_completion_output_params(
        cast(ModelResponse, completion_response),  # cast-ok: non-stream branch guarantees ModelResponse
        tool_name_mapping=tool_name_mapping,
        polyfill_result=None,
    )
    if anthropic_response is None:
        raise ValueError("Failed to transform LiteLLM model response to Anthropic format")
    return stamp_litellm_model_response(
        anthropic_response,
        model,
        source_response=completion_response,
    )


def dispatch_anthropic_messages(
    *,
    model: str,
    messages: list[dict[str, object]],  # mutable-ok: public Anthropic SDK boundary
    max_tokens: int,
    metadata: Mapping[str, object] | None,
    stop_sequences: list[str] | None,  # mutable-ok: public Anthropic SDK boundary
    stream: bool,
    system: str | list[dict[str, object]] | None,  # mutable-ok: public Anthropic SDK boundary
    temperature: float | None,
    thinking: dict[str, object] | None,  # mutable-ok: public Anthropic SDK boundary
    tool_choice: dict[str, object] | None,  # mutable-ok: public Anthropic SDK boundary
    tools: list[dict[str, object]] | None,  # mutable-ok: public Anthropic SDK boundary
    top_k: int | None,
    top_p: float | None,
    request_kwargs: Mapping[str, object],
    litellm_model: BaseLiteLLMModel | None = None,
) -> object:
    return cast(  # cast-ok: shared sync bridge preserves the async function's object response
        object,
        run_async_function(
            adispatch_anthropic_messages,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            metadata=metadata,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            thinking=thinking,
            tool_choice=tool_choice,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
            request_kwargs=request_kwargs,
            litellm_model=litellm_model,
        ),
    )
