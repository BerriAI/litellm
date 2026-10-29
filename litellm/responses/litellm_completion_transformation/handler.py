"""
Handler for transforming responses api requests to litellm.completion requests
"""

from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.responses.additional_tools import HoistedAdditionalTools, hoist_additional_tools
from litellm.responses.litellm_completion_transformation.client_tool_search import bridge_client_tool_search
from litellm.responses.litellm_completion_transformation.compaction_items import holds_compaction_summary
from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler, StoredTurns
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.tool_search import has_client_tool_search
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    with_compaction_items_as_params,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.native.chat_plan import PlannedChat

_REQUEST_KWARGS_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class _BridgedRequest:
    input: str | ResponseInputParam
    responses_api_request: ResponsesAPIOptionalRequestParams


def _bridged_request(
    hoisted: HoistedAdditionalTools,
    responses_api_request: ResponsesAPIOptionalRequestParams,
    loaded_earlier: Sequence[Mapping[str, object]],
) -> _BridgedRequest:
    searched: Final = bridge_client_tool_search(hoisted.input, hoisted.tools, loaded_earlier)
    return _BridgedRequest(
        input=searched.input,
        responses_api_request=(
            {**responses_api_request, "tools": list(searched.tools)}
            if hoisted.hoisted or searched.bridged
            else responses_api_request
        ),
    )


def _plan_bridged_chat(request: Mapping[str, object]) -> "PlannedChat | None":
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.native.chat_plan import plan_chat

    logging_obj: Final = request.get("litellm_logging_obj")
    return plan_chat(request, logging_obj) if isinstance(logging_obj, Logging) else None


def _price_planned_chat_response(
    request: Mapping[str, object], response: ModelResponse | litellm.CustomStreamWrapper
) -> None:
    """Prices the chat response as `acompletion` would have, before it becomes a Responses object."""
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj: Final = request.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return
    if isinstance(response, ModelResponse):
        logging_obj.price_bridged_provider_response(response)
    else:
        logging_obj.price_bridged_provider_stream(response)


class LiteLLMCompletionTransformationHandler:
    def response_api_handler(
        self,
        previous_response_client_ids: tuple[str, ...] = (),
        /,
        *,
        model: str,
        input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: str | None = None,
        _is_async: bool = False,
        stream: bool | None = None,
        extra_headers: Mapping[str, object] | None = None,
        **kwargs,
    ) -> (
        ResponsesAPIResponse
        | BaseResponsesAPIStreamingIterator
        | Coroutine[object, object, ResponsesAPIResponse | BaseResponsesAPIStreamingIterator]
    ):
        hoisted: Final = hoist_additional_tools(
            input,
            responses_api_request.get("tools"),
            drop_params=drops_unsupported_params(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs).get("drop_params")),
        )
        previous_response_id: Final = responses_api_request.get("previous_response_id")
        if _is_async and previous_response_id and has_client_tool_search(hoisted.tools):
            return self._async_response_api_handler_after_earlier_searches(
                previous_response_client_ids,
                previous_response_id,
                hoisted,
                model=model,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                request_kwargs=_REQUEST_KWARGS_ADAPTER.validate_python(kwargs),
            )
        bridged: Final = _bridged_request(hoisted, responses_api_request, ())
        bridged_input: Final = bridged.input
        bridged_request: Final = bridged.responses_api_request
        litellm_completion_request: Final[dict] = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=bridged_input,
                responses_api_request=bridged_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                previous_response_client_ids,
                litellm_completion_request=litellm_completion_request,
                request_input=bridged_input,
                responses_api_request=bridged_request,
                **kwargs,
            )

        completion_args: Final = {**kwargs, **litellm_completion_request, "_skip_responses_api_bridge": True}

        from litellm.native.chat_plan import execute_planned_chat

        planned_chat: Final = _plan_bridged_chat(_REQUEST_KWARGS_ADAPTER.validate_python(completion_args))
        litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = (
            litellm.completion(**completion_args) if planned_chat is None else execute_planned_chat(planned_chat)
        )

        if planned_chat is not None:
            _price_planned_chat_response(completion_args, litellm_completion_response)

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: Final[ResponsesAPIResponse] = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=bridged_input,
                    responses_api_request=bridged_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=bridged_input,
                responses_api_request=bridged_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                logs_completion=planned_chat is not None,
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")

    async def _async_response_api_handler_after_earlier_searches(
        self,
        previous_response_client_ids: tuple[str, ...],
        previous_response_id: str,
        hoisted: HoistedAdditionalTools,
        /,
        *,
        model: str,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: str | None,
        stream: bool | None,
        extra_headers: Mapping[str, object] | None,
        request_kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        stored_turns: Final = await ResponsesSessionHandler.get_stored_turns_for_previous_response_id(
            previous_response_id, previous_response_client_ids
        )
        bridged: Final = _bridged_request(
            hoisted,
            responses_api_request,
            ResponsesSessionHandler.client_tool_search_tools_loaded_before(
                stored_turns, previous_response_id, previous_response_client_ids
            ),
        )
        return await self.async_response_api_handler(
            previous_response_client_ids,
            stored_turns,
            litellm_completion_request=(
                LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                    model=model,
                    input=bridged.input,
                    responses_api_request=bridged.responses_api_request,
                    custom_llm_provider=custom_llm_provider,
                    stream=stream,
                    extra_headers=extra_headers,
                    **request_kwargs,
                )
            ),
            request_input=bridged.input,
            responses_api_request=bridged.responses_api_request,
            **request_kwargs,
        )

    async def async_response_api_handler(
        self,
        previous_response_client_ids: tuple[str, ...] = (),
        stored_turns: StoredTurns | None = None,
        /,
        *,
        litellm_completion_request: dict,
        request_input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        previous_response_id: Final[str | None] = responses_api_request.get("previous_response_id")
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
                instructions=responses_api_request.get("instructions"),
                request_kwargs=_REQUEST_KWARGS_ADAPTER.validate_python(kwargs),
                previous_response_client_ids=previous_response_client_ids,
                stored_turns=stored_turns,
                input_replaces_history=holds_compaction_summary(with_compaction_items_as_params(request_input)),
            )

        acompletion_args: Final = {**kwargs, **litellm_completion_request, "_skip_responses_api_bridge": True}

        from litellm.native.chat_plan import aexecute_planned_chat

        planned_chat: Final = _plan_bridged_chat(_REQUEST_KWARGS_ADAPTER.validate_python(acompletion_args))
        litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = (
            await litellm.acompletion(**acompletion_args)
            if planned_chat is None
            else await aexecute_planned_chat(planned_chat)
        )

        if planned_chat is not None:
            _price_planned_chat_response(acompletion_args, litellm_completion_response)

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: Final[ResponsesAPIResponse] = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=litellm_completion_request.get("model") or "",
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
                custom_llm_provider=litellm_completion_request.get("custom_llm_provider"),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                logs_completion=planned_chat is not None,
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")
