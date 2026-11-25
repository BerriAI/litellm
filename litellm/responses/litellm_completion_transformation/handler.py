"""
Handler for transforming responses api requests to litellm.completion requests
"""

import sys
from collections.abc import Coroutine, Mapping
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.exceptions import UnknownPreviousResponseError, UnresolvablePreviousResponseError
from litellm.litellm_core_utils.allowed_tools_tool_choice import unenforced_allowed_tools_tool_choice
from litellm.litellm_core_utils.core_helpers import attribute_or_none
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.responses.additional_tools import hoist_additional_tools
from litellm.responses.litellm_completion_transformation.error_param import error_param_named_as_a_responses_field
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    holds_an_earlier_reply,
    previous_response_not_found,
    previous_response_refusal,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse

_REQUEST_KWARGS_ADAPTER: Final = TypeAdapter(dict[str, object])


def _with_the_tool_choice_sent(
    responses_api_request: ResponsesAPIOptionalRequestParams,
    custom_llm_provider: str | None,
    request_kwargs: Mapping[str, object],
) -> ResponsesAPIOptionalRequestParams:
    api_version: Final = request_kwargs.get("api_version")
    tool_choice: Final[object] = responses_api_request.get(  # pyright: ignore[reportUnknownMemberType]  # ToolChoice has an untyped member
        "tool_choice"
    )
    if not unenforced_allowed_tools_tool_choice(
        tool_choice, custom_llm_provider, api_version if isinstance(api_version, str) else None
    ):
        return responses_api_request
    return {**responses_api_request, "tool_choice": None}


def _proxy_database_connected() -> bool:
    return attribute_or_none(sys.modules.get("litellm.proxy.proxy_server"), "prisma_client") is not None


def _stored_turns_need_aresponses(
    client_response_ids: tuple[str, ...], model: str, provider: str | None
) -> UnresolvablePreviousResponseError:
    message: Final = (
        f"Previous response with id '{client_response_ids[-1]}' can only be continued with `litellm.aresponses()` in "
        "this process. Its conversation is rebuilt from the LiteLLM proxy's database, which only the proxy's event "
        "loop can query."
    )
    return previous_response_refusal(client_response_ids)(
        message=message,
        model=model,
        llm_provider=provider,
        body={"message": message, "type": "invalid_request_error", "param": "previous_response_id", "code": None},
    )


def _sync_follow_up_refusal(
    client_response_ids: tuple[str, ...], model: str, provider: str | None
) -> UnresolvablePreviousResponseError:
    return (
        _stored_turns_need_aresponses(client_response_ids, model, provider)
        if _proxy_database_connected()
        else previous_response_not_found(client_response_ids, model, provider)
    )


def _invalid_previous_response_id(
    previous_response_id: object, model: str, provider: str | None
) -> UnknownPreviousResponseError | None:
    if previous_response_id is None or isinstance(previous_response_id, str):
        return None
    message: Final = (
        f"Invalid type for 'previous_response_id': expected a string, got {type(previous_response_id).__name__}"
    )
    return UnknownPreviousResponseError(
        message=message,
        model=model,
        llm_provider=provider,
        body={
            "message": message,
            "type": "invalid_request_error",
            "param": "previous_response_id",
            "code": "invalid_type",
        },
    )


def _log_full_tool_names(
    request_kwargs: Mapping[str, object], responses_api_request: ResponsesAPIOptionalRequestParams
) -> None:
    logging_obj: Final = request_kwargs.get("litellm_logging_obj")
    chat_names: Final = LiteLLMCompletionResponsesConfig.member_chat_names(
        LiteLLMCompletionResponsesConfig.request_tools(responses_api_request)
    )
    shortened: Final = {sent_name: full_name for full_name, sent_name in chat_names.items() if sent_name != full_name}
    if shortened and isinstance(logging_obj, LiteLLMLoggingObj):
        logging_obj.record_bridged_tool_names(shortened)


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
        previous_response_id: Final[str | None] = responses_api_request.get("previous_response_id")
        invalid_previous_response_id: Final = _invalid_previous_response_id(
            previous_response_id, model, custom_llm_provider
        )
        if invalid_previous_response_id is not None:
            raise invalid_previous_response_id
        hoisted: Final = hoist_additional_tools(input, responses_api_request.get("tools"))
        bridged_input: Final = hoisted.input
        bridged_request: Final[ResponsesAPIOptionalRequestParams] = (
            {**responses_api_request, "tools": list(hoisted.tools)} if hoisted.hoisted else responses_api_request
        )
        litellm_completion_request: Final = _REQUEST_KWARGS_ADAPTER.validate_python(
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
        echoed_request: Final = _with_the_tool_choice_sent(
            bridged_request, custom_llm_provider, _REQUEST_KWARGS_ADAPTER.validate_python(kwargs)
        )

        if _is_async:
            return self.async_response_api_handler(
                previous_response_client_ids,
                litellm_completion_request=litellm_completion_request,
                request_input=bridged_input,
                responses_api_request=echoed_request,
                previous_response_id=previous_response_id,
                **kwargs,
            )

        if previous_response_id and not holds_an_earlier_reply(bridged_input):
            raise _sync_follow_up_refusal(
                previous_response_client_ids or (previous_response_id,), model, custom_llm_provider
            )
        completion_args: Final = {}
        completion_args.update(kwargs)
        completion_args.update(litellm_completion_request)
        completion_args["_skip_responses_api_bridge"] = True

        _log_full_tool_names(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs), echoed_request)
        with error_param_named_as_a_responses_field(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs), bridged_request):
            litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = litellm.completion(
                **completion_args,
            )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: Final[ResponsesAPIResponse] = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=bridged_input,
                    responses_api_request=echoed_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=bridged_input,
                responses_api_request=echoed_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")

    async def async_response_api_handler(
        self,
        previous_response_client_ids: tuple[str, ...] = (),
        /,
        *,
        litellm_completion_request: dict,
        request_input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        previous_response_id: str | None,
        **kwargs,
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        request_fields: Final = _REQUEST_KWARGS_ADAPTER.validate_python(litellm_completion_request)
        completion_request: Final[Mapping[str, object]] = (
            _REQUEST_KWARGS_ADAPTER.validate_python(
                await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                    previous_response_id=previous_response_id,
                    litellm_completion_request=dict(request_fields),
                    instructions=responses_api_request.get("instructions"),
                    request_kwargs=_REQUEST_KWARGS_ADAPTER.validate_python(kwargs),
                    previous_response_client_ids=previous_response_client_ids,
                    request_input=request_input,
                )
            )
            if previous_response_id
            else request_fields
        )

        acompletion_args: Final = {}
        acompletion_args.update(kwargs)
        acompletion_args.update(completion_request)
        acompletion_args["_skip_responses_api_bridge"] = True

        _log_full_tool_names(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs), responses_api_request)
        with error_param_named_as_a_responses_field(
            _REQUEST_KWARGS_ADAPTER.validate_python(kwargs), responses_api_request
        ):
            litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = await litellm.acompletion(
                **acompletion_args,
            )
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
            )
        raise ValueError(f"Unexpected response type: {type(litellm_completion_response)}")
