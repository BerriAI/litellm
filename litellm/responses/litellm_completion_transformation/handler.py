"""
Handler for transforming responses api requests to litellm.completion requests
"""

import sys
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.exceptions import UnresolvablePreviousResponseError
from litellm.litellm_core_utils.allowed_tools_tool_choice import unenforced_allowed_tools_tool_choice
from litellm.litellm_core_utils.core_helpers import attribute_or_none, drops_unsupported_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.model_param_helper import embeds_thought_signature_in_tool_call_id
from litellm.responses.additional_tools import HoistedAdditionalTools, hoist_additional_tools
from litellm.responses.litellm_completion_transformation.client_tool_search import bridge_client_tool_search
from litellm.responses.litellm_completion_transformation.compaction_items import holds_compaction_summary
from litellm.responses.litellm_completion_transformation.compaction_trigger import (
    CompactionSummaryStream,
    as_compaction_reply,
    ends_in_compaction_trigger,
    with_trailing_compaction_trigger_as_summary_prompt,
    without_a_forced_tool,
)
from litellm.responses.litellm_completion_transformation.error_param import error_param_named_as_a_responses_field
from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler, StoredTurns
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.tool_search import has_client_tool_search
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    holds_an_earlier_reply,
    previous_response_ids_metadata,
    previous_response_not_found,
    with_compaction_items_as_params,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse

_REQUEST_KWARGS_ADAPTER: Final = TypeAdapter(dict[str, object])
_BRIDGE_TRANSFORM_KEYWORDS: Final = frozenset({"caller_input", "names_non_string_fields"})


def _without_bridge_transform_keywords(request_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {key: value for key, value in request_kwargs.items() if key not in _BRIDGE_TRANSFORM_KEYWORDS}
    )


def _chat_verbosity(responses_api_request: Mapping[str, object]) -> Mapping[str, object]:
    match responses_api_request.get("text"):
        case {"verbosity": str() as verbosity}:
            return {"verbosity": verbosity}
        case _:
            return {}


def _proxy_database_connected() -> bool:
    return attribute_or_none(sys.modules.get("litellm.proxy.proxy_server"), "prisma_client") is not None


def _stored_turns_need_aresponses(
    client_response_id: str, model: str, provider: str | None
) -> UnresolvablePreviousResponseError:
    message: Final = (
        f"Previous response with id '{client_response_id}' can only be continued with `litellm.aresponses()` in "
        "this process. Its conversation is rebuilt from the LiteLLM proxy's database, which only the proxy's event "
        "loop can query."
    )
    return UnresolvablePreviousResponseError(
        message=message,
        model=model,
        llm_provider=provider,
        body={"message": message, "type": "invalid_request_error", "param": "previous_response_id", "code": None},
    )


def _sync_follow_up_refusal(
    client_response_id: str, model: str, provider: str | None
) -> UnresolvablePreviousResponseError:
    return (
        _stored_turns_need_aresponses(client_response_id, model, provider)
        if _proxy_database_connected()
        else previous_response_not_found(client_response_id, model, provider)
    )


@dataclass(frozen=True, slots=True)
class _BridgedRequest:
    input: str | ResponseInputParam
    responses_api_request: ResponsesAPIOptionalRequestParams


def _bridged_reply(
    completion_response: object,
    *,
    model: str,
    custom_llm_provider: str | None,
    request_input: str | ResponseInputParam,
    responses_api_request: ResponsesAPIOptionalRequestParams,
    litellm_metadata: dict[str, object] | None,  # mutable-ok: the iterator keeps the caller's metadata dict as sent
    tool_call_provider_fields: bool,
    summarizes_into_compaction: bool,
) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
    if isinstance(completion_response, ModelResponse):
        return LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
            chat_completion_response=(
                as_compaction_reply(completion_response, llm_provider=custom_llm_provider)
                if summarizes_into_compaction
                else completion_response
            ),
            request_input=request_input,
            responses_api_request=responses_api_request,
        )
    if isinstance(completion_response, litellm.CustomStreamWrapper):
        return LiteLLMCompletionStreamingIterator(
            model=model,
            litellm_custom_stream_wrapper=completion_response,
            request_input=request_input,
            responses_api_request=responses_api_request,
            custom_llm_provider=custom_llm_provider,
            litellm_metadata=litellm_metadata,
            tool_call_provider_fields=tool_call_provider_fields,
            compaction_summary=(
                CompactionSummaryStream(llm_provider=custom_llm_provider) if summarizes_into_compaction else None
            ),
        )
    raise ValueError(f"Unexpected response type: {type(completion_response)}")


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
        summarizes: Final = ends_in_compaction_trigger(input)
        request: Final = without_a_forced_tool(responses_api_request) if summarizes else responses_api_request
        hoisted: Final = hoist_additional_tools(
            with_trailing_compaction_trigger_as_summary_prompt(input),
            request.get("tools"),
            drop_params=drops_unsupported_params(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs).get("drop_params")),
        )
        previous_response_id: Final = request.get("previous_response_id")
        if _is_async and previous_response_id and has_client_tool_search(hoisted.tools):
            return self._async_response_api_handler_after_earlier_searches(
                previous_response_client_ids,
                previous_response_id,
                hoisted,
                model=model,
                caller_input=input,
                responses_api_request=request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                request_kwargs=_REQUEST_KWARGS_ADAPTER.validate_python(kwargs),
                summarizes_into_compaction=summarizes,
            )
        bridged: Final = _bridged_request(hoisted, request, ())
        bridged_input: Final = bridged.input
        bridged_request: Final = bridged.responses_api_request
        litellm_completion_request: Final[dict[str, object]] = {
            **LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=bridged_input,
                caller_input=input,
                names_non_string_fields=True,
                responses_api_request=bridged_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **_without_bridge_transform_keywords(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs)),
            ),
            **_chat_verbosity(bridged_request),
        }
        echoed_request: Final = _with_the_tool_choice_sent(
            bridged_request, custom_llm_provider, _REQUEST_KWARGS_ADAPTER.validate_python(kwargs)
        )

        if _is_async:
            return self.async_response_api_handler(
                previous_response_client_ids,
                litellm_completion_request=litellm_completion_request,
                request_input=bridged_input,
                responses_api_request=echoed_request,
                summarizes_into_compaction=summarizes,
                **kwargs,
            )

        if (
            previous_response_id
            and not holds_compaction_summary(with_compaction_items_as_params(bridged_input))
            and not holds_an_earlier_reply(bridged_input)
        ):
            raise _sync_follow_up_refusal(
                previous_response_client_ids[-1] if previous_response_client_ids else previous_response_id,
                model,
                custom_llm_provider,
            )
        _log_full_tool_names(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs), echoed_request)
        with error_param_named_as_a_responses_field(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs), bridged_request):
            litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = litellm.completion(
                **{**kwargs, **litellm_completion_request, "_skip_responses_api_bridge": True},
            )
        return _bridged_reply(
            litellm_completion_response,
            model=model,
            custom_llm_provider=custom_llm_provider,
            request_input=bridged_input,
            responses_api_request=echoed_request,
            litellm_metadata=kwargs.get("litellm_metadata", {}),
            tool_call_provider_fields=not embeds_thought_signature_in_tool_call_id(
                _REQUEST_KWARGS_ADAPTER.validate_python(kwargs)
            ),
            summarizes_into_compaction=summarizes,
        )

    async def _async_response_api_handler_after_earlier_searches(
        self,
        previous_response_client_ids: tuple[str, ...],
        previous_response_id: str,
        hoisted: HoistedAdditionalTools,
        /,
        *,
        model: str,
        caller_input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: str | None,
        stream: bool | None,
        extra_headers: Mapping[str, object] | None,
        request_kwargs: Mapping[str, object],
        summarizes_into_compaction: bool,
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
            litellm_completion_request={
                **LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                    model=model,
                    input=bridged.input,
                    caller_input=caller_input,
                    names_non_string_fields=True,
                    responses_api_request=bridged.responses_api_request,
                    custom_llm_provider=custom_llm_provider,
                    stream=stream,
                    extra_headers=extra_headers,
                    **_without_bridge_transform_keywords(request_kwargs),
                ),
                **_chat_verbosity(bridged.responses_api_request),
            },
            request_input=bridged.input,
            responses_api_request=_with_the_tool_choice_sent(
                bridged.responses_api_request, custom_llm_provider, request_kwargs
            ),
            summarizes_into_compaction=summarizes_into_compaction,
            **request_kwargs,
        )

    async def async_response_api_handler(
        self,
        previous_response_client_ids: tuple[str, ...] = (),
        stored_turns: StoredTurns | None = None,
        /,
        *,
        litellm_completion_request: Mapping[str, object],
        request_input: str | ResponseInputParam,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        summarizes_into_compaction: bool = False,
        **kwargs,
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        previous_response_id: Final[str | None] = responses_api_request.get("previous_response_id")
        request_kwargs: Final = _REQUEST_KWARGS_ADAPTER.validate_python(kwargs)
        completion_request: Final[Mapping[str, object]] = (
            await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=dict(litellm_completion_request),
                instructions=responses_api_request.get("instructions"),
                request_kwargs=request_kwargs,
                previous_response_client_ids=previous_response_client_ids,
                stored_turns=stored_turns,
                input_replaces_history=holds_compaction_summary(with_compaction_items_as_params(request_input)),
                request_input=request_input,
            )
            if previous_response_id
            else {
                **_REQUEST_KWARGS_ADAPTER.validate_python(litellm_completion_request),
                **previous_response_ids_metadata((), request_kwargs),
            }
        )

        completion_kwargs: Final[Mapping[str, object]] = {
            **request_kwargs,
            **completion_request,
            "_skip_responses_api_bridge": True,
        }
        _log_full_tool_names(_REQUEST_KWARGS_ADAPTER.validate_python(kwargs), responses_api_request)
        with error_param_named_as_a_responses_field(
            _REQUEST_KWARGS_ADAPTER.validate_python(kwargs), responses_api_request
        ):
            litellm_completion_response: Final[ModelResponse | litellm.CustomStreamWrapper] = await litellm.acompletion(
                **completion_kwargs
            )
        provider: Final = completion_request.get("custom_llm_provider")
        return _bridged_reply(
            litellm_completion_response,
            model=str(completion_request.get("model") or ""),
            custom_llm_provider=provider if isinstance(provider, str) else None,
            request_input=request_input,
            responses_api_request=responses_api_request,
            litellm_metadata=kwargs.get("litellm_metadata", {}),
            tool_call_provider_fields=not embeds_thought_signature_in_tool_call_id(
                _REQUEST_KWARGS_ADAPTER.validate_python(kwargs)
            ),
            summarizes_into_compaction=summarizes_into_compaction,
        )
