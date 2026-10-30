# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import asyncio
import copy
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm.constants import REDACTED_BY_LITELLM
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.classifier_logging import without_classifier_audit
from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
)
from litellm.litellm_core_utils.served_output_texts import SERVED_OUTPUT_TEXTS_KEY
from litellm.llms.base_llm.guardrail_translation.utils import stream_item_field, stream_item_items
from litellm.llms.vertex_ai.common_utils import (
    redact_vertex_ai_metadata_from_litellm_params,
    redact_vertex_ai_metadata_from_logged_object,
)
from litellm.secret_managers.main import str_to_bool
from litellm.types.router import BaselineRouteStamp
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObject,
    )

    LiteLLMLoggingObject = _LiteLLMLoggingObject
else:
    LiteLLMLoggingObject = Any

MESSAGE_REDACTION_ENABLE_HEADERS: Final = ("litellm-enable-message-redaction", "x-litellm-enable-message-redaction")


def _opted_out_of_message_logging(custom_logger: CustomLogger) -> bool:
    return hasattr(custom_logger, "message_logging") and custom_logger.message_logging is not True


def redact_message_input_output_from_custom_logger(
    litellm_logging_obj: LiteLLMLoggingObject, result, custom_logger: CustomLogger
) -> object:
    if _opted_out_of_message_logging(custom_logger):
        return perform_redaction(litellm_logging_obj.model_call_details, result, redact_streaming_responses=False)
    return result


def redacted_model_call_details_for_custom_logger(model_call_details: dict, custom_logger: CustomLogger) -> dict:
    """
    Returns a copy of model_call_details with the request messages, instructions and streaming responses
    redacted when the custom logger has opted out of message logging. The shared model_call_details is left
    untouched so other callbacks and the spend log still receive the unredacted values.
    """
    if not _opted_out_of_message_logging(custom_logger):
        return model_call_details
    return {
        **{key: value for key, value in model_call_details.items() if key != SERVED_OUTPUT_TEXTS_KEY},
        **_redacted_entries_for_opted_out_logger(model_call_details),
    }


def _redacted_entries_for_opted_out_logger(model_call_details: Mapping[str, object]) -> Mapping[str, object]:
    optional_params: Final = _as_mapping(model_call_details.get("optional_params"))
    litellm_params: Final = _as_mapping(model_call_details.get("litellm_params"))
    standard_logging_object: Final = _as_mapping(model_call_details.get("standard_logging_object"))
    return {
        "messages": [{"role": "user", "content": REDACTED_BY_LITELLM}],
        "prompt": "",
        "input": "",
        **({"instructions": REDACTED_BY_LITELLM} if model_call_details.get("instructions") is not None else {}),
        **({"optional_params": _with_redacted_instructions(optional_params)} if optional_params is not None else {}),
        **({"litellm_params": redacted_litellm_params(litellm_params)} if litellm_params is not None else {}),
        **(
            {"standard_logging_object": _redact_standard_logging_object(standard_logging_object)}
            if standard_logging_object is not None
            else {}
        ),
        **{
            streaming_key: _redacted_streaming_response_copy(model_call_details[streaming_key])
            for streaming_key in (
                "complete_streaming_response",
                "async_complete_streaming_response",
                "complete_response",
            )
            if model_call_details.get(streaming_key) is not None
        },
    }


def redacted_response_obj_for_custom_logger(response_obj: object, custom_logger: CustomLogger) -> object:
    if not _opted_out_of_message_logging(custom_logger) or not _is_redactable_response(response_obj):
        return response_obj
    return perform_redaction({}, response_obj)


def _is_redactable_response(response_obj: object) -> bool:
    return isinstance(
        response_obj, (litellm.ModelResponse, litellm.ResponsesAPIResponse, litellm.EmbeddingResponse)
    ) or (isinstance(response_obj, dict) and ("choices" in response_obj or "output" in response_obj))


def _redacted_streaming_response_copy(streaming_response: object) -> object:
    redacted_response: Final = (
        streaming_response
        if isinstance(streaming_response, litellm.ResponsesAPIResponse)
        else copy.deepcopy(streaming_response)
    )
    return _redact_streaming_response(redacted_response)


BEDROCK_GUARDRAIL_TRACE_FIELDS: Final = ("trace", "amazon-bedrock-trace")
_MATCHED_TEXT_KEYS: Final = frozenset({"match", "regex"})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _without_matched_text(trace: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    redacted: Final = dict(copy.deepcopy(trace))
    stack: Final[list[JsonValue]] = [redacted]  # mutable-ok: iterative walk, recursive_detector bans recursion here
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in _MATCHED_TEXT_KEYS & node.keys():
                node[key] = "[REDACTED]"
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return redacted


def _redacted_bedrock_trace(trace: object) -> object:
    try:
        return _without_matched_text(_JSON_OBJECT.validate_python(trace))
    except ValidationError:
        return trace


def _redact_bedrock_guardrail_trace(response: object) -> None:
    """A Bedrock guardrail trace echoes the matched text of its sensitive-information and word findings back."""
    mapping: Final = _as_mapping(response)
    if isinstance(response, dict):
        traces: Final = {
            field: _redacted_bedrock_trace(value)
            for field, value in (mapping or {}).items()
            if field in BEDROCK_GUARDRAIL_TRACE_FIELDS
        }
        response.update(traces)
        return
    for field, value in ((name, getattr(response, name, None)) for name in BEDROCK_GUARDRAIL_TRACE_FIELDS):
        if value is not None:
            setattr(response, field, _redacted_bedrock_trace(value))


def _redact_streaming_response(streaming_response):
    if hasattr(streaming_response, "choices"):
        for choice in streaming_response.choices:
            _redact_choice_content(choice)
        redact_vertex_ai_metadata_from_logged_object(streaming_response)
        if isinstance(streaming_response, litellm.ModelResponse):
            _redact_bedrock_guardrail_trace(streaming_response)
    elif isinstance(streaming_response, litellm.ResponsesAPIResponse):
        return perform_redaction({}, streaming_response)
    elif hasattr(streaming_response, "output"):
        _redact_responses_api_output(streaming_response.output)
        if hasattr(streaming_response, "reasoning") and streaming_response.reasoning is not None:
            streaming_response.reasoning = None
    return streaming_response


def _redact_tool_calls(tool_calls) -> None:
    """Redact tool call arguments (assistant tool calls carry prompt-derived data)."""
    if not tool_calls:
        return
    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        if function is not None and hasattr(function, "arguments"):
            function.arguments = REDACTED_BY_LITELLM


def _redact_function_call(function_call) -> None:
    """Redact legacy assistant function_call arguments."""
    if function_call is not None and hasattr(function_call, "arguments"):
        function_call.arguments = REDACTED_BY_LITELLM


def _redact_choice_content(choice):
    """Helper to redact content in a choice (message or delta)."""
    if isinstance(choice, litellm.Choices):
        if choice.message.content is not None:
            choice.message.content = REDACTED_BY_LITELLM
        if getattr(choice.message, "reasoning_content", None) is not None:
            choice.message.reasoning_content = REDACTED_BY_LITELLM
        if hasattr(choice.message, "thinking_blocks"):
            choice.message.thinking_blocks = None
        _redact_provider_specific_fields(choice.message)
        _redact_tool_calls(getattr(choice.message, "tool_calls", None))
        _redact_function_call(getattr(choice.message, "function_call", None))
    elif isinstance(choice, litellm.utils.StreamingChoices):
        if choice.delta.content is not None:
            choice.delta.content = REDACTED_BY_LITELLM
        if getattr(choice.delta, "reasoning_content", None) is not None:
            choice.delta.reasoning_content = REDACTED_BY_LITELLM
        if hasattr(choice.delta, "thinking_blocks"):
            choice.delta.thinking_blocks = None
        _redact_provider_specific_fields(choice.delta)
        _redact_tool_calls(getattr(choice.delta, "tool_calls", None))
        _redact_function_call(getattr(choice.delta, "function_call", None))


def _as_mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None  # pyright: ignore[reportUnknownVariableType]  # keys are read one by one


def _redact_provider_specific_fields(message: object) -> None:
    if (fields := _as_mapping(getattr(message, "provider_specific_fields", None))) is not None:
        setattr(message, "provider_specific_fields", _redacted_provider_specific_fields(fields))  # noqa: B010  # typed as object


def _redact_provider_specific_fields_dict(choice: Mapping[str, object]) -> None:
    for message in (choice.get("message"), choice.get("delta")):
        if (
            (readable := _as_mapping(message)) is not None
            and (fields := _as_mapping(readable.get("provider_specific_fields"))) is not None
            and isinstance(message, dict)
        ):
            message["provider_specific_fields"] = _redacted_provider_specific_fields(fields)


def _redacted_provider_specific_fields(fields: Mapping[str, object]) -> Mapping[str, object]:
    """Copy of provider_specific_fields with its reasoning redacted the same way as the message's own."""
    return {
        **fields,
        **({"reasoning_content": REDACTED_BY_LITELLM} if fields.get("reasoning_content") is not None else {}),
        **{
            key: None
            for key in (
                "thinking_blocks",
                "reasoningContentBlocks",
                "citations",
                "compaction_blocks",
                "compaction_delta",
            )
            if key in fields
        },
    }


def _redact_field_when_present(item: object, field: str) -> None:
    if getattr(item, field, None) is not None:
        setattr(item, field, REDACTED_BY_LITELLM)


def _redact_responses_api_output(output_items):
    """Helper to redact ResponsesAPIResponse output items."""
    for output_item in output_items:
        output_type = stream_item_field(output_item, "type")
        if getattr(output_item, "text", None) is not None:
            output_item.text = REDACTED_BY_LITELLM

        for content_part in stream_item_items(output_item, "content"):
            _redact_field_when_present(content_part, "text")
            _redact_field_when_present(content_part, "refusal")

        if output_type == "reasoning":
            for summary_item in stream_item_items(output_item, "summary"):
                _redact_field_when_present(summary_item, "text")

        if output_type == "compaction" and hasattr(output_item, "encrypted_content"):
            output_item.encrypted_content = REDACTED_BY_LITELLM
        if output_type == "function_call" and hasattr(output_item, "arguments"):
            output_item.arguments = REDACTED_BY_LITELLM
        if output_type == "custom_tool_call" and hasattr(output_item, "input"):
            output_item.input = REDACTED_BY_LITELLM


_RESPONSE_ITEMS: Final = TypeAdapter(tuple[object, ...])


def _redacted_responses_api_output_part(part: object, fields: tuple[str, ...], redacted_str: str) -> object:
    if (part_mapping := _as_mapping(part)) is None:
        return part
    return {key: redacted_str if key in fields and value is not None else value for key, value in part_mapping.items()}


def _redacted_responses_api_output_item(item: object, redacted_str: str) -> object:
    if (item_mapping := _as_mapping(item)) is None:
        return item
    content: Final = item_mapping.get("content")
    summary: Final = item_mapping.get("summary")
    return {
        **item_mapping,
        **({"text": redacted_str} if item_mapping.get("text") is not None else {}),
        **(
            {
                "content": [
                    _redacted_responses_api_output_part(part, ("text", "refusal"), redacted_str)
                    for part in _RESPONSE_ITEMS.validate_python(content)
                ]
            }
            if isinstance(content, list)
            else {}
        ),
        **(
            {
                "summary": [
                    _redacted_responses_api_output_part(part, ("text",), redacted_str)
                    for part in _RESPONSE_ITEMS.validate_python(summary)
                ]
            }
            if item_mapping.get("type") == "reasoning" and isinstance(summary, list)
            else {}
        ),
        **(
            {"encrypted_content": redacted_str}
            if item_mapping.get("type") == "compaction" and "encrypted_content" in item_mapping
            else {}
        ),
        **(
            {"arguments": redacted_str}
            if item_mapping.get("type") == "function_call" and "arguments" in item_mapping
            else {}
        ),
        **(
            {"input": redacted_str}
            if item_mapping.get("type") == "custom_tool_call" and "input" in item_mapping
            else {}
        ),
    }


def _redact_responses_api_output_dict(output_items: Sequence[object], redacted_str: str) -> Sequence[object]:
    return [_redacted_responses_api_output_item(item, redacted_str) for item in output_items]


def _with_redacted_instructions(params: Mapping[str, object]) -> Mapping[str, object]:
    return params if params.get("instructions") is None else {**params, "instructions": REDACTED_BY_LITELLM}


def _redacted_responses_api_response(
    response: Mapping[str, object], *, redact_instructions: bool = True
) -> Mapping[str, object]:
    output: Final = response.get("output")
    redacted: Final = {
        **response,
        "output": _redact_responses_api_output_dict(_RESPONSE_ITEMS.validate_python(output), REDACTED_BY_LITELLM)
        if isinstance(output, list)
        else output,
    }
    return _with_redacted_instructions(redacted) if redact_instructions else redacted


def redacted_standard_logging_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    """
    Return a copy of a ``StandardLoggingPayload`` with its messages and response redacted.

    The success path redacts through ``perform_redaction`` before a callback ever sees the
    payload, but the failure path does not, so a callback that batches both has to redact
    the ones it is handed.
    """
    return _redact_standard_logging_object(payload)


def _redact_standard_logging_object(
    payload: Mapping[str, object], *, redact_instructions: bool = True
) -> dict[str, object]:
    standard_logging_object: Final = copy.deepcopy(without_classifier_audit(payload))
    redacted_str: Final = REDACTED_BY_LITELLM

    if standard_logging_object.get("messages") is not None:
        standard_logging_object["messages"] = [{"role": "user", "content": redacted_str}]
    model_parameters: Final = _as_mapping(standard_logging_object.get("model_parameters"))
    if model_parameters is not None and redact_instructions:
        standard_logging_object["model_parameters"] = _with_redacted_instructions(model_parameters)

    response: Final = standard_logging_object.get("response")
    if response is not None:
        if isinstance(response, dict) and "output" in response:
            standard_logging_object["response"] = _redacted_responses_api_response(
                response, redact_instructions=redact_instructions
            )
            redact_vertex_ai_metadata_from_logged_object(standard_logging_object["response"])
        elif isinstance(response, dict) and "choices" in response:
            # ModelResponse dict format - redact content in choices
            if isinstance(response.get("choices"), list):
                _redact_model_response_dict_choices(response["choices"], redacted_str)
            redact_vertex_ai_metadata_from_logged_object(response)
            _redact_bedrock_guardrail_trace(response)
        elif isinstance(response, str):
            standard_logging_object["response"] = redacted_str
        else:
            # For other formats (empty dict, None, etc.), use simple text format
            standard_logging_object["response"] = {"text": redacted_str}
    return standard_logging_object


def _redact_tool_calls_dict(message: Mapping[str, object]) -> None:
    """Redact tool call / function_call arguments in a dict-form message or delta."""
    tool_calls: Final = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and isinstance(tool_call.get("function"), dict):
                tool_call["function"]["arguments"] = REDACTED_BY_LITELLM

    function_call: Final = message.get("function_call")
    if isinstance(function_call, dict) and "arguments" in function_call:
        function_call["arguments"] = REDACTED_BY_LITELLM


def _redact_model_response_dict_choices(choices, redacted_str: str):
    for choice in choices:
        if isinstance(choice, dict):
            _redact_provider_specific_fields_dict(choice)
            if "message" in choice and isinstance(choice["message"], dict):
                if choice["message"].get("content") is not None:
                    choice["message"]["content"] = redacted_str
                if choice["message"].get("reasoning_content") is not None:
                    choice["message"]["reasoning_content"] = redacted_str
                if "thinking_blocks" in choice["message"]:
                    choice["message"]["thinking_blocks"] = None
                if "audio" in choice["message"]:
                    choice["message"]["audio"] = None
                _redact_tool_calls_dict(choice["message"])
            elif "delta" in choice and isinstance(choice["delta"], dict):
                if choice["delta"].get("content") is not None:
                    choice["delta"]["content"] = redacted_str
                if choice["delta"].get("reasoning_content") is not None:
                    choice["delta"]["reasoning_content"] = redacted_str
                if "thinking_blocks" in choice["delta"]:
                    choice["delta"]["thinking_blocks"] = None
                if "audio" in choice["delta"]:
                    choice["delta"]["audio"] = None
                _redact_tool_calls_dict(choice["delta"])
            elif choice.get("text") is not None:
                choice["text"] = redacted_str
        else:
            _redact_choice_content(choice)


def _redacted_baseline_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    route: Final = metadata.get("_autorouter_baseline_route")
    if not isinstance(route, BaselineRouteStamp):
        return metadata
    return {**metadata, "_autorouter_baseline_route": replace(route, request_parameters=None)}


def _redacted_proxy_server_request(request: Mapping[str, object], *, redact_body: bool) -> Mapping[str, object]:
    """Copy of proxy_server_request; redact_body redacts its body the same way as the top-level request."""
    body: Final = _as_mapping(request.get("body"))
    if not redact_body or body is None:
        return without_classifier_audit(request)
    redacted_body: Final = {**body}
    perform_redaction(redacted_body, result=None)
    return {**without_classifier_audit(request), "body": redacted_body}


def _redact_hidden_provider_specific_fields(params: Mapping[str, object]) -> None:
    """Redact the provider_specific_fields copied into metadata hidden_params, keeping each metadata dict."""
    for metadata in (params.get("metadata"), params.get("litellm_metadata")):
        if (
            (readable := _as_mapping(metadata)) is not None
            and (hidden_params := _as_mapping(readable.get("hidden_params"))) is not None
            and (fields := _as_mapping(hidden_params.get("provider_specific_fields"))) is not None
            and isinstance(metadata, dict)
        ):
            metadata["hidden_params"] = {
                **hidden_params,
                "provider_specific_fields": _redacted_provider_specific_fields(fields),
            }


def redacted_litellm_params(params: Mapping[str, object], *, redact_request_body: bool = True) -> dict[str, object]:
    """
    Copy of litellm_params without classifier audit fields or baseline route request parameters.
    redact_request_body also redacts the request instructions and the nested proxy_server_request body.
    """
    request: Final = _as_mapping(params.get("proxy_server_request"))
    return {
        **(_with_redacted_instructions(params) if redact_request_body else params),
        **{
            key: _redacted_baseline_metadata(value)
            for key, value in params.items()
            if key in ("metadata", "litellm_metadata") and isinstance(value, Mapping)
        },
        **(
            {"proxy_server_request": _redacted_proxy_server_request(request, redact_body=redact_request_body)}
            if request is not None
            else {}
        ),
    }


def perform_redaction(model_call_details: dict, result, redact_streaming_responses: bool = True):
    """
    Performs the actual redaction on the logging object and result.

    redact_streaming_responses=False skips the in-place redaction of the shared streaming
    response entries; per-callback redaction hands each opted-out callback its own redacted
    copy via redacted_model_call_details_for_custom_logger instead. It also leaves the shared
    request instructions, proxy request body and hidden provider fields as they are, since the
    spend log and the other callbacks read them.
    """
    # Redact model_call_details
    params: Final = _as_mapping(model_call_details.get("litellm_params"))
    if params is not None:
        if redact_streaming_responses:
            _redact_hidden_provider_specific_fields(params)
        model_call_details["litellm_params"] = redacted_litellm_params(
            params, redact_request_body=redact_streaming_responses
        )
    model_call_details["messages"] = [{"role": "user", "content": REDACTED_BY_LITELLM}]
    model_call_details["prompt"] = ""
    model_call_details["input"] = ""
    optional_params: Final = _as_mapping(model_call_details.get("optional_params"))
    if redact_streaming_responses:
        model_call_details.update(
            {
                **({"instructions": REDACTED_BY_LITELLM} if model_call_details.get("instructions") is not None else {}),
                **({"optional_params": _with_redacted_instructions(optional_params)} if optional_params else {}),
            }
        )
    model_call_details.pop(SERVED_OUTPUT_TEXTS_KEY, None)
    standard_logging_object: Final = model_call_details.get("standard_logging_object")
    if isinstance(standard_logging_object, Mapping):
        model_call_details["standard_logging_object"] = _redact_standard_logging_object(
            standard_logging_object, redact_instructions=redact_streaming_responses
        )
    redact_vertex_ai_metadata_from_litellm_params(model_call_details)

    # Redact streaming response
    if redact_streaming_responses and model_call_details.get("stream", False) is True:
        model_call_details.update(
            {
                streaming_key: _redact_streaming_response(model_call_details[streaming_key])
                for streaming_key in ("complete_streaming_response", "async_complete_streaming_response")
                if model_call_details.get(streaming_key) is not None
            }
        )

    # Redact result
    if result is not None:
        # Check if result is a coroutine, async generator, or other async object - these cannot be deepcopied
        if (
            asyncio.iscoroutine(result)
            or inspect.iscoroutinefunction(result)
            or hasattr(result, "__aiter__")
            or hasattr(result, "__anext__")  # async generator
        ):  # async iterator
            # For async objects, return a simple redacted response without deepcopy
            return {"text": REDACTED_BY_LITELLM}

        if not _is_redactable_response(result):
            return {"text": REDACTED_BY_LITELLM}

        _result: Final = copy.deepcopy(result)
        if isinstance(_result, litellm.ModelResponse):
            if hasattr(_result, "choices") and _result.choices is not None:
                for choice in _result.choices:
                    _redact_choice_content(choice)
            redact_vertex_ai_metadata_from_logged_object(_result)
            _redact_bedrock_guardrail_trace(_result)
        elif isinstance(_result, dict) and "choices" in _result:
            # Handle dict representation of ModelResponse (e.g., from model_dump())
            if _result.get("choices") is not None:
                _redact_model_response_dict_choices(_result["choices"], REDACTED_BY_LITELLM)
            redact_vertex_ai_metadata_from_logged_object(_result)
            _redact_bedrock_guardrail_trace(_result)
        elif isinstance(_result, dict) and "output" in _result:
            return _redacted_responses_api_response(_result, redact_instructions=redact_streaming_responses)
        elif isinstance(_result, litellm.ResponsesAPIResponse):
            if hasattr(_result, "output"):
                _redact_responses_api_output(_result.output)
            # Redact reasoning field in ResponsesAPIResponse
            if hasattr(_result, "reasoning") and _result.reasoning is not None:
                _result.reasoning = None
            return _result.model_copy(
                update={"instructions": REDACTED_BY_LITELLM}
                if redact_streaming_responses and _result.instructions is not None
                else {}
            )
        elif isinstance(_result, litellm.EmbeddingResponse):
            if hasattr(_result, "data") and _result.data is not None:
                _result.data = []
        else:
            return {"text": REDACTED_BY_LITELLM}
        return _result


def should_redact_message_logging(model_call_details: dict) -> bool:
    """
    Determine if message logging should be redacted.

    Priority order:
    1. Dynamic parameter (turn_off_message_logging in request)
    2. Headers (litellm-disable-message-redaction / litellm-enable-message-redaction)
    3. Global setting (litellm.turn_off_message_logging)
    """
    litellm_params: Final = model_call_details.get("litellm_params", {})

    metadata_field: Final = get_metadata_variable_name_from_kwargs(litellm_params)
    metadata = litellm_params.get(metadata_field, {})
    if not isinstance(metadata, dict):
        # Fall back: litellm_metadata was None, try metadata
        metadata = litellm_params.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # Get headers from the metadata
    request_headers: Final = metadata.get("headers", {})

    # Check for headers that explicitly control redaction
    if request_headers and bool(request_headers.get("litellm-disable-message-redaction", False)):
        # User explicitly disabled redaction via header
        return False

    is_redaction_enabled_via_header = False
    for header in MESSAGE_REDACTION_ENABLE_HEADERS:
        if bool(request_headers.get(header, False)):
            is_redaction_enabled_via_header = True
            break

    # Priority 1: Check dynamic parameter first (if explicitly set)
    dynamic_turn_off: Final = _get_turn_off_message_logging_from_dynamic_params(model_call_details)
    if dynamic_turn_off is not None:
        # Dynamic parameter is explicitly set, use it
        return dynamic_turn_off

    # Priority 2: Check if header explicitly enables redaction
    if is_redaction_enabled_via_header:
        return True

    # Priority 3: Fall back to global setting
    return litellm.turn_off_message_logging is True


def redact_message_input_output_from_logging(model_call_details: dict, result, input: object | None = None) -> Any:
    """
    Removes messages, prompts, input, response from logging. This modifies the data in-place
    only redacts when litellm.turn_off_message_logging == True
    """
    if should_redact_message_logging(model_call_details):
        return perform_redaction(model_call_details, result)
    return result


def _get_turn_off_message_logging_from_dynamic_params(
    model_call_details: dict,
) -> bool | None:
    """
    gets the value of `turn_off_message_logging` from the dynamic params, if it exists.

    handles boolean and string values of `turn_off_message_logging`
    """
    standard_callback_dynamic_params: Final[StandardCallbackDynamicParams | None] = model_call_details.get(
        "standard_callback_dynamic_params", None
    )
    if standard_callback_dynamic_params:
        _turn_off_message_logging: Final = standard_callback_dynamic_params.get("turn_off_message_logging")
        if isinstance(_turn_off_message_logging, bool):
            return _turn_off_message_logging
        elif isinstance(_turn_off_message_logging, str):
            return str_to_bool(_turn_off_message_logging)
    return None


def redact_user_api_key_info(metadata: dict) -> dict:
    """
    removes any user_api_key_info before passing to logging object, if flag set

    Usage:

    SDK
    ```python
    litellm.redact_user_api_key_info = True
    ```

    PROXY:
    ```yaml
    litellm_settings:
        redact_user_api_key_info: true
    ```
    """
    if litellm.redact_user_api_key_info is not True:
        return metadata

    new_metadata: Final = {}
    for k, v in metadata.items():
        if isinstance(k, str) and k.startswith("user_api_key"):
            pass
        else:
            new_metadata[k] = v

    return new_metadata
