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
from typing import TYPE_CHECKING, Any, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import (
    get_metadata_variable_name_from_kwargs,
)
from litellm.llms.vertex_ai.common_utils import (
    redact_vertex_ai_metadata_from_litellm_params,
    redact_vertex_ai_metadata_from_logged_object,
)
from litellm.secret_managers.main import str_to_bool
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObject,
    )

    LiteLLMLoggingObject = _LiteLLMLoggingObject
else:
    LiteLLMLoggingObject = Any


def redact_message_input_output_from_custom_logger(
    litellm_logging_obj: LiteLLMLoggingObject, result, custom_logger: CustomLogger
):
    if hasattr(custom_logger, "message_logging") and custom_logger.message_logging is not True:
        return perform_redaction(litellm_logging_obj.model_call_details, result, redact_streaming_responses=False)
    return result


def redact_streaming_responses_for_custom_logger(model_call_details: dict, custom_logger: CustomLogger) -> dict:
    """
    Returns a copy of model_call_details whose streaming response entries are redacted deepcopies
    when the custom logger has opted out of message logging. The shared model_call_details is left
    untouched so other callbacks still receive the unredacted response.
    """
    if not (hasattr(custom_logger, "message_logging") and custom_logger.message_logging is not True):
        return model_call_details
    redacted_entries = {
        streaming_key: _redacted_streaming_response_copy(model_call_details[streaming_key])
        for streaming_key in ("complete_streaming_response", "async_complete_streaming_response")
        if model_call_details.get(streaming_key) is not None
    }
    if not redacted_entries:
        return model_call_details
    return {**model_call_details, **redacted_entries}


def _redacted_streaming_response_copy(streaming_response):
    redacted_response = copy.deepcopy(streaming_response)
    _redact_streaming_response(redacted_response)
    return redacted_response


def _redact_streaming_response(streaming_response):
    if hasattr(streaming_response, "choices"):
        for choice in streaming_response.choices:
            _redact_choice_content(choice)
        redact_vertex_ai_metadata_from_logged_object(streaming_response)
    elif hasattr(streaming_response, "output"):
        _redact_responses_api_output(streaming_response.output)
        if hasattr(streaming_response, "reasoning") and streaming_response.reasoning is not None:
            streaming_response.reasoning = None


def _redact_choice_content(choice):
    """Helper to redact content in a choice (message or delta)."""
    if isinstance(choice, litellm.Choices):
        choice.message.content = "redacted-by-litellm"
        if hasattr(choice.message, "reasoning_content"):
            choice.message.reasoning_content = "redacted-by-litellm"
        if hasattr(choice.message, "thinking_blocks"):
            choice.message.thinking_blocks = None
        _redact_provider_specific_fields(
            getattr(choice.message, "provider_specific_fields", None)
        )
    elif isinstance(choice, litellm.utils.StreamingChoices):
        choice.delta.content = "redacted-by-litellm"
        if hasattr(choice.delta, "reasoning_content"):
            choice.delta.reasoning_content = "redacted-by-litellm"
        if hasattr(choice.delta, "thinking_blocks"):
            choice.delta.thinking_blocks = None
        _redact_provider_specific_fields(
            getattr(choice.delta, "provider_specific_fields", None)
        )


def _redact_provider_specific_fields(psf):
    """Wholesale-clear a response Message/Delta ``provider_specific_fields`` dict.

    This is a provider-NATIVE grab-bag. Beyond the reasoning duplicates
    (``reasoning_content``/``thinking_blocks``/``reasoningContentBlocks``),
    providers stash output-bearing content under ~20 other keys an allowlist can
    never enumerate: Anthropic ``citations``/``web_search_results``/
    ``tool_results``/``code_interpreter_results``/``compaction_blocks``, Bedrock
    ``citationsContent``, Gemini ``thought_signatures``/
    ``server_side_tool_invocations``, Cohere ``tool_plan``, MCP
    ``mcp_call_results``, RAG ``search_results``, and so on — the same
    unenumerable shape as the provider-native request body. Redaction only ever
    runs on the logging copy, and no consumer reads a named key off that copy
    (streaming reassembly and multi-turn replay read the untouched live response
    / request history instead), so we clear the whole dict to close the class
    instead of chasing keys one provider at a time.
    """
    if not isinstance(psf, dict):
        return
    psf.clear()


def _redact_responses_api_output(output_items):
    """Helper to redact ResponsesAPIResponse output items."""
    for output_item in output_items:
        if hasattr(output_item, "text"):
            output_item.text = "redacted-by-litellm"

        if hasattr(output_item, "content") and isinstance(output_item.content, list):
            for content_part in output_item.content:
                if hasattr(content_part, "text"):
                    content_part.text = "redacted-by-litellm"

        # Redact reasoning items in output array
        if hasattr(output_item, "type") and output_item.type == "reasoning":
            if hasattr(output_item, "summary") and isinstance(output_item.summary, list):
                for summary_item in output_item.summary:
                    if hasattr(summary_item, "text"):
                        summary_item.text = "redacted-by-litellm"


def _redact_responses_api_output_dict(output_items, redacted_str: str):
    """Helper to redact ResponsesAPIResponse output items in dict form."""
    for output_item in output_items:
        if not isinstance(output_item, dict):
            continue

        if "text" in output_item:
            output_item["text"] = redacted_str

        if isinstance(output_item.get("content"), list):
            for content_item in output_item["content"]:
                if isinstance(content_item, dict) and "text" in content_item:
                    content_item["text"] = redacted_str

        if output_item.get("type") == "reasoning" and isinstance(output_item.get("summary"), list):
            for summary_item in output_item["summary"]:
                if isinstance(summary_item, dict) and "text" in summary_item:
                    summary_item["text"] = redacted_str


def _redact_standard_logging_object(model_call_details: dict):
    """Redact messages and response inside standard_logging_object if present."""
    standard_logging_object = model_call_details.get("standard_logging_object")
    if standard_logging_object is None:
        return

    redacted_str = "redacted-by-litellm"

    if standard_logging_object.get("messages") is not None:
        standard_logging_object["messages"] = [{"role": "user", "content": redacted_str}]

    response = standard_logging_object.get("response")
    if response is not None:
        if isinstance(response, dict) and "output" in response:
            # ResponsesAPIResponse format - redact content in output items
            if isinstance(response.get("output"), list):
                _redact_responses_api_output_dict(response["output"], redacted_str)
            redact_vertex_ai_metadata_from_logged_object(response)
        elif isinstance(response, dict) and "choices" in response:
            # ModelResponse dict format - redact content in choices
            if isinstance(response.get("choices"), list):
                _redact_model_response_dict_choices(response["choices"], redacted_str)
            redact_vertex_ai_metadata_from_logged_object(response)
        elif isinstance(response, str):
            standard_logging_object["response"] = redacted_str
        else:
            # For other formats (empty dict, None, etc.), use simple text format
            standard_logging_object["response"] = {"text": redacted_str}


# Input-bearing keys that show up in the proxy_server_request.body snapshot.
# This is the LiteLLM-API (OpenAI-compat) request body, so its input keys are
# a finite, stable set; it must stay key-based (not wholesale-redacted) because
# downstream consumers read named keys off it (`user`->Lago billing,
# `tools`->spend-log index, `input`/`messages`->Responses session).
# "messages"/"prompt"/"input" are the OpenAI-style payload entry points;
# "contents" is the Gemini/Vertex native user-turn field.
# "query"/"documents" are the rerank inputs and "document" the OCR input —
# all top-level keys the proxy preserves verbatim in this snapshot.
# "system"/"system_prompt"/"instructions" are the provider-native top-level
# system-prompt fields (Anthropic `system`, Responses API `instructions`);
# "system_instruction"/"systemInstruction" are the Gemini/Vertex equivalents.
# All carry user content just as the messages do.
# (additional_args.complete_input_dict is the provider-native wire body and is
# wholesale-redacted in _redact_additional_args_complete_input_dict instead.)
def _redact_request_body_dict(body: dict):
    """Scrub the input/system-prompt keys on the proxy_server_request body dict."""
    if "messages" in body:
        body["messages"] = [{"role": "user", "content": "redacted-by-litellm"}]
    if "prompt" in body:
        body["prompt"] = ""
    if "input" in body:
        body["input"] = ""
    if "contents" in body:
        body["contents"] = [
            {"role": "user", "parts": [{"text": "redacted-by-litellm"}]}
        ]
    if "query" in body:  # rerank
        body["query"] = "redacted-by-litellm"
    if "documents" in body:  # rerank
        body["documents"] = ["redacted-by-litellm"]
    if "document" in body:  # OCR
        body["document"] = "redacted-by-litellm"
    for key in (
        "system",
        "system_prompt",
        "instructions",
        "system_instruction",
        "systemInstruction",
    ):
        if key in body:
            body[key] = "redacted-by-litellm"


def _redact_proxy_server_request_body(model_call_details: dict):
    """Redact the input-bearing keys inside the proxy's body snapshot.

    ``litellm_params["proxy_server_request"]["body"]`` is a separate copy of
    the request payload built during proxy pre-call (see
    ``litellm_pre_call_utils.add_litellm_data_to_request``). The flat-field
    overwrite in ``perform_redaction`` does not reach it, so custom-logger
    callbacks that inspect this path see the unredacted prompt.
    """
    litellm_params = model_call_details.get("litellm_params")
    if not isinstance(litellm_params, dict):
        return
    proxy_server_request = litellm_params.get("proxy_server_request")
    if not isinstance(proxy_server_request, dict):
        return
    body = proxy_server_request.get("body")
    if not isinstance(body, dict):
        return

    _redact_request_body_dict(body)


def _redact_additional_args_complete_input_dict(model_call_details: dict):
    """Redact the input-bearing keys inside additional_args.complete_input_dict.

    Provider handlers (see ``litellm/llms/<provider>/.../handler.py`` and
    ``litellm/interactions/http_handler.py``) record the provider-native
    request payload at ``additional_args["complete_input_dict"]`` so that
    pre-call logs and OTel spans can show the wire-format request. The
    flat-field overwrite in ``perform_redaction`` does not reach it, so
    custom-logger callbacks (and the OTel exporter) see the unredacted
    prompt even when message logging is disabled.
    """
    additional_args = model_call_details.get("additional_args")
    if not isinstance(additional_args, dict):
        return
    complete_input_dict = additional_args.get("complete_input_dict")
    if not isinstance(complete_input_dict, dict):
        return

    # complete_input_dict is the provider-NATIVE wire body. User input lands
    # under ~20+ provider-specific, often nested key shapes (Vertex embeddings
    # `instances`, rerank `query`/`documents`, image `prompt`, audio `file`,
    # OCR `document`, Bedrock invoke `inputText`, passthrough arbitrary bodies),
    # so a key-allowlist cannot close the class. No consumer reads a named key
    # from this dict (the OTel exporter iterates `.items()` generically; logging
    # treats it as an opaque curl-command payload), so we wholesale-redact it,
    # preserving only the non-input `model` for span/debug usefulness.
    redacted: dict = {"redacted-by-litellm": True}
    model = complete_input_dict.get("model")
    if isinstance(model, str):
        redacted["model"] = model
    additional_args["complete_input_dict"] = redacted


def _redact_model_response_dict_choices(choices, redacted_str: str):
    for choice in choices:
        if isinstance(choice, dict):
            if "message" in choice and isinstance(choice["message"], dict):
                choice["message"]["content"] = redacted_str
                if "reasoning_content" in choice["message"]:
                    choice["message"]["reasoning_content"] = redacted_str
                if "thinking_blocks" in choice["message"]:
                    choice["message"]["thinking_blocks"] = None
                if "audio" in choice["message"]:
                    choice["message"]["audio"] = None
                _redact_provider_specific_fields(
                    choice["message"].get("provider_specific_fields")
                )
            elif "delta" in choice and isinstance(choice["delta"], dict):
                choice["delta"]["content"] = redacted_str
                if "reasoning_content" in choice["delta"]:
                    choice["delta"]["reasoning_content"] = redacted_str
                if "thinking_blocks" in choice["delta"]:
                    choice["delta"]["thinking_blocks"] = None
                if "audio" in choice["delta"]:
                    choice["delta"]["audio"] = None
                _redact_provider_specific_fields(
                    choice["delta"].get("provider_specific_fields")
                )
        else:
            _redact_choice_content(choice)


def perform_redaction(model_call_details: dict, result, redact_streaming_responses: bool = True):
    """
    Performs the actual redaction on the logging object and result.

    redact_streaming_responses=False skips the in-place redaction of the shared streaming
    response entries; per-callback redaction hands each opted-out callback its own redacted
    copy via redact_streaming_responses_for_custom_logger instead.
    """
    # Redact model_call_details
    model_call_details["messages"] = [{"role": "user", "content": "redacted-by-litellm"}]
    model_call_details["prompt"] = ""
    model_call_details["input"] = ""
    _redact_standard_logging_object(model_call_details)
    _redact_proxy_server_request_body(model_call_details)
    _redact_additional_args_complete_input_dict(model_call_details)
    redact_vertex_ai_metadata_from_litellm_params(model_call_details)

    # Redact streaming response
    if redact_streaming_responses and model_call_details.get("stream", False) is True:
        for _streaming_key in ("complete_streaming_response", "async_complete_streaming_response"):
            _redact_streaming_response(model_call_details.get(_streaming_key))

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
            return {"text": "redacted-by-litellm"}

        _result = copy.deepcopy(result)
        if isinstance(_result, litellm.ModelResponse):
            if hasattr(_result, "choices") and _result.choices is not None:
                for choice in _result.choices:
                    _redact_choice_content(choice)
            redact_vertex_ai_metadata_from_logged_object(_result)
        elif isinstance(_result, dict) and "choices" in _result:
            # Handle dict representation of ModelResponse (e.g., from model_dump())
            if _result.get("choices") is not None:
                _redact_model_response_dict_choices(_result["choices"], "redacted-by-litellm")
            redact_vertex_ai_metadata_from_logged_object(_result)
        elif isinstance(_result, dict) and "output" in _result:
            if isinstance(_result.get("output"), list):
                _redact_responses_api_output_dict(_result["output"], "redacted-by-litellm")
        elif isinstance(_result, litellm.ResponsesAPIResponse):
            if hasattr(_result, "output"):
                _redact_responses_api_output(_result.output)
            # Redact reasoning field in ResponsesAPIResponse
            if hasattr(_result, "reasoning") and _result.reasoning is not None:
                _result.reasoning = None
        elif isinstance(_result, litellm.EmbeddingResponse):
            if hasattr(_result, "data") and _result.data is not None:
                _result.data = []
        else:
            return {"text": "redacted-by-litellm"}
        return _result


def should_redact_message_logging(model_call_details: dict) -> bool:
    """
    Determine if message logging should be redacted.

    Priority order:
    1. Dynamic parameter (turn_off_message_logging in request)
    2. Headers (litellm-disable-message-redaction / litellm-enable-message-redaction)
    3. Global setting (litellm.turn_off_message_logging)
    """
    litellm_params = model_call_details.get("litellm_params", {})

    metadata_field = get_metadata_variable_name_from_kwargs(litellm_params)
    metadata = litellm_params.get(metadata_field, {})
    if not isinstance(metadata, dict):
        # Fall back: litellm_metadata was None, try metadata
        metadata = litellm_params.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # Get headers from the metadata
    request_headers = metadata.get("headers", {})

    # Check for headers that explicitly control redaction
    if request_headers and bool(request_headers.get("litellm-disable-message-redaction", False)):
        # User explicitly disabled redaction via header
        return False

    possible_enable_headers = [
        "litellm-enable-message-redaction",  # old header. maintain backwards compatibility
        "x-litellm-enable-message-redaction",  # new header
    ]

    is_redaction_enabled_via_header = False
    for header in possible_enable_headers:
        if bool(request_headers.get(header, False)):
            is_redaction_enabled_via_header = True
            break

    # Priority 1: Check dynamic parameter first (if explicitly set)
    dynamic_turn_off = _get_turn_off_message_logging_from_dynamic_params(model_call_details)
    if dynamic_turn_off is not None:
        # Dynamic parameter is explicitly set, use it
        return dynamic_turn_off

    # Priority 2: Check if header explicitly enables redaction
    if is_redaction_enabled_via_header:
        return True

    # Priority 3: Fall back to global setting
    return litellm.turn_off_message_logging is True


def redact_message_input_output_from_logging(model_call_details: dict, result, input: Optional[Any] = None) -> Any:
    """
    Removes messages, prompts, input, response from logging. This modifies the data in-place
    only redacts when litellm.turn_off_message_logging == True
    """
    if should_redact_message_logging(model_call_details):
        return perform_redaction(model_call_details, result)
    return result


def _get_turn_off_message_logging_from_dynamic_params(
    model_call_details: dict,
) -> Optional[bool]:
    """
    gets the value of `turn_off_message_logging` from the dynamic params, if it exists.

    handles boolean and string values of `turn_off_message_logging`
    """
    standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = model_call_details.get(
        "standard_callback_dynamic_params", None
    )
    if standard_callback_dynamic_params:
        _turn_off_message_logging = standard_callback_dynamic_params.get("turn_off_message_logging")
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

    new_metadata = {}
    for k, v in metadata.items():
        if isinstance(k, str) and k.startswith("user_api_key"):
            pass
        else:
            new_metadata[k] = v

    return new_metadata
