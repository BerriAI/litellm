"""
OpenAI Responses API Handler for Unified Guardrails

This module provides a class-based handler for OpenAI Responses API format.
The class methods can be overridden for custom behavior.

Pattern Overview:
-----------------
1. Extract text content from input/output (both string and list formats)
2. Create async tasks to apply guardrails to each text segment
3. Track mappings to know where each response belongs
4. Apply guardrail responses back to the original structure

Responses API Format:
---------------------
Input: Union[str, List[Dict]] where each dict has:
  - role: str
  - content: Union[str, List[Dict]] (can have text items)
  - type: str (e.g., "message")

Output: response.output is List[GenericResponseOutputItem] where each has:
  - type: str (e.g., "message")
  - id: str
  - status: str
  - role: str
  - content: List[OutputText] where OutputText has:
    - type: str (e.g., "output_text")
    - text: str
"""

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Union, cast

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.tool_param import FunctionToolParam
from pydantic import BaseModel, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
    OpenAiResponsesToChatCompletionStreamIterator,
)
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.llms.base_llm.guardrail_translation.utils import (
    blocked_responses_stream_usage,
    stream_item_field,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    BaseLiteLLMOpenAIResponseObject,
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
    ContentPartAddedEvent,
    ContentPartDoneEvent,
    ContentPartDonePartOutputText,
    ErrorEvent,
    ErrorEventError,
    OpenAIMcpServerTool,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    OutputTextDeltaEvent,
    OutputTextDoneEvent,
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    OutputFunctionToolCall,
    OutputText,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from fastapi import HTTPException

    from litellm.integrations.custom_guardrail import (
        CustomGuardrail,
        ModifyResponseException,
    )
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import ResponseInputParam


class ResponseOutputEnvelope(TypedDict, total=False):
    """Dict form of a Responses API response, as far as guardrail write-back reads it."""

    output: ReadOnly[Sequence[object]]
    model: ReadOnly[str | None]


class ResponsesStreamChunk(TypedDict, total=False):
    """Responses API streaming event, as far as the accumulated-stream helpers read it."""

    type: ReadOnly[str]
    text: ReadOnly[str]
    delta: ReadOnly[str]
    item_id: ReadOnly[str]
    output_index: ReadOnly[int]
    content_index: ReadOnly[int]


_PATCHABLE_ITEM_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {"function_call_output": "output", "message": "content"}
)

_EMPTY_RESPONSES_REQUEST: Final[ResponsesAPIOptionalRequestParams] = {}


def _item_rewrite_field(item: Mapping[str, object]) -> str | None:
    item_type: Final = item.get("type")
    if item_type is None:
        return "content" if "content" in item else None
    if not isinstance(item_type, str):
        return None
    return _PATCHABLE_ITEM_FIELDS.get(item_type)


def _rewritten_input_item(item: Mapping[str, object], rewritten: object) -> Mapping[str, object] | None:
    field: Final = _item_rewrite_field(item)
    if field is None or not isinstance(rewritten, Mapping):
        return None
    rewritten_content: Final = rewritten.get("content")
    if isinstance(item.get(field), str) and isinstance(rewritten_content, str):
        return {**item, field: rewritten_content}  # mutable-ok: request input items must stay JSON-plain dicts
    rewritten_row: Final = cast("AllMessageValues", rewritten)  # cast-ok: guardrails hand back chat-shaped rows
    converted_items, _ = LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(
        [rewritten_row]  # mutable-ok: converter signature takes a list
    )
    if len(converted_items) != 1 or not isinstance(converted_items[0], Mapping):
        return None
    first_converted: Final = cast("Mapping[str, object]", converted_items[0])  # cast-ok: isinstance-checked above
    converted_value: Final = first_converted.get(field)
    if converted_value is None:
        return None
    return {**item, field: converted_value}  # mutable-ok: request input items must stay JSON-plain dicts


def _is_function_call_item(item: object) -> bool:
    return isinstance(item, Mapping) and item.get("type") in ("function_call", "custom_tool_call")


def _last_message_role(messages: Sequence[object]) -> str | None:
    if not messages:
        return None
    last: Final = messages[-1]
    role: Final = last.get("role") if isinstance(last, Mapping) else getattr(last, "role", None)
    return role if isinstance(role, str) else None


def _provenance_unit_bounds(
    raw_input: Sequence[object],
    solo_conversions: Sequence[Sequence[object]],
) -> tuple[tuple[int, int], ...]:
    trailing_roles: Final = tuple(
        accumulate(
            (_last_message_role(messages) for messages in solo_conversions),
            lambda previous, current: current if current is not None else previous,
        )
    )
    start_indexes: Final = tuple(
        index
        for index in range(len(raw_input))
        if index == 0 or not (_is_function_call_item(raw_input[index]) and trailing_roles[index - 1] == "assistant")
    )
    return tuple(zip(start_indexes, (*start_indexes[1:], len(raw_input))))


def _input_item_provenance(
    raw_input: Sequence[object],
    expected_messages: Sequence[object],
) -> tuple[Mapping[int, int], frozenset[int]] | None:
    if not all(isinstance(item, Mapping) for item in raw_input):
        return None
    solo_conversions: Final = tuple(
        LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=cast("ResponseInputParam", [item]),  # cast-ok: items checked as Mappings above
            responses_api_request=_EMPTY_RESPONSES_REQUEST,
        )
        for item in raw_input
    )
    full_conversion: Final = tuple(
        LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=cast("ResponseInputParam", list(raw_input)),  # cast-ok: items checked as Mappings above
            responses_api_request=_EMPTY_RESPONSES_REQUEST,
        )
    )
    if full_conversion != tuple(expected_messages):
        return None
    units: Final = _provenance_unit_bounds(raw_input, solo_conversions)
    unit_messages: Final = tuple(
        tuple(solo_conversions[start])
        if end - start == 1
        else tuple(
            LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=cast("ResponseInputParam", list(raw_input[start:end])),  # cast-ok: checked as Mappings above
                responses_api_request=_EMPTY_RESPONSES_REQUEST,
            )
        )
        for start, end in units
    )
    if tuple(message for messages in unit_messages for message in messages) != full_conversion:
        return None
    boundaries: Final = tuple(accumulate((len(messages) for messages in unit_messages), initial=0))
    item_for_message: Final = MappingProxyType(
        {
            message_index: start
            for unit_index, (start, end) in enumerate(units)
            if end - start == 1
            for message_index in range(boundaries[unit_index], boundaries[unit_index + 1])
        }
    )
    tainted: Final = frozenset(
        message_index
        for unit_index, (start, end) in enumerate(units)
        if end - start > 1
        for message_index in range(boundaries[unit_index], boundaries[unit_index + 1])
    )
    return item_for_message, tainted


class _RequestFields(NamedTuple):
    input: tuple[object, ...]
    instructions: str | None


class _ExtractedInputs(NamedTuple):
    inputs: GenericGuardrailAPIInputs
    task_mappings: tuple[tuple[int, int | None], ...]


def _patched_request_fields(
    raw_input: object,
    instructions: object,
    original_messages: Sequence[object],
    structured_messages: Sequence[object],
) -> _RequestFields | None:
    if not isinstance(raw_input, list) or len(original_messages) != len(structured_messages):
        return None
    offset: Final = 1 if instructions else 0
    provenance: Final = _input_item_provenance(raw_input, tuple(original_messages)[offset:])
    if provenance is None:
        return None
    item_for_message, tainted = provenance
    changed: Final = tuple(
        (index, rewritten)
        for index, (original, rewritten) in enumerate(zip(original_messages, structured_messages))
        if original != rewritten
    )
    instruction_rewrites: Final = tuple(rewritten for index, rewritten in changed if index < offset)
    rewritten_instructions: Final = (
        instruction_rewrites[0].get("content")
        if instruction_rewrites and isinstance(instruction_rewrites[0], Mapping)
        else instructions
    )
    instructions_value: Final = rewritten_instructions if isinstance(rewritten_instructions, str) else None
    if rewritten_instructions is not None and instructions_value is None:
        return None
    body_changes: Final = tuple((index - offset, rewritten) for index, rewritten in changed if index >= offset)
    if any(message_index in tainted or message_index not in item_for_message for message_index, _ in body_changes):
        return None
    replacements: Final = MappingProxyType(
        {
            item_for_message[message_index]: _rewritten_input_item(
                cast("Mapping[str, object]", raw_input[item_for_message[message_index]]),  # cast-ok: checked Mappings
                rewritten,
            )
            for message_index, rewritten in body_changes
        }
    )
    if len(replacements) != len(body_changes) or any(item is None for item in replacements.values()):
        return None
    return _RequestFields(
        input=tuple(replacements.get(index, item) for index, item in enumerate(raw_input)),
        instructions=instructions_value,
    )


def _patch_or_convert_request_fields(
    raw_input: object,
    instructions: object,
    original_messages: Sequence[object],
    structured_messages: Sequence[AllMessageValues],
) -> _RequestFields | None:
    if not isinstance(structured_messages, list):
        return None
    patched: Final = _patched_request_fields(raw_input, instructions, original_messages, structured_messages)
    if patched is not None:
        return patched
    input_items, converted_instructions = (
        LiteLLMResponsesTransformationHandler().convert_chat_completion_messages_to_responses_api(structured_messages)
    )
    return _RequestFields(input=tuple(input_items), instructions=converted_instructions)


def _next_stream_sequence_number(responses_so_far: Sequence[Any] | None) -> int:
    sequence_numbers: Final = (
        item.get("sequence_number") if isinstance(item, dict) else getattr(item, "sequence_number", None)
        for item in reversed(responses_so_far or ())
    )
    return next((n + 1 for n in sequence_numbers if isinstance(n, int)), 0)


class OpenAIResponsesHandler(BaseTranslation):
    """
    Handler for processing OpenAI Responses API with guardrails.

    This class provides methods to:
    1. Process input (pre-call hook)
    2. Process output response (post-call hook)

    Methods can be overridden to customize behavior for different message formats.
    """

    def get_structured_messages(self, data: dict) -> list[AllMessageValues] | None:
        """
        Convert Responses API request data to OpenAI-spec structured messages.

        Transforms `input` (string or ResponseInputParam) and optional
        `instructions` into chat completion messages.
        """
        input_data: Final = data.get("input")
        if input_data is None:
            return None
        messages: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=input_data,
            responses_api_request=data,
        )
        return cast(list[AllMessageValues], messages) if messages else None

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> dict[str, object]:
        """
        Process input by applying guardrails to text content.

        Handles both string input and list of message objects.
        """
        input_data: Final[str | ResponseInputParam | None] = data.get("input")
        if not isinstance(input_data, (str, list)):
            return data
        structured_messages: Final = self.get_structured_messages(data)
        extracted: Final = self._extract_guardrail_inputs(data, input_data)
        if not extracted.inputs.get("texts"):
            return data
        if structured_messages:
            extracted.inputs["structured_messages"] = structured_messages
        original_tools: Final[list[dict[str, object]]] = list(data.get("tools") or [])
        guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
            inputs=extracted.inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        self._apply_guardrailed_tools_to_data(data, original_tools, guardrailed_inputs.get("tools"))
        written_back: Final = self._written_back_request_fields(data, structured_messages, guardrailed_inputs)
        if written_back is not None:
            data["input"] = list(written_back.input)  # mutable-ok: JSON body
            if written_back.instructions is None:
                data.pop("instructions", None)
            else:
                data["instructions"] = written_back.instructions  # rebind-ok: data is an out-param
        elif isinstance(input_data, str):
            guardrailed_texts: Final = guardrailed_inputs.get("texts") or ()
            data["input"] = guardrailed_texts[0] if guardrailed_texts else input_data  # rebind-ok: data is an out-param
        else:
            await self._apply_guardrail_responses_to_input(
                messages=input_data,
                responses=guardrailed_inputs.get("texts") or (),
                task_mappings=extracted.task_mappings,
            )
        verbose_proxy_logger.debug("OpenAI Responses API: Processed input messages: %s", data.get("input"))
        return data

    def _extract_guardrail_inputs(
        self,
        data: Mapping[str, object],
        input_data: "str | ResponseInputParam",
    ) -> _ExtractedInputs:
        texts_to_check: Final[list[str]] = []
        images_to_check: Final[list[str]] = []
        task_mappings: Final[list[tuple[int, int | None]]] = []
        tools_to_check: Final[list[ChatCompletionToolParam]] = []
        if isinstance(input_data, str):
            texts_to_check.append(input_data)
        else:
            for msg_idx, message in enumerate(input_data):
                self._extract_input_text_and_images(
                    message=message,
                    msg_idx=msg_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=images_to_check,
                    task_mappings=task_mappings,
                )
        tools: Final = data.get("tools")
        if tools:
            self._extract_and_transform_tools(
                cast("list[FunctionToolParam | OpenAIMcpServerTool]", tools),  # cast-ok: request body tools
                tools_to_check,
            )
        inputs: Final = GenericGuardrailAPIInputs(texts=texts_to_check)
        if images_to_check:
            inputs["images"] = images_to_check
        if tools_to_check:
            inputs["tools"] = tools_to_check
        model: Final = data.get("model")
        if isinstance(model, str):
            inputs["model"] = model
        return _ExtractedInputs(inputs=inputs, task_mappings=tuple(task_mappings))

    @staticmethod
    def _written_back_request_fields(
        data: Mapping[str, object],
        structured_messages: Sequence[AllMessageValues] | None,
        guardrailed_inputs: GenericGuardrailAPIInputs,
    ) -> _RequestFields | None:
        guardrailed: Final = guardrailed_inputs.get("structured_messages")
        if guardrailed is None or guardrailed is structured_messages:
            return None
        return _patch_or_convert_request_fields(
            data.get("input"),
            data.get("instructions"),
            structured_messages or (),
            guardrailed,
        )

    def extract_request_tool_names(self, data: dict) -> list[str]:
        """Extract tool names from Responses API request (tools[].name for function
        and custom, tools[].server_label for mcp)."""
        names: Final[list[str]] = []
        for tool in data.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") in ("function", "custom") and tool.get("name"):
                names.append(str(tool["name"]))
            elif tool.get("type") == "mcp" and tool.get("server_label"):
                names.append(str(tool["server_label"]))
        return names

    def _extract_and_transform_tools(
        self,
        tools: list[FunctionToolParam | OpenAIMcpServerTool],
        tools_to_check: list[ChatCompletionToolParam],
    ) -> None:
        """
        Extract and transform tools from Responses API format to Chat Completion format.

        Uses the LiteLLM transformation function to convert Responses API tools
        to Chat Completion tools that can be passed to guardrails.
        """
        if tools is not None and isinstance(tools, list):
            # Transform Responses API tools to Chat Completion tools
            (
                transformed_tools,
                _,
            ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(tools)
            tools_to_check.extend(cast(list[ChatCompletionToolParam], transformed_tools))

    def _remap_tools_to_responses_api_format(self, guardrailed_tools: list[Any]) -> list[dict[str, object]]:
        """
        Remap guardrail-returned tools (Chat Completion format) back to
        Responses API request tool format.
        """
        return LiteLLMCompletionResponsesConfig.transform_chat_completion_tool_params_to_responses_api_tools(
            guardrailed_tools
        )

    def _merge_tools_after_guardrail(
        self,
        original_tools: list[dict[str, object]],
        remapped: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """
        Merge remapped guardrailed tools with original tools that were not sent
        to the guardrail (e.g. web_search, web_search_preview), preserving order.
        Tools a guardrail appended (``remapped`` longer than ``original_tools``)
        have no original slot and are kept so an injected tool is not dropped.
        """
        if not original_tools:
            return remapped
        result: Final[list[dict[str, object]]] = []
        j = 0
        for tool in original_tools:
            if isinstance(tool, dict) and tool.get("type") in (
                "web_search",
                "web_search_preview",
            ):
                result.append(tool)
            else:
                if j < len(remapped):
                    result.append(remapped[j])
                    j += 1
        # Keep guardrail-appended tools that matched no original slot above.
        result.extend(remapped[j:])
        return result

    def _apply_guardrailed_tools_to_data(
        self,
        data: dict,
        original_tools: list[dict[str, object]],
        guardrailed_tools: list[ChatCompletionToolParam] | None,
    ) -> None:
        """Remap guardrailed tools to Responses API format and merge with original, then set data['tools']."""
        if guardrailed_tools is not None:
            remapped: Final = self._remap_tools_to_responses_api_format(guardrailed_tools)
            data["tools"] = self._merge_tools_after_guardrail(original_tools, remapped)

    def _extract_input_text_and_images(
        self,
        message: Any,
        msg_idx: int,
        texts_to_check: list[str],
        images_to_check: list[str],
        task_mappings: list[tuple[int, int | None]],
    ) -> None:
        """
        Extract text content and images from an input message.

        Override this method to customize text/image extraction logic.
        """
        content: Final = message.get("content", None)
        if content is None:
            return

        if isinstance(content, str):
            # Simple string content
            texts_to_check.append(content)
            task_mappings.append((msg_idx, None))

        elif isinstance(content, list):
            # List content (e.g., multimodal with text and images)
            for content_idx, content_item in enumerate(content):
                if isinstance(content_item, dict):
                    # Extract text
                    text_str = content_item.get("text", None)
                    if text_str is not None:
                        texts_to_check.append(text_str)
                        task_mappings.append((msg_idx, int(content_idx)))

                    # Extract images
                    if content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url")
                            if url:
                                images_to_check.append(url)

    async def _apply_guardrail_responses_to_input(
        self,
        messages: Any,  # Can be List[Dict[str, Any]] or ResponseInputParam
        responses: Sequence[str],
        task_mappings: Sequence[tuple[int, int | None]],
    ) -> None:
        """
        Apply guardrail responses back to input messages.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            msg_idx = cast(int, mapping[0])
            content_idx_optional = cast(int | None, mapping[1])

            content = messages[msg_idx].get("content", None)
            if content is None:
                continue

            if isinstance(content, str) and content_idx_optional is None:
                # Replace string content with guardrail response
                messages[msg_idx]["content"] = guardrail_response

            elif isinstance(content, list) and content_idx_optional is not None:
                # Replace specific text item in list content
                if isinstance(messages[msg_idx]["content"][content_idx_optional], dict):
                    messages[msg_idx]["content"][content_idx_optional]["text"] = guardrail_response

    async def process_output_response(
        self,
        response: Union["ResponsesAPIResponse", ResponseOutputEnvelope],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict | None = None,
    ) -> Union["ResponsesAPIResponse", ResponseOutputEnvelope]:
        """
        Process output response by applying guardrails to text content and tool calls.

        Args:
            response: LiteLLM ResponsesAPIResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - response.output is a list of output items
            - Each output item can be:
              * GenericResponseOutputItem with a content list of OutputText objects
              * ResponseFunctionToolCall with tool call data
            - Each OutputText object has a text field
        """

        texts_to_check: Final[list[str]] = []
        images_to_check: Final[list[str]] = []
        tool_calls_to_check: Final[list[ChatCompletionToolCallChunk]] = []
        task_mappings: Final[list[tuple[int, int]]] = []
        # Track (output_item_index, content_index) for each text

        # Handle both dict and Pydantic object responses
        response_output: Sequence[object]
        if isinstance(response, dict):
            response_output = response.get("output", [])
        elif hasattr(response, "output"):
            response_output = response.output or []
        else:
            verbose_proxy_logger.debug("OpenAI Responses API: No output found in response")
            return response

        if not response_output:
            verbose_proxy_logger.debug("OpenAI Responses API: Empty output in response")
            return response

        # Step 1: Extract all text content and tool calls from response output
        for output_idx, output_item in enumerate(response_output):
            self._extract_output_text_and_images(
                output_item=output_item,
                output_idx=output_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                task_mappings=task_mappings,
                tool_calls_to_check=tool_calls_to_check,
            )

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check or tool_calls_to_check:
            # Use the real request_data if provided (proxy path), otherwise
            # create a standalone dict (SDK / direct-call path).
            if request_data is None:
                request_data = {"response": response}
            else:
                if "response" not in request_data:
                    request_data["response"] = response

            # Add user API key metadata with prefixed keys
            if "litellm_metadata" not in request_data:
                user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
                if user_metadata:
                    request_data["litellm_metadata"] = user_metadata

            inputs: Final = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if tool_calls_to_check:
                inputs["tool_calls"] = tool_calls_to_check
            # Include model information from the response if available
            response_model: str | None = None
            if isinstance(response, dict):
                response_model = response.get("model")
            elif hasattr(response, "model"):
                response_model = getattr(response, "model", None)
            if response_model:
                inputs["model"] = response_model

            guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts: Final = guardrailed_inputs.get("texts", [])

            # Step 3: Map guardrail responses back to original response structure
            await self._apply_guardrail_responses_to_output(
                response=response,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug("OpenAI Responses API: Processed output response: %s", response)

        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: list[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: "LiteLLMLoggingObj | None" = None,
        user_api_key_dict: "UserAPIKeyAuth | None" = None,
        request_data: dict | None = None,
    ) -> list[Any]:
        """
        Process output streaming response by applying guardrails to text content.

        Mirrors the Chat Completions handler pattern: extract text from the final
        chunk, apply the guardrail, then write the result back in-place so the
        caller sees the modified content (e.g. PII tokens replaced).

        For ``response.completed`` events (the normal end-of-stream signal) we
        use the same per-item extraction + task-mapping approach as
        ``process_output_response`` so that unmasking / blocking works correctly
        for every output item.
        """
        if not responses_so_far:
            return responses_so_far

        final_chunk: Final = responses_so_far[-1]
        # Accept both plain dicts and Pydantic models (BaseLiteLLMOpenAIResponseObject
        # exposes a .get() shim, so all the .get() calls below work for both).
        if not (isinstance(final_chunk, dict) or hasattr(final_chunk, "get")):
            return responses_so_far

        # ------------------------------------------------------------------ #
        # Case 1: response.completed — full response is available in the      #
        # final chunk; iterate output items, apply guardrail, write back.     #
        # ------------------------------------------------------------------ #
        if final_chunk.get("type") == "response.completed":
            response_obj: Final[ResponseOutputEnvelope] = final_chunk.get("response") or {}
            if not hasattr(response_obj, "get"):
                return responses_so_far
            outputs: Final[Sequence[object]] = response_obj.get("output") or []

            texts_to_check: Final[list[str]] = []
            tool_calls_to_check: Final[list[ChatCompletionToolCallChunk]] = []
            task_mappings: Final[list[tuple[int, int]]] = []

            for output_idx, output_item in enumerate(outputs):
                self._extract_output_text_and_images(
                    output_item=output_item,
                    output_idx=output_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=[],
                    task_mappings=task_mappings,
                    tool_calls_to_check=tool_calls_to_check,
                )

            if texts_to_check or tool_calls_to_check:
                if request_data is None:
                    request_data = {}
                if "response" not in request_data:
                    request_data["response"] = response_obj
                if "litellm_metadata" not in request_data:
                    user_metadata: Final = self.transform_user_api_key_dict_to_metadata(user_api_key_dict)
                    if user_metadata:
                        request_data["litellm_metadata"] = user_metadata

                inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
                if tool_calls_to_check:
                    inputs["tool_calls"] = cast(list[ChatCompletionToolCallChunk], tool_calls_to_check)
                response_model = response_obj.get("model")
                if response_model:
                    inputs["model"] = response_model

                guardrailed_inputs: Final = await guardrail_to_apply.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data,
                    input_type="response",
                    logging_obj=litellm_logging_obj,
                )

                guardrailed_texts: Final = guardrailed_inputs.get("texts", [])

                # Write guardrailed texts back into the output items in-place.
                # final_chunk is a reference into responses_so_far so this
                # mutates the list that the caller holds.
                await self._apply_guardrail_responses_to_output(
                    response=response_obj,
                    responses=guardrailed_texts,
                    task_mappings=task_mappings,
                )

            return responses_so_far

        # ------------------------------------------------------------------ #
        # Case 2: response.output_item.done — extract tool calls only.        #
        # ------------------------------------------------------------------ #
        if final_chunk.get("type") == "response.output_item.done":
            model_response_stream: Final = (
                OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(final_chunk)
            )
            tool_calls: Final = model_response_stream.choices[0].delta.tool_calls
            if tool_calls:
                inputs = GenericGuardrailAPIInputs()
                inputs["tool_calls"] = cast(list[ChatCompletionToolCallChunk], tool_calls)
                if hasattr(model_response_stream, "model") and model_response_stream.model:
                    inputs["model"] = model_response_stream.model
                await guardrail_to_apply.apply_guardrail(
                    inputs=inputs,
                    request_data=request_data if request_data is not None else {},
                    input_type="response",
                    logging_obj=litellm_logging_obj,
                )
            return responses_so_far

        # ------------------------------------------------------------------ #
        # Fallback: apply guardrail to the accumulated text string.           #
        # No structured write-back is possible here; guardrails that only     #
        # need to block/flag (not rewrite) still work correctly.             #
        # ------------------------------------------------------------------ #
        string_so_far: Final = self.get_streaming_string_so_far(responses_so_far)
        if string_so_far:
            fallback_inputs: Final = GenericGuardrailAPIInputs(texts=[string_so_far])
            response_model = (
                final_chunk.get("response", {}).get("model") if isinstance(final_chunk.get("response"), dict) else None
            )
            if response_model:
                fallback_inputs["model"] = response_model
            await guardrail_to_apply.apply_guardrail(
                inputs=fallback_inputs,
                request_data=request_data if request_data is not None else {},
                input_type="response",
                logging_obj=litellm_logging_obj,
            )
        return responses_so_far

    def _check_streaming_has_ended(self, responses_so_far: Sequence[ResponsesStreamChunk]) -> bool:
        """
        Check if the streaming has ended.
        """
        if not responses_so_far:
            return False
        terminal_types: Final = {
            ResponsesAPIStreamEvents.RESPONSE_COMPLETED.value,
            ResponsesAPIStreamEvents.RESPONSE_FAILED.value,
            ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE.value,
        }
        return responses_so_far[-1].get("type") in terminal_types

    def build_stream_error_items(
        self,
        exc: "HTTPException",
        responses_so_far: Sequence[Any] | None = None,
    ) -> Sequence[Any] | None:
        from litellm.proxy.common_request_processing import (
            serialize_http_exception_detail,
        )

        message, _ = serialize_http_exception_detail(exc.detail)
        return (
            ErrorEvent(
                type=ResponsesAPIStreamEvents.ERROR,
                sequence_number=_next_stream_sequence_number(responses_so_far),
                error=ErrorEventError(
                    type="guardrail_error",
                    code=str(exc.status_code),
                    message=message,
                    param=None,
                ),
            ),
        )

    def get_streaming_string_so_far(self, responses_so_far: Sequence[ResponsesStreamChunk]) -> str:
        """
        Get the string so far from the responses so far.

        ``response.output_text.done`` events carry the whole part in ``text``, while
        ``response.output_text.delta`` events carry fragments in ``delta``. A stream
        that dies before its done event (``response.failed`` / ``response.incomplete``)
        has text only in deltas, so per content part the done text wins when present
        and the joined deltas fill in otherwise, never both.
        """
        keyed_events: Final = tuple(
            (
                (event.get("item_id"), event.get("output_index"), event.get("content_index")),
                event.get("text"),
                event.get("delta"),
            )
            for event in responses_so_far
            if isinstance(event.get("text"), str) or isinstance(event.get("delta"), str)
        )

        def part_text(part_key: tuple[object, object, object]) -> str:
            done_texts: Final = tuple(
                text for key, text, _ in keyed_events if key == part_key and isinstance(text, str)
            )
            if done_texts:
                return done_texts[-1]
            return "".join(delta for key, _, delta in keyed_events if key == part_key and isinstance(delta, str))

        return "".join(part_text(key) for key in dict.fromkeys(key for key, _, _ in keyed_events))

    def _has_text_content(self, response: "ResponsesAPIResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        if not hasattr(response, "output") or response.output is None:
            return False

        for output_item in response.output:
            if isinstance(output_item, BaseModel):
                try:
                    generic_response_output_item = GenericResponseOutputItem.model_validate(output_item.model_dump())
                    if generic_response_output_item.content:
                        output_item = generic_response_output_item
                except Exception:
                    continue
            if isinstance(output_item, (GenericResponseOutputItem, dict)):
                content = (
                    output_item.content
                    if isinstance(output_item, GenericResponseOutputItem)
                    else output_item.get("content", [])
                )
                if content:
                    for content_item in content:
                        # Check if it's an OutputText with text
                        if isinstance(content_item, OutputText):
                            if content_item.text:
                                return True
                        elif isinstance(content_item, dict):
                            if content_item.get("text"):
                                return True
        return False

    def _extract_output_text_and_images(
        self,
        output_item: object,
        output_idx: int,
        texts_to_check: list[str],
        images_to_check: list[str],
        task_mappings: list[tuple[int, int]],
        tool_calls_to_check: list[ChatCompletionToolCallChunk] | None = None,
    ) -> None:
        """
        Extract text content, images, and tool calls from a response output item.

        Override this method to customize text/image/tool extraction logic.
        """

        # Check if this is a tool call (OutputFunctionToolCall)
        if isinstance(output_item, OutputFunctionToolCall) or (
            isinstance(output_item, BaseModel)
            and hasattr(output_item, "type")
            and getattr(output_item, "type") == "function_call"
        ):
            if tool_calls_to_check is not None:
                tool_call_dict = (
                    LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
                        tool_call_item=output_item,
                        index=output_idx,
                    )
                )
                tool_calls_to_check.append(cast(ChatCompletionToolCallChunk, tool_call_dict))
            return
        elif isinstance(output_item, dict) and output_item.get("type") == "function_call":
            # Handle dict representation of tool call
            if tool_calls_to_check is not None:
                # Convert dict to ResponseFunctionToolCall for processing
                try:
                    tool_call_obj: Final = ResponseFunctionToolCall(**output_item)
                    tool_call_dict = LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
                        tool_call_item=tool_call_obj,
                        index=output_idx,
                    )
                    tool_calls_to_check.append(cast(ChatCompletionToolCallChunk, tool_call_dict))
                except Exception:
                    pass
            return

        # Handle both GenericResponseOutputItem and dict
        content: list[OutputText] | list[dict] | None = None
        if isinstance(output_item, BaseModel):
            try:
                output_item_dump: Final = output_item.model_dump()
                generic_response_output_item: Final = GenericResponseOutputItem.model_validate(output_item_dump)
                if generic_response_output_item.content:
                    content = generic_response_output_item.content
            except Exception:
                # Try to extract content directly from output_item if validation fails
                if hasattr(output_item, "content") and output_item.content:
                    content = output_item.content
                else:
                    return
        elif isinstance(output_item, dict):
            content = output_item.get("content", [])
        else:
            return

        if not content:
            return

        verbose_proxy_logger.debug("OpenAI Responses API: Processing output item: %s", output_item)

        # Iterate through content items (list of OutputText objects)
        for content_idx, content_item in enumerate(content):
            # Handle both OutputText objects and dicts
            if isinstance(content_item, OutputText):
                text_content = content_item.text
            elif isinstance(content_item, dict):
                text_content = content_item.get("text")
            else:
                continue

            if text_content:
                texts_to_check.append(text_content)
                task_mappings.append((output_idx, int(content_idx)))

    async def _apply_guardrail_responses_to_output(
        self,
        response: Union["ResponsesAPIResponse", ResponseOutputEnvelope],
        responses: list[str],
        task_mappings: list[tuple[int, int]],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Override this method to customize how responses are applied.
        """
        # Handle both dict and Pydantic object responses
        if isinstance(response, dict):
            response_output = response.get("output", [])
        elif hasattr(response, "output"):
            response_output = response.output or []
        else:
            return

        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            output_idx = cast(int, mapping[0])
            content_idx = cast(int, mapping[1])

            if output_idx >= len(response_output):
                continue

            output_item = response_output[output_idx]

            # Handle both GenericResponseOutputItem, BaseModel, and dict
            if isinstance(output_item, GenericResponseOutputItem):
                if output_item.content and content_idx < len(output_item.content):
                    content_item = output_item.content[content_idx]
                    if isinstance(content_item, OutputText):
                        content_item.text = guardrail_response
                    elif isinstance(content_item, dict):
                        content_item["text"] = guardrail_response
            elif isinstance(output_item, BaseModel):
                # Handle other Pydantic models by converting to GenericResponseOutputItem
                try:
                    generic_item = GenericResponseOutputItem.model_validate(output_item.model_dump())
                    if generic_item.content and content_idx < len(generic_item.content):
                        content_item = generic_item.content[content_idx]
                        if isinstance(content_item, OutputText):
                            content_item.text = guardrail_response
                            # Update the original response output
                            if hasattr(output_item, "content") and output_item.content:
                                original_content = output_item.content[content_idx]
                                if hasattr(original_content, "text"):
                                    original_content.text = guardrail_response
                except Exception:
                    pass
            elif isinstance(output_item, dict):
                content = output_item.get("content", [])
                if content and content_idx < len(content):
                    if isinstance(content[content_idx], dict):
                        content[content_idx]["text"] = guardrail_response
                    elif hasattr(content[content_idx], "text"):
                        content[content_idx].text = guardrail_response

    def build_block_sse_chunks(
        self,
        exc: "ModifyResponseException",
        stream_started: bool = False,
        responses_so_far: Sequence[object] | None = None,
    ) -> Sequence[bytes]:
        """
        Build Responses API SSE events that deliver the guardrail block message
        and terminate the stream cleanly, mirroring the non-streaming block
        response: a completed response whose only output is the violation text,
        with the real usage the upstream call consumed.

        - ``stream_started`` False (buffered / pre-stream): nothing has been
          sent, so emit the full synthetic sequence (``response.created``
          through ``response.completed``).
        - ``stream_started`` True (sampling / mid-stream): events already
          reached the client, so continue the in-progress response: close the
          output item still open on the wire, deliver the block message as a
          new output item under the same response id, and close with a
          ``response.completed`` carrying only the replacement item.

        The proxy's data generator appends ``data: [DONE]`` itself.
        """
        events: Final = (
            self._block_continuation_events(exc, responses_so_far or ())
            if stream_started
            else self._standalone_block_events(exc)
        )
        return tuple(
            f"data: {event.model_dump_json(exclude_none=True, exclude_unset=True, serialize_as_any=True)}\n\n".encode()
            for event in events
        )

    @staticmethod
    def _standalone_block_events(exc: "ModifyResponseException") -> Sequence[ResponsesAPIStreamingResponse]:
        from litellm.responses.streaming_iterator import build_synthetic_response_events

        return build_synthetic_response_events(
            transformed=_blocked_response(exc, response_id=f"resp_{uuid.uuid4()}", model=exc.model),
            logging_obj=None,
            chunk_size=max(len(exc.message), 1),
        )

    @staticmethod
    def _block_continuation_events(
        exc: "ModifyResponseException", responses_so_far: Sequence[object]
    ) -> Sequence[ResponsesAPIStreamingResponse]:
        response_id, model, output_index = _continuation_identity(exc, responses_so_far)
        item: Final = _blocked_output_item(exc)
        item_id: Final = item.id
        part: Final[_BlockedContentPart] = {"type": "output_text", "text": exc.message, "annotations": ()}
        done_part: Final[_BlockedDoneContentPart] = {
            "type": "output_text",
            "text": exc.message,
            "annotations": (),
            "logprobs": None,
        }
        return (
            *_open_item_closing_events(responses_so_far),
            OutputItemAddedEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=output_index,
                item=item,
            ),
            ContentPartAddedEvent(
                type=ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                part=BaseLiteLLMOpenAIResponseObject.model_validate(part),
            ),
            OutputTextDeltaEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                delta=exc.message,
            ),
            OutputTextDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                text=exc.message,
            ),
            ContentPartDoneEvent(
                type=ResponsesAPIStreamEvents.CONTENT_PART_DONE,
                item_id=item_id,
                output_index=output_index,
                content_index=0,
                part=ContentPartDonePartOutputText.model_validate(done_part),
            ),
            OutputItemDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                output_index=output_index,
                item=item,
            ),
            ResponseCompletedEvent(
                type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
                response=_blocked_response(exc, response_id=response_id, model=model, output_item=item),
            ),
        )


class _BlockedContentPart(TypedDict):
    type: ReadOnly[str]
    text: ReadOnly[str]
    annotations: ReadOnly[tuple[object, ...]]


class _BlockedDoneContentPart(TypedDict):
    type: ReadOnly[str]
    text: ReadOnly[str]
    annotations: ReadOnly[tuple[object, ...]]
    logprobs: ReadOnly[None]


class _BlockedItemPayload(TypedDict):
    type: ReadOnly[str]
    id: ReadOnly[str]
    status: ReadOnly[str]
    role: ReadOnly[str]
    content: ReadOnly[tuple[_BlockedContentPart, ...]]


class _BlockedResponsePayload(TypedDict):
    id: ReadOnly[str]
    object: ReadOnly[str]
    created_at: ReadOnly[int]
    model: ReadOnly[str]
    output: ReadOnly[tuple[GenericResponseOutputItem, ...]]
    status: ReadOnly[str]
    usage: ReadOnly[ResponseAPIUsage]


def _blocked_output_item(exc: "ModifyResponseException") -> GenericResponseOutputItem:
    payload: Final[_BlockedItemPayload] = {
        "type": "message",
        "id": f"msg_{uuid.uuid4()}",
        "status": "completed",
        "role": "assistant",
        "content": ({"type": "output_text", "text": exc.message, "annotations": ()},),
    }
    return GenericResponseOutputItem.model_validate(payload)


def _blocked_response(
    exc: "ModifyResponseException",
    response_id: str,
    model: str,
    output_item: GenericResponseOutputItem | None = None,
) -> ResponsesAPIResponse:
    payload: Final[_BlockedResponsePayload] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "output": (output_item if output_item is not None else _blocked_output_item(exc),),
        "status": "completed",
        "usage": blocked_responses_stream_usage(exc.original_response),
    }
    return ResponsesAPIResponse.model_validate(payload)


def _continuation_identity(exc: "ModifyResponseException", responses_so_far: Sequence[object]) -> tuple[str, str, int]:
    responses: Final = tuple(
        response for item in responses_so_far if (response := stream_item_field(item, "response")) is not None
    )
    response_id: Final = next(
        (rid for response in responses if isinstance(rid := stream_item_field(response, "id"), str) and rid),
        f"resp_{uuid.uuid4()}",
    )
    model: Final = next(
        (m for response in responses if isinstance(m := stream_item_field(response, "model"), str) and m),
        exc.model,
    )
    indices: Final = tuple(
        index for item in responses_so_far if isinstance(index := stream_item_field(item, "output_index"), int)
    )
    return response_id, model, max(indices) + 1 if indices else 0


@dataclass(frozen=True, slots=True)
class _OpenItemState:
    item_id: str
    item_type: str
    role: str
    output_index: int
    content_index: int
    text: str
    part_open: bool
    payload: object


def _open_item_state(responses_so_far: Sequence[object]) -> _OpenItemState | None:
    typed: Final = tuple((stream_item_field(event, "type"), event) for event in responses_so_far)
    added: Final = tuple(
        (added_index, stream_item_field(event, "item"))
        for event_type, event in typed
        if event_type == "response.output_item.added"
        and isinstance(added_index := stream_item_field(event, "output_index"), int)
    )
    done_indices: Final = frozenset(
        done_index
        for event_type, event in typed
        if event_type == "response.output_item.done"
        and isinstance(done_index := stream_item_field(event, "output_index"), int)
    )
    open_added: Final = tuple((index, payload) for index, payload in added if index not in done_indices)
    if not open_added:
        return None
    output_index, item_payload = open_added[-1]
    if item_payload is None:
        return None
    item_id: Final = stream_item_field(item_payload, "id")
    if not isinstance(item_id, str) or not item_id:
        return None
    raw_type: Final = stream_item_field(item_payload, "type")
    raw_role: Final = stream_item_field(item_payload, "role")
    part_added: Final = tuple(
        part_index
        for event_type, event in typed
        if event_type == "response.content_part.added"
        and stream_item_field(event, "item_id") == item_id
        and isinstance(part_index := stream_item_field(event, "content_index"), int)
    )
    part_done: Final = frozenset(
        part_done_index
        for event_type, event in typed
        if event_type == "response.content_part.done"
        and stream_item_field(event, "item_id") == item_id
        and isinstance(part_done_index := stream_item_field(event, "content_index"), int)
    )
    open_parts: Final = tuple(index for index in part_added if index not in part_done)
    text: Final = "".join(
        delta
        for event_type, event in typed
        if event_type == "response.output_text.delta"
        and stream_item_field(event, "item_id") == item_id
        and isinstance(delta := stream_item_field(event, "delta"), str)
    )
    return _OpenItemState(
        item_id=item_id,
        item_type=raw_type if isinstance(raw_type, str) and raw_type else "message",
        role=raw_role if isinstance(raw_role, str) and raw_role else "assistant",
        output_index=output_index,
        content_index=open_parts[-1] if open_parts else 0,
        text=text,
        part_open=bool(open_parts),
        payload=item_payload,
    )


_item_fields_adapter: Final = TypeAdapter(Mapping[str, object])
_no_item_fields: Final[Mapping[str, object]] = MappingProxyType({})


def _incomplete_item_fields(payload: object) -> Mapping[str, object]:
    raw: Final = payload.model_dump() if isinstance(payload, BaseModel) else payload
    if not isinstance(raw, dict):
        return _no_item_fields
    return _item_fields_adapter.validate_python(raw)


def _open_item_closing_events(responses_so_far: Sequence[object]) -> Sequence[ResponsesAPIStreamingResponse]:
    """Close the output item still in progress on the relayed stream before the
    block item is appended: strict Responses clients reject a
    ``response.completed`` that arrives while an earlier ``output_item.added``
    was never closed. A message item closes ``completed`` with exactly the text
    the client has received so far; any other item type (a function call the
    guardrail rejected, for instance) closes ``incomplete`` so the synthetic
    done event can never authorize acting on it."""
    open_item: Final = _open_item_state(responses_so_far)
    if open_item is None:
        return ()
    if open_item.item_type != "message":
        return (
            OutputItemDoneEvent(
                type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                output_index=open_item.output_index,
                item=BaseLiteLLMOpenAIResponseObject.model_validate(
                    MappingProxyType({**_incomplete_item_fields(open_item.payload), "status": "incomplete"})
                ),
            ),
        )
    partial_part: Final[_BlockedContentPart] = {
        "type": "output_text",
        "text": open_item.text,
        "annotations": (),
    }
    closed_payload: Final[_BlockedItemPayload] = {
        "type": open_item.item_type,
        "id": open_item.item_id,
        "status": "completed",
        "role": open_item.role,
        "content": (partial_part,),
    }
    item_done: Final = OutputItemDoneEvent(
        type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        output_index=open_item.output_index,
        item=GenericResponseOutputItem.model_validate(closed_payload),
    )
    if not open_item.part_open:
        return (item_done,)
    partial_done_part: Final[_BlockedDoneContentPart] = {
        "type": "output_text",
        "text": open_item.text,
        "annotations": (),
        "logprobs": None,
    }
    return (
        OutputTextDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
            item_id=open_item.item_id,
            output_index=open_item.output_index,
            content_index=open_item.content_index,
            text=open_item.text,
        ),
        ContentPartDoneEvent(
            type=ResponsesAPIStreamEvents.CONTENT_PART_DONE,
            item_id=open_item.item_id,
            output_index=open_item.output_index,
            content_index=open_item.content_index,
            part=ContentPartDonePartOutputText.model_validate(partial_done_part),
        ),
        item_done,
    )
