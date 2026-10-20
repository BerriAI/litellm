"""
OpenAI Responses API token counting transformation logic.

This module handles the transformation of requests to OpenAI's /v1/responses/input_tokens endpoint.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.llms.anthropic.pass_through.messages.utils import normalize_native_messages
from litellm.llms.anthropic.pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.anthropic import AnthropicMessagesRequest, AnthropicOutputSchema


class ResponsesInputTextPart(TypedDict):
    type: ReadOnly[Literal["input_text"]]
    text: ReadOnly[str]


class ResponsesInputImagePart(TypedDict):
    type: ReadOnly[Literal["input_image"]]
    image_url: ReadOnly[str]
    detail: ReadOnly[str]


class ResponsesInputFilePart(TypedDict):
    type: ReadOnly[Literal["input_file"]]
    filename: ReadOnly[str]
    file_data: ReadOnly[str]


ResponsesInputPart = ResponsesInputTextPart | ResponsesInputImagePart | ResponsesInputFilePart

ResponsesContentRole = Literal["user", "assistant"]

ResponsesCountRequestParts: TypeAlias = tuple[
    tuple[Mapping[str, object], ...],
    str | None,
    tuple[Mapping[str, object], ...] | None,
    Mapping[str, object] | None,
]
_RESPONSES_COUNT_REQUEST_PARTS: Final = TypeAdapter(ResponsesCountRequestParts)
_TRANSLATED_REQUEST: Final = TypeAdapter(Mapping[str, object])


class _AnthropicCountMessage(TypedDict):
    role: ReadOnly[str]


_ANTHROPIC_COUNT_MESSAGES: Final = TypeAdapter(tuple[_AnthropicCountMessage, ...])
_ANTHROPIC_OUTPUT_SCHEMA: Final = TypeAdapter(AnthropicOutputSchema)


def _anthropic_output_schema(output_format: JsonValue) -> AnthropicOutputSchema | None:
    try:
        return _ANTHROPIC_OUTPUT_SCHEMA.validate_python(output_format)
    except ValidationError:
        return None


def _chat_image_block_to_responses_part(image_url: object) -> ResponsesInputImagePart | None:
    url: Final = image_url.get("url") if isinstance(image_url, Mapping) else image_url
    if not isinstance(url, str) or not url:
        return None
    detail: Final = image_url.get("detail") if isinstance(image_url, Mapping) else None
    part: Final[ResponsesInputImagePart] = {
        "type": "input_image",
        "image_url": url,
        "detail": detail if isinstance(detail, str) and detail else "auto",
    }
    return part


def _chat_file_block_to_responses_part(file_value: object) -> ResponsesInputFilePart | None:
    """Only an inline file round trips: OpenAI rejects `file_data` without the `filename` beside it."""
    if not isinstance(file_value, Mapping):
        return None
    filename: Final = file_value.get("filename")
    file_data: Final = file_value.get("file_data")
    if not isinstance(filename, str) or not filename or not isinstance(file_data, str) or not file_data:
        return None
    part: Final[ResponsesInputFilePart] = {
        "type": "input_file",
        "filename": filename,
        "file_data": file_data,
    }
    return part


def _chat_block_to_responses_part(block: object, role: ResponsesContentRole) -> ResponsesInputPart | None:
    if isinstance(block, str):
        bare: Final[ResponsesInputTextPart] = {"type": "input_text", "text": block}
        return bare
    if not isinstance(block, Mapping):
        return None
    match block.get("type"):
        case "text":
            text_value: Final = block.get("text")
            text: Final[ResponsesInputTextPart] = {
                "type": "input_text",
                "text": text_value if isinstance(text_value, str) else "",
            }
            return text
        case "image_url" if role == "user":
            return _chat_image_block_to_responses_part(block.get("image_url"))
        case "file" if role == "user":
            return _chat_file_block_to_responses_part(block.get("file"))
        case _:
            return None


def chat_content_blocks_to_responses_content(
    content: Sequence[object],
    role: ResponsesContentRole,
) -> str | tuple[ResponsesInputPart, ...]:
    """Text-only content collapses to a joined string, which every role accepts and counts identically.

    Only a user turn may carry an image or file part: the Responses API rejects any part but
    output_text and refusal inside an assistant turn.
    """
    parts: Final = tuple(
        part for part in (_chat_block_to_responses_part(block, role) for block in content) if part is not None
    )
    if any(part["type"] != "input_text" for part in parts):
        return parts
    return "\n".join(part["text"] for part in parts if part["type"] == "input_text")


class OpenAICountTokensConfig:
    """
    Configuration and transformation logic for OpenAI Responses API token counting.

    OpenAI Responses API Token Counting Specification:
    - Endpoint: POST https://api.openai.com/v1/responses/input_tokens
    - Response: {"input_tokens": <number>}
    """

    def get_openai_count_tokens_endpoint(self, api_base: str | None = None) -> str:
        base = api_base or "https://api.openai.com/v1"
        base = base.rstrip("/")
        return f"{base}/responses/input_tokens"

    def transform_request_to_count_tokens(
        self,
        model: str,
        input: str | Sequence[object],
        tools: Sequence[Mapping[str, object]] | None = None,
        instructions: str | None = None,
        text: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """
        Transform request to OpenAI Responses API token counting format.

        The Responses API uses `input` (not `messages`) and `instructions` (not `system`).
        """
        request: Final[dict[str, object]] = {
            "model": model,
            "input": input,
        }

        if instructions is not None:
            request["instructions"] = instructions

        if tools is not None:
            request["tools"] = self._transform_tools_for_responses_api([dict(tool) for tool in tools])

        if text is not None:
            request["text"] = text

        return request

    def get_required_headers(self, api_key: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def validate_request(self, model: str, input: str | Sequence[object]) -> None:
        if not model:
            raise ValueError("model parameter is required")

        if not input:
            raise ValueError("input parameter is required")

    @staticmethod
    def _transform_tools_for_responses_api(
        tools: list[dict[str, Any]],
    ) -> list[dict[str, object]]:
        """
        Transform OpenAI chat tools format to Responses API tools format.

        Chat format:  {"type": "function", "function": {"name": "...", "parameters": {...}}}
        Responses format: {"type": "function", "name": "...", "parameters": {...}}
        """
        transformed: Final[list[dict[str, object]]] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                item: dict[str, object] = {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
                if "strict" in func:
                    item["strict"] = func["strict"]
                transformed.append(item)
            else:
                # Pass through non-function tools (e.g., web_search, file_search)
                transformed.append(tool)
        return transformed

    @staticmethod
    def messages_to_responses_input(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, object]], str | None]:
        """
        Convert standard chat messages format to OpenAI Responses API input format.

        Returns:
            (input_items, instructions) tuple where instructions is extracted
            from system/developer messages.
        """
        input_items: Final[list[dict[str, object]]] = []
        instructions_parts: Final[list[str]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            if role in ("system", "developer"):
                # Extract system/developer messages as instructions
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    # Handle content blocks - extract text
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    instructions_parts.append("\n".join(text_parts))
            elif role == "user":
                if isinstance(content, list):
                    content = chat_content_blocks_to_responses_content(content, "user")
                input_items.append({"role": "user", "content": content})
            elif role == "assistant":
                # Map tool_calls to Responses API function_call items
                tool_calls = msg.get("tool_calls")
                if isinstance(content, list):
                    content = chat_content_blocks_to_responses_content(content, "assistant")
                if content:
                    input_items.append({"role": "assistant", "content": content})
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "arguments": func.get("arguments", ""),
                            }
                        )
                elif not content:
                    input_items.append({"role": "assistant", "content": content})
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": content if isinstance(content, str) else str(content),
                    }
                )

        instructions: Final = "\n".join(instructions_parts) if instructions_parts else None
        return input_items, instructions

    @staticmethod
    def malformed_anthropic_messages_error(messages: Sequence[Mapping[str, object]]) -> str | None:
        try:
            _ANTHROPIC_COUNT_MESSAGES.validate_python(messages)
        except ValidationError as e:
            return "; ".join(
                f"messages.{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in e.errors()
            )
        return None

    @staticmethod
    def anthropic_messages_to_responses_input(
        model: str,
        messages: Sequence[Mapping[str, object]],
        system: object,
        tools: Sequence[Mapping[str, object]] | None,
        output_format: JsonValue = None,
    ) -> ResponsesCountRequestParts:
        """Convert with the adapter /v1/messages inference uses, so the count sees the same input it bills."""
        anthropic_request: Final = AnthropicMessagesRequest(
            model=model,
            messages=normalize_native_messages(messages),
            system=system if isinstance(system, (str, list)) else None,
            tools=[dict(tool) for tool in tools] if tools else None,
            output_format=_anthropic_output_schema(output_format),
        )
        translated: Final = _TRANSLATED_REQUEST.validate_python(
            LiteLLMAnthropicToResponsesAPIAdapter().translate_request(
                anthropic_request, include_encrypted_reasoning=False
            )
        )
        return OpenAICountTokensConfig.count_request_parts(
            ResponsesAPIRequestUtils._restore_encrypted_content_item_ids_in_input(translated.get("input")),  # pyright: ignore[reportPrivateUsage]  # the input restore inference runs
            translated.get("instructions"),
            translated.get("tools"),
            translated.get("text"),
        )

    @staticmethod
    def count_request_parts(
        input_items: object, instructions: object, tools: object, text: object = None
    ) -> ResponsesCountRequestParts:
        return _RESPONSES_COUNT_REQUEST_PARTS.validate_python((input_items, instructions, tools, text))
