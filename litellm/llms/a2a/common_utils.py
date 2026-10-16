"""
Common utilities for A2A (Agent-to-Agent) Protocol
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.azure_ai.common_utils import has_azure_entra_params, resolve_azure_ai_agent_auth_header
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues


@with_config(ConfigDict(extra="allow", strict=True))
class _A2AMessage(TypedDict):
    parts: ReadOnly[NotRequired[Sequence[object]]]


@with_config(ConfigDict(extra="allow", strict=True))
class _A2AMessagePart(TypedDict):
    kind: ReadOnly[NotRequired[object]]
    text: ReadOnly[NotRequired[object]]
    parts: ReadOnly[NotRequired[Sequence[object]]]


@with_config(ConfigDict(extra="allow", strict=True))
class _A2AResponseResult(TypedDict):
    parts: ReadOnly[NotRequired[Sequence[object]]]
    message: ReadOnly[NotRequired[object]]
    artifact: ReadOnly[NotRequired[object]]
    status: ReadOnly[NotRequired[object]]
    artifacts: ReadOnly[NotRequired[Sequence[object]]]


@with_config(ConfigDict(extra="allow", strict=True))
class _A2AStatus(TypedDict):
    message: ReadOnly[NotRequired[object]]


_A2A_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(strict=True))
_A2A_TEXT_ADAPTER: Final = TypeAdapter(str, config=ConfigDict(strict=True))
_A2A_MESSAGE_ADAPTER: Final = TypeAdapter(_A2AMessage)
_A2A_MESSAGE_PART_ADAPTER: Final = TypeAdapter(_A2AMessagePart)
_A2A_RESPONSE_RESULT_ADAPTER: Final = TypeAdapter(_A2AResponseResult)
_A2A_STATUS_ADAPTER: Final = TypeAdapter(_A2AStatus)


class A2AError(BaseLLMException):
    """Base exception for A2A protocol errors"""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict[str, str] = {},
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


def convert_messages_to_prompt(messages: list[AllMessageValues]) -> str:
    """
    Convert OpenAI messages to a single prompt string for A2A agent.

    Formats each message as "{role}: {content}" and joins with newlines
    to preserve conversation history. Handles both string and list content.

    Args:
        messages: List of OpenAI-format messages

    Returns:
        Formatted prompt string with full conversation context
    """
    conversation_parts: Final = []
    for msg in messages:
        # Use LiteLLM's helper to extract text from content (handles both str and list)
        content_text = convert_content_list_to_str(message=msg)

        # Get role
        if isinstance(msg, BaseModel):
            role = _A2A_MAPPING_ADAPTER.validate_python(msg.model_dump()).get("role", "user")
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
        else:
            role = dict(msg).get("role", "user")

        if content_text:
            conversation_parts.append(f"{role}: {content_text}")

    return "\n".join(conversation_parts)


def extract_text_from_a2a_message(message: Mapping[str, object] | None, depth: int = 0, max_depth: int = 10) -> str:
    """
    Extract text content from A2A message parts.

    Args:
        message: A2A message dict with 'parts' containing text parts
        depth: Current recursion depth (internal use)
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Concatenated text from all text parts
    """
    if message is None or depth >= max_depth:
        return ""

    typed_message: Final = _A2A_MESSAGE_ADAPTER.validate_python(message)
    parts: Final = typed_message.get("parts", [])
    typed_parts: Final = tuple(_A2A_MESSAGE_PART_ADAPTER.validate_python(raw_part) for raw_part in parts)
    text_parts: Final[list[str]] = []

    for part in typed_parts:
        if part.get("kind") == "text":
            text_parts.append(_A2A_TEXT_ADAPTER.validate_python(part.get("text", "")))
        # Handle nested parts if they exist
        elif "parts" in part:
            nested_text = extract_text_from_a2a_message(part, depth + 1, max_depth)
            if nested_text:
                text_parts.append(nested_text)

    return " ".join(text_parts)


def extract_text_from_a2a_response(response_dict: Mapping[str, object], max_depth: int = 10) -> str:
    """
    Extract text content from A2A response result.

    Args:
        response_dict: A2A response dict with 'result' containing message
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Text from response message parts
    """
    result: Final = response_dict.get("result", {})
    if not isinstance(result, dict):
        return ""
    typed_result: Final = _A2A_RESPONSE_RESULT_ADAPTER.validate_python(result)

    # A2A response can have different formats:
    # 1. Direct message: {"result": {"kind": "message", "parts": [...]}}
    # 2. Nested message: {"result": {"message": {"parts": [...]}}}
    # 3. Task with artifacts: {"result": {"kind": "task", "artifacts": [{"parts": [...]}]}}
    # 4. Task with status message: {"result": {"kind": "task", "status": {"message": {"parts": [...]}}}}
    # 5. Streaming artifact-update: {"result": {"kind": "artifact-update", "artifact": {"parts": [...]}}}

    # Check if result itself has parts (direct message)
    if "parts" in typed_result:
        return extract_text_from_a2a_message(typed_result, depth=0, max_depth=max_depth)

    # Check for nested message
    message: Final = typed_result.get("message")
    if message:
        return extract_text_from_a2a_message(
            _A2A_MESSAGE_ADAPTER.validate_python(message), depth=0, max_depth=max_depth
        )

    # Check for streaming artifact-update (singular artifact)
    artifact: Final = typed_result.get("artifact")
    if artifact and isinstance(artifact, dict):
        return extract_text_from_a2a_message(
            _A2A_MESSAGE_ADAPTER.validate_python(artifact), depth=0, max_depth=max_depth
        )

    # Check for task status message (common in Gemini A2A agents)
    status: Final = typed_result.get("status", {})
    if isinstance(status, dict):
        typed_status: Final = _A2A_STATUS_ADAPTER.validate_python(status)
        status_message: Final = typed_status.get("message")
        if status_message:
            return extract_text_from_a2a_message(
                _A2A_MESSAGE_ADAPTER.validate_python(status_message), depth=0, max_depth=max_depth
            )

    # Handle task result with artifacts (plural, array)
    artifacts: Final = typed_result.get("artifacts", [])
    if artifacts:
        first_artifact: Final = _A2A_MESSAGE_ADAPTER.validate_python(artifacts[0])
        return extract_text_from_a2a_message(first_artifact, depth=0, max_depth=max_depth)

    return ""


AgentAuthHeaderResolver = Callable[[Mapping[str, object]], Awaitable[Mapping[str, str]]]


def a2a_hop_uses_entra(litellm_params: Mapping[str, object], custom_llm_provider: object) -> bool:
    return not custom_llm_provider and has_azure_entra_params(litellm_params)


async def resolve_a2a_hop_auth_header(
    litellm_params: Mapping[str, object],
    custom_llm_provider: object,
    resolve_entra_header: AgentAuthHeaderResolver = resolve_azure_ai_agent_auth_header,
) -> Mapping[str, str] | None:
    """Entra credentials authenticate the A2A hop only; a completion-bridge agent hands them to the model provider it bridges to."""
    if not a2a_hop_uses_entra(litellm_params, custom_llm_provider):
        return None
    return await resolve_entra_header(litellm_params)
