"""
Utility functions for A2A protocol.
"""

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, has_attributes

if TYPE_CHECKING:
    from a2a.types import SendMessageRequest, SendStreamingMessageRequest

_OBJECT_ITEMS: Final = TypeAdapter(tuple[object, ...])


class _SupportsModelDump(Protocol):
    def model_dump(self, *, mode: str) -> object: ...


class A2ARequestUtils:
    """Utility class for A2A request/response processing."""

    @staticmethod
    def extract_text_from_message(message: object) -> str:
        """
        Extract text content from A2A message parts.

        Args:
            message: A2A message dict or object with 'parts' containing text parts

        Returns:
            Concatenated text from all text parts
        """
        if message is None:
            return ""
        # a2a-sdk Part RootModels keep kind/text on part.root, so read pydantic messages as dicts
        if has_attributes(message, _SupportsModelDump, "model_dump"):
            return A2ARequestUtils.extract_text_from_message(message.model_dump(mode="json"))

        # Handle both dict and object access
        if isinstance(message, dict):
            parts = message.get("parts", [])
        else:
            parts = getattr(message, "parts", []) or []

        text_parts: Final[list[str]] = []
        for part in parts:
            if isinstance(part, dict):
                if part.get("kind") == "text":
                    text_parts.append(part.get("text", ""))
            else:
                if getattr(part, "kind", None) == "text":
                    text_parts.append(getattr(part, "text", ""))

        return " ".join(text_parts)

    @staticmethod
    def extract_text_from_response(response_dict: Mapping[str, object]) -> str:
        """
        Extract text content from A2A response result.

        Args:
            response_dict: A2A response dict with 'result' containing message

        Returns:
            Text from response message parts
        """
        result: Final = as_str_mapping(response_dict.get("result"))
        if result is None:
            return ""

        # Direct message format (A2A spec): detect by explicit kind tag only.
        # The "parts" heuristic is too broad and would match any future result
        # type that happens to include a "parts" field.
        if result.get("kind") == "message":
            return A2ARequestUtils.extract_text_from_message(result)
        if result.get("kind") == "task":
            return A2ARequestUtils.extract_text_from_task(result)

        message: Final = result.get("message", {})
        return A2ARequestUtils.extract_text_from_message(message)

    @staticmethod
    def extract_text_from_task(task: Mapping[str, object]) -> str:
        artifacts: Final = task.get("artifacts")
        artifact_items: Final = _OBJECT_ITEMS.validate_python(artifacts) if isinstance(artifacts, list) else ()
        artifact_texts: Final = tuple(
            A2ARequestUtils.extract_text_from_message(artifact) for artifact in artifact_items
        )
        if any(artifact_texts):
            return " ".join(text for text in artifact_texts if text)
        status: Final = as_str_mapping(task.get("status"))
        return A2ARequestUtils.extract_text_from_message(status.get("message")) if status is not None else ""

    @staticmethod
    def extract_text_from_stream_event(event: Mapping[str, object]) -> str:
        result: Final = as_str_mapping(event.get("result"))
        if result is None:
            return ""
        if result.get("kind") == "artifact-update":
            return A2ARequestUtils.extract_text_from_message(result.get("artifact"))
        if result.get("kind") == "status-update":
            status: Final = as_str_mapping(result.get("status"))
            return A2ARequestUtils.extract_text_from_message(status.get("message")) if status is not None else ""
        return A2ARequestUtils.extract_text_from_response(event)

    @staticmethod
    def get_input_message_from_request(
        request: "SendMessageRequest | SendStreamingMessageRequest",
    ) -> object:
        """
        Extract the input message from an A2A request.

        Args:
            request: The A2A SendMessageRequest or SendStreamingMessageRequest

        Returns:
            The message object/dict or None
        """
        params: Final[object] = getattr(request, "params", None)
        if params is None:
            return None
        return getattr(params, "message", None)

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count tokens in text using litellm.token_counter.

        Args:
            text: Text to count tokens for

        Returns:
            Token count, or 0 if counting fails
        """
        if not text:
            return 0
        try:
            return litellm.token_counter(text=text)
        except Exception:
            verbose_logger.debug("Failed to count tokens")
            return 0

    @staticmethod
    def calculate_usage_from_request_response(
        request: "SendMessageRequest | SendStreamingMessageRequest",
        response_dict: Mapping[str, object],
    ) -> tuple[int, int, int]:
        """
        Calculate token usage from A2A request and response.

        Args:
            request: The A2A SendMessageRequest or SendStreamingMessageRequest
            response_dict: The A2A response as a dict

        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens)
        """
        input_message: Final = A2ARequestUtils.get_input_message_from_request(request)
        prompt_tokens: Final = A2ARequestUtils.count_tokens(A2ARequestUtils.extract_text_from_message(input_message))

        # Count output tokens
        output_text: Final = A2ARequestUtils.extract_text_from_response(response_dict)
        completion_tokens: Final = A2ARequestUtils.count_tokens(output_text)

        total_tokens: Final = prompt_tokens + completion_tokens

        return prompt_tokens, completion_tokens, total_tokens


def get_session_id_from_a2a_params(params: Mapping[str, object]) -> str | None:
    message: Final = params.get("message", {})
    if isinstance(message, dict):
        return message.get("contextId")
    return getattr(message, "contextId", None)


def scope_session_to_principal(session_id: str, principal: str | None) -> str:
    """
    Bind a client-supplied A2A contextId to the authenticated principal.

    Without this, two distinct keys authorized for the same agent could set the
    same contextId and read/append to each other's backend memory. The
    principal is hashed (it is already a hashed token) so the raw value is never
    sent to the agent backend, while the original contextId is kept as a suffix
    for operator-side correlation.
    """
    if not principal:
        return session_id
    principal_prefix: Final = hashlib.sha256(principal.encode("utf-8")).hexdigest()[:16]
    return f"{principal_prefix}-{session_id}"


# Backwards compatibility aliases
def extract_text_from_a2a_message(message: object) -> str:
    return A2ARequestUtils.extract_text_from_message(message)


def extract_text_from_a2a_response(response_dict: Mapping[str, object]) -> str:
    return A2ARequestUtils.extract_text_from_response(response_dict)
