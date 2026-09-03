"""
Transformation utilities for A2A <-> OpenAI message format conversion.

A2A Message Format:
{
    "role": "user",
    "parts": [{"kind": "text", "text": "Hello!"}],
    "messageId": "abc123"
}

OpenAI Message Format:
{"role": "user", "content": "Hello!"}

A2A Streaming Events:
- Task event (kind: "task") - Initial task creation with status "submitted"
- Status update (kind: "status-update") - Status changes (working, completed)
- Artifact update (kind: "artifact-update") - Content/artifact delivery
"""

from collections.abc import Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_STR_KEY_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def _as_object_mapping(value: object) -> Mapping[str, object]:
    try:
        return _STR_KEY_MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return {}


class A2AStreamingContext:
    """
    Context holder for A2A streaming state.
    Tracks task_id, context_id, and message accumulation.
    """

    def __init__(self, request_id: str, input_message: Mapping[str, JsonValue]):
        self.request_id = request_id
        self.task_id = str(uuid4())
        self.context_id = str(uuid4())
        self.input_message = input_message
        self.accumulated_text = ""
        self.has_emitted_task = False
        self.has_emitted_working = False


class A2ACompletionBridgeTransformation:
    """
    Static methods for transforming between A2A and OpenAI message formats.
    """

    @staticmethod
    def _text_from_a2a_part(part: JsonValue) -> str | None:
        if not isinstance(part, dict):
            return None
        text: Final = part.get("text")
        if text is None:
            return None
        if part.get("kind") not in (None, "", "text"):
            return None
        return str(text)

    @staticmethod
    def _extract_text_from_a2a_parts(parts: Sequence[JsonValue]) -> str:
        """Extract text from A2A parts (with or without explicit ``kind``)."""
        extracted: Final = (A2ACompletionBridgeTransformation._text_from_a2a_part(part) for part in parts)
        return "\n".join(text for text in extracted if text is not None)

    @staticmethod
    def get_forward_metadata(
        a2a_message: Mapping[str, JsonValue],
        params: Mapping[str, JsonValue] | None = None,
    ) -> Mapping[str, JsonValue] | None:
        """
        Merge A2A metadata from MessageSendParams and the message for downstream providers.

        Forwarded once on the LangGraph run payload (``metadata``), not duplicated on
        each input message — see ``apply_forward_metadata_to_completion_params``.
        """
        params_metadata: Final = params.get("metadata") if params else None
        message_metadata: Final = a2a_message.get("metadata")
        merged: Final[dict[str, JsonValue]] = {
            **(params_metadata if isinstance(params_metadata, dict) else {}),
            **(message_metadata if isinstance(message_metadata, dict) else {}),
        }
        return merged or None

    @staticmethod
    def apply_forward_metadata_to_completion_params(
        completion_params: MutableMapping[str, object],
        a2a_message: Mapping[str, JsonValue],
        params: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """
        Attach A2A metadata to completion kwargs for provider bridges (e.g. LangGraph).

        Uses ``extra_body`` so we do not collide with LiteLLM's spend-log ``metadata`` kwarg.
        """
        forward_metadata: Final = A2ACompletionBridgeTransformation.get_forward_metadata(
            a2a_message=a2a_message,
            params=params,
        )
        if not forward_metadata:
            return

        extra_body: Final = _as_object_mapping(completion_params.get("extra_body"))
        # Layer client-supplied A2A metadata under any agent-owner-configured
        # ``extra_body.metadata`` so the configured keys remain authoritative
        # and an A2A caller cannot overwrite server-set run metadata.
        existing_dict: Final = _as_object_mapping(extra_body.get("metadata"))
        merged_metadata: Final[dict[str, object]] = {**forward_metadata, **existing_dict}
        completion_params["extra_body"] = {**extra_body, "metadata": merged_metadata}

        verbose_logger.debug("A2A -> completion forward metadata keys=%s", list(forward_metadata.keys()))

    @staticmethod
    def a2a_message_to_openai_messages(
        a2a_message: Mapping[str, JsonValue],
    ) -> list[dict[str, object]]:
        """
        Transform an A2A message to OpenAI message format.

        Args:
            a2a_message: A2A message with role, parts, and messageId

        Returns:
            List of OpenAI-format messages
        """
        role: Final = a2a_message.get("role", "user")
        raw_parts: Final = a2a_message.get("parts", [])

        # Map A2A roles to OpenAI roles
        openai_role: Final = (
            "user" if role == "user" else "assistant" if role == "assistant" else "system" if role == "system" else role
        )
        parts: Final = raw_parts if isinstance(raw_parts, list) else []

        content: Final = A2ACompletionBridgeTransformation._extract_text_from_a2a_parts(parts)

        # Do not attach A2A message.metadata here — the completion bridge forwards it
        # once at run level via extra_body.metadata (LangGraph POST /runs/wait shape).
        openai_message: Final[dict[str, object]] = {"role": openai_role, "content": content}

        verbose_logger.debug(
            "A2A -> OpenAI transform: role=%s -> %s, content_length=%s", role, openai_role, len(content)
        )

        return [openai_message]

    @staticmethod
    def _extract_response_content(response: "ModelResponse | CustomStreamWrapper") -> str:
        if not isinstance(response, ModelResponse) or not response.choices:
            return ""
        choice: Final = response.choices[0]
        if not choice.message:
            return ""
        return choice.message.content or ""

    @staticmethod
    def openai_response_to_a2a_response(
        response: "ModelResponse | CustomStreamWrapper",
        request_id: str | None = None,
    ) -> dict[str, object]:
        """
        Transform a LiteLLM ModelResponse to A2A SendMessageResponse format.

        Args:
            response: LiteLLM ModelResponse object
            request_id: Original A2A request ID

        Returns:
            A2A SendMessageResponse dict
        """
        content: Final = A2ACompletionBridgeTransformation._extract_response_content(response)

        # Build A2A message
        a2a_message: Final = {
            "kind": "message",
            "role": "agent",
            "parts": [{"kind": "text", "text": content}],
            "messageId": uuid4().hex,
        }

        # Build A2A response
        a2a_response: Final[dict[str, object]] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": a2a_message,
        }

        verbose_logger.debug("OpenAI -> A2A transform: content_length=%s", len(content))

        return a2a_response

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format with timezone."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def create_task_event(
        ctx: A2AStreamingContext,
    ) -> dict[str, object]:
        """
        Create the initial task event with status 'submitted'.

        This is the first event emitted in an A2A streaming response.
        """
        return {
            "id": ctx.request_id,
            "jsonrpc": "2.0",
            "result": {
                "contextId": ctx.context_id,
                "history": [
                    {
                        "contextId": ctx.context_id,
                        "kind": "message",
                        "messageId": ctx.input_message.get("messageId", uuid4().hex),
                        "parts": ctx.input_message.get("parts", []),
                        "role": ctx.input_message.get("role", "user"),
                        "taskId": ctx.task_id,
                    }
                ],
                "id": ctx.task_id,
                "kind": "task",
                "status": {
                    "state": "submitted",
                },
            },
        }

    @staticmethod
    def create_status_update_event(
        ctx: A2AStreamingContext,
        state: str,
        final: bool = False,
        message_text: str | None = None,
    ) -> dict[str, object]:
        """
        Create a status update event.

        Args:
            ctx: Streaming context
            state: Status state ('working', 'completed')
            final: Whether this is the final event
            message_text: Optional message text for 'working' status
        """
        status: Final[dict[str, object]] = {
            "state": state,
            "timestamp": A2ACompletionBridgeTransformation._get_timestamp(),
        }

        # Add message for 'working' status
        if state == "working" and message_text:
            status["message"] = {
                "contextId": ctx.context_id,
                "kind": "message",
                "messageId": str(uuid4()),
                "parts": [{"kind": "text", "text": message_text}],
                "role": "agent",
                "taskId": ctx.task_id,
            }

        return {
            "id": ctx.request_id,
            "jsonrpc": "2.0",
            "result": {
                "contextId": ctx.context_id,
                "final": final,
                "kind": "status-update",
                "status": status,
                "taskId": ctx.task_id,
            },
        }

    @staticmethod
    def create_artifact_update_event(
        ctx: A2AStreamingContext,
        text: str,
    ) -> dict[str, object]:
        """
        Create an artifact update event with content.

        Args:
            ctx: Streaming context
            text: The text content for the artifact
        """
        return {
            "id": ctx.request_id,
            "jsonrpc": "2.0",
            "result": {
                "artifact": {
                    "artifactId": str(uuid4()),
                    "name": "response",
                    "parts": [{"kind": "text", "text": text}],
                },
                "contextId": ctx.context_id,
                "kind": "artifact-update",
                "taskId": ctx.task_id,
            },
        }
