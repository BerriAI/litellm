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

from datetime import datetime, timezone
from typing import Any, Final
from uuid import uuid4

from litellm._logging import verbose_logger


class A2AStreamingContext:
    """
    Context holder for A2A streaming state.
    Tracks task_id, context_id, and message accumulation.
    """

    def __init__(self, request_id: str, input_message: dict[str, Any]):
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
    def _extract_text_from_a2a_parts(parts: list[dict[str, Any]]) -> str:
        """Extract text from A2A parts (with or without explicit ``kind``)."""
        content_parts: Final[list[str]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = part.get("kind")
            text = part.get("text")
            if text is None:
                continue
            if kind in (None, "", "text"):
                content_parts.append(str(text))
        return "\n".join(content_parts)

    @staticmethod
    def get_forward_metadata(
        a2a_message: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Merge A2A metadata from MessageSendParams and the message for downstream providers.

        Forwarded once on the LangGraph run payload (``metadata``), not duplicated on
        each input message — see ``apply_forward_metadata_to_completion_params``.
        """
        merged: Final[dict[str, Any]] = {}
        if params and isinstance(params.get("metadata"), dict):
            merged.update(params["metadata"])
        message_metadata: Final = a2a_message.get("metadata")
        if isinstance(message_metadata, dict):
            merged.update(message_metadata)
        return merged or None

    @staticmethod
    def apply_forward_metadata_to_completion_params(
        completion_params: dict[str, Any],
        a2a_message: dict[str, Any],
        params: dict[str, Any] | None = None,
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

        extra_body = completion_params.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        # Layer client-supplied A2A metadata under any agent-owner-configured
        # ``extra_body.metadata`` so the configured keys remain authoritative
        # and an A2A caller cannot overwrite server-set run metadata.
        existing_metadata: Final = extra_body.get("metadata")
        existing_dict: Final[dict[str, Any]] = existing_metadata if isinstance(existing_metadata, dict) else {}
        merged_metadata: Final[dict[str, Any]] = {**forward_metadata, **existing_dict}
        extra_body = {**extra_body, "metadata": merged_metadata}
        completion_params["extra_body"] = extra_body

        verbose_logger.debug("A2A -> completion forward metadata keys=%s", list(forward_metadata.keys()))

    @staticmethod
    def a2a_message_to_openai_messages(
        a2a_message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Transform an A2A message to OpenAI message format.

        Args:
            a2a_message: A2A message with role, parts, and messageId

        Returns:
            List of OpenAI-format messages
        """
        role: Final = a2a_message.get("role", "user")
        parts = a2a_message.get("parts", [])

        # Map A2A roles to OpenAI roles
        openai_role = role
        if role == "user":
            openai_role = "user"
        elif role == "assistant":
            openai_role = "assistant"
        elif role == "system":
            openai_role = "system"

        if not isinstance(parts, list):
            parts = []

        content: Final = A2ACompletionBridgeTransformation._extract_text_from_a2a_parts(parts)

        # Do not attach A2A message.metadata here — the completion bridge forwards it
        # once at run level via extra_body.metadata (LangGraph POST /runs/wait shape).
        openai_message: Final[dict[str, Any]] = {"role": openai_role, "content": content}

        verbose_logger.debug(
            "A2A -> OpenAI transform: role=%s -> %s, content_length=%s", role, openai_role, len(content)
        )

        return [openai_message]

    @staticmethod
    def openai_response_to_a2a_response(
        response: Any,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Transform a LiteLLM ModelResponse to A2A SendMessageResponse format.

        Args:
            response: LiteLLM ModelResponse object
            request_id: Original A2A request ID

        Returns:
            A2A SendMessageResponse dict
        """
        # Extract content from response
        content = ""
        if hasattr(response, "choices") and response.choices:
            choice: Final = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content = choice.message.content or ""

        # Build A2A message
        a2a_message: Final = {
            "kind": "message",
            "role": "agent",
            "parts": [{"kind": "text", "text": content}],
            "messageId": uuid4().hex,
        }

        # Build A2A response
        a2a_response: Final = {
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
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        """
        Create a status update event.

        Args:
            ctx: Streaming context
            state: Status state ('working', 'completed')
            final: Whether this is the final event
            message_text: Optional message text for 'working' status
        """
        status: Final[dict[str, Any]] = {
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
    ) -> dict[str, Any]:
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
