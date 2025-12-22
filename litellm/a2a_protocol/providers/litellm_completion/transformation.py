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
from typing import Any, Dict, List, Optional
from uuid import uuid4

from litellm._logging import verbose_logger


class A2AStreamingContext:
    """
    Context holder for A2A streaming state.
    Tracks task_id, context_id, and message accumulation.
    """

    def __init__(self, request_id: str, input_message: Dict[str, Any]):
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
    def a2a_message_to_openai_messages(
        a2a_message: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Transform an A2A message to OpenAI message format.

        Args:
            a2a_message: A2A message with role, parts, and messageId

        Returns:
            List of OpenAI-format messages
        """
        role = a2a_message.get("role", "user")
        parts = a2a_message.get("parts", [])

        # Map A2A roles to OpenAI roles
        openai_role = role
        if role == "user":
            openai_role = "user"
        elif role == "assistant":
            openai_role = "assistant"
        elif role == "system":
            openai_role = "system"

        # Extract text content from parts
        content_parts = []
        for part in parts:
            kind = part.get("kind", "")
            if kind == "text":
                text = part.get("text", "")
                content_parts.append(text)

        content = "\n".join(content_parts) if content_parts else ""

        verbose_logger.debug(
            f"A2A -> OpenAI transform: role={role} -> {openai_role}, content_length={len(content)}"
        )

        return [{"role": openai_role, "content": content}]

    @staticmethod
    def openai_response_to_a2a_response(
        response: Any,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content = choice.message.content or ""

        # Build A2A message
        a2a_message = {
            "role": "agent",
            "parts": [{"kind": "text", "text": content}],
            "messageId": uuid4().hex,
        }

        # Build A2A response
        a2a_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "message": a2a_message,
            },
        }

        verbose_logger.debug(
            f"OpenAI -> A2A transform: content_length={len(content)}"
        )

        return a2a_response

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp in ISO format with timezone."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def create_task_event(
        ctx: A2AStreamingContext,
    ) -> Dict[str, Any]:
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
        message_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a status update event.

        Args:
            ctx: Streaming context
            state: Status state ('working', 'completed')
            final: Whether this is the final event
            message_text: Optional message text for 'working' status
        """
        status: Dict[str, Any] = {
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
    ) -> Dict[str, Any]:
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

    @staticmethod
    def openai_chunk_to_a2a_chunk(
        chunk: Any,
        request_id: Optional[str] = None,
        is_final: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a LiteLLM streaming chunk to A2A streaming format.

        NOTE: This method is deprecated for streaming. Use the event-based
        methods (create_task_event, create_status_update_event,
        create_artifact_update_event) instead for proper A2A streaming.

        Args:
            chunk: LiteLLM ModelResponse chunk
            request_id: Original A2A request ID
            is_final: Whether this is the final chunk

        Returns:
            A2A streaming chunk dict or None if no content
        """
        # Extract delta content
        content = ""
        if chunk is not None and hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                content = choice.delta.content or ""

        if not content and not is_final:
            return None

        # Build A2A streaming chunk (legacy format)
        a2a_chunk = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": content}],
                    "messageId": uuid4().hex,
                },
                "final": is_final,
            },
        }

        return a2a_chunk
