"""
Transformation layer for Pydantic AI agents.

Pydantic AI agents follow A2A protocol but don't support streaming.
This module provides fake streaming by converting non-streaming responses into streaming chunks.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Final, Protocol, cast, runtime_checkable
from uuid import uuid4

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)

_ANY_KEY_DICT_ADAPTER: Final = TypeAdapter(dict[object, object])
_STR_KEY_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_LIST_ADAPTER: Final = TypeAdapter(list[object])
_TEXT_ADAPTER: Final = TypeAdapter(str)


@runtime_checkable
class _SupportsModelDump(Protocol):
    def model_dump(self, *, mode: str, exclude_none: bool) -> Mapping[str, object]: ...


@runtime_checkable
class _SupportsPydanticDict(Protocol):
    def dict(self, *, exclude_none: bool) -> Mapping[str, object]: ...


class PydanticAITransformation:
    """
    Transformation layer for Pydantic AI agents.

    Handles:
    - Direct A2A requests to Pydantic AI endpoints
    - Polling for task completion (since Pydantic AI doesn't support streaming)
    - Fake streaming by chunking non-streaming responses
    """

    @staticmethod
    def _remove_none_values(obj: object) -> object:
        """
        Recursively remove None values from a dict/list structure.

        FastA2A/Pydantic AI servers don't accept None values for optional fields -
        they expect those fields to be omitted entirely.

        Args:
            obj: Dict, list, or other value to clean

        Returns:
            Cleaned object with None values removed
        """
        if isinstance(obj, dict):
            typed_dict: Final = _ANY_KEY_DICT_ADAPTER.validate_python(obj)
            return {k: PydanticAITransformation._remove_none_values(v) for k, v in typed_dict.items() if v is not None}
        elif isinstance(obj, list):
            typed_list: Final = _LIST_ADAPTER.validate_python(obj)
            return [PydanticAITransformation._remove_none_values(item) for item in typed_list if item is not None]
        else:
            return obj

    @staticmethod
    def _params_to_dict(
        params: "_SupportsModelDump | _SupportsPydanticDict | Mapping[str, object]",
    ) -> Mapping[str, object]:
        """
        Convert params to a dict, handling Pydantic models.

        Args:
            params: Dict or Pydantic model

        Returns:
            Dict representation of params
        """
        if isinstance(params, _SupportsModelDump):
            # Pydantic v2 model
            return params.model_dump(mode="python", exclude_none=True)
        elif isinstance(params, _SupportsPydanticDict):
            # Pydantic v1 model
            return params.dict(exclude_none=True)
        elif isinstance(params, dict):
            return params
        else:
            # Try to convert to dict
            return dict(params)

    @staticmethod
    async def _poll_for_completion(
        client: AsyncHTTPHandler,
        endpoint: str,
        task_id: object,
        request_id: str,
        max_attempts: int = 30,
        poll_interval: float = 0.5,
        agent_extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """
        Poll for task completion using tasks/get method.

        Args:
            client: HTTPX async client
            endpoint: API endpoint URL
            task_id: Task ID to poll for
            request_id: JSON-RPC request ID
            max_attempts: Maximum polling attempts
            poll_interval: Seconds between poll attempts

        Returns:
            Completed task response
        """
        for attempt in range(max_attempts):
            poll_request = {
                "jsonrpc": "2.0",
                "id": f"{request_id}-poll-{attempt}",
                "method": "tasks/get",
                "params": {"id": task_id},
            }

            response = await client.post(
                endpoint,
                json=poll_request,
                headers={
                    **(agent_extra_headers or {}),
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            poll_data = _STR_KEY_DICT_ADAPTER.validate_python(response.json())

            result = _STR_KEY_DICT_ADAPTER.validate_python(poll_data.get("result", {}))
            status = _STR_KEY_DICT_ADAPTER.validate_python(result.get("status", {}))
            state = status.get("state", "")

            verbose_logger.debug("Pydantic AI: Poll attempt %s/%s, state=%s", attempt + 1, max_attempts, state)

            if state == "completed":
                return poll_data
            elif state in ("failed", "canceled"):
                raise Exception(f"Task {task_id} ended with state: {state}")

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {max_attempts * poll_interval} seconds")

    @staticmethod
    async def _send_and_poll_raw(
        api_base: str,
        request_id: str,
        params: "_SupportsModelDump | _SupportsPydanticDict | Mapping[str, object]",
        timeout: float = 60.0,
        agent_extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """
        Send a request to Pydantic AI agent and return the raw task response.

        This is an internal method used by both non-streaming and streaming handlers.
        Returns the raw Pydantic AI task format with history/artifacts.

        Args:
            api_base: Base URL of the Pydantic AI agent
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            timeout: Request timeout in seconds

        Returns:
            Raw Pydantic AI task response (with history/artifacts)
        """
        # Convert params to dict if it's a Pydantic model
        # Remove None values - FastA2A doesn't accept null for optional fields
        params_dict: Final = _ANY_KEY_DICT_ADAPTER.validate_python(
            PydanticAITransformation._remove_none_values(PydanticAITransformation._params_to_dict(params))
        )

        # Ensure the message has 'kind': 'message' as required by FastA2A/Pydantic AI
        if "message" in params_dict:
            message_value: Final = _ANY_KEY_DICT_ADAPTER.validate_python(params_dict["message"])
            message_value["kind"] = "message"
            params_dict["message"] = message_value

        # Build A2A JSON-RPC request using message/send method for FastA2A compatibility
        a2a_request: Final = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": params_dict,
        }

        # FastA2A uses root endpoint (/) not /messages
        endpoint: Final = api_base.rstrip("/")

        verbose_logger.info("Pydantic AI: Sending non-streaming request to %s", endpoint)

        # Send request to Pydantic AI agent using shared async HTTP client
        client: Final = get_async_httpx_client(
            llm_provider=cast(Any, "pydantic_ai_agent"),
            params={"timeout": timeout},
        )
        response: Final = await client.post(
            endpoint,
            json=a2a_request,
            headers={
                **(agent_extra_headers or {}),
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        response_data = _STR_KEY_DICT_ADAPTER.validate_python(response.json())

        # Check if task is already completed
        result: Final = _STR_KEY_DICT_ADAPTER.validate_python(response_data.get("result", {}))
        status: Final = _STR_KEY_DICT_ADAPTER.validate_python(result.get("status", {}))
        state: Final = status.get("state", "")

        if state != "completed":
            # Need to poll for completion
            task_id: Final = result.get("id")
            if task_id:
                verbose_logger.info("Pydantic AI: Task %s submitted, polling for completion...", task_id)
                response_data = await PydanticAITransformation._poll_for_completion(
                    client=client,
                    endpoint=endpoint,
                    task_id=task_id,
                    request_id=request_id,
                    agent_extra_headers=agent_extra_headers,
                )

        verbose_logger.info("Pydantic AI: Received completed response for request_id=%s", request_id)

        return response_data

    @staticmethod
    async def send_non_streaming_request(
        api_base: str,
        request_id: str,
        params: "_SupportsModelDump | _SupportsPydanticDict | Mapping[str, object]",
        timeout: float = 60.0,
        agent_extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """
        Send a non-streaming A2A request to Pydantic AI agent and wait for completion.

        Args:
            api_base: Base URL of the Pydantic AI agent (e.g., "http://localhost:9999")
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message (dict or Pydantic model)
            timeout: Request timeout in seconds
            agent_extra_headers: Per-request headers to forward on the upstream HTTP call.

        Returns:
            Standard A2A non-streaming response format with message
        """
        # Get raw task response
        raw_response: Final = await PydanticAITransformation._send_and_poll_raw(
            api_base=api_base,
            request_id=request_id,
            params=params,
            timeout=timeout,
            agent_extra_headers=agent_extra_headers,
        )

        # Transform to standard A2A non-streaming format
        return PydanticAITransformation._transform_to_a2a_response(
            response_data=raw_response,
            request_id=request_id,
        )

    @staticmethod
    async def send_and_get_raw_response(
        api_base: str,
        request_id: str,
        params: "_SupportsModelDump | _SupportsPydanticDict | Mapping[str, object]",
        timeout: float = 60.0,
        agent_extra_headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """
        Send a request to Pydantic AI agent and return the raw task response.

        Used by streaming handler to get raw response for fake streaming.

        Args:
            api_base: Base URL of the Pydantic AI agent
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            timeout: Request timeout in seconds
            agent_extra_headers: Per-request headers to forward on the upstream HTTP call.

        Returns:
            Raw Pydantic AI task response (with history/artifacts)
        """
        return await PydanticAITransformation._send_and_poll_raw(
            api_base=api_base,
            request_id=request_id,
            params=params,
            timeout=timeout,
            agent_extra_headers=agent_extra_headers,
        )

    @staticmethod
    def _transform_to_a2a_response(
        response_data: Mapping[str, object],
        request_id: str,
    ) -> dict[str, object]:
        """
        Transform Pydantic AI task response to standard A2A non-streaming format.

        Pydantic AI returns a task with history/artifacts, but the standard A2A
        non-streaming format expects ``result`` to be the Message directly
        (``kind="message"``), per the A2A spec / ``SendMessageResponse``:
        {
            "jsonrpc": "2.0",
            "id": "...",
            "result": {
                "kind": "message",
                "role": "agent",
                "parts": [{"kind": "text", "text": "..."}],
                "messageId": "..."
            }
        }

        Args:
            response_data: Pydantic AI task response
            request_id: Original request ID

        Returns:
            Standard A2A non-streaming response format
        """
        # Extract the agent response text
        full_text, message_id, parts = PydanticAITransformation._extract_response_text(response_data)

        # Build standard A2A message
        a2a_message: Final = {
            "kind": "message",
            "role": "agent",
            "parts": parts if parts else [{"kind": "text", "text": full_text}],
            "messageId": message_id,
        }

        # Return standard A2A non-streaming format
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": a2a_message,
        }

    @staticmethod
    def _extract_response_text(response_data: Mapping[str, object]) -> tuple[object, object, Sequence[object]]:
        """
        Extract response text from completed task response.

        Pydantic AI returns completed tasks with:
        - history: list of messages (user and agent)
        - artifacts: list of result artifacts

        Args:
            response_data: Completed task response

        Returns:
            Tuple of (full_text, message_id, parts)
        """
        result: Final = _STR_KEY_DICT_ADAPTER.validate_python(response_data.get("result", {}))

        # Try to extract from artifacts first (preferred for results)
        artifacts: Final = result.get("artifacts", [])
        if artifacts:
            for artifact in _LIST_ADAPTER.validate_python(artifacts):
                parts = _LIST_ADAPTER.validate_python(_STR_KEY_DICT_ADAPTER.validate_python(artifact).get("parts", []))
                for part in parts:
                    if (part_dict := _STR_KEY_DICT_ADAPTER.validate_python(part)).get("kind") == "text":
                        text = part_dict.get("text", "")
                        if text:
                            return text, str(uuid4()), parts

        # Fall back to history - get the last agent message
        history: Final = _LIST_ADAPTER.validate_python(result.get("history", []))
        for msg in reversed(history):
            if (msg_dict := _STR_KEY_DICT_ADAPTER.validate_python(msg)).get("role") == "agent":
                parts = _LIST_ADAPTER.validate_python(msg_dict.get("parts", []))
                message_id = msg_dict.get("messageId", str(uuid4()))
                full_text = ""
                for part in parts:
                    if (part_dict := _STR_KEY_DICT_ADAPTER.validate_python(part)).get("kind") == "text":
                        full_text += _TEXT_ADAPTER.validate_python(part_dict.get("text", ""))
                if full_text:
                    return full_text, message_id, parts

        # Fall back to message field (original format)
        message: Final = result.get("message", {})
        if message:
            message_dict: Final = _STR_KEY_DICT_ADAPTER.validate_python(message)
            parts = _LIST_ADAPTER.validate_python(message_dict.get("parts", []))
            message_id = message_dict.get("messageId", str(uuid4()))
            full_text = ""
            for part in parts:
                if (part_dict := _STR_KEY_DICT_ADAPTER.validate_python(part)).get("kind") == "text":
                    full_text += _TEXT_ADAPTER.validate_python(part_dict.get("text", ""))
            return full_text, message_id, parts

        return "", str(uuid4()), []

    @staticmethod
    async def fake_streaming_from_response(
        response_data: Mapping[str, object],
        request_id: str,
        chunk_size: int = 50,
        delay_ms: int = 10,
    ) -> AsyncIterator[dict[str, object]]:
        """
        Convert a non-streaming A2A response into fake streaming chunks.

        Emits proper A2A streaming events:
        1. Task event (kind: "task") - Initial task with status "submitted"
        2. Status update (kind: "status-update") - Status "working"
        3. Artifact update chunks (kind: "artifact-update") - Content delivery in chunks
        4. Status update (kind: "status-update") - Final "completed" status

        Args:
            response_data: Non-streaming A2A response dict (completed task)
            request_id: A2A JSON-RPC request ID
            chunk_size: Number of characters per chunk (default: 50)
            delay_ms: Delay between chunks in milliseconds (default: 10)

        Yields:
            A2A streaming response events
        """
        # Extract the response text from completed task
        full_text, message_id, parts = PydanticAITransformation._extract_response_text(response_data)

        # Extract input message from raw response for history
        result: Final = _STR_KEY_DICT_ADAPTER.validate_python(response_data.get("result", {}))
        history: Final = _LIST_ADAPTER.validate_python(result.get("history", []))
        input_message = _STR_KEY_DICT_ADAPTER.validate_python({})
        for msg in history:
            if (msg_dict := _STR_KEY_DICT_ADAPTER.validate_python(msg)).get("role") == "user":
                input_message = msg_dict
                break

        # Generate IDs for streaming events
        task_id: Final = str(uuid4())
        context_id: Final = str(uuid4())
        artifact_id: Final = str(uuid4())
        input_message_id: Final = input_message.get("messageId", str(uuid4()))

        # 1. Emit initial task event (kind: "task", status: "submitted")
        # Format matches A2ACompletionBridgeTransformation.create_task_event
        task_event: Final = _STR_KEY_DICT_ADAPTER.validate_python(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contextId": context_id,
                    "history": [
                        {
                            "contextId": context_id,
                            "kind": "message",
                            "messageId": input_message_id,
                            "parts": input_message.get("parts", [{"kind": "text", "text": ""}]),
                            "role": "user",
                            "taskId": task_id,
                        }
                    ],
                    "id": task_id,
                    "kind": "task",
                    "status": {
                        "state": "submitted",
                    },
                },
            }
        )
        yield task_event

        # 2. Emit status update (kind: "status-update", status: "working")
        # Format matches A2ACompletionBridgeTransformation.create_status_update_event
        working_event: Final = _STR_KEY_DICT_ADAPTER.validate_python(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contextId": context_id,
                    "final": False,
                    "kind": "status-update",
                    "status": {
                        "state": "working",
                    },
                    "taskId": task_id,
                },
            }
        )
        yield working_event

        # Small delay to simulate processing
        await asyncio.sleep(delay_ms / 1000.0)

        # 3. Emit artifact update chunks (kind: "artifact-update")
        # Format matches A2ACompletionBridgeTransformation.create_artifact_update_event
        if full_text:
            full_text_str: Final = _TEXT_ADAPTER.validate_python(full_text)
            # Split text into chunks
            for i in range(0, len(full_text_str), chunk_size):
                chunk_text = full_text_str[i : i + chunk_size]
                is_last_chunk = (i + chunk_size) >= len(full_text_str)

                artifact_event = _STR_KEY_DICT_ADAPTER.validate_python(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "contextId": context_id,
                            "kind": "artifact-update",
                            "taskId": task_id,
                            "artifact": {
                                "artifactId": artifact_id,
                                "parts": [
                                    {
                                        "kind": "text",
                                        "text": chunk_text,
                                    }
                                ],
                            },
                        },
                    }
                )
                yield artifact_event

                # Add delay between chunks (except for last chunk)
                if not is_last_chunk:
                    await asyncio.sleep(delay_ms / 1000.0)

        # 4. Emit final status update (kind: "status-update", status: "completed", final: true)
        completed_event: Final = _STR_KEY_DICT_ADAPTER.validate_python(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contextId": context_id,
                    "final": True,
                    "kind": "status-update",
                    "status": {
                        "state": "completed",
                    },
                    "taskId": task_id,
                },
            }
        )
        yield completed_event

        verbose_logger.info("Pydantic AI: Fake streaming completed for request_id=%s", request_id)
