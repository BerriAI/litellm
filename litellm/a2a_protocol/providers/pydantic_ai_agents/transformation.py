"""
Transformation layer for Pydantic AI agents.

Pydantic AI agents follow A2A protocol but don't support streaming.
This module provides fake streaming by converting non-streaming responses into streaming chunks.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, cast
from uuid import uuid4

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client


class PydanticAITransformation:
    """
    Transformation layer for Pydantic AI agents.
    
    Handles:
    - Direct A2A requests to Pydantic AI endpoints
    - Polling for task completion (since Pydantic AI doesn't support streaming)
    - Fake streaming by chunking non-streaming responses
    """

    @staticmethod
    def _remove_none_values(obj: Any) -> Any:
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
            return {
                k: PydanticAITransformation._remove_none_values(v)
                for k, v in obj.items()
                if v is not None
            }
        elif isinstance(obj, list):
            return [
                PydanticAITransformation._remove_none_values(item)
                for item in obj
                if item is not None
            ]
        else:
            return obj

    @staticmethod
    def _params_to_dict(params: Any) -> Dict[str, Any]:
        """
        Convert params to a dict, handling Pydantic models.
        
        Args:
            params: Dict or Pydantic model
            
        Returns:
            Dict representation of params
        """
        if hasattr(params, "model_dump"):
            # Pydantic v2 model
            return params.model_dump(mode="python", exclude_none=True)
        elif hasattr(params, "dict"):
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
        task_id: str,
        request_id: str,
        max_attempts: int = 30,
        poll_interval: float = 0.5,
    ) -> Dict[str, Any]:
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
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            poll_data = response.json()
            
            result = poll_data.get("result", {})
            status = result.get("status", {})
            state = status.get("state", "")
            
            verbose_logger.debug(
                f"Pydantic AI: Poll attempt {attempt + 1}/{max_attempts}, state={state}"
            )
            
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
        params: Any,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
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
        params_dict = PydanticAITransformation._params_to_dict(params)
        
        # Remove None values - FastA2A doesn't accept null for optional fields
        params_dict = PydanticAITransformation._remove_none_values(params_dict)
        
        # Ensure the message has 'kind': 'message' as required by FastA2A/Pydantic AI
        if "message" in params_dict:
            params_dict["message"]["kind"] = "message"

        # Build A2A JSON-RPC request using message/send method for FastA2A compatibility
        a2a_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": params_dict,
        }

        # FastA2A uses root endpoint (/) not /messages
        endpoint = api_base.rstrip("/")

        verbose_logger.info(
            f"Pydantic AI: Sending non-streaming request to {endpoint}"
        )

        # Send request to Pydantic AI agent using shared async HTTP client
        client = get_async_httpx_client(
            llm_provider=cast(Any, "pydantic_ai_agent"),
            params={"timeout": timeout},
        )
        response = await client.post(
            endpoint,
            json=a2a_request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        response_data = response.json()
        
        # Check if task is already completed
        result = response_data.get("result", {})
        status = result.get("status", {})
        state = status.get("state", "")
        
        if state != "completed":
            # Need to poll for completion
            task_id = result.get("id")
            if task_id:
                verbose_logger.info(
                    f"Pydantic AI: Task {task_id} submitted, polling for completion..."
                )
                response_data = await PydanticAITransformation._poll_for_completion(
                    client=client,
                    endpoint=endpoint,
                    task_id=task_id,
                    request_id=request_id,
                )

        verbose_logger.info(f"Pydantic AI: Received completed response for request_id={request_id}")

        return response_data

    @staticmethod
    async def send_non_streaming_request(
        api_base: str,
        request_id: str,
        params: Any,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Send a non-streaming A2A request to Pydantic AI agent and wait for completion.

        Args:
            api_base: Base URL of the Pydantic AI agent (e.g., "http://localhost:9999")
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message (dict or Pydantic model)
            timeout: Request timeout in seconds

        Returns:
            Standard A2A non-streaming response format with message
        """
        # Get raw task response
        raw_response = await PydanticAITransformation._send_and_poll_raw(
            api_base=api_base,
            request_id=request_id,
            params=params,
            timeout=timeout,
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
        params: Any,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Send a request to Pydantic AI agent and return the raw task response.
        
        Used by streaming handler to get raw response for fake streaming.

        Args:
            api_base: Base URL of the Pydantic AI agent
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            timeout: Request timeout in seconds

        Returns:
            Raw Pydantic AI task response (with history/artifacts)
        """
        return await PydanticAITransformation._send_and_poll_raw(
            api_base=api_base,
            request_id=request_id,
            params=params,
            timeout=timeout,
        )

    @staticmethod
    def _transform_to_a2a_response(
        response_data: Dict[str, Any],
        request_id: str,
    ) -> Dict[str, Any]:
        """
        Transform Pydantic AI task response to standard A2A non-streaming format.
        
        Pydantic AI returns a task with history/artifacts, but the standard A2A
        non-streaming format expects:
        {
            "jsonrpc": "2.0",
            "id": "...",
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": "..."}],
                    "messageId": "..."
                }
            }
        }
        
        Args:
            response_data: Pydantic AI task response
            request_id: Original request ID
            
        Returns:
            Standard A2A non-streaming response format
        """
        # Extract the agent response text
        full_text, message_id, parts = PydanticAITransformation._extract_response_text(
            response_data
        )
        
        # Build standard A2A message
        a2a_message = {
            "role": "agent",
            "parts": parts if parts else [{"kind": "text", "text": full_text}],
            "messageId": message_id,
        }
        
        # Return standard A2A non-streaming format
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "message": a2a_message,
            },
        }

    @staticmethod
    def _extract_response_text(response_data: Dict[str, Any]) -> tuple[str, str, list]:
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
        result = response_data.get("result", {})
        
        # Try to extract from artifacts first (preferred for results)
        artifacts = result.get("artifacts", [])
        if artifacts:
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                for part in parts:
                    if part.get("kind") == "text":
                        text = part.get("text", "")
                        if text:
                            return text, str(uuid4()), parts
        
        # Fall back to history - get the last agent message
        history = result.get("history", [])
        for msg in reversed(history):
            if msg.get("role") == "agent":
                parts = msg.get("parts", [])
                message_id = msg.get("messageId", str(uuid4()))
                full_text = ""
                for part in parts:
                    if part.get("kind") == "text":
                        full_text += part.get("text", "")
                if full_text:
                    return full_text, message_id, parts
        
        # Fall back to message field (original format)
        message = result.get("message", {})
        if message:
            parts = message.get("parts", [])
            message_id = message.get("messageId", str(uuid4()))
            full_text = ""
            for part in parts:
                if part.get("kind") == "text":
                    full_text += part.get("text", "")
            return full_text, message_id, parts
        
        return "", str(uuid4()), []

    @staticmethod
    async def fake_streaming_from_response(
        response_data: Dict[str, Any],
        request_id: str,
        chunk_size: int = 50,
        delay_ms: int = 10,
    ) -> AsyncIterator[Dict[str, Any]]:
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
        full_text, message_id, parts = PydanticAITransformation._extract_response_text(
            response_data
        )
        
        # Extract input message from raw response for history
        result = response_data.get("result", {})
        history = result.get("history", [])
        input_message = {}
        for msg in history:
            if msg.get("role") == "user":
                input_message = msg
                break

        # Generate IDs for streaming events
        task_id = str(uuid4())
        context_id = str(uuid4())
        artifact_id = str(uuid4())
        input_message_id = input_message.get("messageId", str(uuid4()))

        # 1. Emit initial task event (kind: "task", status: "submitted")
        # Format matches A2ACompletionBridgeTransformation.create_task_event
        task_event = {
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
        yield task_event

        # 2. Emit status update (kind: "status-update", status: "working")
        # Format matches A2ACompletionBridgeTransformation.create_status_update_event
        working_event = {
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
        yield working_event

        # Small delay to simulate processing
        await asyncio.sleep(delay_ms / 1000.0)

        # 3. Emit artifact update chunks (kind: "artifact-update")
        # Format matches A2ACompletionBridgeTransformation.create_artifact_update_event
        if full_text:
            # Split text into chunks
            for i in range(0, len(full_text), chunk_size):
                chunk_text = full_text[i:i + chunk_size]
                is_last_chunk = (i + chunk_size) >= len(full_text)

                artifact_event = {
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
                yield artifact_event

                # Add delay between chunks (except for last chunk)
                if not is_last_chunk:
                    await asyncio.sleep(delay_ms / 1000.0)

        # 4. Emit final status update (kind: "status-update", status: "completed", final: true)
        completed_event = {
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
        yield completed_event

        verbose_logger.info(
            f"Pydantic AI: Fake streaming completed for request_id={request_id}"
        )


