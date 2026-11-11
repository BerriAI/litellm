"""
Common utilities for A2A (Agent-to-Agent) Protocol
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import BaseModel

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.azure_ai.common_utils import has_azure_entra_params, resolve_azure_ai_agent_auth_header
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

A2A_TASK_STATE_COMPLETED: Final = "completed"
A2A_TASK_FAILURE_STATES: Final = frozenset({"failed", "canceled", "rejected"})
A2A_TASK_RECOVERABLE_STATES: Final = frozenset({"submitted", "working"})
A2A_TASK_INTERRUPTED_STATES: Final = frozenset({"input-required", "auth-required"})


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


def _message_role(message: Mapping[str, object]) -> object:
    return message.get("role", "user")


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
            role = _message_role(msg.model_dump())
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
        else:
            role = dict(msg).get("role", "user")

        if content_text:
            conversation_parts.append(f"{role}: {content_text}")

    return "\n".join(conversation_parts)


def extract_text_from_a2a_message(message: dict[str, Any], depth: int = 0, max_depth: int = 10) -> str:
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

    parts: Final = message.get("parts", [])
    text_parts: Final[list[str]] = []

    for part in parts:
        if part.get("kind") == "text":
            text_parts.append(part.get("text", ""))
        # Handle nested parts if they exist
        elif "parts" in part:
            nested_text = extract_text_from_a2a_message(part, depth + 1, max_depth)
            if nested_text:
                text_parts.append(nested_text)

    return " ".join(text_parts)


def a2a_task_state(result: Mapping[str, object]) -> str | None:
    if result.get("kind") != "task":
        return None
    status: Final = result.get("status")
    if not isinstance(status, dict):
        return None
    state: Final = status.get("state")
    return state if isinstance(state, str) else None


def _extract_text_from_a2a_result(result: Mapping[str, object], max_depth: int = 10) -> str:
    """
    Extract text from an A2A ``result`` object without applying task-state suppression.

    Used for guardrail scanning so nonterminal status messages are still checked.
    """
    # A2A response can have different formats:
    # 1. Direct message: {"result": {"kind": "message", "parts": [...]}}
    # 2. Nested message: {"result": {"message": {"parts": [...]}}}
    # 3. Task with artifacts: {"result": {"kind": "task", "artifacts": [{"parts": [...]}]}}
    # 4. Task with status message: {"result": {"kind": "task", "status": {"message": {"parts": [...]}}}}
    # 5. Streaming artifact-update: {"result": {"kind": "artifact-update", "artifact": {"parts": [...]}}}

    if "parts" in result:
        return extract_text_from_a2a_message(dict(result), depth=0, max_depth=max_depth)

    message: Final = result.get("message")
    if message and isinstance(message, dict):
        return extract_text_from_a2a_message(message, depth=0, max_depth=max_depth)

    artifact: Final = result.get("artifact")
    if artifact and isinstance(artifact, dict):
        return extract_text_from_a2a_message(artifact, depth=0, max_depth=max_depth)

    status: Final = result.get("status", {})
    if isinstance(status, dict):
        status_message: Final = status.get("message")
        if status_message and isinstance(status_message, dict):
            return extract_text_from_a2a_message(status_message, depth=0, max_depth=max_depth)

    artifacts: Final = result.get("artifacts", [])
    if isinstance(artifacts, list) and artifacts:
        first_artifact: Final = artifacts[0]
        if isinstance(first_artifact, dict):
            return extract_text_from_a2a_message(first_artifact, depth=0, max_depth=max_depth)

    return ""


def extract_text_from_a2a_response_for_guardrails(
    response_dict: Mapping[str, object],
    max_depth: int = 10,
) -> str:
    """Extract text for output guardrails (includes nonterminal task status messages)."""
    result: Final = response_dict.get("result", {})
    if not isinstance(result, dict):
        return ""
    return _extract_text_from_a2a_result(result, max_depth=max_depth)


def extract_text_from_a2a_response(response_dict: Mapping[str, object], max_depth: int = 10) -> str:
    """
    Extract text content from A2A response result.

    Suppresses most text on nonterminal tasks so chat completions do not leak
    in-progress agent output (use ``extract_text_from_a2a_response_for_guardrails``
    when scanning for policy).

    Args:
        response_dict: A2A response dict with 'result' containing message
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Text from response message parts
    """
    result: Final = response_dict.get("result", {})
    if not isinstance(result, dict):
        return ""

    task_state: Final[str | None] = a2a_task_state(result)
    if task_state is not None and task_state != A2A_TASK_STATE_COMPLETED:
        artifacts: Final = result.get("artifacts", [])
        if isinstance(artifacts, list) and artifacts:
            first_artifact: Final = artifacts[0]
            if isinstance(first_artifact, dict):
                return extract_text_from_a2a_message(first_artifact, depth=0, max_depth=max_depth)
        return ""

    return _extract_text_from_a2a_result(result, max_depth=max_depth)


def _handle_a2a_poll_data(poll_data: Mapping[str, object], status_code: int) -> Mapping[str, object] | None:
    if "error" in poll_data:
        error: Final = poll_data.get("error")
        message: Final = error.get("message", "Unknown error") if isinstance(error, dict) else "Unknown error"
        raise A2AError(status_code=status_code, message=f"A2A error: {message}")

    result: Final = poll_data.get("result", {})
    state: Final = a2a_task_state(result) if isinstance(result, dict) else None
    if state == A2A_TASK_STATE_COMPLETED:
        return poll_data
    if state in A2A_TASK_FAILURE_STATES:
        raise A2AError(status_code=status_code, message=f"A2A task ended with state: {state}")
    if state in A2A_TASK_INTERRUPTED_STATES:
        raise A2AError(status_code=status_code, message=f"A2A task requires further action (state: {state})")
    return None


def poll_a2a_task_completion(
    *,
    api_base: str,
    task_id: str,
    request_id: str,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> Mapping[str, object]:
    """Poll ``tasks/get`` until the task reaches a terminal state (sync completions only)."""
    endpoint: Final = api_base.rstrip("/")
    deadline: Final = time.monotonic() + timeout
    request_headers: Final = {"Content-Type": "application/json", **(dict(headers) if headers else {})}
    attempt: Final = 0

    with httpx.Client(timeout=timeout) as client:
        while time.monotonic() < deadline:
            poll_request: Final = {
                "jsonrpc": "2.0",
                "id": f"{request_id}-poll-{attempt}",
                "method": "tasks/get",
                "params": {"id": task_id},
            }
            response: Final = client.post(endpoint, json=poll_request, headers=request_headers)
            response.raise_for_status()
            poll_data: Final = response.json()
            if not isinstance(poll_data, dict):
                raise A2AError(status_code=response.status_code, message="Invalid tasks/get response")
            completed: Final = _handle_a2a_poll_data(poll_data, response.status_code)
            if completed is not None:
                return completed
            attempt += 1
            time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

    raise A2AError(
        status_code=504,
        message=f"A2A task {task_id} did not complete within {timeout} seconds",
    )


async def async_poll_a2a_task_completion(
    *,
    api_base: str,
    task_id: str,
    request_id: str,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
    client: "AsyncHTTPHandler | None" = None,
) -> Mapping[str, object]:
    """Poll ``tasks/get`` without blocking the event loop."""
    from litellm import LlmProviders
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client

    endpoint: Final = api_base.rstrip("/")
    deadline: Final = asyncio.get_running_loop().time() + timeout
    request_headers: Final = {"Content-Type": "application/json", **(dict(headers) if headers else {})}
    attempt: Final = 0
    async_client: Final = client or get_async_httpx_client(llm_provider=LlmProviders.A2A)
    owns_client: Final = client is None

    try:
        while asyncio.get_running_loop().time() < deadline:
            poll_request: Final = {
                "jsonrpc": "2.0",
                "id": f"{request_id}-poll-{attempt}",
                "method": "tasks/get",
                "params": {"id": task_id},
            }
            if not isinstance(async_client, AsyncHTTPHandler):
                raise A2AError(status_code=500, message="A2A async poll requires AsyncHTTPHandler")
            response: Final = await async_client.post(
                endpoint, json=poll_request, headers=request_headers, timeout=timeout
            )
            response.raise_for_status()
            poll_data: Final = response.json()
            if not isinstance(poll_data, dict):
                raise A2AError(status_code=response.status_code, message="Invalid tasks/get response")
            completed: Final = _handle_a2a_poll_data(poll_data, response.status_code)
            if completed is not None:
                return completed
            attempt += 1
            await asyncio.sleep(min(poll_interval, max(0.0, deadline - asyncio.get_running_loop().time())))
    finally:
        if owns_client:
            await async_client.aclose()

    raise A2AError(
        status_code=504,
        message=f"A2A task {task_id} did not complete within {timeout} seconds",
    )


AgentAuthHeaderResolver = Callable[[Mapping[str, object]], Awaitable[Mapping[str, str]]]


def a2a_hop_uses_entra(litellm_params: Mapping[str, object], custom_llm_provider: object) -> bool:
    return not custom_llm_provider and has_azure_entra_params(litellm_params)


async def resolve_a2a_hop_auth_header(
    litellm_params: Mapping[str, object],
    custom_llm_provider: object,
    resolve_entra_header: AgentAuthHeaderResolver = resolve_azure_ai_agent_auth_header,
    agent_extra_headers: Mapping[str, str] | None = None,
) -> Mapping[str, str] | None:
    if a2a_hop_uses_entra(litellm_params, custom_llm_provider):
        return await resolve_entra_header(litellm_params)
    if custom_llm_provider or any(name.lower() == "authorization" for name in (agent_extra_headers or {})):
        return None
    api_key: Final = litellm_params.get("api_key")
    return {"Authorization": f"Bearer {api_key}"} if isinstance(api_key, str) and api_key else None
