"""
Common utilities for A2A (Agent-to-Agent) Protocol
"""
from typing import Any, Dict, List

from pydantic import BaseModel

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues


class A2AError(BaseLLMException):
    """Base exception for A2A protocol errors"""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Dict[str, Any] = {},
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            headers=headers,
        )


def convert_messages_to_prompt(messages: List[AllMessageValues]) -> str:
    """
    Convert OpenAI messages to a single prompt string for A2A agent.
    
    Formats each message as "{role}: {content}" and joins with newlines
    to preserve conversation history. Handles both string and list content.
    
    Args:
        messages: List of OpenAI-format messages
    
    Returns:
        Formatted prompt string with full conversation context
    """
    conversation_parts = []
    for msg in messages:
        # Use LiteLLM's helper to extract text from content (handles both str and list)
        content_text = convert_content_list_to_str(message=msg)
        
        # Get role
        if isinstance(msg, BaseModel):
            role = msg.model_dump().get("role", "user")
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
        else:
            role = dict(msg).get("role", "user")  # type: ignore
        
        if content_text:
            conversation_parts.append(f"{role}: {content_text}")
    
    return "\n".join(conversation_parts)


def extract_text_from_a2a_message(
    message: Dict[str, Any], depth: int = 0, max_depth: int = 10
) -> str:
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
    
    parts = message.get("parts", [])
    text_parts: List[str] = []
    
    for part in parts:
        if part.get("kind") == "text":
            text_parts.append(part.get("text", ""))
        # Handle nested parts if they exist
        elif "parts" in part:
            nested_text = extract_text_from_a2a_message(part, depth + 1, max_depth)
            if nested_text:
                text_parts.append(nested_text)
    
    return " ".join(text_parts)


def extract_text_from_a2a_response(
    response_dict: Dict[str, Any], max_depth: int = 10
) -> str:
    """
    Extract text content from A2A response result.
    
    Args:
        response_dict: A2A response dict with 'result' containing message
        max_depth: Maximum recursion depth to prevent infinite loops
    
    Returns:
        Text from response message parts
    """
    result = response_dict.get("result", {})
    if not isinstance(result, dict):
        return ""
    
    # A2A response can have different formats:
    # 1. Direct message: {"result": {"kind": "message", "parts": [...]}}
    # 2. Nested message: {"result": {"message": {"parts": [...]}}}
    # 3. Task with artifacts: {"result": {"kind": "task", "artifacts": [{"parts": [...]}]}}
    # 4. Task with status message: {"result": {"kind": "task", "status": {"message": {"parts": [...]}}}}
    # 5. Streaming artifact-update: {"result": {"kind": "artifact-update", "artifact": {"parts": [...]}}}
    
    # Check if result itself has parts (direct message)
    if "parts" in result:
        return extract_text_from_a2a_message(result, depth=0, max_depth=max_depth)
    
    # Check for nested message
    message = result.get("message")
    if message:
        return extract_text_from_a2a_message(message, depth=0, max_depth=max_depth)
    
    # Check for streaming artifact-update (singular artifact)
    artifact = result.get("artifact")
    if artifact and isinstance(artifact, dict):
        return extract_text_from_a2a_message(
            artifact, depth=0, max_depth=max_depth
        )
    
    # Check for task status message (common in Gemini A2A agents)
    status = result.get("status", {})
    if isinstance(status, dict):
        status_message = status.get("message")
        if status_message:
            return extract_text_from_a2a_message(
                status_message, depth=0, max_depth=max_depth
            )
    
    # Handle task result with artifacts (plural, array)
    artifacts = result.get("artifacts", [])
    if artifacts and len(artifacts) > 0:
        first_artifact = artifacts[0]
        return extract_text_from_a2a_message(
            first_artifact, depth=0, max_depth=max_depth
        )
    
    return ""
