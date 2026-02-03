"""
Common utilities for A2A (Agent-to-Agent) Protocol
"""
from typing import Any, Dict, List

from litellm.llms.base_llm.chat.transformation import BaseLLMException


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


def extract_text_from_a2a_message(message: Dict[str, Any]) -> str:
    """
    Extract text content from A2A message parts.
    
    Args:
        message: A2A message dict with 'parts' containing text parts
    
    Returns:
        Concatenated text from all text parts
    """
    if message is None:
        return ""
    
    parts = message.get("parts", [])
    text_parts: List[str] = []
    
    for part in parts:
        if part.get("kind") == "text":
            text_parts.append(part.get("text", ""))
    
    return " ".join(text_parts)


def extract_text_from_a2a_response(response_dict: Dict[str, Any]) -> str:
    """
    Extract text content from A2A response result.
    
    Args:
        response_dict: A2A response dict with 'result' containing message
    
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
    
    # Check if result itself has parts (direct message)
    if "parts" in result:
        return extract_text_from_a2a_message(result)
    
    # Check for nested message
    message = result.get("message")
    if message:
        return extract_text_from_a2a_message(message)
    
    # Handle task result with artifacts
    artifacts = result.get("artifacts", [])
    if artifacts and len(artifacts) > 0:
        first_artifact = artifacts[0]
        return extract_text_from_a2a_message(first_artifact)
    
    return ""
