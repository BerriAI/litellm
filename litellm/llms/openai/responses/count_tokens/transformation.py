"""
OpenAI Responses API token counting transformation logic.

This module handles the transformation of requests to OpenAI's /v1/responses/input_tokens endpoint.
"""

from typing import Any, Dict, List, Optional, Union


class OpenAICountTokensConfig:
    """
    Configuration and transformation logic for OpenAI Responses API token counting.

    OpenAI Responses API Token Counting Specification:
    - Endpoint: POST https://api.openai.com/v1/responses/input_tokens
    - Response: {"input_tokens": <number>}
    """

    def get_openai_count_tokens_endpoint(self, api_base: Optional[str] = None) -> str:
        base = api_base or "https://api.openai.com/v1"
        base = base.rstrip("/")
        return f"{base}/responses/input_tokens"

    def transform_request_to_count_tokens(
        self,
        model: str,
        input: Union[str, List[Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transform request to OpenAI Responses API token counting format.

        The Responses API uses `input` (not `messages`) and `instructions` (not `system`).
        """
        request: Dict[str, Any] = {
            "model": model,
            "input": input,
        }

        if instructions is not None:
            request["instructions"] = instructions

        if tools is not None:
            request["tools"] = self._transform_tools_for_responses_api(tools)

        return request

    def get_required_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def validate_request(
        self, model: str, input: Union[str, List[Any]]
    ) -> None:
        if not model:
            raise ValueError("model parameter is required")

        if not input:
            raise ValueError("input parameter is required")

    @staticmethod
    def _transform_tools_for_responses_api(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Transform OpenAI chat tools format to Responses API tools format.

        Chat format:  {"type": "function", "function": {"name": "...", "parameters": {...}}}
        Responses format: {"type": "function", "name": "...", "parameters": {...}}
        """
        transformed = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                transformed.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            else:
                # Pass through non-function tools (e.g., web_search, file_search)
                transformed.append(tool)
        return transformed

    @staticmethod
    def messages_to_responses_input(
        messages: List[Dict[str, Any]],
    ) -> tuple:
        """
        Convert standard chat messages format to OpenAI Responses API input format.

        Returns:
            (input_items, instructions) tuple where instructions is extracted
            from system/developer messages.
        """
        input_items: List[Dict[str, Any]] = []
        instructions_parts: List[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            if role in ("system", "developer"):
                # Extract system/developer messages as instructions
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    # Handle content blocks - extract text
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    instructions_parts.append("\n".join(text_parts))
            elif role == "user":
                input_items.append({"role": "user", "content": content})
            elif role == "assistant":
                # Map tool_calls to Responses API function_call items
                tool_calls = msg.get("tool_calls")
                if content:
                    input_items.append({"role": "assistant", "content": content})
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", ""),
                        })
                elif not content:
                    input_items.append({"role": "assistant", "content": content})
            elif role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else str(content),
                })

        instructions = "\n".join(instructions_parts) if instructions_parts else None
        return input_items, instructions
