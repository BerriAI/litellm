"""
AWS Bedrock CountTokens API transformation logic.

This module handles the transformation of requests from Anthropic Messages API format
to AWS Bedrock's CountTokens API format and vice versa.
"""

import re
from collections.abc import Mapping
from typing import Final, Literal

from pydantic import JsonValue

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import get_bedrock_base_model

# Placeholder satisfying the Anthropic InvokeModel schema's required
# max_tokens field; CountTokens only counts input, so it has no effect
# on any generation.
DEFAULT_ANTHROPIC_INVOKE_MODEL_MAX_TOKENS: Final = 1024


def _json_dict(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _json_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _to_converse_content(content: JsonValue) -> list[JsonValue]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        return content
    return []


def _to_converse_message(message: JsonValue) -> dict[str, JsonValue]:
    fields: Final = _json_dict(message)
    return {
        "role": fields.get("role"),
        "content": _to_converse_content(fields.get("content", "")),
    }


def _sanitized_bedrock_tool_name(raw_name: JsonValue) -> str:
    name: Final = re.sub(r"[^a-zA-Z0-9_]", "_", raw_name if isinstance(raw_name, str) else "")
    prefixed: Final = name if not name or name[0].isalpha() else f"t_{name}"
    return prefixed[:64]


def _to_bedrock_tool_spec(tool: JsonValue) -> dict[str, JsonValue]:
    fields: Final = _json_dict(tool)
    name: Final = _sanitized_bedrock_tool_name(fields.get("name", ""))
    return {
        "toolSpec": {
            "name": name,
            "description": fields.get("description") or name,
            "inputSchema": {"json": fields.get("input_schema", {"type": "object", "properties": {}})},
        }
    }


class BedrockCountTokensConfig(BaseAWSLLM):
    """
    Configuration and transformation logic for AWS Bedrock CountTokens API.

    AWS Bedrock CountTokens API Specification:
    - Endpoint: POST /model/{modelId}/count-tokens
    - Input formats: 'invokeModel' or 'converse'
    - Response: {"inputTokens": <number>}
    """

    def _detect_input_type(self, request_data: Mapping[str, JsonValue]) -> Literal["converse", "invokeModel"]:
        """
        Detect whether to use 'converse' or 'invokeModel' input format.

        Args:
            request_data: The original request data

        Returns:
            'converse' or 'invokeModel'
        """
        messages: Final = request_data.get("messages")
        if isinstance(messages, list):
            # Anthropic content blocks carry a "type" key ({"type": "text", ...});
            # Converse blocks don't ({"text": ...}, {"toolUse": ...}). Converse
            # rejects Anthropic-shape blocks, so route those to invokeModel,
            # which forwards the body verbatim.
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, list) and any(isinstance(block, dict) and "type" in block for block in content):
                    return "invokeModel"
            return "converse"

        # For raw text or other formats, use invokeModel
        # This handles cases where the input is prompt-based or already in raw Bedrock format
        return "invokeModel"

    def transform_anthropic_to_bedrock_count_tokens(
        self,
        request_data: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """
        Transform request to Bedrock CountTokens format.
        Supports both Converse and InvokeModel input types.

        Input (Anthropic format):
        {
            "model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello!"}]
        }

        Output (Bedrock CountTokens format for Converse):
        {
            "input": {
                "converse": {
                    "messages": [...],
                    "system": [...] (if present)
                }
            }
        }

        Output (Bedrock CountTokens format for InvokeModel):
        {
            "input": {
                "invokeModel": {
                    "body": "<base64-encoded raw model input>"
                }
            }
        }
        """
        input_type: Final = self._detect_input_type(request_data)

        if input_type == "converse":
            return self._transform_to_converse_format(request_data)
        else:
            return self._transform_to_invoke_model_format(request_data)

    def _transform_to_converse_format(self, request_data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Transform to Converse input format, including system and tools."""
        messages: Final = _json_list(request_data.get("messages"))
        system: Final = request_data.get("system")
        tools: Final = request_data.get("tools")

        # Transform messages
        user_messages: Final[list[JsonValue]] = [_to_converse_message(message) for message in messages]

        converse_input: Final[dict[str, JsonValue]] = {"messages": user_messages}

        # Transform system prompt (string or list of blocks → Bedrock format)
        system_blocks: Final = self._transform_system(system)
        if system_blocks:
            converse_input["system"] = system_blocks

        # Transform tools (Anthropic format → Bedrock toolConfig)
        tool_config: Final = self._transform_tools(tools)
        if tool_config:
            converse_input["toolConfig"] = tool_config

        return {"input": {"converse": converse_input}}

    def _transform_system(self, system: JsonValue) -> list[JsonValue]:
        """Transform Anthropic system prompt to Bedrock system blocks."""
        if system is None:
            return []
        if isinstance(system, str):
            return [{"text": system}]
        if isinstance(system, list):
            # Already in blocks format (e.g. [{"type": "text", "text": "..."}])
            return [{"text": block.get("text", "")} for block in system if isinstance(block, dict)]
        return []

    def _transform_tools(self, tools: JsonValue) -> dict[str, JsonValue] | None:
        """Transform Anthropic tools to Bedrock toolConfig format."""
        if not tools:
            return None

        bedrock_tools: Final[list[JsonValue]] = [_to_bedrock_tool_spec(tool) for tool in _json_list(tools)]

        return {"tools": bedrock_tools}

    def _transform_to_invoke_model_format(self, request_data: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Transform to InvokeModel input format."""
        import base64
        import json

        # For InvokeModel, we need to provide the raw body that would be sent to the model
        # Remove the 'model' field from the body as it's not part of the model input
        body_data: Final = {k: v for k, v in request_data.items() if k != "model"}

        if "messages" in body_data:
            # Bedrock validates the body against the model's InvokeModel schema;
            # Anthropic Messages bodies require these fields.
            body_data.setdefault("anthropic_version", "bedrock-2023-05-31")
            body_data.setdefault("max_tokens", DEFAULT_ANTHROPIC_INVOKE_MODEL_MAX_TOKENS)

        # The CountTokens API expects invokeModel.body as a base64-encoded blob
        encoded_body: Final = base64.b64encode(json.dumps(body_data).encode()).decode()
        return {"input": {"invokeModel": {"body": encoded_body}}}

    def get_bedrock_count_tokens_endpoint(
        self,
        model: str,
        aws_region_name: str,
        api_base: str | None = None,
        aws_bedrock_runtime_endpoint: str | None = None,
    ) -> str:
        """
        Construct the AWS Bedrock CountTokens API endpoint using existing LiteLLM functions.

        Args:
            model: The resolved model ID from router lookup
            aws_region_name: AWS region (e.g., "eu-west-1")
            api_base: Optional custom API base URL (takes highest priority)
            aws_bedrock_runtime_endpoint: Optional custom Bedrock runtime endpoint

        Returns:
            Complete endpoint URL for CountTokens API
        """
        # Use existing LiteLLM function to get the base model ID (removes region prefix)
        model_id = get_bedrock_base_model(model)

        # Remove bedrock/ prefix if present
        model_id = model_id.removeprefix("bedrock/")  # Remove "bedrock/" prefix
        encoded_model_id: Final = self.encode_model_id(model_id=model_id)

        base_url, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=aws_region_name,
        )
        endpoint: Final = f"{base_url}/model/{encoded_model_id}/count-tokens"

        return endpoint

    def transform_bedrock_response_to_anthropic(
        self, bedrock_response: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """
        Transform Bedrock CountTokens response to Anthropic format.

        Input (Bedrock response):
        {
            "inputTokens": 123
        }

        Output (Anthropic format):
        {
            "input_tokens": 123
        }
        """
        input_tokens: Final = bedrock_response.get("inputTokens", 0)

        return {"input_tokens": input_tokens}

    def validate_count_tokens_request(self, request_data: Mapping[str, JsonValue]) -> None:
        """
        Validate the incoming count tokens request.
        Supports both Converse and InvokeModel input formats.

        Args:
            request_data: The request payload

        Raises:
            ValueError: If the request is invalid
        """
        if not request_data.get("model"):
            raise ValueError("model parameter is required")

        input_type: Final = self._detect_input_type(request_data)

        if input_type == "converse":
            # Validate Converse format (messages-based)
            messages: Final = request_data.get("messages", [])
            if not messages:
                raise ValueError("messages parameter is required for Converse input")

            if not isinstance(messages, list):
                raise ValueError("messages must be a list")

            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(f"Message {i} must be a dictionary")

                if "role" not in message:
                    raise ValueError(f"Message {i} must have a 'role' field")

                if "content" not in message:
                    raise ValueError(f"Message {i} must have a 'content' field")
        else:
            # For InvokeModel format, we need at least some content to count tokens
            # The content structure varies by model, so we do minimal validation
            if len(request_data) <= 1:  # Only has 'model' field
                raise ValueError("Request must contain content to count tokens")
