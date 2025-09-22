"""
AWS Bedrock CountTokens API transformation logic.

This module handles the transformation of requests from Anthropic Messages API format
to AWS Bedrock's CountTokens API format and vice versa.
"""

from typing import Any, Dict, List

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockModelInfo


class BedrockCountTokensConfig(BaseAWSLLM):
    """
    Configuration and transformation logic for AWS Bedrock CountTokens API.

    AWS Bedrock CountTokens API Specification:
    - Endpoint: POST /model/{modelId}/count-tokens
    - Input formats: 'invokeModel' or 'converse'
    - Response: {"inputTokens": <number>}
    """

    def _detect_input_type(self, request_data: Dict[str, Any]) -> str:
        """
        Detect whether to use 'converse' or 'invokeModel' input format.

        Args:
            request_data: The original request data

        Returns:
            'converse' or 'invokeModel'
        """
        # If the request has messages in the expected Anthropic format, use converse
        if "messages" in request_data and isinstance(request_data["messages"], list):
            return "converse"

        # For raw text or other formats, use invokeModel
        # This handles cases where the input is prompt-based or already in raw Bedrock format
        return "invokeModel"

    def transform_anthropic_to_bedrock_count_tokens(
        self,
        request_data: Dict[str, Any],
    ) -> Dict[str, Any]:
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
                    "body": "{...raw model input...}"
                }
            }
        }
        """
        input_type = self._detect_input_type(request_data)

        if input_type == "converse":
            return self._transform_to_converse_format(request_data.get("messages", []))
        else:
            return self._transform_to_invoke_model_format(request_data)

    def _transform_to_converse_format(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Transform to Converse input format."""
        # Extract system messages if present
        system_messages = []
        user_messages = []

        for message in messages:
            if message.get("role") == "system":
                system_messages.append({"text": message.get("content", "")})
            else:
                # Transform message content to Bedrock format
                transformed_message: Dict[str, Any] = {"role": message.get("role"), "content": []}

                # Handle content - ensure it's in the correct array format
                content = message.get("content", "")
                if isinstance(content, str):
                    # String content -> convert to text block
                    transformed_message["content"].append({"text": content})
                elif isinstance(content, list):
                    # Already in blocks format - use as is
                    transformed_message["content"] = content

                user_messages.append(transformed_message)

        # Build the converse input format
        converse_input = {"messages": user_messages}

        # Add system messages if present
        if system_messages:
            converse_input["system"] = system_messages

        # Build the complete request
        return {"input": {"converse": converse_input}}

    def _transform_to_invoke_model_format(
        self, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform to InvokeModel input format."""
        import json

        # For InvokeModel, we need to provide the raw body that would be sent to the model
        # Remove the 'model' field from the body as it's not part of the model input
        body_data = {k: v for k, v in request_data.items() if k != "model"}

        return {"input": {"invokeModel": {"body": json.dumps(body_data)}}}

    def get_bedrock_count_tokens_endpoint(
        self, model: str, aws_region_name: str
    ) -> str:
        """
        Construct the AWS Bedrock CountTokens API endpoint using existing LiteLLM functions.

        Args:
            model: The resolved model ID from router lookup
            aws_region_name: AWS region (e.g., "eu-west-1")

        Returns:
            Complete endpoint URL for CountTokens API
        """
        # Use existing LiteLLM function to get the base model ID (removes region prefix)
        model_id = BedrockModelInfo.get_base_model(model)

        # Remove bedrock/ prefix if present
        if model_id.startswith("bedrock/"):
            model_id = model_id[8:]  # Remove "bedrock/" prefix

        base_url = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"
        endpoint = f"{base_url}/model/{model_id}/count-tokens"

        return endpoint

    def transform_bedrock_response_to_anthropic(
        self, bedrock_response: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        input_tokens = bedrock_response.get("inputTokens", 0)

        return {"input_tokens": input_tokens}

    def validate_count_tokens_request(self, request_data: Dict[str, Any]) -> None:
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

        input_type = self._detect_input_type(request_data)

        if input_type == "converse":
            # Validate Converse format (messages-based)
            messages = request_data.get("messages", [])
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
