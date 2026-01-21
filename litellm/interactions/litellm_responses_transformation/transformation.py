"""
Transformation utilities for bridging Interactions API to Responses API.

This module handles transforming between:
- Interactions API format (Google's format with Turn[], system_instruction, etc.)
- Responses API format (OpenAI's format with input[], instructions, etc.)
"""

from typing import Any, Dict, List, Optional, cast

from litellm.types.interactions import (
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    Turn,
)
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIResponse,
)


class LiteLLMResponsesInteractionsConfig:
    """Configuration class for transforming between Interactions API and Responses API."""

    @staticmethod
    def transform_interactions_request_to_responses_request(
        model: str,
        input: Optional[InteractionInput],
        optional_params: InteractionsAPIOptionalRequestParams,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transform an Interactions API request to a Responses API request.
        
        Key transformations:
        - system_instruction -> instructions
        - input (string | Turn[]) -> input (ResponseInputParam)
        - tools -> tools (similar format)
        - generation_config -> temperature, top_p, etc.
        """
        responses_request: Dict[str, Any] = {
            "model": model,
        }
        
        # Transform input
        if input is not None:
            responses_request["input"] = (
                LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(
                    input
                )
            )
        
        # Transform system_instruction -> instructions
        if optional_params.get("system_instruction"):
            responses_request["instructions"] = optional_params["system_instruction"]
        
        # Transform tools (similar format, pass through for now)
        if optional_params.get("tools"):
            responses_request["tools"] = optional_params["tools"]
        
        # Transform generation_config to temperature, top_p, etc.
        generation_config = optional_params.get("generation_config")
        if generation_config:
            if isinstance(generation_config, dict):
                if "temperature" in generation_config:
                    responses_request["temperature"] = generation_config["temperature"]
                if "top_p" in generation_config:
                    responses_request["top_p"] = generation_config["top_p"]
                if "top_k" in generation_config:
                    # Responses API doesn't have top_k, skip it
                    pass
                if "max_output_tokens" in generation_config:
                    responses_request["max_output_tokens"] = generation_config["max_output_tokens"]
        
        # Pass through other optional params that match
        passthrough_params = ["stream", "store", "metadata", "user"]
        for param in passthrough_params:
            if param in optional_params and optional_params[param] is not None:
                responses_request[param] = optional_params[param]
        
        # Add any extra kwargs
        responses_request.update(kwargs)
        
        return responses_request

    @staticmethod
    def _transform_interactions_input_to_responses_input(
        input: InteractionInput,
    ) -> ResponseInputParam:
        """
        Transform Interactions API input to Responses API input format.
        
        Interactions API input can be:
        - string: "Hello"
        - Turn[]: [{"role": "user", "content": [...]}]
        - Content object
        
        Responses API input is:
        - string: "Hello"
        - Message[]: [{"role": "user", "content": [...]}]
        """
        if isinstance(input, str):
            # ResponseInputParam accepts str
            return cast(ResponseInputParam, input)
        
        if isinstance(input, list):
            # Turn[] format - convert to Responses API Message[] format
            messages = []
            for turn in input:
                if isinstance(turn, dict):
                    role = turn.get("role", "user")
                    content = turn.get("content", [])
                    
                    # Transform content array
                    transformed_content = (
                        LiteLLMResponsesInteractionsConfig._transform_content_array(content)
                    )
                    
                    messages.append({
                        "role": role,
                        "content": transformed_content,
                    })
                elif isinstance(turn, Turn):
                    # Pydantic model
                    role = turn.role if hasattr(turn, "role") else "user"
                    content = turn.content if hasattr(turn, "content") else []
                    
                    # Ensure content is a list for _transform_content_array
                    # Cast to List[Any] to handle various content types
                    if isinstance(content, list):
                        content_list: List[Any] = list(content)
                    elif content is not None:
                        content_list = [content]
                    else:
                        content_list = []
                    
                    transformed_content = (
                        LiteLLMResponsesInteractionsConfig._transform_content_array(content_list)
                    )
                    
                    messages.append({
                        "role": role,
                        "content": transformed_content,
                    })
            
            return cast(ResponseInputParam, messages)
        
        # Single content object - wrap in message
        if isinstance(input, dict):
            return cast(ResponseInputParam, [{
                "role": "user",
                "content": LiteLLMResponsesInteractionsConfig._transform_content_array(
                    input.get("content", []) if isinstance(input.get("content"), list) else [input]
                ),
            }])
        
        # Fallback: convert to string
        return cast(ResponseInputParam, str(input))

    @staticmethod
    def _transform_content_array(content: List[Any]) -> List[Dict[str, Any]]:
        """Transform Interactions API content array to Responses API format."""
        if not isinstance(content, list):
            # Single content item - wrap in array
            content = [content]
        
        transformed: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                # Already in dict format, pass through
                transformed.append(item)
            elif isinstance(item, str):
                # Plain string - wrap in text format
                transformed.append({"type": "text", "text": item})
            else:
                # Pydantic model or other - convert to dict
                if hasattr(item, "model_dump"):
                    dumped = item.model_dump()
                    if isinstance(dumped, dict):
                        transformed.append(dumped)
                    else:
                        # Fallback: wrap in text format
                        transformed.append({"type": "text", "text": str(dumped)})
                elif hasattr(item, "dict"):
                    dumped = item.dict()
                    if isinstance(dumped, dict):
                        transformed.append(dumped)
                    else:
                        # Fallback: wrap in text format
                        transformed.append({"type": "text", "text": str(dumped)})
                else:
                    # Fallback: wrap in text format
                    transformed.append({"type": "text", "text": str(item)})
        
        return transformed

    @staticmethod
    def transform_responses_response_to_interactions_response(
        responses_response: ResponsesAPIResponse,
        model: Optional[str] = None,
    ) -> InteractionsAPIResponse:
        """
        Transform a Responses API response to an Interactions API response.
        
        Key transformations:
        - Extract text from output[].content[].text
        - Convert created_at (int) to created (ISO string)
        - Map status
        - Extract usage
        """
        # Extract text from outputs
        outputs = []
        if hasattr(responses_response, "output") and responses_response.output:
            for output_item in responses_response.output:
                # Use getattr with None default to safely access content
                content = getattr(output_item, "content", None)
                if content is not None:
                    content_items = content if isinstance(content, list) else [content]
                    for content_item in content_items:
                        # Check if content_item has text attribute
                        text = getattr(content_item, "text", None)
                        if text is not None:
                            outputs.append({
                                "type": "text",
                                "text": text,
                            })
                        elif isinstance(content_item, dict) and content_item.get("type") == "text":
                            outputs.append(content_item)
        
        # Convert created_at to ISO string
        created_at = getattr(responses_response, "created_at", None)
        if isinstance(created_at, int):
            from datetime import datetime
            created = datetime.fromtimestamp(created_at).isoformat()
        elif created_at is not None and hasattr(created_at, "isoformat"):
            created = created_at.isoformat()
        else:
            created = None
        
        # Map status
        status = getattr(responses_response, "status", "completed")
        if status == "completed":
            interactions_status = "completed"
        elif status == "in_progress":
            interactions_status = "in_progress"
        else:
            interactions_status = status
        
        # Build interactions response
        interactions_response_dict: Dict[str, Any] = {
            "id": getattr(responses_response, "id", ""),
            "object": "interaction",
            "status": interactions_status,
            "outputs": outputs,
            "model": model or getattr(responses_response, "model", ""),
            "created": created,
        }
        
        # Add usage if available
        # Map Responses API usage (input_tokens, output_tokens) to Interactions API spec format
        # (total_input_tokens, total_output_tokens)
        usage = getattr(responses_response, "usage", None)
        if usage:
            interactions_response_dict["usage"] = {
                "total_input_tokens": getattr(usage, "input_tokens", 0),
                "total_output_tokens": getattr(usage, "output_tokens", 0),
            }
        
        # Add role
        interactions_response_dict["role"] = "model"
        
        # Add updated (same as created for now)
        interactions_response_dict["updated"] = created
        
        return InteractionsAPIResponse(**interactions_response_dict)

