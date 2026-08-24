"""
Transformation utilities for bridging Interactions API to Responses API.

This module handles transforming between:
- Interactions API format (Google's format with Step[]/Turn[], system_instruction, etc.)
- Responses API format (OpenAI's format with input[], instructions, etc.)
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, cast

from pydantic import BaseModel

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

_STEP_TYPE_ROLES: Final = MappingProxyType({"user_input": "user", "model_output": "assistant"})


class LiteLLMResponsesInteractionsConfig:
    """Configuration class for transforming between Interactions API and Responses API."""

    @staticmethod
    def transform_interactions_request_to_responses_request(
        model: str,
        input: InteractionInput | None,
        optional_params: InteractionsAPIOptionalRequestParams,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Transform an Interactions API request to a Responses API request.

        Key transformations:
        - system_instruction -> instructions
        - input (string | Turn[]) -> input (ResponseInputParam)
        - tools -> tools (similar format)
        - generation_config -> temperature, top_p, etc.
        """
        responses_request: Final[dict[str, Any]] = {
            "model": model,
        }

        # Transform input
        if input is not None:
            responses_request["input"] = (
                LiteLLMResponsesInteractionsConfig._transform_interactions_input_to_responses_input(input)
            )

        # Transform system_instruction -> instructions
        if optional_params.get("system_instruction"):
            responses_request["instructions"] = optional_params["system_instruction"]

        # Transform tools (similar format, pass through for now)
        if optional_params.get("tools"):
            responses_request["tools"] = optional_params["tools"]

        # Transform generation_config to temperature, top_p, etc.
        generation_config: Final = optional_params.get("generation_config")
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
        passthrough_params: Final = ["stream", "store", "metadata", "user"]
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
        - Step[]: [{"type": "user_input", "content": [...]}, {"type": "model_output", "content": [...]}]
        - Turn[] (legacy): [{"role": "user", "content": [...]}]
        - Content | Content[]: one user message worth of content parts

        Responses API input is:
        - string: "Hello"
        - Message[]: [{"role": "user", "content": [{"type": "input_text", ...}]}]
        """
        if isinstance(input, str):
            return cast(ResponseInputParam, input)

        if isinstance(input, list):
            transformed: Final = (
                [
                    LiteLLMResponsesInteractionsConfig._transform_history_item(item)
                    for item in input
                    if LiteLLMResponsesInteractionsConfig._is_history_item(item)
                ]
                if any(LiteLLMResponsesInteractionsConfig._is_history_item(item) for item in input)
                else [
                    {
                        "role": "user",
                        "content": LiteLLMResponsesInteractionsConfig._transform_content_array(input, "user"),
                    }
                ]
            )
            return cast(ResponseInputParam, transformed)

        if isinstance(input, dict):
            raw_content: Final = input.get("content")
            content_items: Final = raw_content if isinstance(raw_content, list) else [input]
            return cast(
                ResponseInputParam,
                [
                    {
                        "role": "user",
                        "content": LiteLLMResponsesInteractionsConfig._transform_content_array(content_items, "user"),
                    }
                ],
            )

        return cast(ResponseInputParam, str(input))

    @staticmethod
    def _is_history_item(item: object) -> bool:
        if isinstance(item, Turn):
            return True
        return isinstance(item, dict) and ("role" in item or item.get("type") in _STEP_TYPE_ROLES)

    @staticmethod
    def _transform_history_item(item: object) -> Mapping[str, object]:
        raw: Final = item.model_dump(exclude_none=True) if isinstance(item, Turn) else item
        fields: Final = raw if isinstance(raw, Mapping) else {}
        role: Final = LiteLLMResponsesInteractionsConfig._responses_role(fields)
        raw_content: Final = fields.get("content")
        content_items: Final = (
            raw_content if isinstance(raw_content, list) else [] if raw_content is None else [raw_content]
        )
        return {
            "role": role,
            "content": LiteLLMResponsesInteractionsConfig._transform_content_array(content_items, role),
        }

    @staticmethod
    def _responses_role(item: Mapping[str, object]) -> str:
        step_role: Final = _STEP_TYPE_ROLES.get(str(item.get("type", "")))
        if step_role is not None:
            return step_role
        raw_role: Final = str(item.get("role") or "user")
        return "assistant" if raw_role == "model" else raw_role

    @staticmethod
    def _transform_content_array(content: Sequence[object], role: str) -> Sequence[Mapping[str, object]]:
        """Transform Interactions API content parts to Responses API parts for the given role."""
        return [LiteLLMResponsesInteractionsConfig._transform_content_item(item, role) for item in content]

    @staticmethod
    def _transform_content_item(item: object, role: str) -> Mapping[str, object]:
        text_type: Final = "output_text" if role == "assistant" else "input_text"
        if isinstance(item, str):
            return {"type": text_type, "text": item}
        if isinstance(item, Mapping):
            if item.get("type") == "text":
                return {"type": text_type, "text": str(item.get("text", ""))}
            return item
        if isinstance(item, BaseModel):
            return LiteLLMResponsesInteractionsConfig._transform_content_item(item.model_dump(exclude_none=True), role)
        return {"type": text_type, "text": str(item)}

    @staticmethod
    def transform_responses_response_to_interactions_response(
        responses_response: ResponsesAPIResponse,
        model: str | None = None,
    ) -> InteractionsAPIResponse:
        """
        Transform a Responses API response to an Interactions API response.

        Key transformations:
        - Extract text from output[].content[].text
        - Convert created_at (int) to created (ISO string)
        - Map status
        - Extract usage
        """
        # Extract text from outputs and build both `outputs` (legacy) and `steps` (new schema).
        outputs: Final[list[dict[str, Any]]] = []
        steps: Final[list[dict[str, Any]]] = []
        if hasattr(responses_response, "output") and responses_response.output:
            for output_item in responses_response.output:
                # Use getattr with None default to safely access content
                content = getattr(output_item, "content", None)
                if content is not None:
                    content_items = content if isinstance(content, list) else [content]
                    model_output_contents: list[dict[str, Any]] = []
                    for content_item in content_items:
                        # Check if content_item has text attribute
                        text = getattr(content_item, "text", None)
                        if text is not None:
                            # Use independent dict instances so mutations to one
                            # of `outputs` / `steps` don't leak into the other.
                            outputs.append({"type": "text", "text": text})
                            model_output_contents.append({"type": "text", "text": text})
                        elif isinstance(content_item, dict) and content_item.get("type") == "text":
                            outputs.append({**content_item})
                            model_output_contents.append({**content_item})
                    if model_output_contents:
                        steps.append(
                            {
                                "type": "model_output",
                                "content": model_output_contents,
                            }
                        )

        # Convert created_at to ISO string
        created_at: Final = getattr(responses_response, "created_at", None)
        if isinstance(created_at, int):
            from datetime import datetime

            created = datetime.fromtimestamp(created_at).isoformat()
        elif created_at is not None and hasattr(created_at, "isoformat"):
            created = created_at.isoformat()
        else:
            created = None

        # Map status
        status: Final = getattr(responses_response, "status", "completed")
        if status == "completed":
            interactions_status = "completed"
        elif status == "in_progress":
            interactions_status = "in_progress"
        else:
            interactions_status = status

        # Build interactions response — populate both `outputs` (legacy schema) and
        # `steps` (new schema) so callers work regardless of which schema they expect.
        interactions_response_dict: Final[dict[str, Any]] = {
            "id": getattr(responses_response, "id", ""),
            "object": "interaction",
            "status": interactions_status,
            "outputs": outputs,
            "steps": steps,
            "model": model or getattr(responses_response, "model", ""),
            "created": created,
        }

        # Add usage if available
        # Map Responses API usage (input_tokens, output_tokens) to Interactions API spec format
        # (total_input_tokens, total_output_tokens)
        usage: Final = getattr(responses_response, "usage", None)
        if usage:
            interactions_response_dict["usage"] = {
                "total_input_tokens": getattr(usage, "input_tokens", 0),
                "total_output_tokens": getattr(usage, "output_tokens", 0),
            }

        # Add updated (same as created for now)
        interactions_response_dict["updated"] = created

        return InteractionsAPIResponse(**interactions_response_dict)
