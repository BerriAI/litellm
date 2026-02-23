"""
Transforms OpenAI-style requests into TwelveLabs Pegasus 1.2 requests for Bedrock.

Reference:
https://docs.twelvelabs.io/docs/models/pegasus
"""

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage
from litellm.utils import get_base64_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonTwelveLabsPegasusConfig(AmazonInvokeConfig, BaseConfig):
    """
    Handles transforming OpenAI-style requests into Bedrock InvokeModel requests for
    `twelvelabs.pegasus-1-2-v1:0`.

    Pegasus 1.2 requires an `inputPrompt` and a `mediaSource` that either references
    an S3 object or a base64-encoded clip. Optional OpenAI params (temperature,
    response_format, max_tokens) are translated to the TwelveLabs schema.
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "response_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param in {"max_tokens", "max_completion_tokens"}:
                optional_params["maxOutputTokens"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "response_format":
                optional_params["responseFormat"] = self._normalize_response_format(
                    value
                )
        return optional_params

    def _normalize_response_format(self, value: Any) -> Any:
        """Normalize response_format to TwelveLabs format.
        
        TwelveLabs expects:
        {
            "jsonSchema": {...}
        }
        
        But OpenAI format is:
        {
            "type": "json_schema",
            "json_schema": {
                "name": "...",
                "schema": {...}
            }
        }
        """
        if isinstance(value, dict):
            # If it has json_schema field, extract and transform it
            if "json_schema" in value:
                json_schema = value["json_schema"]
                # Extract the schema if nested
                if isinstance(json_schema, dict) and "schema" in json_schema:
                    return {"jsonSchema": json_schema["schema"]}
                # Otherwise use json_schema directly
                return {"jsonSchema": json_schema}
            # If it already has jsonSchema, return as is
            if "jsonSchema" in value:
                return value
            # Otherwise return the dict as is
            return value
        return type_to_response_format_param(response_format=value) or value

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        input_prompt = self._convert_messages_to_prompt(messages=messages)
        request_data: Dict[str, Any] = {"inputPrompt": input_prompt}

        media_source = self._build_media_source(optional_params)
        if media_source is not None:
            request_data["mediaSource"] = media_source

        # Handle temperature and maxOutputTokens
        for key in ("temperature", "maxOutputTokens"):
            if key in optional_params:
                request_data[key] = optional_params.get(key)
        
        # Handle responseFormat - transform to TwelveLabs format
        if "responseFormat" in optional_params:
            response_format = optional_params["responseFormat"]
            transformed_format = self._normalize_response_format(response_format)
            if transformed_format:
                request_data["responseFormat"] = transformed_format
        
        return request_data

    def _build_media_source(self, optional_params: dict) -> Optional[dict]:
        direct_source = optional_params.get("mediaSource") or optional_params.get(
            "media_source"
        )
        if isinstance(direct_source, dict):
            return direct_source

        base64_input = optional_params.get("video_base64") or optional_params.get(
            "base64_string"
        )
        if base64_input:
            return {"base64String": get_base64_str(base64_input)}

        s3_uri = (
            optional_params.get("video_s3_uri")
            or optional_params.get("s3_uri")
            or optional_params.get("media_source_s3_uri")
        )
        if s3_uri:
            s3_location = {"uri": s3_uri}
            bucket_owner = (
                optional_params.get("video_s3_bucket_owner")
                or optional_params.get("s3_bucket_owner")
                or optional_params.get("media_source_bucket_owner")
            )
            if bucket_owner:
                s3_location["bucketOwner"] = bucket_owner
            return {"s3Location": s3_location}
        return None

    def _convert_messages_to_prompt(self, messages: List[AllMessageValues]) -> str:
        prompt_parts: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                text_fragments = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type == "text":
                            text_fragments.append(item.get("text", ""))
                        elif item_type == "image_url":
                            text_fragments.append("<image>")
                        elif item_type == "video_url":
                            text_fragments.append("<video>")
                        elif item_type == "audio_url":
                            text_fragments.append("<audio>")
                    elif isinstance(item, str):
                        text_fragments.append(item)
                content = " ".join(text_fragments)
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(part for part in prompt_parts if part).strip()

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform TwelveLabs Pegasus response to LiteLLM format.
        
        TwelveLabs response format:
        {
            "message": "...",
            "finishReason": "stop" | "length"
        }
        
        LiteLLM format:
        ModelResponse with choices[0].message.content and finish_reason
        """
        try:
            completion_response = raw_response.json()
        except Exception as e:
            raise BedrockError(
                message=f"Error parsing response: {raw_response.text}, error: {str(e)}",
                status_code=raw_response.status_code,
            )
        
        verbose_logger.debug(
            "twelvelabs pegasus response: %s",
            json.dumps(completion_response, indent=4, default=str),
        )
        
        # Extract message content
        message_content = completion_response.get("message", "")
        
        # Extract finish reason and map to LiteLLM format
        finish_reason_raw = completion_response.get("finishReason", "stop")
        finish_reason = map_finish_reason(finish_reason_raw)
        
        # Set the response content
        try:
            if (
                message_content
                and hasattr(model_response.choices[0], "message")
                and getattr(model_response.choices[0].message, "tool_calls", None) is None
            ):
                model_response.choices[0].message.content = message_content  # type: ignore
                model_response.choices[0].finish_reason = finish_reason
            else:
                raise Exception("Unable to set message content")
        except Exception as e:
            raise BedrockError(
                message=f"Error setting response content: {str(e)}. Response: {completion_response}",
                status_code=raw_response.status_code,
            )
        
        # Calculate usage from headers
        bedrock_input_tokens = raw_response.headers.get(
            "x-amzn-bedrock-input-token-count", None
        )
        bedrock_output_tokens = raw_response.headers.get(
            "x-amzn-bedrock-output-token-count", None
        )
        
        prompt_tokens = int(
            bedrock_input_tokens or litellm.token_counter(messages=messages)
        )
        
        completion_tokens = int(
            bedrock_output_tokens
            or litellm.token_counter(
                text=model_response.choices[0].message.content,  # type: ignore
                count_response_tokens=True,
            )
        )
        
        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        
        return model_response

