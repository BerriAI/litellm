"""
Transforms OpenAI-style requests into TwelveLabs Pegasus 1.2 requests for Bedrock.

Reference:
https://docs.twelvelabs.io/docs/models/pegasus
"""

from typing import Any, Dict, List, Optional

from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import get_base64_str


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
        if isinstance(value, dict):
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

        for key in ("temperature", "maxOutputTokens", "responseFormat"):
            if key in optional_params:
                request_data[key] = optional_params.get(key)
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

