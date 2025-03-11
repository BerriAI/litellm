from typing import Optional, Union

import litellm
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIRequestParams,
)
from litellm.types.router import GenericLiteLLMParams


class OpenAIResponsesAPIConfig(BaseResponsesAPIConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        All OpenAI Responses API params are supported
        """
        return [
            "input",
            "model",
            "include",
            "instructions",
            "max_output_tokens",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "reasoning",
            "store",
            "stream",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "truncation",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]

    def map_openai_params(
        self,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> ResponsesAPIOptionalRequestParams:

        return ResponsesAPIOptionalRequestParams(
            include=optional_params.get("include"),
            instructions=optional_params.get("instructions"),
            max_output_tokens=optional_params.get("max_output_tokens"),
            metadata=optional_params.get("metadata"),
            parallel_tool_calls=optional_params.get("parallel_tool_calls"),
            previous_response_id=optional_params.get("previous_response_id"),
            reasoning=optional_params.get("reasoning"),
            store=optional_params.get("store"),
            stream=optional_params.get("stream"),
            temperature=optional_params.get("temperature"),
            text=optional_params.get("text"),
            tool_choice=optional_params.get("tool_choice"),
            tools=optional_params.get("tools"),
            top_p=optional_params.get("top_p"),
            truncation=optional_params.get("truncation"),
            user=optional_params.get("user"),
            extra_headers=optional_params.get("extra_headers"),
            extra_query=optional_params.get("extra_query"),
            extra_body=optional_params.get("extra_body"),
            timeout=optional_params.get("timeout"),
        )

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: ResponsesAPIOptionalRequestParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> ResponsesAPIRequestParams:
        return ResponsesAPIRequestParams(
            model=model, input=input, **response_api_optional_request_params
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the endpoint for OpenAI responses API
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"
