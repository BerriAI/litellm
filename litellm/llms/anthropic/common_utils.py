"""
This file contains common utils for anthropic calls.
"""

from typing import Dict, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
)
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.anthropic import AllAnthropicToolsValues
from litellm.types.llms.openai import AllMessageValues


class AnthropicError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message,
        headers: Optional[httpx.Headers] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AnthropicModelInfo(BaseLLMModelInfo):
    def is_cache_control_set(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"cache_control": ..} in message content block

        Used to check if anthropic prompt caching headers need to be set.
        """
        for message in messages:
            if message.get("cache_control", None) is not None:
                return True
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                for content in _message_content:
                    if "cache_control" in content:
                        return True

        return False

    def is_file_id_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"source": {"type": "file", "file_id": ..}} in message content block
        """
        file_ids = get_file_ids_from_messages(messages)
        return len(file_ids) > 0

    def is_computer_tool_used(
        self, tools: Optional[List[AllAnthropicToolsValues]]
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:
            if "type" in tool and tool["type"].startswith("computer_"):
                return True
        return False

    def is_pdf_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Set to true if media passed into messages.

        """
        for message in messages:
            if (
                "content" in message
                and message["content"] is not None
                and isinstance(message["content"], list)
            ):
                for content in message["content"]:
                    if "type" in content and content["type"] != "text":
                        return True
        return False

    def _get_user_anthropic_beta_headers(
        self, anthropic_beta_header: Optional[str]
    ) -> Optional[List[str]]:
        if anthropic_beta_header is None:
            return None
        return anthropic_beta_header.split(",")

    def get_anthropic_headers(
        self,
        api_key: str,
        anthropic_version: Optional[str] = None,
        computer_tool_used: bool = False,
        prompt_caching_set: bool = False,
        pdf_used: bool = False,
        file_id_used: bool = False,
        is_vertex_request: bool = False,
        user_anthropic_beta_headers: Optional[List[str]] = None,
    ) -> dict:
        betas = set()
        if prompt_caching_set:
            betas.add("prompt-caching-2024-07-31")
        if computer_tool_used:
            betas.add("computer-use-2024-10-22")
        # if pdf_used:
        #     betas.add("pdfs-2024-09-25")
        if file_id_used:
            betas.add("files-api-2025-04-14")
            betas.add("code-execution-2025-05-22")
        headers = {
            "anthropic-version": anthropic_version or "2023-06-01",
            "x-api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }

        if user_anthropic_beta_headers is not None:
            betas.update(user_anthropic_beta_headers)

        # Don't send any beta headers to Vertex, Vertex has failed requests when they are sent
        if is_vertex_request is True:
            pass
        elif len(betas) > 0:
            headers["anthropic-beta"] = ",".join(betas)

        return headers

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        if api_key is None:
            raise litellm.AuthenticationError(
                message="Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` in your environment vars",
                llm_provider="anthropic",
                model=model,
            )

        tools = optional_params.get("tools")
        prompt_caching_set = self.is_cache_control_set(messages=messages)
        computer_tool_used = self.is_computer_tool_used(tools=tools)
        pdf_used = self.is_pdf_used(messages=messages)
        file_id_used = self.is_file_id_used(messages=messages)
        user_anthropic_beta_headers = self._get_user_anthropic_beta_headers(
            anthropic_beta_header=headers.get("anthropic-beta")
        )
        anthropic_headers = self.get_anthropic_headers(
            computer_tool_used=computer_tool_used,
            prompt_caching_set=prompt_caching_set,
            pdf_used=pdf_used,
            api_key=api_key,
            file_id_used=file_id_used,
            is_vertex_request=optional_params.get("is_vertex_request", False),
            user_anthropic_beta_headers=user_anthropic_beta_headers,
        )

        headers = {**headers, **anthropic_headers}

        return headers

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or get_secret_str("ANTHROPIC_API_BASE")
            or "https://api.anthropic.com"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("ANTHROPIC_API_KEY")

    @staticmethod
    def get_base_model(model: Optional[str] = None) -> Optional[str]:
        return model.replace("anthropic/", "") if model else None

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = AnthropicModelInfo.get_api_base(api_base)
        api_key = AnthropicModelInfo.get_api_key(api_key)
        if api_base is None or api_key is None:
            raise ValueError(
                "ANTHROPIC_API_BASE or ANTHROPIC_API_KEY is not set. Please set the environment variable, to query Anthropic's `/models` endpoint."
            )
        response = litellm.module_level_client.get(
            url=f"{api_base}/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from Anthropic. Status code: {response.status_code}, Response: {response.text}"
            )

        models = response.json()["data"]

        litellm_model_names = []
        for model in models:
            stripped_model_name = model["id"]
            litellm_model_name = "anthropic/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names


def process_anthropic_headers(headers: Union[httpx.Headers, dict]) -> dict:
    openai_headers = {}
    if "anthropic-ratelimit-requests-limit" in headers:
        openai_headers["x-ratelimit-limit-requests"] = headers[
            "anthropic-ratelimit-requests-limit"
        ]
    if "anthropic-ratelimit-requests-remaining" in headers:
        openai_headers["x-ratelimit-remaining-requests"] = headers[
            "anthropic-ratelimit-requests-remaining"
        ]
    if "anthropic-ratelimit-tokens-limit" in headers:
        openai_headers["x-ratelimit-limit-tokens"] = headers[
            "anthropic-ratelimit-tokens-limit"
        ]
    if "anthropic-ratelimit-tokens-remaining" in headers:
        openai_headers["x-ratelimit-remaining-tokens"] = headers[
            "anthropic-ratelimit-tokens-remaining"
        ]

    llm_response_headers = {
        "{}-{}".format("llm_provider", k): v for k, v in headers.items()
    }

    additional_headers = {**llm_response_headers, **openai_headers}
    return additional_headers
