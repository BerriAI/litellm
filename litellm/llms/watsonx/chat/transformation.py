"""
Translation from OpenAI's `/chat/completions` endpoint to IBM WatsonX's `/text/chat` endpoint.

Docs: https://cloud.ibm.com/apidocs/watsonx-ai#text-chat
"""

from typing import List, Optional, Tuple, Union

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.watsonx import WatsonXAIEndpoint

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import IBMWatsonXMixin


class IBMWatsonXChatConfig(IBMWatsonXMixin, OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> List:
        return [
            "temperature",  # equivalent to temperature
            "max_tokens",  # equivalent to max_new_tokens
            "top_p",  # equivalent to top_p
            "frequency_penalty",  # equivalent to repetition_penalty
            "stop",  # equivalent to stop_sequences
            "seed",  # equivalent to random_seed
            "stream",  # equivalent to stream
            "tools",
            "tool_choice",  # equivalent to tool_choice + tool_choice_options
            "logprobs",
            "top_logprobs",
            "n",
            "presence_penalty",
            "response_format",
        ]

    def is_tool_choice_option(self, tool_choice: Optional[Union[str, dict]]) -> bool:
        if tool_choice is None:
            return False
        if isinstance(tool_choice, str):
            return tool_choice in ["auto", "none", "required"]
        return False

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        ## TOOLS ##
        _tools = non_default_params.pop("tools", None)
        if _tools is not None:
            # remove 'additionalProperties' from tools
            _tools = _remove_additional_properties(_tools)
            # remove 'strict' from tools
            _tools = _remove_strict_from_schema(_tools)
        if _tools is not None:
            non_default_params["tools"] = _tools

        ## TOOL CHOICE ##

        _tool_choice = non_default_params.pop("tool_choice", None)
        if self.is_tool_choice_option(_tool_choice):
            optional_params["tool_choice_options"] = _tool_choice
        elif _tool_choice is not None:
            optional_params["tool_choice"] = _tool_choice
        return super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("HOSTED_VLLM_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("HOSTED_VLLM_API_KEY") or ""
        )  # vllm does not require an api key
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        url = self._get_base_url(api_base=api_base)
        if model.startswith("deployment/"):
            deployment_id = "/".join(model.split("/")[1:])
            endpoint = (
                WatsonXAIEndpoint.DEPLOYMENT_CHAT_STREAM.value
                if stream
                else WatsonXAIEndpoint.DEPLOYMENT_CHAT.value
            )
            endpoint = endpoint.format(deployment_id=deployment_id)
        else:
            endpoint = (
                WatsonXAIEndpoint.CHAT_STREAM.value
                if stream
                else WatsonXAIEndpoint.CHAT.value
            )
        url = url.rstrip("/") + endpoint

        ## add api version
        url = self._add_api_version_to_url(
            url=url, api_version=optional_params.pop("api_version", None)
        )
        return url
