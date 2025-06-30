"""
Translate from OpenAI's `/v1/chat/completions` to DigitalOcean AI Agent's `/v1/chat/completions`
"""

from typing import List, Optional, Tuple, Union, Dict

from pydantic import BaseModel

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class DigitalOceanConfig(OpenAILikeChatConfig):

    # DigitalOcean-specific parameters:
    stream_options: Optional[Dict] = None
    kb_filters: Optional[List[Dict]] = None
    filter_kb_content_by_query_metadata: Optional[bool] = None
    instruction_override: Optional[str] = None
    include_functions_info: Optional[bool] = None
    include_retrieval_info: Optional[bool] = None
    include_guardrails_info: Optional[bool] = None
    provide_citations: Optional[bool] = None

    def __init__(
        self,
        frequency_penalty: Optional[float] = None,
        function_call: Optional[Union[str, Dict]] = None,
        functions: Optional[List] = None,
        logit_bias: Optional[Dict] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,  # Additional field; interchangeable with max_tokens per spec
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        retrieval_method: Optional[str] = None,  # expected values: "rewrite", "step_back", "sub_queries", "none"
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Dict] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, Dict]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        base_params = super().get_supported_openai_params(model)
        try:
            base_params.remove("max_retries")
        except ValueError:
            pass
        return base_params

    def validate_environment(self,
                             headers: dict,
                             model: str,
                             messages: List[AllMessageValues],
                             optional_params: dict,
                             litellm_params: dict,
                             api_key: Optional[str] = None,
                             api_base: Optional[str] = None):
        api_key = api_key or get_secret_str("DIGITALOCEAN_API_KEY")
        if api_key is None:
            raise ValueError("DigitalOcean API key not found")
        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers


    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:

        complete_url = f"{api_base}/api/v1/chat/completions"
        return complete_url

    def _transform_messages(self, messages: List[AllMessageValues], model: str) -> List:
        for idx, message in enumerate(messages):
            """
            1. Don't pass 'null' function_call assistant message to groq - https://github.com/BerriAI/litellm/issues/5839
            """
            if isinstance(message, BaseModel):
                _message = message.model_dump()
            else:
                _message = message
            assistant_message = _message.get("role") == "assistant"
            if assistant_message:
                new_message = ChatCompletionAssistantMessage(role="assistant")
                for k, v in _message.items():
                    if v is not None:
                        new_message[k] = v  # type: ignore
                messages[idx] = new_message

        return messages

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("DIGITALOCEAN_AGENT_ENDPOINT")
            or get_secret_str("DO_AGENT_ENDPOINT")
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DO_API_KEY") or get_secret_str("DIGITALOCEAN_API_KEY")
        return api_base, dynamic_api_key



    def _create_json_tool_call_for_response_format(
        self,
        json_schema: dict,
    ):
        """
        Handles creating a tool call for getting responses in JSON format.

        Args:
            json_schema (Optional[dict]): The JSON schema the response should be in

        Returns:
            AnthropicMessagesTool: The tool call to send to Anthropic API to get responses in JSON format
        """
        return ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name="json_tool_call",
                parameters=json_schema,
            ),
        )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
        replace_max_completion_tokens_with_max_tokens: bool = False,  # groq supports max_completion_tokens
    ) -> dict:
        _response_format = non_default_params.get("response_format")
        if _response_format is not None and isinstance(_response_format, dict):
            json_schema: Optional[dict] = None
            if "response_schema" in _response_format:
                json_schema = _response_format["response_schema"]
            elif "json_schema" in _response_format:
                json_schema = _response_format["json_schema"]["schema"]
            """
            When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
            - You usually want to provide a single tool
            - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
            - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
            """
            if json_schema is not None:
                _tool_choice = {
                    "type": "function",
                    "function": {"name": "json_tool_call"},
                }
                _tool = self._create_json_tool_call_for_response_format(
                    json_schema=json_schema,
                )
                optional_params["tools"] = [_tool]
                optional_params["tool_choice"] = _tool_choice
                optional_params["json_mode"] = True
                non_default_params.pop(
                    "response_format", None
                )  # only remove if it's a json_schema - handled via using groq's tool calling params.
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        return optional_params
