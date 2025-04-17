"""
Translate from OpenAI's `/v1/chat/completions` to Groq's `/v1/chat/completions`
"""

from typing import List, Optional, Tuple, Union

from pydantic import BaseModel

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class GroqChatConfig(OpenAILikeChatConfig):
    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
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
        # groq is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.groq.com/openai/v1
        api_base = (
            api_base
            or get_secret_str("GROQ_API_BASE")
            or "https://api.groq.com/openai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("GROQ_API_KEY")
        return api_base, dynamic_api_key

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Groq doesn't support 'response_format' while streaming
        """
        if optional_params.get("response_format") is not None:
            return True

        return False

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
        if self._should_fake_stream(non_default_params):
            optional_params["fake_stream"] = True
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
