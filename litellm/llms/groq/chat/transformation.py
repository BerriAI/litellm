"""
Translate from OpenAI's `/v1/chat/completions` to Groq's `/v1/chat/completions`
"""

import types
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel

import litellm
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionAssistantMessage

from ...OpenAI.chat.gpt_transformation import OpenAIGPTConfig


class GroqChatConfig(OpenAIGPTConfig):

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
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def _transform_messages(self, messages: List[AllMessageValues]) -> List:
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
