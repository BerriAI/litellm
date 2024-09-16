"""
Support for o1 model family 

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- modalities: image => drop param (if user opts in to dropping param)  
- role: system ==> translate to role 'user' 
- streaming => faked by LiteLLM 
- Tools, response_format =>  drop param (if user opts in to dropping param) 
- Logprobs => drop param (if user opts in to dropping param) 
"""

import types
from typing import Any, List, Optional, Union

import litellm
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage

from .gpt_transformation import OpenAIGPTConfig


class OpenAIO1Config(OpenAIGPTConfig):
    """
    Reference: https://platform.openai.com/docs/guides/reasoning
    """

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

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model

        """

        all_openai_params = super().get_supported_openai_params(model=model)
        non_supported_params = [
            "logprobs",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "function_call",
            "functions",
            "temperature",
            "top_p",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "response_format",
            "stop",
        ]

        return [
            param for param in all_openai_params if param not in non_supported_params
        ]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, model: str
    ):
        if "max_tokens" in non_default_params:
            optional_params["max_completion_tokens"] = non_default_params.pop(
                "max_tokens"
            )
        return super()._map_openai_params(non_default_params, optional_params, model)

    def is_model_o1_reasoning_model(self, model: str) -> bool:
        if model in litellm.open_ai_chat_completion_models and "o1" in model:
            return True
        return False

    def o1_prompt_factory(self, messages: List[AllMessageValues]):
        """
        Handles limitations of O-1 model family.
        - modalities: image => drop param (if user opts in to dropping param)
        - role: system ==> translate to role 'user'
        """

        for i, message in enumerate(messages):
            if message["role"] == "system":
                new_message = ChatCompletionUserMessage(
                    content=message["content"], role="user"
                )
                messages[i] = new_message  # Replace the old message with the new one

            if "content" in message and isinstance(message["content"], list):
                new_content = []
                for content_item in message["content"]:
                    if content_item.get("type") == "image_url":
                        if litellm.drop_params is not True:
                            raise ValueError(
                                "Image content is not supported for O-1 models. Set litellm.drop_param to True to drop image content."
                            )
                        # If drop_param is True, we simply don't add the image content to new_content
                    else:
                        new_content.append(content_item)
                message["content"] = new_content

        return messages
