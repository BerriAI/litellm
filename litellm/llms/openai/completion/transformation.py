"""
Support for gpt model family 
"""

import types
from typing import List, Optional, Union, cast

import litellm
from litellm.llms.base_llm.transformation import BaseConfig
from litellm.types.llms.openai import (
    AllMessageValues,
    AllPromptValues,
    OpenAITextCompletionUserMessage,
)
from litellm.types.utils import Choices, Message, ModelResponse, TextCompletionResponse

from litellm.litellm_core_utils.prompt_templates.common_utils import convert_content_list_to_str
from ..chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import OpenAIError
from .utils import is_tokens_or_list_of_tokens


class OpenAITextCompletionConfig(OpenAIGPTConfig):
    """
    Reference: https://platform.openai.com/docs/api-reference/completions/create

    The class `OpenAITextCompletionConfig` provides configuration for the OpenAI's text completion API interface. Below are the parameters:

    - `best_of` (integer or null): This optional parameter generates server-side completions and returns the one with the highest log probability per token.

    - `echo` (boolean or null): This optional parameter will echo back the prompt in addition to the completion.

    - `frequency_penalty` (number or null): Defaults to 0. It is a numbers from -2.0 to 2.0, where positive values decrease the model's likelihood to repeat the same line.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `logprobs` (integer or null): This optional parameter includes the log probabilities on the most likely tokens as well as the chosen tokens.

    - `max_tokens` (integer or null): This optional parameter sets the maximum number of tokens to generate in the completion.

    - `n` (integer or null): This optional parameter sets how many completions to generate for each prompt.

    - `presence_penalty` (number or null): Defaults to 0 and can be between -2.0 and 2.0. Positive values increase the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `suffix` (string or null): Defines the suffix that comes after a completion of inserted text.

    - `temperature` (number or null): This optional parameter defines the sampling temperature to use.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """

    best_of: Optional[int] = None
    echo: Optional[bool] = None
    frequency_penalty: Optional[int] = None
    logit_bias: Optional[dict] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    suffix: Optional[str] = None

    def __init__(
        self,
        best_of: Optional[int] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[int] = None,
        logit_bias: Optional[dict] = None,
        logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def _transform_prompt(
        self,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
    ) -> AllPromptValues:
        if len(messages) == 1:  # base case
            message_content = messages[0].get("content")
            if (
                message_content
                and isinstance(message_content, list)
                and is_tokens_or_list_of_tokens(message_content)
            ):
                openai_prompt: AllPromptValues = cast(AllPromptValues, message_content)
            else:
                openai_prompt = ""
                content = convert_content_list_to_str(
                    cast(AllMessageValues, messages[0])
                )
                openai_prompt += content
        else:
            prompt_str_list: List[str] = []
            for m in messages:
                try:  # expect list of int/list of list of int to be a 1 message array only.
                    content = convert_content_list_to_str(cast(AllMessageValues, m))
                    prompt_str_list.append(content)
                except Exception as e:
                    raise e
            openai_prompt = prompt_str_list
        return openai_prompt

    def convert_to_chat_model_response_object(
        self,
        response_object: Optional[TextCompletionResponse] = None,
        model_response_object: Optional[ModelResponse] = None,
    ):
        try:
            ## RESPONSE OBJECT
            if response_object is None or model_response_object is None:
                raise ValueError("Error in response object format")
            choice_list = []
            for idx, choice in enumerate(response_object["choices"]):
                message = Message(
                    content=choice["text"],
                    role="assistant",
                )
                choice = Choices(
                    finish_reason=choice["finish_reason"], index=idx, message=message
                )
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object:
                setattr(model_response_object, "usage", response_object["usage"])

            if "id" in response_object:
                model_response_object.id = response_object["id"]

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            model_response_object._hidden_params["original_response"] = (
                response_object  # track original response, if users make a litellm.text_completion() request, we can return the original response
            )
            return model_response_object
        except Exception as e:
            raise e

    def get_supported_openai_params(self, model: str) -> List:
        return [
            "functions",
            "function_call",
            "temperature",
            "top_p",
            "n",
            "stream",
            "stream_options",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "max_retries",
            "logprobs",
            "top_logprobs",
            "extra_headers",
        ]
