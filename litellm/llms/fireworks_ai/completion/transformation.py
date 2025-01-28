from typing import List, Union

from litellm.types.llms.openai import AllMessageValues, OpenAITextCompletionUserMessage

from ...base_llm.completion.transformation import BaseTextCompletionConfig
from ...openai.completion.utils import _transform_prompt
from ..common_utils import FireworksAIMixin


class FireworksAITextCompletionConfig(FireworksAIMixin, BaseTextCompletionConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        See how LiteLLM supports Provider-specific parameters - https://docs.litellm.ai/docs/completion/provider_specific_params#proxy-usage
        """
        return [
            "max_tokens",
            "logprobs",
            "echo",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "stop",
            "response_format",
            "stream",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

    def transform_text_completion_request(
        self,
        model: str,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        prompt = _transform_prompt(messages=messages)

        if not model.startswith("accounts/"):
            model = f"accounts/fireworks/models/{model}"

        data = {
            "model": model,
            "prompt": prompt,
            **optional_params,
        }
        return data
