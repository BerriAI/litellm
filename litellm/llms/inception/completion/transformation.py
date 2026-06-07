"""
Inception fill-in-the-middle (FIM) completions.

Inception's FIM endpoint is OpenAI text-completion compatible: it takes a
`prompt` (prefix) plus an optional `suffix` and returns standard
`choices[].text`. It is served at `/v1/fim/completions` rather than
`/v1/completions`, so routing points the OpenAI client at the `/v1/fim` base
(see the `text-completion-inception` branch in `main.py`).
"""

from typing import List

from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig


class InceptionTextCompletionConfig(OpenAITextCompletionConfig):
    def get_supported_openai_params(self, model: str) -> List:
        return [
            "suffix",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "stream",
            "stream_options",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_params:
                optional_params[param] = value
        return optional_params
