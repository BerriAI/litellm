"""
Translates calls from OpenAI's `/v1/completions` endpoint to TogetherAI's `/v1/completions` endpoint.

Calls done in OpenAI/openai.py as TogetherAI is openai-compatible.

Docs: https://docs.together.ai/reference/completions-1
"""

from typing import List

from litellm.types.llms.openai import AllMessageValues

from ...OpenAI.openai import OpenAITextCompletionConfig


class TogetherAITextCompletionConfig(OpenAITextCompletionConfig):
    def _transform_prompt(self, messages: List[AllMessageValues]) -> str:
        """
        TogetherAI expects a string prompt.
        """
        together_prompt = super()._transform_prompt(messages)
        if isinstance(together_prompt, list) and len(together_prompt) == 1:
            together_prompt = together_prompt[0]
        elif isinstance(together_prompt, list):
            raise ValueError("TogetherAI does not support multiple prompts.")

        return together_prompt
