"""
Translates calls from OpenAI's `/v1/completions` endpoint to TogetherAI's `/v1/completions` endpoint.

Calls done in OpenAI/openai.py as TogetherAI is openai-compatible.

Docs: https://docs.together.ai/reference/completions-1
"""

from typing import List, Union, cast

from litellm.llms.OpenAI.completion.utils import is_tokens_or_list_of_tokens
from litellm.types.llms.openai import (
    AllMessageValues,
    AllPromptValues,
    OpenAITextCompletionUserMessage,
)

from ...OpenAI.openai import OpenAITextCompletionConfig


class TogetherAITextCompletionConfig(OpenAITextCompletionConfig):
    def _transform_prompt(
        self,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
    ) -> AllPromptValues:
        """
        TogetherAI expects a string prompt.
        """
        initial_prompt: AllPromptValues = super()._transform_prompt(messages)
        ## TOGETHER AI SPECIFIC VALIDATION ##
        if isinstance(initial_prompt, list) and is_tokens_or_list_of_tokens(
            value=initial_prompt
        ):
            raise ValueError("TogetherAI does not support integers as input")
        if (
            isinstance(initial_prompt, list)
            and len(initial_prompt) == 1
            and isinstance(initial_prompt[0], str)
        ):
            together_prompt = initial_prompt[0]
        elif isinstance(initial_prompt, list):
            raise ValueError("TogetherAI does not support multiple prompts.")
        else:
            together_prompt = cast(str, initial_prompt)

        return together_prompt
