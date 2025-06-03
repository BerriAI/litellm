"""
Support for CentML's `/v1/completions` endpoint. 

Calls done in OpenAI/openai.py as CentML is openai-compatible.

Docs: https://docs.centml.ai/reference/completions
"""

from typing import List, Union, cast

from litellm.llms.openai.completion.utils import is_tokens_or_list_of_tokens
from litellm.types.llms.openai import (
    AllMessageValues,
    AllPromptValues,
    OpenAITextCompletionUserMessage,
)

from ...openai.completion.transformation import OpenAITextCompletionConfig
from ...openai.completion.utils import _transform_prompt


class CentmlTextCompletionConfig(OpenAITextCompletionConfig):
    def _transform_prompt(
        self,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
    ) -> AllPromptValues:
        """
        CentML expects a string prompt.
        """
        initial_prompt: AllPromptValues = _transform_prompt(messages)
        ## CENTML SPECIFIC VALIDATION ##
        if isinstance(initial_prompt, list) and is_tokens_or_list_of_tokens(
            value=initial_prompt
        ):
            raise ValueError("CentML does not support integers as input")
        if (
            isinstance(initial_prompt, list)
            and len(initial_prompt) == 1
            and isinstance(initial_prompt[0], str)
        ):
            centml_prompt = initial_prompt[0]
        elif isinstance(initial_prompt, list):
            raise ValueError("CentML does not support multiple prompts.")
        else:
            centml_prompt = cast(str, initial_prompt)

        return centml_prompt

    def transform_text_completion_request(
        self,
        model: str,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        prompt = self._transform_prompt(messages)
        return {
            "model": model,
            "prompt": prompt,
            **optional_params,
        } 