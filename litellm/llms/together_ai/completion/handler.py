"""
Support for OpenAI's `/v1/completions` endpoint. 

Calls done in OpenAI/openai.py as TogetherAI is openai-compatible.

Docs: https://docs.together.ai/reference/completions-1
"""

from typing import Any, Callable, List, Optional, Union

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.openai import AllMessageValues, OpenAITextCompletionUserMessage
from litellm.utils import ModelResponse

from ...OpenAI.openai import OpenAITextCompletion
from .transformation import TogetherAITextCompletionConfig

together_ai_text_completion_global_config = TogetherAITextCompletionConfig()


class TogetherAITextCompletion(OpenAITextCompletion):

    def completion(
        self,
        model_response: ModelResponse,
        api_key: str,
        model: str,
        messages: Union[List[AllMessageValues], List[OpenAITextCompletionUserMessage]],
        timeout: float,
        logging_obj: Logging,
        optional_params: dict,
        print_verbose: Optional[Callable[..., Any]] = None,
        api_base: Optional[str] = None,
        acompletion: bool = False,
        litellm_params=None,
        logger_fn=None,
        client=None,
        organization: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        prompt = together_ai_text_completion_global_config._transform_prompt(messages)

        message = OpenAITextCompletionUserMessage(role="user", content=prompt)
        new_messages = [message]
        return super().completion(
            model_response=model_response,
            api_key=api_key,
            model=model,
            messages=new_messages,
            timeout=timeout,
            logging_obj=logging_obj,
            optional_params=optional_params,
            print_verbose=print_verbose,
            api_base=api_base,
            acompletion=acompletion,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            client=client,
            organization=organization,
            headers=headers,
        )
