"""
Calling logic for Nebius chat completions
"""

from typing import Optional

from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.llms.nebius import NebiusAIEndpoint
from litellm.utils import ModelResponse


def nebius_chat_completions(
    model: str,
    messages: list,
    api_key: str,
    api_base: str,
    logging_obj: Logging = None,
    model_response: Optional[ModelResponse] = None,
    optional_params: Optional[dict] = None,
    litellm_params: Optional[dict] = None,
    logger_fn=None,
    stream: bool = False,
) -> ModelResponse:
    """
    Handle chat completions for Nebius models
    """
    # Initialize OpenAILikeChatHandler for Nebius chat - using the OpenAI-compatible endpoint
    handler = OpenAILikeChatHandler(
        custom_llm_provider="nebius",
        api_base=api_base,
        api_key=api_key
    )

    # Use the handler to make the API call
    response = handler.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        api_key=api_key,
        logging_obj=logging_obj,
        model_response=model_response,
        print_verbose=logger_fn,
        optional_params=optional_params or {},
        litellm_params=litellm_params or {},
        stream=stream,
    )
    
    return response 