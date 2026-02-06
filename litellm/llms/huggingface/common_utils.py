import os
from functools import lru_cache
from typing import Literal, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException

HF_HUB_URL = "https://huggingface.co"


class HuggingFaceError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[httpx.Headers, dict]] = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
        )


hf_tasks = Literal[
    "text-generation-inference",
    "conversational",
    "text-classification",
    "text-generation",
]

hf_task_list = [
    "text-generation-inference",
    "conversational",
    "text-classification",
    "text-generation",
]


def output_parser(generated_text: str):
    """
    Parse the output text to remove any special characters. In our current approach we just check for ChatML tokens.

    Initial issue that prompted this - https://github.com/BerriAI/litellm/issues/763
    """
    chat_template_tokens = ["<|assistant|>", "<|system|>", "<|user|>", "<s>", "</s>"]
    for token in chat_template_tokens:
        if generated_text.strip().startswith(token):
            generated_text = generated_text.replace(token, "", 1)
        if generated_text.endswith(token):
            generated_text = generated_text[::-1].replace(token[::-1], "", 1)[::-1]
    return generated_text


@lru_cache(maxsize=128)
def _fetch_inference_provider_mapping(model: str) -> dict:
    """
    Fetch provider mappings for a model from the Hugging Face Hub.

    Args:
        model: The model identifier (e.g., 'meta-llama/Llama-2-7b')

    Returns:
        dict: The inference provider mapping for the model

    Raises:
        ValueError: If no provider mapping is found
        HuggingFaceError: If the API request fails
    """
    headers = {"Accept": "application/json"}
    if os.getenv("HUGGINGFACE_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"

    path = f"{HF_HUB_URL}/api/models/{model}"
    params = {"expand": ["inferenceProviderMapping"]}

    try:
        response = httpx.get(path, headers=headers, params=params)
        response.raise_for_status()
        provider_mapping = response.json().get("inferenceProviderMapping")

        if provider_mapping is None:
            raise ValueError(f"No provider mapping found for model {model}")

        return provider_mapping
    except httpx.HTTPError as e:
        if hasattr(e, "response"):
            status_code = getattr(e.response, "status_code", 500)
            headers = getattr(e.response, "headers", {})
        else:
            status_code = 500
            headers = {}
        raise HuggingFaceError(
            message=f"Failed to fetch provider mapping: {str(e)}",
            status_code=status_code,
            headers=headers,
        )
