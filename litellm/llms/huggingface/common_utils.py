from typing import Dict, Literal, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException


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


def _validate_environment(
    headers: Dict,
    api_key: Optional[str] = None,
) -> Dict:
    default_headers = {
        "content-type": "application/json",
    }
    if api_key is not None:
        default_headers["Authorization"] = f"Bearer {api_key}"

    headers = {**headers, **default_headers}
    return headers
