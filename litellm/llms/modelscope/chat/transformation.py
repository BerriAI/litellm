"""
Translates from OpenAI's `/v1/chat/completions` to ModelScope's `/v1/chat/completions`
"""

from typing import Any, Coroutine, Literal, Optional, Tuple, Union, cast, overload

from typing_extensions import override

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


def _has_non_text_content(message: AllMessageValues) -> bool:
    """Check if a message has non-text content items (e.g. image_url)."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(item.get("type") != "text" for item in content)


class ModelScopeChatConfig(OpenAIGPTConfig):
    DEFAULT_BASE_URL: str = "https://api-inference.modelscope.cn/v1"

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[list[AllMessageValues], Coroutine[Any, Any, list[AllMessageValues]]]:
        """
        Flatten text-only content lists to strings for ModelScope.

        Messages with non-text content (e.g. image_url for vision models)
        are kept as lists so the parent class can normalize them properly.
        """
        messages = [cast(AllMessageValues, {**m}) for m in messages]
        for message in messages:
            if _has_non_text_content(message):
                continue
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = "".join(item.get("text") or "" for item in content)

        if is_async:
            return super()._transform_messages(messages=messages, model=model, is_async=True)
        else:
            return super()._transform_messages(messages=messages, model=model, is_async=False)

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("MODELSCOPE_API_BASE") or self.DEFAULT_BASE_URL  # type: ignore
        dynamic_api_key = api_key or get_secret_str("MODELSCOPE_API_KEY")
        return api_base, dynamic_api_key

    @override
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default ModelScope /chat/completions endpoint.
        """
        if not api_base:
            api_base = self.DEFAULT_BASE_URL

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
