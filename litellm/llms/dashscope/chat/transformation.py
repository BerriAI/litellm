"""
Translates from OpenAI's `/v1/chat/completions` to DashScope's `/v1/chat/completions`
"""

from typing import TYPE_CHECKING, Any, Coroutine, List, Literal, Optional, Tuple, Union

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.types.llms.openai import ChatCompletionToolParam


class DashScopeChatConfig(OpenAIGPTConfig):
    """
    DashScope configuration.

    DashScope supports content in list format with cache_control metadata.
    See: https://github.com/BerriAI/litellm/issues/18165
    """

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("DASHSCOPE_API_BASE")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DASHSCOPE_API_KEY")
        return api_base, dynamic_api_key

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
        If api_base is not provided, use the default DashScope /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
