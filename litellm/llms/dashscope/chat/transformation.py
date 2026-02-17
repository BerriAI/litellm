"""
Translates from OpenAI's `/v1/chat/completions` to DashScope's `/v1/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DashScopeChatConfig(OpenAIGPTConfig):
    """
    DashScope's OpenAI-compatible endpoint supports content in list format.
    Parent class transformation is used for proper handling of multimodal messages.
    No need to call handle_messages_with_content_list_to_str_conversion.
    """

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
