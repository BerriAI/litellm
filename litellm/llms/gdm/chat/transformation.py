"""
GDM Chat Transformation - Uses OpenAI-compatible transformation
"""

from typing import Optional, Tuple

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class GDMChatConfig(OpenAILikeChatConfig):
    """
    GDM AI Chat Configuration

    GDM provides an OpenAI-compatible API at https://ai.gdm.se/api/v1
    """

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns the API base and API key for GDM.
        """
        import os

        api_base = api_base or os.environ.get("GDM_API_BASE") or "https://ai.gdm.se/api/v1"
        api_key = api_key or os.environ.get("GDM_API_KEY")

        return api_base, api_key
