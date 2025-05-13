from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LlamafileChatConfig(OpenAIGPTConfig):
    """LlamafileChatConfig is used to provide configuration for the LlamaFile's chat API."""

    @staticmethod
    def _resolve_api_key(api_key: Optional[str] = None) -> str:
        """Attempt to ensure that the API key is set, preferring the user-provided key
        over the secret manager key (``LLAMAFILE_API_KEY``).

        If both are None, a fake API key is returned.
        """
        return api_key or get_secret_str("LLAMAFILE_API_KEY") or "fake-api-key"  # llamafile does not require an API key

    @staticmethod
    def _resolve_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """Attempt to ensure that the API base is set, preferring the user-provided key
        over the secret manager key (``LLAMAFILE_API_BASE``).

        If both are None, a default Llamafile server URL is returned.
        See: https://github.com/Mozilla-Ocho/llamafile/blob/bd1bbe9aabb1ee12dbdcafa8936db443c571eb9d/README.md#L61
        """
        return api_base or get_secret_str("LLAMAFILE_API_BASE") or "http://127.0.0.1:8080/v1" # type: ignore


    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Attempts to ensure that the API base and key are set, preferring user-provided values,
        before falling back to secret manager values (``LLAMAFILE_API_BASE`` and ``LLAMAFILE_API_KEY``
        respectively).

        If an API key cannot be resolved via either method, a fake key is returned. Llamafile
        does not require an API key, but the underlying OpenAI library may expect one anyway.
        """
        api_base = LlamafileChatConfig._resolve_api_base(api_base)
        dynamic_api_key = LlamafileChatConfig._resolve_api_key(api_key)

        return api_base, dynamic_api_key
