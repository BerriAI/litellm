from typing import Final

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LlmmanChatConfig(OpenAIGPTConfig):
    """Configuration for llmman's OpenAI-compatible chat API.

    llmman is a local model runner that serves OpenAI-, Ollama- and
    Anthropic-compatible APIs. See https://github.com/llmmanorg/llmman
    """

    @staticmethod
    def _resolve_api_key(api_key: str | None = None) -> str:
        """Resolve the API key, preferring the user-provided value over
        ``LLMMAN_API_KEY``.

        Returns a placeholder when neither is set: llmman does not require a
        key, but the underlying OpenAI library expects a non-None value.
        """
        return api_key or get_secret_str("LLMMAN_API_KEY") or "fake-api-key"

    @staticmethod
    def _resolve_api_base(api_base: str | None = None) -> str | None:
        """Resolve the API base, preferring the user-provided value over
        ``LLMMAN_API_BASE``, then falling back to the default `llmman serve`
        address.

        See: https://github.com/llmmanorg/llmman#serve
        """
        return (
            api_base or get_secret_str("LLMMAN_API_BASE") or "http://127.0.0.1:17434/v1"
        )

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = LlmmanChatConfig._resolve_api_base(api_base)
        dynamic_api_key: Final = LlmmanChatConfig._resolve_api_key(api_key)

        return api_base, dynamic_api_key
