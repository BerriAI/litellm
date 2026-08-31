"""Turing Engine Chat Completions API Config.

Provides OpenAI-compatible adapter configuration for Turing Engine runtime.
"""

from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class TuringConfig(OpenAIGPTConfig):
    """Configuration class for Turing Engine OpenAI-compatible serving runtime."""

    max_tokens: int | None = None
    temperature: int | None = None
    top_p: int | None = None
    stream: bool | None = None
    sparsity_ratio: float | None = None
    use_svd_kv: bool | None = None

    def __init__(
        self,
        max_tokens: int | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        stream: bool | None = None,
        sparsity_ratio: float | None = None,
        use_svd_kv: bool | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

        self.__class__._is_base_class = False

    def _get_openai_compatible_provider_info(
        self, api_base: str | None = None, api_key: str | None = None
    ) -> tuple[str | None, str | None]:
        default_base = api_base or "http://localhost:8000/v1"
        default_key = api_key or "turing-local"
        return default_base, default_key
