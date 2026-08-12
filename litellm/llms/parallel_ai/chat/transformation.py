"""
Translate from OpenAI's `/v1/chat/completions` to Parallel AI's `/chat/completions`.

Parallel AI Chat API Reference: https://docs.parallel.ai/chat-api/chat-quickstart

Research models (lite/base/core) return a top-level `basis` field (per-field
citations, reasoning, and confidence); the OpenAI-compatible response path
preserves it on the returned ModelResponse as an extra field automatically.
"""

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.parallel_ai.common_utils import resolve_parallel_ai_credentials


class ParallelAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "parallel_ai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return resolve_parallel_ai_credentials(api_base=api_base, api_key=api_key)

    def get_supported_openai_params(self, model: str) -> list:
        """
        Parallel's Chat API supports a subset of OpenAI params.

        Ref: https://docs.parallel.ai/chat-api/chat-quickstart

        Sampling params (temperature, top_p, penalties, ...) and tool calling are not
        supported; the research models ground every answer with built-in web research.
        """
        return [
            "stream",
            "response_format",
            "max_retries",
            "extra_headers",
        ]
