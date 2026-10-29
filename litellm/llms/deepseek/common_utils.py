from collections.abc import Mapping, Sequence
from typing import Final

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.deepseek.messages.transformation import DeepSeekAnthropicMessagesConfig, deepseek_anthropic_base
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders


class DeepSeekModelInfo(BaseLLMModelInfo):
    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        from litellm.utils import get_valid_models

        return get_valid_models(custom_llm_provider=LlmProviders.DEEPSEEK.value)

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return DeepSeekAnthropicMessagesConfig.get_api_key(api_key)

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return deepseek_anthropic_base(api_base)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: matches validate_environment's dict contract shared by every provider override
        resolved_api_key: Final = self.get_api_key(api_key)
        return {**headers, **({} if resolved_api_key is None else {"x-api-key": resolved_api_key})}

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model.removeprefix(f"{LlmProviders.DEEPSEEK.value}/")

    def get_token_counter(self) -> BaseTokenCounter | None:
        from litellm.llms.deepseek.count_tokens.token_counter import DeepSeekTokenCounter

        return DeepSeekTokenCounter()
