from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.openai.common_utils import get_openai_credentials
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from httpx import URL

    from litellm.types.llms.openai import AllMessageValues


class OpenAIPassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: Mapping[str, object]) -> bool:
        return False

    def required_custom_llm_provider(self) -> str | None:
        return LlmProviders.OPENAI.value

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: Mapping[str, object] | None,
        litellm_params: Mapping[str, object],
    ) -> tuple["URL", str]:
        base_target_url: Final = self.get_api_base(api_base)
        return self.format_url(endpoint, base_target_url, request_query_params), base_target_url

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return get_openai_credentials(api_key=api_key).api_key

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return get_openai_credentials(api_base=api_base).api_base

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: BaseLLMModelInfo requires a list result
        return []

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence["AllMessageValues"],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base class contract returns dict for httpx
        return {**headers, "Authorization": f"Bearer {api_key}"} if api_key else dict(headers)
