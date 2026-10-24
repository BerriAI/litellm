from collections.abc import Mapping

from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.opencode.common_utils import go_api_base, go_headers
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class OpenCodeGoResponsesConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENCODE_GO

    def supports_native_websocket(self) -> bool:
        return False

    def get_complete_url(self, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return ensure_api_route(go_api_base(api_base).rstrip("/").removesuffix("/v1"), "v1/responses")

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, object]:  # mutable-ok: protocol handlers mutate the returned request and headers
        return go_headers(headers, litellm_params, litellm_params.api_key if litellm_params else None, model)
