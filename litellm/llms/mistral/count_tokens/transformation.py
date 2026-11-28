from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.mistral.common_utils import get_mistral_api_base
from litellm.llms.openai.responses.count_tokens.transformation import OpenAICountTokensConfig


class MistralCountTokensConfig(OpenAICountTokensConfig):
    def get_mistral_count_tokens_endpoint(self, api_base: str | None = None) -> str:
        return join_api_base(get_mistral_api_base(api_base), "v1/responses/input_tokens")

    def get_required_headers(
        self, api_key: str
    ) -> dict[str, str]:  # mutable-ok: matches OpenAICountTokensConfig contract
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
