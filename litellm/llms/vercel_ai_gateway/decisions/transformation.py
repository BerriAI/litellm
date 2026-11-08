from collections.abc import Mapping

from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig


class VercelAIGatewayDecisionsConfig(OpenAIDecisionsConfig):
    """get_llm_provider already resolves the gateway's base and key, so never fall back to OpenAI credentials."""

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return api_key
