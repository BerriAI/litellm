from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class PerplexityDecisionsConfig(BaseDecisionsConfig):
    path = "/v1/decisions"
    api_key_env = ("PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY")
    api_base_env = ("PERPLEXITY_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.perplexity.ai"
