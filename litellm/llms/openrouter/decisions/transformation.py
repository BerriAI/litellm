from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class OpenRouterDecisionsConfig(BaseDecisionsConfig):
    path = "/alpha/decisions"
    api_key_env = ("OPENROUTER_API_KEY",)
    api_base_env = ("OPENROUTER_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://openrouter.ai/api"
