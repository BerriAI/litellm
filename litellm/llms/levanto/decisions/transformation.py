from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class LevantoDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("LEVANTO_API_KEY",)
    api_base_env = ("LEVANTO_BASE_URL",)

    def get_default_api_base(self) -> str | None:
        return "https://api.levanto.ai"
