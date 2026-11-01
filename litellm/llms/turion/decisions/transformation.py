from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class TurionDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("TURION_API_KEY", "TURION_API_TOKEN")
    api_base_env = ("TURION_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.turion.ai"
