from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class CortecsDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("CORTECS_API_KEY",)
    api_base_env = ("CORTECS_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.cortecs.ai/v1"
