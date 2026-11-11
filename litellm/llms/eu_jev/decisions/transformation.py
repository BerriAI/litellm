from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class EuJevDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("EU_JEV_API_KEY",)
    api_base_env = ("EU_JEV_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://eu-jev.com"
