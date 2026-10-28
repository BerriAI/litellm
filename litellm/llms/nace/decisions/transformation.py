from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class NaceDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("NACE_API_KEY",)
    api_base_env = ("NACE_API_BASE", "NACE_BASE_URL")

    def get_default_api_base(self) -> str | None:
        return "https://console.nace.ai"
