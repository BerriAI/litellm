from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class EmpirioLabsDecisionsConfig(BaseDecisionsConfig):
    path = "/v1/decisions"
    api_key_env = ("EMPIRIOLABS_API_KEY",)
    api_base_env = ("EMPIRIOLABS_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.empiriolabs.ai"
