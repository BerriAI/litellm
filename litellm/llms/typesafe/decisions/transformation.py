from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class TypeSafeDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("TYPESAFE_API_KEY",)
    api_base_env = ("TYPESAFE_API_BASE",)
    supports_assets = True
    supports_box_questions = True

    def get_default_api_base(self) -> str | None:
        return "https://api.typesafe.ai"
