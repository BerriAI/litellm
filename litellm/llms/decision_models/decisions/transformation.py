from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class DecisionModelsDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("DECISION_MODELS_API_KEY",)
    api_base_env = ("DECISION_MODELS_API_BASE",)

    def get_default_api_base(self) -> str | None:
        return "https://api.decisionmodels.io"
