from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class BespokeDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("BESPOKE_API_KEY",)
    api_base_env = ("BESPOKE_API_BASE",)
    api_key_required = False
