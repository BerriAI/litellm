from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class StrandsDeciderDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("STRANDS_DECIDER_API_KEY",)
    api_base_env = ("STRANDS_DECIDER_API_BASE",)
    api_key_required = False
