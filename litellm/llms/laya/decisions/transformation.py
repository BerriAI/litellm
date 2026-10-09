from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig


class LayaDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("LAYA_API_KEY",)
    api_base_env = ("LAYA_API_BASE",)
    api_key_required = False
