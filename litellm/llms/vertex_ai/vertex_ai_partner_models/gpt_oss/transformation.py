import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class VertexAIGPTOSSTransformation(OpenAIGPTConfig):
    """
    Transformation for GPT-OSS model from VertexAI

    https://console.cloud.google.com/vertex-ai/publishers/openai/model-garden/gpt-oss-120b-maas?hl=id
    """
    def __init__(self):
        super().__init__()
    
    def get_supported_openai_params(self, model: str) -> list:
        base_gpt_series_params = super().get_supported_openai_params(model=model)
        gpt_oss_only_params = ["reasoning_effort"]
        base_gpt_series_params.extend(gpt_oss_only_params)

        #########################################################
        # VertexAI - GPT-OSS does not support tool calls
        #########################################################
        if litellm.supports_function_calling(model=model) is False:
            TOOL_CALLING_PARAMS_TO_REMOVE = ["tool", "tool_choice", "function_call", "functions"]
            base_gpt_series_params = [param for param in base_gpt_series_params if param not in TOOL_CALLING_PARAMS_TO_REMOVE]

        return base_gpt_series_params

