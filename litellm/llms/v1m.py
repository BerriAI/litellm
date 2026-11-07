from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class V1MError(BaseLLMException):
    pass


class V1MConfig(OpenAIGPTConfig):
    """Configuration for v1m (System One) calibrated decision engine.

    Website: https://v1m.ir
    """
