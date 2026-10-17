from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class SubQError(BaseLLMException):
    pass


class SubQConfig(OpenAIGPTConfig):
    """Configuration for SubQ (System One) calibrated decision engine.

    Website: https://subq.ai
    """
