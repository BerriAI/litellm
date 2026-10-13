from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class V1MError(BaseLLMException):
    pass


class V1MConfig(OpenAIGPTConfig):
    """Configuration for v1m (System One) calibrated decision engine.

    Website: https://v1m.ir
    """

    def __init__(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool | None = None,
       ) -> None:
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if stream is not None:
            self.stream = stream
