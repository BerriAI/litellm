from litellm.llms.base_llm.chat.transformation import BaseLLMException


class JinaAIError(BaseLLMException):
    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)
