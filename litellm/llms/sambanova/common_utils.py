from litellm.llms.base_llm.chat.transformation import BaseLLMException


class SambaNovaError(BaseLLMException):
    def __init__(self, status_code, message, headers):
        super().__init__(status_code=status_code, message=message, headers=headers)
