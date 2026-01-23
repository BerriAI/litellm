"""
Sarvam AI common utilities
"""

from litellm.llms.base_llm.chat.transformation import BaseLLMException


SARVAM_API_BASE = "https://api.sarvam.ai"


class SarvamException(BaseLLMException):
    """Sarvam AI exception handling class"""
    pass
