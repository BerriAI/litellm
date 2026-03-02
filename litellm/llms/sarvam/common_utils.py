import platform

from litellm._version import version as litellm_version
from litellm.llms.base_llm.chat.transformation import BaseLLMException

SARVAM_API_BASE = "https://api.sarvam.ai"


def get_sarvam_user_agent() -> str:
    python_version = f"{platform.python_version()}"
    return (
        f"LiteLLM/{litellm_version} "
        f"python/{python_version}"
    )


class SarvamException(BaseLLMException):
    pass
