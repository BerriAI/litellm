from typing import Optional, List, Union

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class BytezError(BaseLLMException):
    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)


def validate_environment(
    api_key: Optional[str] = None, messages: Union[List, None] = None
) -> None:
    if not messages:
        raise Exception(
            "kwarg `messages` must be an array of messages that follow the openai chat standard"
        )

    if not api_key:
        raise Exception("Missing api_key, make sure you pass in your api key")
