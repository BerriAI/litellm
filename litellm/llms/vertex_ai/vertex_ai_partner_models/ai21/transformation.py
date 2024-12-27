import types
from typing import Optional

from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig


class VertexAIAi21Config(OpenAILikeChatConfig):
    """
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/ai21

    The class `VertexAIAi21Config` provides configuration for the VertexAI's AI21 API interface

    -> Supports all OpenAI parameters
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }
