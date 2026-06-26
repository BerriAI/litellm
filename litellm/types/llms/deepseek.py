from typing_extensions import Literal, TypedDict


class DeepSeekThinkingParam(TypedDict, total=False):
    type: Literal["enabled", "disabled"]
