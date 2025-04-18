from typing import Literal, Optional, TypedDict, Union

from litellm.types.llms.openai import ChatCompletionCachedContent


class CacheControlMessageInjectionPoint(TypedDict):
    """Type for message-level injection points."""

    location: Literal["message"]
    role: Optional[
        Literal["user", "system", "assistant"]
    ]  # Optional: target by role (user, system, assistant)
    index: Optional[Union[int, str]]  # Optional: target by specific index
    control: Optional[ChatCompletionCachedContent]


CacheControlInjectionPoint = CacheControlMessageInjectionPoint
