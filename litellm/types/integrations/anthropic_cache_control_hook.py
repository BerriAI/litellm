from typing import Literal, Optional, Union

from typing_extensions import TypedDict

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
