from typing import Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict

from litellm.types.llms.openai import ChatCompletionCachedContent


class CacheControlMessageInjectionPoint(TypedDict):
    """Type for message-level injection points."""

    location: Literal["message"]
    role: Optional[Literal["user", "system", "assistant"]]  # Optional: target by role (user, system, assistant)
    index: Optional[Union[int, str]]  # Optional: target by specific index
    control: Optional[ChatCompletionCachedContent]
    _litellm_judged: NotRequired[bool]  # Internal: written back by litellm once the client cache_control judgment ran


class CacheControlToolConfigInjectionPoint(TypedDict):
    """Type for tool_config-level injection points (Bedrock)."""

    location: Literal["tool_config"]
    control: Optional[ChatCompletionCachedContent]
    _litellm_judged: NotRequired[bool]  # Internal: written back by litellm once the client cache_control judgment ran


CacheControlInjectionPoint = Union[
    CacheControlMessageInjectionPoint,
    CacheControlToolConfigInjectionPoint,
]
