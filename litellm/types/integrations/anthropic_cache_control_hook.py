from typing import Final, Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.types.llms.openai import ChatCompletionCachedContent

INJECTED_CACHE_BREAKPOINTS_METADATA_KEY: Final = "litellm_injected_cache_breakpoints"


class CacheControlMessageInjectionPoint(TypedDict):
    """Type for message-level injection points."""

    location: Literal["message"]
    role: Literal["user", "system", "assistant"] | None  # Optional: target by role (user, system, assistant)
    index: int | str | None  # Optional: target by specific index
    control: ChatCompletionCachedContent | None
    _litellm_judged: NotRequired[bool]  # Internal: written back by litellm once the client cache_control judgment ran
    _litellm_openai_dialect: NotRequired[ReadOnly[bool]]


class CacheControlToolConfigInjectionPoint(TypedDict):
    """Type for tool_config-level injection points (Bedrock)."""

    location: Literal["tool_config"]
    control: ChatCompletionCachedContent | None
    _litellm_judged: NotRequired[bool]  # Internal: written back by litellm once the client cache_control judgment ran
    _litellm_openai_dialect: NotRequired[ReadOnly[bool]]


CacheControlInjectionPoint = CacheControlMessageInjectionPoint | CacheControlToolConfigInjectionPoint
