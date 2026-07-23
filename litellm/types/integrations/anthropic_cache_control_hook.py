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


class CacheControlSystemInjectionPoint(TypedDict, total=False):
    """Type for system-level injection points (Anthropic Messages API).

    Anthropic's /v1/messages keeps the system prompt as a top-level ``system``
    parameter — it is NOT a message in the ``messages`` array. Use this
    injection-point shape to cache the system prompt (the common big win for
    Claude Code: long system prompt cached across turns).
    """

    location: Literal["system"]
    control: Optional[ChatCompletionCachedContent]


class CacheControlToolsInjectionPoint(TypedDict, total=False):
    """Type for tools-level injection points (Anthropic Messages API).

    Marks the last tool in the ``tools`` array with ``cache_control`` so the
    entire tool list participates in the prompt cache. Anthropic only honors
    ``cache_control`` on the final tool entry; the marker covers all tools
    that precede it.
    """

    location: Literal["tools"]
    control: Optional[ChatCompletionCachedContent]


CacheControlInjectionPoint = Union[
    CacheControlMessageInjectionPoint,
    CacheControlToolConfigInjectionPoint,
    CacheControlSystemInjectionPoint,
    CacheControlToolsInjectionPoint,
]
