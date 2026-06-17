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
    control: ChatCompletionCachedContent | None


class CacheControlToolConfigInjectionPoint(TypedDict):
    """Type for tool_config-level injection points (Bedrock)."""

    location: Literal["tool_config"]


class _CacheControlSystemInjectionPointBase(TypedDict):
    location: Literal["system"]


class CacheControlSystemInjectionPoint(
    _CacheControlSystemInjectionPointBase, total=False
):
    """Type for system-prompt injection points (Anthropic ``/v1/messages``).

    On the native Anthropic Messages endpoint the system prompt is a top-level
    ``system`` parameter rather than a ``role: system`` entry inside
    ``messages`` (as it is for OpenAI chat/completions), so it needs its own
    targetable location. ``location`` is required (it lives on the base class);
    only ``control`` is optional.
    """

    control: ChatCompletionCachedContent | None


class _CacheControlToolsInjectionPointBase(TypedDict):
    location: Literal["tools"]


class CacheControlToolsInjectionPoint(
    _CacheControlToolsInjectionPointBase, total=False
):
    """Type for tool-list injection points (Anthropic ``/v1/messages``).

    Caches the (typically long, static) tool list sent by Anthropic-native
    clients such as Claude Code. Distinct from the Bedrock-only ``tool_config``
    location, which is consumed by the Converse transform. ``location`` is
    required (it lives on the base class); only ``control`` is optional.
    """

    control: ChatCompletionCachedContent | None


CacheControlInjectionPoint = Union[
    CacheControlMessageInjectionPoint,
    CacheControlToolConfigInjectionPoint,
    CacheControlSystemInjectionPoint,
    CacheControlToolsInjectionPoint,
]
