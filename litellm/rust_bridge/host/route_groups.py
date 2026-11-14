from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from litellm.proxy._types import LiteLLMRoutes


def route_groups() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({member.name: tuple(member.value) for member in LiteLLMRoutes})
