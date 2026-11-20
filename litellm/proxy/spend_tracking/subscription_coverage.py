from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.constants import SUBSCRIPTION_BUCKET_SUFFIX
from litellm.litellm_core_utils.core_helpers import as_str_mapping


def _claude_allowance_covers(metadata: Mapping[str, object]) -> bool:
    usage: Final = as_str_mapping(metadata.get("claude_subscription_usage"))
    return usage is not None and usage.get("is_overage") is False


CLIENT_SUBSCRIPTION_COVERAGE: Final[Mapping[str, Callable[[Mapping[str, object]], bool]]] = MappingProxyType(
    {"anthropic": _claude_allowance_covers}
)


def daily_spend_provider(provider: str | None, metadata: Mapping[str, object]) -> str | None:
    covers: Final = None if provider is None else CLIENT_SUBSCRIPTION_COVERAGE.get(provider)
    if provider is None or covers is None or not covers(metadata):
        return provider
    return f"{provider}{SUBSCRIPTION_BUCKET_SUFFIX}"
