"""
Providers whose requests are covered by a flat subscription, each with the way to read
the account id of the login the proxy is signed into.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.chatgpt.authenticator import Authenticator


def _chatgpt_account_id() -> str | None:
    return Authenticator().get_account_id()


SUBSCRIPTION_ACCOUNT_ID_RESOLVERS: Final[Mapping[str, Callable[[], str | None]]] = MappingProxyType(
    {"chatgpt": _chatgpt_account_id}
)


def resolve_subscription_account_id(custom_llm_provider: str) -> str | None:
    resolver: Final = SUBSCRIPTION_ACCOUNT_ID_RESOLVERS.get(custom_llm_provider)
    return None if resolver is None else resolver()
