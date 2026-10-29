"""
Providers whose requests are covered by a flat subscription, each with the way to read
the account id of the login a deployment is signed into.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.chatgpt.authenticator import Authenticator, get_cached_authenticator, get_chatgpt_auth_file


def _chatgpt_account_id(litellm_params: Mapping[str, object]) -> str | None:
    auth_file: Final = get_chatgpt_auth_file(litellm_params)
    if auth_file:
        return get_cached_authenticator(auth_file).read_account_id()
    api_key: Final = litellm_params.get("api_key")
    return Authenticator().read_account_id(api_key if isinstance(api_key, str) and api_key else None)


SUBSCRIPTION_ACCOUNT_ID_RESOLVERS: Final[Mapping[str, Callable[[Mapping[str, object]], str | None]]] = MappingProxyType(
    {"chatgpt": _chatgpt_account_id}
)


def resolve_subscription_account_id(custom_llm_provider: str, litellm_params: Mapping[str, object]) -> str | None:
    resolver: Final = SUBSCRIPTION_ACCOUNT_ID_RESOLVERS.get(custom_llm_provider)
    return None if resolver is None else resolver(litellm_params)
