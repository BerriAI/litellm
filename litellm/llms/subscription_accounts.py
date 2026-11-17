"""
Providers whose requests are covered by a flat subscription, each with the way to read
the account id of the login a deployment is signed into.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.llms.chatgpt.authenticator import Authenticator, get_cached_authenticator, get_chatgpt_auth_file
from litellm.secret_managers.main import UnresolvableSecretReference, resolve_local_oidc_reference
from litellm.utils import with_named_credential


def _chatgpt_account_id(litellm_params: Mapping[str, object]) -> str | None:
    params: Final = with_named_credential(litellm_params)
    auth_file: Final = get_chatgpt_auth_file(params)
    if auth_file:
        return get_cached_authenticator(auth_file).read_account_id()
    api_key: Final = params.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        return Authenticator().read_account_id()
    match resolve_local_oidc_reference(api_key):
        case UnresolvableSecretReference():
            return None
        case str() as access_token:
            return Authenticator().read_account_id(access_token)


SUBSCRIPTION_ACCOUNT_ID_RESOLVERS: Final[Mapping[str, Callable[[Mapping[str, object]], str | None]]] = MappingProxyType(
    {"chatgpt": _chatgpt_account_id}
)


def resolve_subscription_account_id(custom_llm_provider: str, litellm_params: Mapping[str, object]) -> str | None:
    resolver: Final = SUBSCRIPTION_ACCOUNT_ID_RESOLVERS.get(custom_llm_provider)
    return None if resolver is None else resolver(litellm_params)
