from collections.abc import Mapping
from typing import Final

from litellm.proxy.auth.customer_hierarchy_auth import (
    customer_identity_from_headers,
    customer_member_cache_key,
    customer_org_alias_cache_key,
    select_customer_team,
)
from litellm.proxy.auth.customer_spend import customer_spend_counter
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth


def cached_customer_spend_keys(headers: Mapping[str, object] | None, cache: UserApiKeyCache) -> frozenset[str]:
    if headers is None or not any(
        name.lower() in ("x-litellm-customer-member-id", "x-litellm-customer-org-alias") for name in headers
    ):
        return frozenset()
    identity: Final = customer_identity_from_headers(headers)
    if identity is None:
        return frozenset()
    key: Final = (
        customer_member_cache_key(identity.customer_member_id)
        if identity.customer_member_id is not None
        else customer_org_alias_cache_key(identity.customer_org_alias)
        if identity.customer_org_alias is not None
        else None
    )
    if key is None:
        return frozenset()
    cached: Final = cache.get_local_frozen_model(key, CustomerHierarchyAuth)
    if cached is None:
        return frozenset()
    selected: Final = (
        select_customer_team(cached, identity.customer_team_id)
        if identity.customer_member_id is not None and identity.customer_team_id is not None
        else cached
    )
    return (
        frozenset(customer_spend_counter(scope) for scope in selected.scopes) if selected is not None else frozenset()
    )
