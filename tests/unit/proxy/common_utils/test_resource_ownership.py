from typing import Final

from litellm.litellm_core_utils import resource_ownership as core_resource_ownership
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils import resource_ownership as proxy_resource_ownership

PREVIOUSLY_EXPOSED_NAMES: Final = (
    "Final",
    "LitellmUserRoles",
    "UserAPIKeyAuth",
    "get_primary_resource_owner_scope",
    "get_resource_owner_scopes",
    "is_proxy_admin",
    "user_can_access_resource_owner",
)


def test_proxy_path_reexports_core_objects() -> None:
    mismatched: Final = [
        name
        for name in PREVIOUSLY_EXPOSED_NAMES
        if getattr(proxy_resource_ownership, name) is not getattr(core_resource_ownership, name)
    ]
    assert mismatched == []


def test_moved_module_uses_the_proxy_types_objects() -> None:
    assert core_resource_ownership.LitellmUserRoles is LitellmUserRoles
    assert core_resource_ownership.UserAPIKeyAuth is UserAPIKeyAuth
