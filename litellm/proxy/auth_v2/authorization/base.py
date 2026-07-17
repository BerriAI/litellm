from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from litellm.proxy.auth_v2.authorization.roles import Role
    from litellm.proxy.auth_v2.models import Principal


@runtime_checkable
class Authorizer(Protocol):
    """Decides what an authenticated principal is allowed to do.

    This is the extension point for authorization methods. RBAC is the only
    implementation today; add others (ABAC, ReBAC, an external PDP, ...) by
    implementing this protocol and passing the instance to
    ``AuthSecurity(..., authorizer=...)``.
    """

    def enforce(self, principal: "Principal", obj: str, act: str) -> bool:
        """Return True if ``principal`` may perform ``act`` on resource ``obj``."""
        ...

    def has_any_role(self, principal: "Principal", allowed: "Tuple[Role, ...]") -> bool:
        """Return True if ``principal`` holds (or inherits) any of ``allowed``."""
        ...
