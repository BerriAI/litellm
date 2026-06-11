from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.security import SecurityScopes

if TYPE_CHECKING:
    from litellm.proxy.auth_v2.models import Principal


def has_required_scopes(security_scopes: SecurityScopes, principal: "Principal") -> bool:
    return set(security_scopes.scopes).issubset(set(principal.scopes))
