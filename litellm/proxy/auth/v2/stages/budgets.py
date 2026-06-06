from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth

    from ..authn.authenticators import AuthContext


async def enforce_hierarchy_budgets(
    identity: UserAPIKeyAuth, route: str, ctx: AuthContext
) -> None:
    """Enforce team, organization, and global budgets for an auth_v2 request.

    These hierarchy caps live in v1's ``common_checks`` (not the pre-call hooks),
    which auth_v2 does not run -- so without this they would go unenforced under
    v2. Rather than reimplement them (divergent logic, wrong counter keys), this
    calls the exact same functions v1 uses, so there is one budget implementation
    with two callers. Raises ``litellm.BudgetExceededError`` when a cap is
    exceeded; the key/user budgets are already handled by the pre-call hooks.
    """
    from litellm.proxy.auth.auth_checks import (
        _global_proxy_budget_check,
        _organization_max_budget_check,
        _team_max_budget_check,
        get_team_object,
    )
    from litellm.proxy.auth.user_api_key_auth import get_global_proxy_spend
    from litellm.proxy.proxy_server import litellm_proxy_admin_name

    team_id = getattr(identity, "team_id", None)
    team_object: Optional[LiteLLM_TeamTable] = None
    if team_id is not None:
        team_object = await get_team_object(
            team_id=team_id,
            prisma_client=ctx.prisma_client,
            user_api_key_cache=ctx.user_api_key_cache,
            parent_otel_span=ctx.parent_otel_span,
            proxy_logging_obj=ctx.proxy_logging_obj,
        )

    await _team_max_budget_check(
        team_object=team_object,
        valid_token=identity,
        proxy_logging_obj=ctx.proxy_logging_obj,
    )
    await _organization_max_budget_check(
        valid_token=identity,
        team_object=team_object,
        prisma_client=ctx.prisma_client,
        user_api_key_cache=ctx.user_api_key_cache,
        proxy_logging_obj=ctx.proxy_logging_obj,
    )

    global_proxy_spend = await get_global_proxy_spend(
        litellm_proxy_admin_name=litellm_proxy_admin_name,
        user_api_key_cache=ctx.user_api_key_cache,
        prisma_client=ctx.prisma_client,
        token=getattr(identity, "token", None) or "",
        proxy_logging_obj=ctx.proxy_logging_obj,
    )
    _global_proxy_budget_check(
        global_proxy_spend=global_proxy_spend, skip_budget_checks=False, route=route
    )
