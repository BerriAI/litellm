"""The per-model key and user budget checks the Python auth host still runs for gateway-auth-core, and the reads
and alerts the core's key path budgets ask for: whether the request is free, the team member's budget, the
global proxy spend, and the key, team member and proxy budget alerts.

TODO: moves to gateway-auth-core one check at a time.

Blocked on:
- the model budget fallback rewrites the request body through the router
- per-model spend lives in `PROXY_VirtualKeyModelMaxBudgetLimiter`
- request pricing, which decides a free request, is read from the router

After the move:
- `key_model_budgets` and the budget half of `finish_jwt` become reads of per-model spend and a comparison in
  the core, and this module keeps only the reads and alerts"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final

from fastapi import Request

import litellm
from litellm.constants import GLOBAL_PROXY_SPEND_CACHE_KEY
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import (
    request_skips_budget_checks,
    team_member_max_budget_alert_check,
)
from litellm.proxy.auth.host.facts import with_owner
from litellm.proxy.auth.user_api_key_auth import (
    budget_served_model,
    check_key_model_budget_with_fallback,
    check_user_model_budget,
    fetch_global_spend_with_event_coordination,
    get_model_from_request_context,
    get_model_names_for_budget_checks,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, team_membership_auth_cache_key
from litellm.repositories.table_repositories import TeamMembershipRepository
from litellm.types.integrations.slack_alerting import CallInfo, Litellm_EntityType
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

if TYPE_CHECKING:
    from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router


@dataclass(frozen=True, slots=True)
class BudgetChecks:
    """Each check raises `litellm.BudgetExceededError` for the first budget the request is over."""

    request: Request
    request_data: dict[str, object]  # mutable-ok: the model budget fallback rewrites the request body
    route: str
    prisma_client: PrismaClient | None
    user_api_key_cache: UserApiKeyCache
    proxy_logging_obj: ProxyLogging
    llm_router: Router | None
    llm_model_list: Sequence[object] | None
    model_max_budget_limiter: PROXY_VirtualKeyModelMaxBudgetLimiter

    def _requested_model(self, token: UserAPIKeyAuth) -> str | list[str] | None:
        return get_model_from_request_context(
            request_data=self.request_data,
            route=self.route,
            request=self.request,
            llm_router=self.llm_router,
            team_id=token.team_id,
        )

    def _skips_budget_checks_for(self, model: str | Sequence[str] | None) -> bool:
        return request_skips_budget_checks(
            route=self.route, model=model, llm_router=self.llm_router, request_body=self.request_data
        )

    async def _key_skips_budget_checks(self, token: UserAPIKeyAuth) -> bool:
        return self._skips_budget_checks_for(
            await budget_served_model(
                self._requested_model(token), self.request_data, self.request, self.llm_router, token
            )
        )

    def _request_models(self, token: UserAPIKeyAuth) -> Sequence[str]:
        return get_model_names_for_budget_checks(model=self._requested_model(token))

    async def _team_membership(
        self, prisma_client: PrismaClient, user_id: str, team_id: str
    ) -> LiteLLM_TeamMembership | None:
        cache_key: Final = team_membership_auth_cache_key(team_id=team_id, user_id=user_id)
        cached: Final = await self.user_api_key_cache.async_get_cache(key=cache_key, model_type=LiteLLM_TeamMembership)
        if cached is not None:
            return cached
        db_member: Final = await TeamMembershipRepository(prisma_client).table.find_first(
            where={"user_id": user_id, "team_id": team_id},
            include={"litellm_budget_table": True},
        )
        if db_member is None:
            return None
        membership: Final = LiteLLM_TeamMembership.model_validate(db_member.model_dump())
        await self.user_api_key_cache.async_set_cache(
            key=cache_key, value=membership, model_type=LiteLLM_TeamMembership, ttl=5
        )
        return membership

    async def key_skips_budget_checks(self, token: UserAPIKeyAuth) -> bool:
        return await self._key_skips_budget_checks(token)

    async def team_member_max_budget(self, user_id: str, team_id: str) -> float | None:
        """The membership's budget with an unexpired temporary increase applied, `None` when it has none."""
        if self.prisma_client is None:
            return None
        membership: Final = await self._team_membership(self.prisma_client, user_id, team_id)
        if membership is None or membership.litellm_budget_table is None:
            return None
        return membership.litellm_budget_table.effective_max_budget(now=datetime.now(timezone.utc))

    def team_member_alert(
        self, token: UserAPIKeyAuth, owner: LiteLLM_UserTable | None, spend: float, max_budget: float
    ) -> None:
        if token.user_id is None or token.team_id is None:
            return
        team_member_max_budget_alert_check(
            team_id=token.team_id,
            team_alias=token.team_alias,
            team_metadata=token.team_metadata,
            organization_id=token.org_id,
            user_id=token.user_id,
            user_email=owner.user_email if owner is not None else None,
            proxy_logging_obj=self.proxy_logging_obj,
            spend=spend,
            max_budget=max_budget,
        )

    async def key_model_budgets(self, key: UserAPIKeyAuth, owner: LiteLLM_UserTable | None) -> None:
        """The key's per-model budgets with the router's model fallback, then its owner's per-model budget."""
        token: Final = with_owner(key, owner)
        models: Final = self._request_models(token)
        if token.model_max_budget and self.prisma_client is not None and models and token.token is not None:
            for model_name in models:
                await check_key_model_budget_with_fallback(
                    valid_token=token,
                    model_max_budget_limiter=self.model_max_budget_limiter,
                    model_name=model_name,
                    request_data=self.request_data,
                    request=self.request,
                    llm_model_list=self.llm_model_list,
                    llm_router=self.llm_router,
                )
            await self._user_model(token, self._request_models(token))
            return
        await self._user_model(token, models)

    async def jwt_user_model(self, token: UserAPIKeyAuth) -> None:
        """The per-model user budget the JWT path checks, under the same zero-cost model skip as the key path."""
        if not self._skips_budget_checks_for(self._requested_model(token)):
            await self._user_model(token, self._request_models(token))

    async def _user_model(self, token: UserAPIKeyAuth, models: Sequence[str]) -> None:
        await check_user_model_budget(
            valid_token=token, model_max_budget_limiter=self.model_max_budget_limiter, models=models
        )


async def global_proxy_spend(prisma_client: PrismaClient, user_api_key_cache: UserApiKeyCache) -> float | None:
    with tracer.trace("litellm.proxy.auth.get_global_proxy_spend"):
        return await fetch_global_spend_with_event_coordination(
            cache_key=GLOBAL_PROXY_SPEND_CACHE_KEY,
            user_api_key_cache=user_api_key_cache,
            prisma_client=prisma_client,
        )


def proxy_budget_alert(
    token: UserAPIKeyAuth, spend: float, *, proxy_logging_obj: ProxyLogging, litellm_proxy_admin_name: str
) -> None:
    call_info: Final = CallInfo(
        token=token.token,
        spend=spend,
        max_budget=litellm.max_budget,
        user_id=litellm_proxy_admin_name,
        team_id=token.team_id,
        event_group=Litellm_EntityType.PROXY,
    )
    _ = asyncio.create_task(proxy_logging_obj.budget_alerts(type="proxy_budget", user_info=call_info))
