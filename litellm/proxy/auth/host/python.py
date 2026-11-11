"""``PythonAuthHost``: what gateway-auth-core reaches outside itself, served from the proxy's Python state.

The core parks on a `RustAuthStep`, and `answer.answer` awaits the matching method here on the caller's task.
A method reads or writes one named thing and returns the row it found, or runs a step `steps/` owns. Each
method answers the `AuthHost` method of the same name in `gateway-auth-core/src/host.rs`, which says what it is
for and when it leaves Python."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol

import httpx
from fastapi import HTTPException, Request
from typing_extensions import assert_never

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.tag import LiteLLM_TagTable
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import (
    LiteLLM_JWTAuth,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    ProxyErrorTypes,
    ProxyException,
    TeamMemberAddRequest,
)
from litellm.proxy.agent_endpoints.identity import has_legacy_identity
from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore, resolve_managed_agent
from litellm.proxy.auth import identity as auth_identity
from litellm.proxy.auth.auth_checks import (
    ExperimentalUIJWTToken,
    TeamNotFoundError,
    agent_requires_trace_id,
    can_team_access_model,
    delete_cache_key_object,
    get_default_end_user_budget,
    get_end_user_object,
    get_jwt_key_mapping_object,
    get_model_access_group_budgets_batch,
    get_object_permission,
    get_org_object,
    get_org_object_by_alias,
    get_project_object,
    get_tag_objects_batch,
    get_team_member_default_budget,
    get_team_membership,
    get_team_object,
    get_team_object_by_alias,
    get_user_object,
    get_user_role,
    resolve_and_validate_end_user_id,
    send_key_max_budget_warning,
    send_key_soft_budget_alert,
    send_key_token_budget_alert,
)
from litellm.proxy.auth.auth_object_prefetch import prefetch_identity_keys
from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
from litellm.proxy.auth.host.facts import (
    AliasMissing,
    EndUserRow,
    FoundTeam,
    RegisteredAgentFacts,
    TeamMissing,
    team_facts,
    with_project,
)
from litellm.proxy.auth.host.steps import jwt_auto_register
from litellm.proxy.auth.host.steps.authorization import (
    AlertKind,
    Authorization,
    MemberBudget,
    SpendScope,
    member_budget,
)
from litellm.proxy.auth.host.steps.budgets import BudgetChecks, global_proxy_spend, proxy_budget_alert
from litellm.proxy.auth.resolvers.exceptions import KeyNotFoundError
from litellm.proxy.auth.resolvers.grants import GrantResolver, UserLookup, user_models
from litellm.proxy.auth.resolvers.store import IdentityStore
from litellm.proxy.auth.team_grants import team_grants, team_model_aliases
from litellm.proxy.auth.user_api_key_auth import (
    apply_budget_limits_to_end_user_params,
    arm_routing_cooldowns,
    identity_cache_keys,
    normalize_model_for_auth,
    prefetch_referenced_auth_objects,
    update_key_budget_with_temp_budget_increase,
)
from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body, safe_get_request_headers
from litellm.proxy.common_utils.user_api_key_cache import (
    AUTH_OBJECTS_TARGET,
    UserApiKeyCache,
    get_management_object_ttl,
)
from litellm.proxy.spend_tracking.carried_budget_state import carry_organization_budget_state
from litellm.redis.stores.spend_counters import key_counter, team_member_counter
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.rust_bridge._native import RustIdentity
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityFailure, ManagedAgentContext
from litellm.types.proxy.auth.auth_checks import UserNotFoundError
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router

JWKS_FETCH_ATTEMPTS: Final = 3
JWKS_FETCH_RETRY_BACKOFF_SECONDS: Final = 0.25
OIDC_DISCOVERY_PATH: Final = ".well-known/openid-configuration"


class AgentLookup(Protocol):
    """The registered-agent lookups a JWT agent claim is matched against."""

    def get_agent_by_id(self, agent_id: str) -> AgentResponse | None: ...

    def get_agent_by_name(self, agent_name: str) -> AgentResponse | None: ...


class NoRegisteredAgents:
    """The lookup in force until the proxy binds its agent registry: no agent is registered."""

    def get_agent_by_id(self, agent_id: str) -> None:
        return None

    def get_agent_by_name(self, agent_name: str) -> None:
        return None


@dataclass(frozen=True, slots=True)
class HttpCall:
    """The proxy request a key or JWT admission authenticates. Calls that only verify or map a JWT (SSO,
    MCP token flows, pass-through) have none, and the core never asks them a question that needs one."""

    request: Request
    request_data: dict[str, object]  # mutable-ok: model normalization and budget fallbacks rewrite the request body
    start_time: datetime


def _attributes(team: LiteLLM_TeamTableCachedObj) -> Mapping[str, object]:
    return vars(team)


@dataclass(frozen=True, slots=True)
class PythonAuthHost:
    route: str
    parent_otel_span: Span | None
    http: HttpCall | None
    prisma_client: PrismaClient | None
    user_api_key_cache: UserApiKeyCache
    proxy_logging_obj: ProxyLogging
    llm_router: Router | None
    llm_model_list: Sequence[object] | None
    model_max_budget_limiter: PROXY_VirtualKeyModelMaxBudgetLimiter
    litellm_proxy_admin_name: str
    litellm_jwtauth: LiteLLM_JWTAuth
    agent_lookup: AgentLookup
    http_handler: HTTPHandler
    api_key: str | None
    authorization: Authorization | None = None

    def call(self) -> HttpCall:
        if self.http is None:
            raise RuntimeError("rust auth asked about the proxy request on a call that has none")
        return self.http

    def _authorization(self) -> Authorization:
        if self.authorization is None:
            raise RuntimeError("rust auth asked an authorization question on a call that authorizes nothing")
        return self.authorization

    def _db(self) -> PrismaClient:
        if self.prisma_client is None:
            raise RuntimeError("rust auth asked for a database read on a proxy without a database")
        return self.prisma_client

    def _budgets(self) -> BudgetChecks:
        http: Final = self.call()
        return BudgetChecks(
            request=http.request,
            request_data=http.request_data,
            route=self.route,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            proxy_logging_obj=self.proxy_logging_obj,
            llm_router=self.llm_router,
            llm_model_list=self.llm_model_list,
            model_max_budget_limiter=self.model_max_budget_limiter,
        )

    def _store(self, *, check_cache_only: bool = False) -> IdentityStore:
        return IdentityStore(
            self.prisma_client,
            self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
            check_cache_only=check_cache_only,
        )

    def _grants(self, prisma_client: PrismaClient | None) -> GrantResolver:
        return GrantResolver(
            prisma_client,
            self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
            load_user=get_user_object,
            load_membership=get_team_membership,
        )

    # Key path

    async def end_user(self, token: str, resolved: UserAPIKeyAuth | None) -> EndUserRow:
        """Raises `litellm.BudgetExceededError` for an end user over budget."""
        http: Final = self.call()
        raw_end_user_id: Final = (
            resolved.end_user_id if resolved is not None else None
        ) or get_end_user_id_from_request_body(http.request_data, safe_get_request_headers(http.request))
        end_user_id: Final = await resolve_and_validate_end_user_id(
            raw_end_user_id=raw_end_user_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
            route=self.route,
        )
        if self.prisma_client is not None:
            await prefetch_identity_keys(
                identity_cache_keys(token, end_user_id=end_user_id, key_is_resolved=resolved is not None),
                user_api_key_cache=self.user_api_key_cache,
                before_read=arm_routing_cooldowns,
            )
        if not end_user_id:
            return EndUserRow(end_user_id=end_user_id, end_user=None, params=MappingProxyType({}))
        return await self._load_end_user(end_user_id)

    async def _load_end_user(self, end_user_id: str) -> EndUserRow:
        params: Final[dict[str, object]] = {"end_user_id": end_user_id}  # mutable-ok: the budget helper fills it
        end_user: LiteLLM_EndUserTable | None = None  # rebind-ok: stays None when the end user cannot be read
        try:
            with tracer.trace("litellm.proxy.auth.get_end_user_object"):
                end_user = await get_end_user_object(  # rebind-ok: see above
                    end_user_id=end_user_id,
                    prisma_client=self.prisma_client,
                    user_api_key_cache=self.user_api_key_cache,
                    parent_otel_span=self.parent_otel_span,
                    proxy_logging_obj=self.proxy_logging_obj,
                    route=self.route,
                )
            if end_user is not None:
                params["allowed_model_region"] = end_user.allowed_model_region
                if end_user.litellm_budget_table is not None:
                    apply_budget_limits_to_end_user_params(
                        end_user_params=params, budget_info=end_user.litellm_budget_table, end_user_id=end_user_id
                    )
            elif litellm.max_end_user_budget_id is not None:
                default_budget: Final = await get_default_end_user_budget(
                    prisma_client=self.prisma_client,
                    user_api_key_cache=self.user_api_key_cache,
                    parent_otel_span=self.parent_otel_span,
                )
                if default_budget is not None:
                    apply_budget_limits_to_end_user_params(
                        end_user_params=params, budget_info=default_budget, end_user_id=end_user_id
                    )
        except litellm.BudgetExceededError:
            raise
        except Exception as e:  # noqa: BLE001  # an unreadable end user leaves the request without end-user limits
            verbose_proxy_logger.debug("Unable to find user in db. Error - %s", e)
        return EndUserRow(end_user_id=end_user_id, end_user=end_user, params=MappingProxyType(params))

    async def cached_key(self, hash: str) -> UserAPIKeyAuth | None:
        """Raises when the key is not cached; the core reads that as a miss."""
        with tracer.trace("litellm.proxy.auth.get_key_object_check_cache"):
            token: Final = IdentityStore.key_from_principal(
                await self._store(check_cache_only=True).resolve(hashed_token=hash)
            )
        token.via_virtual_key = True
        return token

    async def ui_key(self, token: str) -> UserAPIKeyAuth | None:
        return ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(token)

    async def with_session_grants(
        self,
        token: UserAPIKeyAuth,
        user: LiteLLM_UserTable,
        team: LiteLLM_TeamTableCachedObj | None,
        membership: LiteLLM_TeamMembership | None,
    ) -> UserAPIKeyAuth:
        return UserAPIKeyAuth.model_validate(
            MappingProxyType(
                {
                    **token.model_dump(exclude_none=True),
                    **team_grants(team, membership, user.user_id),
                    "user_role": get_user_role(user),
                    "models": () if team is not None else user_models(user),
                }
            )
        )

    async def evict_key(self, hash: str) -> None:
        await delete_cache_key_object(
            hashed_token=hash, user_api_key_cache=self.user_api_key_cache, proxy_logging_obj=self.proxy_logging_obj
        )

    async def cached_team(self, team_id: str) -> LiteLLM_TeamTableCachedObj:
        """Raises when the team is not cached; the core reads that as no cached team."""
        return await get_team_object(
            team_id=team_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
            check_cache_only=True,
        )

    async def with_team(self, token: UserAPIKeyAuth, team: LiteLLM_TeamTableCachedObj) -> UserAPIKeyAuth:
        fields: Final = UserAPIKeyAuth.model_fields
        return token.model_copy(
            update={f"team_{name}": value for name, value in _attributes(team).items() if f"team_{name}" in fields}
        )

    async def db_key(self, hash: str, end_user: EndUserRow) -> UserAPIKeyAuth | None:
        try:
            with tracer.trace("litellm.proxy.auth.get_key_object_from_db"):
                token: Final = IdentityStore.key_from_principal(await self._store().resolve(hashed_token=hash))
        except KeyNotFoundError:
            return None
        params: Final = end_user.params
        per_model_limits: Final = {
            name: params[name] for name in ("end_user_model_tpm_limit", "end_user_model_rpm_limit") if name in params
        }
        return token.model_copy(
            update={
                "end_user_id": params.get("end_user_id"),
                "end_user_tpm_limit": params.get("end_user_tpm_limit"),
                "end_user_rpm_limit": params.get("end_user_rpm_limit"),
                "end_user_tpd_limit": params.get("end_user_tpd_limit"),
                "allowed_model_region": params.get("allowed_model_region"),
                **per_model_limits,
            }
        )

    async def with_temp_budget(self, token: UserAPIKeyAuth) -> UserAPIKeyAuth:
        return update_key_budget_with_temp_budget_increase(token)

    async def normalize_model(self, token: UserAPIKeyAuth) -> None:
        http: Final = self.call()
        await normalize_model_for_auth(http.request_data, token, http.request, self.route)

    async def prefetch_auth_objects(self, token: UserAPIKeyAuth, end_user: EndUserRow) -> None:
        await prefetch_referenced_auth_objects(
            token,
            end_user_id=end_user.end_user_id,
            user_api_key_cache=self.user_api_key_cache,
            prisma_client=self.prisma_client,
        )

    async def key_owner(self, user_id: str) -> LiteLLM_UserTable | None:
        """Raises when the owner cannot be read; the core treats that as no owner."""
        with tracer.trace("litellm.proxy.auth.get_user_object"):
            return await get_user_object(
                user_id=user_id,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
                user_id_upsert=False,
                parent_otel_span=self.parent_otel_span,
                proxy_logging_obj=self.proxy_logging_obj,
            )

    async def key_skips_budget_checks(self, token: UserAPIKeyAuth) -> bool:
        return await self._budgets().key_skips_budget_checks(token)

    async def team_member_max_budget(self, user_id: str, team_id: str) -> float | None:
        return await self._budgets().team_member_max_budget(user_id, team_id)

    async def key_model_budgets(self, key: UserAPIKeyAuth, owner: LiteLLM_UserTable | None) -> None:
        await self._budgets().key_model_budgets(key, owner)

    async def throttle_key(self, token: UserAPIKeyAuth, share: float) -> None:
        token.budget_throttle_pct = share  # rebind-ok: the rate limiter reads the request's throttle off the token

    async def key_team(self, team_id: str) -> LiteLLM_TeamTableCachedObj | None:
        try:
            with tracer.trace("litellm.proxy.auth.get_team_object"):
                return await get_team_object(
                    team_id=team_id,
                    prisma_client=self.prisma_client,
                    user_api_key_cache=self.user_api_key_cache,
                    parent_otel_span=self.parent_otel_span,
                    proxy_logging_obj=self.proxy_logging_obj,
                )
        except HTTPException:
            return None

    async def object_permission(self, object_permission_id: str) -> LiteLLM_ObjectPermissionTable | None:
        if self.prisma_client is None:
            return None
        return await get_object_permission(
            object_permission_id=object_permission_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
        )

    async def project(self, project_id: str) -> LiteLLM_ProjectTableCachedObj | None:
        return await get_project_object(
            project_id=project_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            proxy_logging_obj=self.proxy_logging_obj,
        )

    async def global_proxy_spend(self) -> float | None:
        return await global_proxy_spend(self._db(), self.user_api_key_cache)

    async def key_budget_alert(
        self,
        token: UserAPIKeyAuth,
        owner: LiteLLM_UserTable | None,
        kind: Literal["team_member", "proxy", "max_budget_warning", "token_budget", "soft_budget"],
        spend: float | None,
        max_budget: float | None,
    ) -> None:
        match kind:
            case "team_member":
                self._budgets().team_member_alert(token, owner, spend or 0.0, max_budget or 0.0)
            case "max_budget_warning":
                send_key_max_budget_warning(token, owner, self.proxy_logging_obj)
            case "token_budget":
                send_key_token_budget_alert(token, owner, spend or 0.0, self.proxy_logging_obj)
            case "soft_budget":
                send_key_soft_budget_alert(token, owner, self.proxy_logging_obj)
            case "proxy":
                proxy_budget_alert(
                    token,
                    spend or 0.0,
                    proxy_logging_obj=self.proxy_logging_obj,
                    litellm_proxy_admin_name=self.litellm_proxy_admin_name,
                )
            case unreachable:
                assert_never(unreachable)

    # JWT path

    # TODO: the JWT IO is answered here until the bridge answers it natively, with `HttpClientPool` for the
    # fetches and `redis-store` for the JWT keyspaces

    async def _get_with_transient_retries(self, url: str) -> httpx.Response | None:
        for attempt in range(1, JWKS_FETCH_ATTEMPTS):
            try:
                return await self.http_handler.get(url)
            except httpx.TransportError as e:
                verbose_proxy_logger.warning(
                    "JWT Auth: %s fetching %s (attempt %s/%s), retrying: %s",
                    type(e).__name__,
                    url,
                    attempt,
                    JWKS_FETCH_ATTEMPTS,
                    e,
                )
                await asyncio.sleep(JWKS_FETCH_RETRY_BACKOFF_SECONDS * attempt)
        try:
            return await self.http_handler.get(url)
        except httpx.TransportError as e:
            verbose_proxy_logger.error(
                "JWT Auth: %s fetching %s after %s attempts", type(e).__name__, url, JWKS_FETCH_ATTEMPTS
            )
            return None

    async def fetch_json(self, url: str) -> object:
        response: Final = await self._get_with_transient_retries(url)
        if response is None:
            return None
        discovery: Final = OIDC_DISCOVERY_PATH in url
        if response.status_code != 200:
            what: Final = "OIDC discovery endpoint" if discovery else "JWKS endpoint"
            raise Exception(f"JWT Auth: {what} {url} returned status {response.status_code}: {response.text}")
        try:
            return response.json()
        except Exception as e:
            if discovery:
                raise Exception(f"JWT Auth: Failed to parse OIDC discovery document at {url}: {e}")
            verbose_proxy_logger.error("Error parsing response: %s. Original Response: %s", e, response.text)
            raise Exception(f"Error parsing response: {e}. Check server logs for original response.")

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def fetch_userinfo(self, endpoint: str, token: str) -> object:
        verbose_proxy_logger.debug("Calling OIDC UserInfo endpoint: %s", endpoint)
        try:
            response: Final = await self.http_handler.get(
                url=endpoint, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
            )
            if response.status_code != 200:
                raise Exception(f"OIDC UserInfo endpoint returned status {response.status_code}: {response.text}")
            return response.json()
        except Exception as e:
            verbose_proxy_logger.error("Error fetching OIDC UserInfo: %s", e)
            raise Exception(f"Failed to fetch OIDC UserInfo: {e}")

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def cached_json(self, key: str) -> object:
        return await self.user_api_key_cache.async_get_cache(key)

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def cache_json(self, key: str, value: object, ttl_seconds: float) -> None:
        await self.user_api_key_cache.async_set_cache(key=key, value=value, ttl=ttl_seconds)

    async def forget_json(self, key: str) -> None:
        await self.user_api_key_cache.async_delete_cache(key)

    async def _team(
        self, team_id: str, *, upsert: bool = False, check_db_only: bool = False
    ) -> FoundTeam | TeamMissing:
        try:
            team: Final = await get_team_object(
                team_id=team_id,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
                parent_otel_span=self.parent_otel_span,
                proxy_logging_obj=self.proxy_logging_obj,
                check_db_only=check_db_only or None,
                team_id_upsert=upsert,
            )
        except TeamNotFoundError:
            return TeamMissing.ABSENT
        except HTTPException:
            return TeamMissing.UNREADABLE
        return team_facts(team)

    async def team(self, team_id: str, upsert: bool) -> FoundTeam | TeamMissing:
        return await self._team(team_id, upsert=upsert)

    async def granting_team(self, team_id: str) -> FoundTeam | TeamMissing:
        return await self._team(team_id, check_db_only=True)

    async def team_by_alias(self, alias: str) -> FoundTeam | AliasMissing:
        if self.prisma_client is None:
            return AliasMissing.NO_DB
        try:
            team: Final = await get_team_object_by_alias(
                team_alias=alias,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
                parent_otel_span=self.parent_otel_span,
                proxy_logging_obj=self.proxy_logging_obj,
            )
        except HTTPException as e:
            match e.status_code:
                case 404:
                    return AliasMissing.ABSENT
                case 400:
                    return AliasMissing.AMBIGUOUS
                case _:
                    raise
        return team_facts(team)

    async def load_team_membership(self, user_id: str, team_id: str) -> LiteLLM_TeamMembership | None:
        return await get_team_membership(
            user_id=user_id,
            team_id=team_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
        )

    async def team_model_access(self, team: LiteLLM_TeamTable, model: str) -> bool:
        try:
            await can_team_access_model(
                model=model, team_object=team, llm_router=self.llm_router, team_model_aliases=team_model_aliases(team)
            )
        except ProxyException:
            return False
        return True

    async def user(
        self, user_id: str, user_email: str | None, sso_user_id: str | None, upsert: bool
    ) -> LiteLLM_UserTable | None:
        try:
            user: Final = await get_user_object(
                user_id=user_id,
                user_email=user_email,
                sso_user_id=sso_user_id,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
                user_id_upsert=upsert,
                parent_otel_span=self.parent_otel_span,
                proxy_logging_obj=self.proxy_logging_obj,
            )
        except UserNotFoundError:
            return None
        return user

    async def resolve_user_identity(
        self, user_id: str, user_email: str | None, sso_user_id: str | None, upsert: bool, team_id: str | None
    ) -> tuple[LiteLLM_UserTable, LiteLLM_TeamMembership | None] | None:
        try:
            user, membership, _ = await self._grants(self.prisma_client).resolve_identity(
                UserLookup(user_id=user_id, user_email=user_email, sso_user_id=sso_user_id, upsert=upsert),
                team_id=team_id,
            )
        except UserNotFoundError:
            return None
        return user, membership

    async def _recache_user(self, user: LiteLLM_UserTable) -> LiteLLM_UserTable:
        await self.user_api_key_cache.async_set_cache(
            key=user.user_id,
            value=user,
            model_type=LiteLLM_UserTable,
            ttl=get_management_object_ttl(self.user_api_key_cache),
        )
        return user

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def set_user_role(self, user: LiteLLM_UserTable, role: str) -> LiteLLM_UserTable:
        await UserRepository(self._db()).table.update(where={"user_id": user.user_id}, data={"user_role": role})
        return await self._recache_user(user.model_copy(update={"user_role": role}))

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def patch_team_membership(
        self, user: LiteLLM_UserTable, add: Sequence[str], remove: Sequence[str]
    ) -> LiteLLM_UserTable:
        from litellm.proxy.management_endpoints.scim.scim_v2 import (  # noqa: PLC0415  # scim imports the proxy
            patch_team_membership,
        )

        await patch_team_membership(
            user_id=user.user_id, teams_ids_to_add_user_to=list(add), teams_ids_to_remove_user_from=list(remove)
        )
        kept: Final = tuple(team_id for team_id in dict.fromkeys(user.teams or []) if team_id not in remove)
        return await self._recache_user(user.model_copy(update={"teams": [*kept, *add]}))

    async def add_team_member(self, user_id: str, team_id: str) -> None:
        from litellm.proxy.management_endpoints.team_endpoints import (  # noqa: PLC0415  # team endpoints import auth
            team_member_add,
        )

        try:
            await team_member_add(
                data=TeamMemberAddRequest(member=Member(user_id=user_id, role="user"), team_id=team_id),
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
            )
            verbose_proxy_logger.debug("Successfully added user %s to team %s", user_id, team_id)
        except ProxyException as e:
            if e.type != ProxyErrorTypes.team_member_already_in_team:
                raise
            verbose_proxy_logger.debug("User %s is already a member of team %s", user_id, team_id)

    async def org(self, org_id: str) -> LiteLLM_OrganizationTable | None:
        return await get_org_object(
            org_id=org_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
        )

    async def org_by_alias(self, alias: str) -> LiteLLM_OrganizationTable | None:
        verbose_proxy_logger.info("JWT Auth: Resolving org by alias: '%s'", alias)
        org: Final = await get_org_object_by_alias(
            org_alias=alias,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
        )
        if org is not None:
            verbose_proxy_logger.info("JWT Auth: Resolved org_alias='%s' to org_id='%s'", alias, org.organization_id)
        return org

    async def jwt_end_user(self, end_user_id: str) -> EndUserRow:
        end_user: Final = await get_end_user_object(
            end_user_id=end_user_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
            route=self.route,
        )
        return EndUserRow(end_user_id=end_user_id, end_user=end_user, params=MappingProxyType({}))

    async def finish_jwt(self, identity: RustIdentity) -> UserAPIKeyAuth:
        """The JWT path's project lookup and per-model user budget check, which the key path runs as its
        own steps. Raises `litellm.BudgetExceededError` for a user over its model budget."""
        budgets: Final = self._budgets()
        token: Final = auth_identity.finished_jwt_token(identity, self.api_key, self.parent_otel_span)
        project_id: Final = token.project_id
        finished: Final = with_project(token, None if project_id is None else await self.project(project_id))
        await budgets.jwt_user_model(finished)
        return finished

    async def key_mapping_hash(self, claim_field: str, claim_value: str, issuer: str | None) -> str | None:
        return await get_jwt_key_mapping_object(
            jwt_claim_name=claim_field, jwt_claim_value=claim_value, prisma_client=self._db(), jwt_issuer=issuer
        )

    async def key_by_hash(self, hash: str) -> UserAPIKeyAuth:
        return IdentityStore.key_from_principal(await self._store().resolve(hashed_token=hash))

    async def reusable_key_hash(self, user_id: str, team_id: str | None) -> str | None:
        key: Final = await VerificationTokenRepository(self._db()).find_newest_reusable_key(user_id, team_id)
        return None if key is None else key.token

    async def create_key_mapping(
        self,
        claim_field: str,
        claim_value: str,
        issuer: str | None,
        behavior: str,
        existing_hash: str | None,
        team_id: str | None,
        user_id: str | None,
        org_id: str | None,
        agent_id: str | None,
    ) -> tuple[str | None, bool]:
        return await jwt_auto_register.create_key_mapping(
            self._db(), claim_field, claim_value, issuer, behavior, existing_hash, team_id, user_id, org_id, agent_id
        )

    async def managed_agent(self, claims: Mapping[str, object]) -> ManagedAgentContext | None:
        return await resolve_managed_agent(claims, self.prisma_client, cache=self.user_api_key_cache)

    async def record_agent_authentication(self, agent: ManagedAgentContext) -> AgentIdentityFailure | None:
        return await AgentIdentityStore.from_client(self.prisma_client).record_authentication(agent)

    async def registered_agent(self, claim: str) -> RegisteredAgentFacts | None:
        agent: Final = self.agent_lookup.get_agent_by_id(claim) or self.agent_lookup.get_agent_by_name(claim)
        if agent is None:
            return None
        return RegisteredAgentFacts(
            agent_id=agent.agent_id,
            identity_managed=bool(agent.identity_managed),
            has_identity=agent.identity is not None,
            has_legacy_identity=has_legacy_identity(agent.litellm_params),
        )

    async def delegated_agent_team(
        self,
        user_id: str | None,
        agent_id: str,
        team_id: str | None,
        explicit_team: bool,
        allowed_team_ids: Sequence[str] | None,
    ) -> str | None:
        from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (  # noqa: PLC0415  # imports auth
            resolve_delegated_agent_team,
        )

        return await resolve_delegated_agent_team(
            user_id,
            agent_id,
            team_id,
            explicit_team=explicit_team,
            allowed_team_ids=None if allowed_team_ids is None else frozenset(allowed_team_ids),
        )

    # Authorization

    async def unpriced_models(self) -> Sequence[str]:
        return self._authorization().unpriced_models()

    async def agent_requires_trace_id(self, agent_id: str) -> bool:
        return agent_requires_trace_id(agent_id)

    async def has_trace_id(self, headers_only: bool) -> bool:
        return self._authorization().has_trace_id(headers_only)

    async def team_organization(self, team_id: str) -> str | None:
        try:
            team: Final = await get_team_object(
                team_id=team_id,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
                proxy_logging_obj=self.proxy_logging_obj,
            )
        except HTTPException:
            return None
        return team.organization_id

    async def merge_budget_tags(self) -> None:
        self._authorization().merge_budget_tags()

    async def check_team_model_access(self, membership_read: bool, membership: LiteLLM_TeamMembership | None) -> None:
        await self._authorization().check_team_model_access(membership_read, membership)

    async def check_model_grants(self) -> None:
        await self._authorization().check_model_grants()

    async def check_project(self) -> None:
        await self._authorization().check_project()

    async def stamp_access_groups(
        self, membership_read: bool, membership: LiteLLM_TeamMembership | None
    ) -> Sequence[str]:
        return await self._authorization().stamp_access_groups(membership_read, membership)

    async def spend(
        self,
        scope: SpendScope,
        entity_id: str,
        team_id: str | None,
        window: str | None,
        recorded: float,
        max_budget: float,
    ) -> float:
        if self.authorization is None and scope in ("team_member", "key"):
            from litellm.proxy.proxy_server import get_current_spend  # noqa: PLC0415  # proxy global

            return await get_current_spend(
                counter_key=team_member_counter(entity_id, team_id or "")
                if scope == "team_member"
                else key_counter(entity_id),
                fallback_spend=recorded,
                max_budget=max_budget,
            )
        return await self._authorization().spend(scope, entity_id, team_id, window, recorded, max_budget)

    async def organization_budget(self, org_id: str) -> tuple[float, float] | None:
        token: Final = self._authorization().valid_token
        if token is None or self.prisma_client is None:
            return None
        try:
            org: Final = await get_org_object(
                org_id=org_id,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
                proxy_logging_obj=self.proxy_logging_obj,
                include_budget_table=True,
            )
        except Exception:  # noqa: BLE001  # an organization that does not load sets no budget, as common_checks had it
            return None
        if org is None:
            return None
        carry_organization_budget_state(valid_token=token, org_table=org)
        max_budget: Final = org.litellm_budget_table.max_budget if org.litellm_budget_table is not None else None
        return None if max_budget is None else (max_budget, org.spend or 0.0)

    async def tag_budgets(self) -> Sequence[tuple[str, float, float]]:
        tags: Final = get_tags_from_request_body(request_body=self._authorization().request_body)
        if self.prisma_client is None or not tags:
            return ()
        objects: Final = await get_tag_objects_batch(
            tag_names=tags,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            proxy_logging_obj=self.proxy_logging_obj,
        )
        return tuple(budget for budget in (_tag_budget(tag, objects.get(tag)) for tag in tags) if budget is not None)

    async def model_access_group_budgets(self, groups: Sequence[str]) -> Sequence[tuple[str, float, float]]:
        if self.prisma_client is None or not groups:
            return ()
        budgets: Final = await get_model_access_group_budgets_batch(
            access_group_names=list(groups),
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
        )
        return tuple(
            budget for budget in (_group_budget(group, budgets.get(group)) for group in groups) if budget is not None
        )

    async def team_member_budget_state(self, membership: LiteLLM_TeamMembership | None) -> MemberBudget:
        team: Final = self._authorization().team_object
        own: Final = membership.litellm_budget_table if membership is not None else None
        default_budget_id: Final = (team.metadata or {}).get("team_member_budget_id") if team is not None else None
        team_default: Final = (
            await get_team_member_default_budget(
                budget_id=default_budget_id,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
            )
            if (own is None or own.max_budget is None) and isinstance(default_budget_id, str)
            else None
        )
        return member_budget(membership, team_default)

    async def budget_alert(
        self, kind: AlertKind, org_id: str | None, spend: float | None, max_budget: float | None
    ) -> None:
        await self._authorization().budget_alert(kind, org_id, spend, max_budget)

    async def check_guardrail_modification(self) -> None:
        self._authorization().check_guardrail_modification()

    async def check_object_access(self) -> None:
        await self._authorization().check_object_access()

    async def custom_validate(self, claims: Mapping[str, object]) -> bool:
        validate: Final = self.litellm_jwtauth.custom_validate
        if validate is None:
            raise RuntimeError("rust auth ran custom_validate on a proxy that configures none")
        return bool(validate(dict(claims)))


def _tag_budget(tag: str, tag_object: LiteLLM_TagTable | None) -> tuple[str, float, float] | None:
    budget: Final = tag_object.litellm_budget_table if tag_object is not None else None
    if tag_object is None or budget is None or budget.max_budget is None:
        return None
    return tag, budget.max_budget, tag_object.spend or 0.0


def _group_budget(group: str, budget: ModelAccessGroupBudget | None) -> tuple[str, float, float] | None:
    if budget is None or budget.max_budget is None:
        return None
    return group, budget.max_budget, budget.spend


def proxy_host(
    route: str,
    api_key: str | None,
    *,
    authorization: Authorization | None = None,
    parent_otel_span: Span | None = None,
    http: HttpCall | None = None,
) -> PythonAuthHost:
    """The host for one call, over the proxy's live globals."""
    from litellm.proxy.agent_endpoints.agent_registry import (  # noqa: PLC0415  # the registry imports the proxy
        global_agent_registry,
    )
    from litellm.proxy.proxy_server import (  # noqa: PLC0415  # runtime globals, rebound on every config load
        jwt_http_handler,
        litellm_jwtauth,
        litellm_proxy_admin_name,
        llm_model_list,
        llm_router,
        model_max_budget_limiter,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    return PythonAuthHost(
        route=route,
        parent_otel_span=parent_otel_span,
        http=http,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        llm_router=llm_router,
        llm_model_list=llm_model_list,
        model_max_budget_limiter=model_max_budget_limiter,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
        litellm_jwtauth=litellm_jwtauth,
        agent_lookup=global_agent_registry,
        http_handler=jwt_http_handler,
        api_key=api_key,
        authorization=authorization,
    )
