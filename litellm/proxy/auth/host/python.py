"""``PythonAuthHost``: what gateway-auth-core reaches outside itself, served from the proxy's Python state.

The core parks on a `RustAuthStep`, and `answer.answer` awaits the matching method here on the caller's task.
A method reads or writes one named thing and returns the row it found, or runs a step `steps/` owns. Each
method answers the `AuthHost` method of the same name in `gateway-auth-core/src/host.rs`, which says what it is
for and when it leaves Python."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from fastapi import HTTPException, Request

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
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
from litellm.proxy.auth.auth_checks import (
    ExperimentalUIJWTToken,
    TeamNotFoundError,
    can_team_access_model,
    delete_cache_key_object,
    get_default_end_user_budget,
    get_end_user_object,
    get_jwt_key_mapping_object,
    get_object_permission,
    get_org_object,
    get_org_object_by_alias,
    get_project_object,
    get_team_membership,
    get_team_object,
    get_team_object_by_alias,
    get_user_object,
    get_user_role,
    resolve_and_validate_end_user_id,
)
from litellm.proxy.auth.auth_object_prefetch import prefetch_identity_keys
from litellm.proxy.auth.host.facts import (
    AliasMissing,
    EndUserRow,
    FoundTeam,
    RegisteredAgentFacts,
    TeamMissing,
    end_user_params,
    team_facts,
    with_project,
)
from litellm.proxy.auth.host.identity import auto_registered_key
from litellm.proxy.auth.host.steps import jwt_auto_register
from litellm.proxy.auth.host.steps.budgets import BudgetChecks, proxy_budget_alert
from litellm.proxy.auth.jwt import auth_from_result, jwt_result
from litellm.proxy.auth.resolvers.exceptions import KeyNotFoundError
from litellm.proxy.auth.resolvers.grants import (
    GrantResolver,
    LookupDegraded,
    ResolvedGrants,
    UserLookup,
    raise_public,
    user_models,
)
from litellm.proxy.auth.resolvers.store import IdentityStore
from litellm.proxy.auth.team_grants import team_grants, team_model_aliases
from litellm.proxy.auth.user_api_key_auth import (
    arm_routing_cooldowns,
    identity_cache_keys,
    normalize_model_for_auth,
    prefetch_referenced_auth_objects,
    update_key_budget_with_temp_budget_increase,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    AUTH_OBJECTS_TARGET,
    UserApiKeyCache,
    get_management_object_ttl,
)
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.rust_bridge._native import RustIdentity
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentIdentityFailure, ManagedAgentContext
from litellm.types.proxy.auth.auth_checks import UserNotFoundError
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

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


class SharedFetches:
    """Callers that fetch one URL at the same time share the first caller's request and its outcome.

    When a cached JWKS expires, every JWT request in flight misses the cache together, and without this each
    one would fetch the identity provider itself. The core cannot hold a lock from its cache read to the fetch,
    since it only ever waits on a host answer, so the sharing lives in the host. The first caller fetches on its
    own task and the rest wait on its future. When it is cancelled, one of them fetches instead."""

    def __init__(self) -> None:
        self._running: Final[dict[str, asyncio.Future[object]]] = {}  # mutable-ok: fetches in flight, keyed by url

    async def fetch(self, url: str, fetch: Callable[[str], Awaitable[object]]) -> object:
        running: Final = self._running.get(url)
        if running is not None:
            _ = await asyncio.wait((running,))
            return await self.fetch(url, fetch) if running.cancelled() else running.result()
        shared: Final[asyncio.Future[object]] = asyncio.get_running_loop().create_future()
        self._running[url] = shared
        try:
            body: Final = await fetch(url)
        except asyncio.CancelledError:
            _ = shared.cancel()
            raise
        except Exception as e:
            shared.set_exception(e)
            _ = shared.exception()
            raise
        finally:
            del self._running[url]
        shared.set_result(body)
        return body


_JWT_FETCHES: Final = SharedFetches()

_PER_MODEL_END_USER_LIMITS: Final = frozenset({"end_user_model_tpm_limit", "end_user_model_rpm_limit"})


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
    shared_fetches: SharedFetches
    api_key: str | None

    def call(self) -> HttpCall:
        if self.http is None:
            raise RuntimeError("rust auth asked about the proxy request on a call that has none")
        return self.http

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
            load_team=get_team_object,
            load_membership=get_team_membership,
        )

    # Key path

    async def validated_end_user_id(self, end_user_id: str) -> str | None:
        return await resolve_and_validate_end_user_id(
            raw_end_user_id=end_user_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
            proxy_logging_obj=self.proxy_logging_obj,
            route=self.route,
        )

    async def prefetch_identity(self, token: str | None, end_user_id: str | None) -> None:
        await prefetch_identity_keys(
            identity_cache_keys(token or "", end_user_id=end_user_id, key_is_resolved=token is None),
            user_api_key_cache=self.user_api_key_cache,
            before_read=arm_routing_cooldowns,
        )

    async def end_user(self, end_user_id: str | None) -> EndUserRow:
        """Raises `litellm.BudgetExceededError` for an end user over budget."""
        if not end_user_id:
            return EndUserRow(end_user_id=end_user_id, end_user=None, params={})
        return await self._load_end_user(end_user_id)

    async def _load_end_user(self, end_user_id: str) -> EndUserRow:
        try:
            end_user, budget = await self._end_user_and_budget(end_user_id)
        except litellm.BudgetExceededError:
            raise
        except Exception as e:  # noqa: BLE001  # an unreadable end user leaves the request without end-user limits
            verbose_proxy_logger.debug("Unable to find user in db. Error - %s", e)
            return EndUserRow(end_user_id=end_user_id, end_user=None, params={"end_user_id": end_user_id})
        return EndUserRow(
            end_user_id=end_user_id, end_user=end_user, params=end_user_params(end_user_id, end_user, budget)
        )

    async def _end_user_and_budget(
        self, end_user_id: str
    ) -> tuple[LiteLLM_EndUserTable | None, LiteLLM_BudgetTable | None]:
        """The end user with its own budget, or the default end-user budget for one the proxy has no row for."""
        with tracer.trace("litellm.proxy.auth.get_end_user_object"):
            end_user: Final = await get_end_user_object(
                end_user_id=end_user_id,
                prisma_client=self.prisma_client,
                user_api_key_cache=self.user_api_key_cache,
                parent_otel_span=self.parent_otel_span,
                proxy_logging_obj=self.proxy_logging_obj,
                route=self.route,
            )
        if end_user is not None:
            return end_user, end_user.litellm_budget_table
        if litellm.max_end_user_budget_id is None:
            return None, None
        return None, await get_default_end_user_budget(
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            parent_otel_span=self.parent_otel_span,
        )

    async def cached_key(self, hash: str) -> UserAPIKeyAuth | None:
        """Raises when the key is not cached; the core reads that as a miss."""
        with tracer.trace("litellm.proxy.auth.get_key_object_check_cache"):
            token: Final = IdentityStore.key_from_principal(
                await self._store(check_cache_only=True).resolve(hashed_token=hash)
            )
        token.via_virtual_key = True
        return token

    async def ui_key(self, token: str) -> UserAPIKeyAuth | None:
        key: Final = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(token)
        if key is None or not key.is_session_token or self.prisma_client is None:
            return key
        return await self._with_current_grants(key)

    async def _with_current_grants(self, key: UserAPIKeyAuth) -> UserAPIKeyAuth:
        """A ``lite login`` key with the grants of its user's and team's current rows, so a `/team/update` or a
        demotion applies without a re-login. When a row cannot be read for a reason unrelated to the caller, the
        minted grants stand."""
        outcome: Final = await self._grants(self.prisma_client).resolve(
            UserLookup(user_id=key.user_id), team_id=key.team_id
        )
        match outcome:
            case ResolvedGrants(user_object=LiteLLM_UserTable() as user, team_object=team, team_membership=membership):
                return UserAPIKeyAuth.model_validate(
                    MappingProxyType(
                        {
                            **key.model_dump(exclude_none=True),
                            **team_grants(team, membership, user.user_id),
                            "user_role": get_user_role(user),
                            "models": () if team is not None else user_models(user),
                        }
                    )
                )
            case ResolvedGrants():
                return key
            case LookupDegraded(error=error):
                verbose_proxy_logger.debug("Session token grants not refreshed, keeping minted grants: %s", error)
                return key
            case _:
                raise_public(outcome)

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
        return token.model_copy(
            update={
                "end_user_id": params.get("end_user_id"),
                "end_user_tpm_limit": params.get("end_user_tpm_limit"),
                "end_user_rpm_limit": params.get("end_user_rpm_limit"),
                "end_user_tpd_limit": params.get("end_user_tpd_limit"),
                "allowed_model_region": params.get("allowed_model_region"),
                **{name: limit for name, limit in params.items() if name in _PER_MODEL_END_USER_LIMITS},
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

    async def team_member_budget(self, key: UserAPIKeyAuth, owner: LiteLLM_UserTable | None) -> None:
        await self._budgets().team_member_budget(key, owner)

    async def key_budgets(self, key: UserAPIKeyAuth, owner: LiteLLM_UserTable | None, is_llm_api_route: bool) -> None:
        await self._budgets().key_budgets(key, owner, is_llm_api_route)

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

    async def proxy_budget_alert(self, token: UserAPIKeyAuth) -> None:
        await proxy_budget_alert(
            token,
            prisma_client=self._db(),
            user_api_key_cache=self.user_api_key_cache,
            proxy_logging_obj=self.proxy_logging_obj,
            litellm_proxy_admin_name=self.litellm_proxy_admin_name,
        )

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
        return await self.shared_fetches.fetch(url, self._fetch_json)

    async def _fetch_json(self, url: str) -> object:
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
        return EndUserRow(end_user_id=end_user_id, end_user=end_user, params={})

    async def finish_jwt(self, identity: RustIdentity.Jwt) -> UserAPIKeyAuth:
        """Raises `litellm.BudgetExceededError` for a user over its model budget."""
        return await self._finish(auth_from_result(jwt_result(identity, self.api_key), self.parent_otel_span))

    async def finish_auto_registered_jwt(
        self,
        key: UserAPIKeyAuth,
        claims: Mapping[str, object],
        end_user_id: str | None,
        user_email: str | None,
        org_id: str | None,
        user: LiteLLM_UserTable | None,
    ) -> UserAPIKeyAuth:
        """Raises `litellm.BudgetExceededError` for a user over its model budget."""
        return await self._finish(auto_registered_key(key, claims, end_user_id, user_email, org_id, user))

    async def _finish(self, token: UserAPIKeyAuth) -> UserAPIKeyAuth:
        """The JWT path's project lookup and per-model user budget check, which the key path runs as its
        own steps."""
        budgets: Final = self._budgets()
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

    async def custom_validate(self, claims: Mapping[str, object]) -> bool:
        validate: Final = self.litellm_jwtauth.custom_validate
        if validate is None:
            raise RuntimeError("rust auth ran custom_validate on a proxy that configures none")
        return bool(validate(dict(claims)))


def proxy_host(
    route: str,
    api_key: str | None,
    *,
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
        shared_fetches=_JWT_FETCHES,
        api_key=api_key,
    )
