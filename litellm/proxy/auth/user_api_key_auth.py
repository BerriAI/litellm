"""
This file handles authentication for the LiteLLM Proxy.

it checks if the user passed a valid API Key to the LiteLLM Proxy

Returns a UserAPIKeyAuth object if the API key is valid

"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import Final, NamedTuple, Protocol, Union, assert_never, cast

import fastapi
from fastapi import HTTPException, Request, WebSocket, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import TypeAdapter
from starlette.exceptions import WebSocketException

import litellm
from litellm._internal_context import service_target, set_request_parent_otel_span
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._service_logger import ServiceLogging
from litellm.constants import (
    CLIENT_REQUESTED_MODEL_SCOPE_KEY,
    GLOBAL_PROXY_SPEND_CACHE_KEY,
    LITELLM_PROXY_BUDGET_NAME,
    LITELLM_PROXY_MASTER_KEY_ALIAS,
    MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY,
)
from litellm.integrations.otel.model.config import is_otel_v2_enabled
from litellm.integrations.otel.runtime import phase_event, phase_span, seed_request_identity
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.dot_notation_indexing import get_nested_value
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias, resolve_served_model
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.proxy._types import *
from litellm.proxy.agent_endpoints.auth.agent_caller import agent_caller_from_headers
from litellm.proxy.auth.access_group_rules import access_group_ids_for_identity
from litellm.proxy.auth.auth_checks import (  # noqa: F401  # legacy module exports
    ExperimentalUIJWTToken,
    TeamNotFoundError,
    _cache_key_object,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _can_object_call_model,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _check_end_user_budget,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _delete_cache_key_object,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _get_user_role,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _is_model_cost_zero,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _is_user_proxy_admin,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _model_access_group_max_budget_check,  # pyright: ignore[reportPrivateUsage]  # reuse common-check group budget enforcement
    _team_member_max_budget_alert_check,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _unpriced_models_in_request,  # pyright: ignore[reportPrivateUsage]  # reuse common-check pricing enforcement
    _virtual_key_max_budget_alert_check,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _virtual_key_max_budget_check,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _virtual_key_soft_budget_check,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    cache_key_object,
    can_customer_access_model,
    can_key_call_model,
    can_object_call_model,
    can_project_access_model,
    can_team_access_model,
    check_end_user_budget,
    common_checks,
    delete_cache_key_object,
    get_default_end_user_budget,
    get_end_user_object,
    get_jwt_key_mapping_object,
    get_key_end_user_budget_id,
    get_object_permission,
    get_org_object_for_request,
    get_project_object,
    get_team_member_default_budget,
    get_team_membership,
    get_team_object,
    get_user_object,
    get_user_role,
    is_model_cost_zero,
    is_user_proxy_admin,
    is_valid_fallback_model,
    jwt_key_mapping_cache_key,
    key_access_group_grants_model,
    key_model_aliases_for_auth_check,
    model_max_budget_of,
    request_skips_budget_checks,
    resolve_and_validate_end_user_id,
    resolve_default_end_user_budget,
    route_skips_budget_checks,
    send_soft_budget_alert,
    send_token_budget_alert,
    stamp_matched_model_access_groups,
    team_member_max_budget_alert_check,
    virtual_key_max_budget_alert_check,
    virtual_key_max_budget_check,
    virtual_key_soft_budget_check,  # pyright: ignore[reportUnusedImport]  # backwards-compatible package export
)
from litellm.proxy.auth.auth_denial import denial_exception
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler
from litellm.proxy.auth.auth_method import AuthMethod
from litellm.proxy.auth.auth_object_prefetch import (
    AUTH_OBJECTS_TARGET,
    AuthObjectRefs,
    prefetch_auth_objects,
    prefetch_identity_keys,
)
from litellm.proxy.auth.auth_utils import (
    abbreviate_api_key,
    fallback_target_model_name,
    get_end_user_id_from_request_body,
    get_model_from_request,
    get_request_route,
    get_request_route_template,
    is_invalid_virtual_key_error,
    iter_request_fallback_targets,
    log_model_access_denial,
    normalize_request_route,
    pre_db_read_auth_checks,
    request_dispatched_to_pass_through_endpoint,
    request_dispatched_to_provider_pass_through,
    route_in_additonal_public_routes,
)
from litellm.proxy.auth.budget_throttle import budget_throttle_percentage, should_throttle_budget_exceeded
from litellm.proxy.auth.core.answers import (
    AdminKey,
    FallbackTarget,
    KeyFacts,
    KeyModelBudgets,
    ModelBudgetScope,
    NonAdminKey,
    OwnerFacts,
    PresentedKey,
    TeamMember,
    TeamMemberBudget,
    TeamMemberModelBudgetScope,
)
from litellm.proxy.auth.core.credential import (
    PresentedKeys,
    WebSocketDenial,
    WebSocketKeys,
    scheme_credential,
    select_api_key,
    select_websocket_api_key,
)
from litellm.proxy.auth.core.host import ModelBudgetHost
from litellm.proxy.auth.core.inputs import KeySettings, PlanFacts, PlanSettings, RequestFacts, RoutingOverride
from litellm.proxy.auth.core.outcome import (
    Allow,
    Anonymous,
    AnonymousReason,
    AnonymousValidator,
    CachedAdminKey,
    CredentialPlan,
    CustomAuthResult,
    CustomIdentity,
    CustomValidator,
    Deny,
    Grant,
    JwtValidator,
    KeyValidator,
    MasterKey,
    ModelBudgetExceeded,
    ModelBudgetResult,
    NoCustomIdentity,
    NoMasterKey,
    OAuth2Mode,
    OAuth2Validator,
    Outcome,
    PremiumFeature,
    PremiumRequired,
    ReplacedKey,
    UserAuthRoute,
    VirtualKey,
    WithinModelBudgets,
)
from litellm.proxy.auth.handle_jwt import JWTAuthManager, JWTHandler
from litellm.proxy.auth.network import TrustedProxyConfig, resolve_network_context
from litellm.proxy.auth.oauth2_check import Oauth2Handler
from litellm.proxy.auth.oauth2_proxy_hook import handle_oauth2_proxy_request
from litellm.proxy.auth.resolvers import CredentialRef, Principal
from litellm.proxy.auth.resolvers.grants import (
    GrantResolver,
    LookupDegraded,
    ResolvedGrants,
    UserLookup,
    raise_public,
    user_models,
)
from litellm.proxy.auth.resolvers.store import IdentityStore
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.selected_core import AUTH_CORE
from litellm.proxy.auth.team_grants import team_grants
from litellm.proxy.auth.trusted_proxy_utils import get_trusted_proxy_cidrs
from litellm.proxy.common_utils.cache_coordinator import EventDrivenCacheCoordinator
from litellm.proxy.common_utils.http_parsing_utils import (  # noqa: F401  # legacy module exports
    _read_request_body,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _safe_get_request_headers,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _safe_get_request_query_params,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _safe_set_request_parsed_body,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    is_opaque_pass_through_request,
    populate_request_with_path_params,
    read_raw_json_body,
    read_request_body,
    resolve_realtime_route_model,
    rewrite_request_model,
    safe_get_request_headers,
    safe_get_request_query_params,
    safe_set_request_parsed_body,
)
from litellm.proxy.common_utils.model_listing_utils import claude_code_requested_group
from litellm.proxy.common_utils.realtime_utils import (  # noqa: F401  # legacy module exports
    _realtime_request_body,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    realtime_request_body,
)
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    end_user_cache_key,
    end_user_restricted_registry_cache_key,
    model_access_group_registry_cache_key,
    team_membership_auth_cache_key,
)
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.proxy.spend_tracking.carried_budget_state import (
    carry_committed_budget_spend,
    carry_team_and_user_budget_state,
)
from litellm.proxy.spend_tracking.committed_spend import (
    bind_committed_admission_counters,
    committed_spend_tracked,
    read_admission_committed_spend,
)
from litellm.proxy.spend_tracking.spend_counter_batch import (
    bind_admission_counter_keys,
    release_spend_counter_batch,
    spend_counter_batch_scope,
)
from litellm.proxy.utils import (
    PrismaClient,
    ProxyLogging,
    normalize_route_for_root_path,
)
from litellm.redis.stores.spend_counters import (
    SpendCounterStore,
    key_counter,
    team_member_counter,
)
from litellm.repositories.table_repositories import JWTKeyMappingRepository, TeamMembershipRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.router_api import (
    RoutingContext,
    fallbacks_disabled,
    references_provider_scoped_resource,
    request_fallback_chain,
)
from litellm.secret_managers.main import get_secret_bool
from litellm.types.services import ServiceTypes

try:
    from litellm_enterprise.proxy.auth.user_api_key_auth import (
        enterprise_custom_auth as _enterprise_custom_auth,
    )

    enterprise_custom_auth: Callable | None = _enterprise_custom_auth
except ImportError as e:
    verbose_proxy_logger.debug("Error in enterprise custom auth: %s", e)
    enterprise_custom_auth = None

user_api_key_service_logger_obj: Final = ServiceLogging()  # used for tracking latency on OTEL


custom_litellm_key_header: Final = APIKeyHeader(
    name=SpecialHeaders.custom_litellm_api_key.value,
    auto_error=False,
    description="Bearer token",
)
api_key_header: Final = APIKeyHeader(
    name=SpecialHeaders.openai_authorization.value,
    auto_error=False,
    description="Bearer token",
)
azure_api_key_header: Final = APIKeyHeader(
    name=SpecialHeaders.azure_authorization.value,
    auto_error=False,
    description="Some older versions of the openai Python package will send an API-Key header with just the API key ",
)
anthropic_api_key_header: Final = APIKeyHeader(
    name=SpecialHeaders.anthropic_authorization.value,
    auto_error=False,
    description="If anthropic client used.",
)
google_ai_studio_api_key_header: Final = APIKeyHeader(
    name=SpecialHeaders.google_ai_studio_authorization.value,
    auto_error=False,
    description="If google ai studio client used.",
)
azure_apim_header: Final = APIKeyHeader(
    name=SpecialHeaders.azure_apim_authorization.value,
    auto_error=False,
    description="The default name of the subscription key header of Azure",
)


def _get_model_from_request_context(
    request_data: dict,
    route: str,
    request: Request | None,
    llm_router: litellm.Router | None = None,
    team_id: str | None = None,
) -> str | list[str] | None:
    return get_model_from_request(
        request_data=request_data,
        route=route,
        request_headers=safe_get_request_headers(request=request),
        request_query_params=safe_get_request_query_params(request=request),
        llm_router=llm_router,
        request=request,
        team_id=team_id,
    )


_CLAUDE_MODEL_ROUTES: Final = frozenset(
    f"/{prefix}{endpoint}" for prefix in ("", "v1/") for endpoint in ("messages", "chat/completions", "responses")
)
_CLAUDE_MODEL_NORMALIZED: Final = "litellm.claude_model_normalized"
_GENERAL_SETTINGS_ADAPTER: Final = TypeAdapter(dict[str, object])
_PASS_THROUGH_HEADERS_ADAPTER: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(
    Mapping[str, object] | None
)


async def _normalize_claude_model(
    request_data: dict, valid_token: UserAPIKeyAuth, request: Request | None, route: str
) -> None:
    from litellm.proxy.proxy_server import llm_router, prisma_client, proxy_config, proxy_logging_obj

    if route not in _CLAUDE_MODEL_ROUTES or llm_router is None:
        return
    if request is not None and request.scope.get(_CLAUDE_MODEL_NORMALIZED) is True:
        return
    requested: Final = _get_model_from_request_context(request_data, route, request, llm_router, valid_token.team_id)
    if not isinstance(requested, str) or requested != request_data.get("model"):
        return
    if not requested.startswith("claude-router-") and not requested.lower().endswith("[1m]"):
        return
    settings: Final = await proxy_config.get_hierarchical_router_settings(
        user_api_key_dict=valid_token, prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
    )
    aliases: Final = settings.get("model_group_alias") if isinstance(settings, Mapping) else None
    source: Final = claude_code_requested_group(
        requested, llm_router, valid_token.team_id, (valid_token.aliases, valid_token.team_model_aliases, aliases)
    )
    if request is not None:
        request.scope[_CLAUDE_MODEL_NORMALIZED] = True
    if source is None:
        return
    rewrite_request_model(request_data, request, source)


_CODEX_NATIVE_RESPONSES_ROUTE: Final = "/backend-api/codex/responses"


def _is_codex_native_litellm_turn(route: str, model: object, llm_router: litellm.Router, team_id: str | None) -> bool:
    if route != _CODEX_NATIVE_RESPONSES_ROUTE:
        return False
    from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import routes_to_litellm

    return routes_to_litellm(model, llm_router, team_id)


async def _resolve_router_settings_model_group_alias(
    request_data: dict[str, object],  # mutable-ok: the request body is rewritten in place for every downstream reader
    valid_token: UserAPIKeyAuth,
    request: Request | None,
    route: str,
) -> None:
    """Rewrite the requested model through the key's or team's ``router_settings.model_group_alias``
    before the allowlist checks, so they authorize the model group the request is routed to.
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client, proxy_config, proxy_logging_obj

    if request is None or llm_router is None or not RouteChecks.is_llm_api_route(route=route):
        return
    if request.scope.get(MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY) is True:
        return
    request.scope[MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY] = True
    requested: Final = request_data.get("model")
    if request_dispatched_to_pass_through_endpoint(request) or (
        request_dispatched_to_provider_pass_through(request)
        and not _is_codex_native_litellm_turn(route, requested, llm_router, valid_token.team_id)
    ):
        return
    if not isinstance(requested, str) or await read_raw_json_body(request=request) is None:
        return
    settings: Final = await proxy_config.get_hierarchical_router_settings(
        user_api_key_dict=valid_token, prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
    )
    if not isinstance(settings, Mapping):
        return
    target: Final = resolve_model_group_alias(settings.get("model_group_alias"), requested)
    if target is None or target == requested:
        return
    if request_dispatched_to_provider_pass_through(request):
        from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import native_turn_serves_through_litellm

        if not await native_turn_serves_through_litellm(requested, llm_router, request, valid_token):
            return
    verbose_proxy_logger.debug(
        "router_settings.model_group_alias resolved %s -> %s before auth",
        requested.replace("\r", "").replace("\n", ""),
        target.replace("\r", "").replace("\n", ""),
    )
    request.scope.setdefault(CLIENT_REQUESTED_MODEL_SCOPE_KEY, requested)
    rewrite_request_model(request_data, request, target)


def _resolve_codex_native_chatgpt_model(
    request_data: dict[str, object],  # mutable-ok: the request body is rewritten in place for every downstream reader
    valid_token: UserAPIKeyAuth,
    request: Request | None,
    route: str,
) -> None:
    requested: Final = request_data.get("model")
    if route != _CODEX_NATIVE_RESPONSES_ROUTE or request is None or not isinstance(requested, str):
        return
    from litellm.proxy.pass_through_endpoints.codex_passthrough_endpoints import native_chatgpt_model
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        return
    target: Final = native_chatgpt_model(requested, llm_router, valid_token)
    if target is None:
        return
    request.scope.setdefault(CLIENT_REQUESTED_MODEL_SCOPE_KEY, requested)
    rewrite_request_model(request_data, request, target)


def _get_model_names_for_budget_checks(
    model: str | list[str] | None,
) -> list[str]:
    if model is None:
        return []
    if isinstance(model, str):
        return [model]
    return model


async def _normalize_model_for_auth(
    request_data: dict[str, object],  # mutable-ok: normalization updates dispatch data
    valid_token: UserAPIKeyAuth,
    request: Request | None,
    route: str,
) -> None:
    await _normalize_claude_model(request_data, valid_token, request, route)
    await _resolve_router_settings_model_group_alias(request_data, valid_token, request, route)
    _resolve_codex_native_chatgpt_model(request_data, valid_token, request, route)


def _budget_fallback_target_model(target: object) -> str | None:
    if isinstance(target, str):
        return target
    if not isinstance(target, Mapping):
        return None
    target_mapping: Final = cast(Mapping[str, object], target)  # cast-ok: isinstance cannot type the container's items
    if set(target_mapping) != {"model"}:
        return None
    model: Final = target_mapping.get("model")
    return model if isinstance(model, str) else None


class _KeyAndUserModelBudgetLimiter(Protocol):
    async def is_key_within_model_budget(self, user_api_key_dict: UserAPIKeyAuth, model: str) -> bool: ...

    async def is_user_within_model_budget(
        self, user_id: str, user_model_max_budget: Mapping[str, object], model: str, team_id: str | None = None
    ) -> bool: ...


class _TeamModelBudgetLimiter(Protocol):
    async def is_team_within_model_budget(
        self,
        team_id: str,
        team_model_max_budget: Mapping[str, object],
        key_model_max_budget: Mapping[str, object] | None,
        model: str,
    ) -> bool: ...


class _EndUserModelBudgetLimiter(Protocol):
    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: Mapping[str, object],
        model: str,
        team_id: str | None = None,
    ) -> bool: ...


class _EveryModelBudgetLimiter(Protocol):
    async def is_key_within_model_budget(self, user_api_key_dict: UserAPIKeyAuth, model: str) -> bool: ...

    async def is_user_within_model_budget(
        self, user_id: str, user_model_max_budget: Mapping[str, object], model: str, team_id: str | None = None
    ) -> bool: ...

    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: Mapping[str, object],
        model: str,
        team_id: str | None = None,
    ) -> bool: ...

    async def is_team_within_model_budget(
        self,
        team_id: str,
        team_model_max_budget: Mapping[str, object],
        key_model_max_budget: Mapping[str, object] | None,
        model: str,
    ) -> bool: ...

    async def is_team_member_within_model_budget(
        self,
        user_id: str,
        team_id: str,
        team_member_model_max_budget: Mapping[str, object],
        model: str,
    ) -> bool: ...


class _TokenTeamModels(Protocol):
    @property
    def team_models(self) -> list[str]: ...


class _RawCacheRead(Protocol):
    async def async_get_cache(self, *, key: str) -> object: ...


def _raw_cache(cache: _RawCacheRead) -> _RawCacheRead:
    """View an untyped cache object's ``async_get_cache`` as returning ``object``
    instead of ``Any``, so a caller can ``isinstance``-narrow it without paying
    the ``reportAny`` cost of the underlying (unannotated) cache implementation."""
    return cache


def _token_team_models(valid_token: _TokenTeamModels) -> list[str]:
    return valid_token.team_models


class _TokenModelBudgetFields(Protocol):
    @property
    def model_max_budget(self) -> Mapping[str, object] | None: ...

    @property
    def end_user_model_max_budget(self) -> Mapping[str, object] | None: ...

    @property
    def team_model_aliases(self) -> dict[str, str] | None: ...  # mutable-ok: mirrors _can_object_call_model's contract


def _token_model_budget_fields(valid_token: _TokenModelBudgetFields) -> _TokenModelBudgetFields:
    return valid_token


def _router_fallbacks(router: litellm.Router) -> Sequence[object] | None:
    fallbacks: Final[Sequence[object] | None] = router.settings().fallbacks
    return fallbacks


async def _read_user_model_max_budget(
    user_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> Mapping[str, object] | None:
    """The user row's `model_max_budget`, or None when the row cannot be read.

    A user whose row is missing must not be refused: this is a budget lookup,
    and the main auth path likewise treats an unreadable user as no user.
    """
    if user_id is None or prisma_client is None:
        return None
    try:
        user_obj: Final = await get_user_object(
            user_id=user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as e:  # noqa: BLE001  # mirrors the main path's tolerance
        verbose_logger.debug("Unable to read user for the per-model budget check: %s", e)
        return None
    return user_obj.model_max_budget if user_obj is not None else None


async def _check_team_model_budget(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _TeamModelBudgetLimiter,
    models: list[str],
) -> None:
    """Enforce the team's `model_max_budget` for every requested model the key does not override."""
    team_model_max_budget: Final = valid_token.team_model_max_budget
    if valid_token.team_id is None or not team_model_max_budget:
        return
    key_model_max_budget: Final[Mapping[str, object] | None] = valid_token.model_max_budget
    for model_name in models:
        await model_max_budget_limiter.is_team_within_model_budget(
            team_id=valid_token.team_id,
            team_model_max_budget=team_model_max_budget,
            key_model_max_budget=key_model_max_budget,
            model=model_name,
        )


async def _end_user_model_max_budget(
    end_user_id: str | None,
    end_user_object: LiteLLM_EndUserTable | None,
    key_end_user_budget_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
) -> Mapping[str, object] | None:
    budget_table: Final = (
        end_user_object.litellm_budget_table
        if end_user_object is not None
        else (
            await resolve_default_end_user_budget(
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                key_end_user_budget_id=key_end_user_budget_id,
                parent_otel_span=parent_otel_span,
            )
            if end_user_id and prisma_client is not None
            else None
        )
    )
    return model_max_budget_of(budget_table) if budget_table is not None else None


async def _check_end_user_model_budget(
    end_user_id: str | None,
    end_user_model_max_budget: Mapping[str, object] | None,
    model_max_budget_limiter: _EndUserModelBudgetLimiter,
    models: Sequence[str],
    team_id: str | None,
) -> None:
    if not end_user_id or not end_user_model_max_budget:
        return
    for model_name in models:
        await model_max_budget_limiter.is_end_user_within_model_budget(
            end_user_id=end_user_id,
            end_user_model_max_budget=end_user_model_max_budget,
            model=model_name,
            team_id=team_id,
        )


async def _is_within_every_model_budget(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _EveryModelBudgetLimiter,
    model: str,
) -> bool:
    budget_fields: Final = _token_model_budget_fields(valid_token)
    key_model_max_budget: Final = budget_fields.model_max_budget
    user_model_max_budget: Final = valid_token.user_model_max_budget
    end_user_model_max_budget: Final = budget_fields.end_user_model_max_budget
    team_model_max_budget: Final = valid_token.team_model_max_budget
    team_member_model_max_budget: Final = valid_token.team_member_model_max_budget
    try:
        if valid_token.token is not None and isinstance(key_model_max_budget, Mapping) and key_model_max_budget:
            await model_max_budget_limiter.is_key_within_model_budget(
                user_api_key_dict=valid_token,
                model=model,
            )
        if valid_token.user_id is not None and isinstance(user_model_max_budget, Mapping) and user_model_max_budget:
            await model_max_budget_limiter.is_user_within_model_budget(
                user_id=valid_token.user_id,
                user_model_max_budget=user_model_max_budget,
                model=model,
                team_id=valid_token.team_id,
            )
        if (
            valid_token.end_user_id is not None
            and isinstance(end_user_model_max_budget, Mapping)
            and end_user_model_max_budget
        ):
            await model_max_budget_limiter.is_end_user_within_model_budget(
                end_user_id=valid_token.end_user_id,
                end_user_model_max_budget=end_user_model_max_budget,
                model=model,
                team_id=valid_token.team_id,
            )
        if valid_token.team_id is not None and isinstance(team_model_max_budget, Mapping) and team_model_max_budget:
            await model_max_budget_limiter.is_team_within_model_budget(
                team_id=valid_token.team_id,
                team_model_max_budget=team_model_max_budget,
                key_model_max_budget=key_model_max_budget,
                model=model,
            )
        if (
            valid_token.user_id is not None
            and valid_token.team_id is not None
            and isinstance(team_member_model_max_budget, Mapping)
            and team_member_model_max_budget
        ):
            await model_max_budget_limiter.is_team_member_within_model_budget(
                user_id=valid_token.user_id,
                team_id=valid_token.team_id,
                team_member_model_max_budget=team_member_model_max_budget,
                model=model,
            )
    except litellm.BudgetExceededError:
        return False
    return True


async def _is_fallback_model_authorized(
    model: str,
    valid_token: UserAPIKeyAuth,
    llm_model_list: Sequence[object] | None,
    llm_router: litellm.Router | None,
    prisma_client: PrismaClient | None = None,
    team_membership: LiteLLM_TeamMembership | None = None,
    project_object: LiteLLM_ProjectTableCachedObj | None = None,
    refuse_agent_keys: bool = False,
    team_object: LiteLLM_TeamTable | None = None,
) -> bool:
    if refuse_agent_keys and valid_token.agent_id is not None:
        return False
    try:
        await can_key_call_model(
            model=model,
            llm_model_list=llm_model_list,
            valid_token=valid_token,
            llm_router=llm_router,
            prisma_client=prisma_client,
        )
        if team_object is not None:
            try:
                await can_team_access_model(
                    model=model,
                    team_object=team_object,
                    llm_router=llm_router,
                    team_model_aliases=_token_model_budget_fields(valid_token).team_model_aliases,
                    key_model_aliases=key_model_aliases_for_auth_check(valid_token),
                    prisma_client=prisma_client,
                )
            except ProxyException as team_denial:
                if team_denial.type != ProxyErrorTypes.team_model_access_denied:
                    raise
                if not await key_access_group_grants_model(
                    model=model,
                    valid_token=valid_token,
                    team_object=team_object,
                    llm_router=llm_router,
                    key_model_aliases=key_model_aliases_for_auth_check(valid_token),
                ):
                    raise
        else:
            team_models: Final = _token_team_models(valid_token)
            if team_models:
                can_object_call_model(
                    model=model,
                    llm_router=llm_router,
                    models=team_models,
                    team_model_aliases=_token_model_budget_fields(valid_token).team_model_aliases,
                    team_id=valid_token.team_id,
                    key_model_aliases=key_model_aliases_for_auth_check(valid_token),
                    object_type="team",
                )
        member_allowed_models: Final = (
            team_membership.litellm_budget_table.allowed_models
            if team_membership is not None and team_membership.litellm_budget_table is not None
            else None
        )
        if member_allowed_models:
            can_object_call_model(
                model=model,
                llm_router=llm_router,
                models=member_allowed_models,
                team_id=valid_token.team_id,
                key_model_aliases=key_model_aliases_for_auth_check(valid_token),
                object_type="team",
            )
        if project_object is not None and len(project_object.models) > 0:
            can_project_access_model(
                model=model,
                project_object=project_object,
                llm_router=llm_router,
                key_model_aliases=key_model_aliases_for_auth_check(valid_token),
            )
    except ProxyException:
        return False
    return True


async def _is_within_all_fallback_model_budgets(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _EveryModelBudgetLimiter,
    models: Sequence[str],
) -> bool:
    for model in models:
        if not await _is_within_every_model_budget(
            valid_token=valid_token,
            model_max_budget_limiter=model_max_budget_limiter,
            model=model,
        ):
            return False
    return True


async def _model_budget_verdict(check: Awaitable[bool]) -> ModelBudgetExceeded | None:
    try:
        with tracer.trace("litellm.proxy.auth.budget_checks"):
            _ = await check
    except litellm.BudgetExceededError as e:
        return ModelBudgetExceeded(
            message=e.message,
            current_cost=e.current_cost,
            max_budget=e.max_budget,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
        )
    return None


class _ModelBudgetHost:
    def __init__(
        self,
        *,
        request: Request,
        request_data: dict[str, object],  # mutable-ok: budget fallbacks rewrite the request model in place
        valid_token: UserAPIKeyAuth,
        route: str,
        limiter: _KeyAndUserModelBudgetLimiter,
        llm_router: litellm.Router | None,
        llm_model_list: Sequence[object] | None,
    ) -> None:
        self._request: Final = request
        self._request_data: Final = request_data
        self._valid_token: Final = valid_token
        self._route: Final = route
        self._limiter: Final = limiter
        self._llm_router: Final = llm_router
        self._llm_model_list: Final = llm_model_list

    @property
    def valid_token(self) -> UserAPIKeyAuth:
        return self._valid_token

    @property
    def route(self) -> str:
        return self._route

    async def read_model_budget_scope(self) -> ModelBudgetScope:
        requested: Final = self._request_data.get("model")
        user_model_max_budget: Final = self._valid_token.user_model_max_budget
        budget_fallbacks: Final[Mapping[str, list[str]]] = self._valid_token.budget_fallbacks or {}
        return ModelBudgetScope(
            models=tuple(
                _get_model_names_for_budget_checks(
                    model=_get_model_from_request_context(
                        request_data=self._request_data,
                        route=self._route,
                        request=self._request,
                        llm_router=self._llm_router,
                        team_id=self._valid_token.team_id,
                    )
                )
            ),
            rewritable_model=requested if isinstance(requested, str) else None,
            key=KeyModelBudgets(
                fallbacks=MappingProxyType({model: tuple(chain) for model, chain in budget_fallbacks.items()})
            )
            if self._valid_token.model_max_budget and self._valid_token.token is not None
            else None,
            user_has_model_budgets=self._valid_token.user_id is not None
            and isinstance(user_model_max_budget, Mapping)
            and bool(user_model_max_budget),
        )

    async def check_key_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        return await _model_budget_verdict(
            self._limiter.is_key_within_model_budget(user_api_key_dict=self._valid_token, model=model)
        )

    async def authorize_budget_fallback(self, model: str) -> bool:
        return await _is_fallback_model_authorized(
            model=model,
            valid_token=self._valid_token,
            llm_model_list=self._llm_model_list,
            llm_router=self._llm_router,
        )

    async def rewrite_request_model(self, model: str) -> None:
        _reroute_request(self._request_data, self._request, model)

    async def check_user_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        user_id: Final = self._valid_token.user_id
        user_model_max_budget: Final = self._valid_token.user_model_max_budget
        if user_id is None or not isinstance(user_model_max_budget, Mapping):
            return None
        return await _model_budget_verdict(
            self._limiter.is_user_within_model_budget(
                user_id=user_id,
                user_model_max_budget=user_model_max_budget,
                model=model,
                team_id=self._valid_token.team_id,
            )
        )


async def _enforce_model_budgets(
    check: Callable[[ModelBudgetHost], Awaitable[ModelBudgetResult]], host: _ModelBudgetHost
) -> None:
    match await check(host):
        case ModelBudgetExceeded() as exceeded:
            raise denial_exception(
                exceeded,
                token=host.valid_token.token,
                route=host.route,
                jwt_auth_enabled=False,
                key=host.valid_token,
            )
        case WithinModelBudgets():
            return


def _reroute_request(
    request_data: dict[str, object],  # mutable-ok: the fallback model replaces the requested one in place
    request: Request,
    model: str,
) -> None:
    rewrite_request_model(request_data=request_data, request=request, model=model)
    if "path_params" in request.scope:
        request.scope["path_params"]["model"] = model  # rebind-ok: route handlers use the fallback model


def _budget_fallback_candidates(targets: Sequence[object]) -> tuple[str, ...]:
    return tuple(model for target in targets if (model := _budget_fallback_target_model(target)) is not None)


class _TeamMemberModelBudgetHost:
    def __init__(
        self,
        *,
        valid_token: UserAPIKeyAuth,
        limiter: _EveryModelBudgetLimiter,
        models: Sequence[str],
        request_data: dict[str, object],  # mutable-ok: fallback rewrites the request body
        request: Request,
        llm_model_list: Sequence[object] | None,
        llm_router: litellm.Router | None,
        project_object: LiteLLM_ProjectTableCachedObj | None,
        team_object: LiteLLM_TeamTable | None,
        end_user_object: LiteLLM_EndUserTable | None,
        route: str,
        prisma_client: PrismaClient | None,
        user_api_key_cache: UserApiKeyCache | None,
        proxy_logging_obj: ProxyLogging | None,
    ) -> None:
        self._valid_token: Final = valid_token
        self._limiter: Final = limiter
        self._models: Final = tuple(models)
        self._request_data: Final = request_data
        self._request: Final = request
        self._llm_model_list: Final = llm_model_list
        self._llm_router: Final = llm_router
        self._project_object: Final = project_object
        self._team_object: Final = team_object
        self._end_user_object: Final = end_user_object
        self._route: Final = route
        self._prisma_client: Final = prisma_client
        self._user_api_key_cache: Final = user_api_key_cache
        self._proxy_logging_obj: Final = proxy_logging_obj

    def _resolved_dependencies(self) -> tuple[PrismaClient | None, UserApiKeyCache, ProxyLogging]:
        from litellm.proxy import proxy_server

        return (
            self._prisma_client if self._prisma_client is not None else proxy_server.prisma_client,
            self._user_api_key_cache if self._user_api_key_cache is not None else proxy_server.user_api_key_cache,
            self._proxy_logging_obj if self._proxy_logging_obj is not None else proxy_server.proxy_logging_obj,
        )

    async def read_team_member_model_budget_scope(self) -> TeamMemberModelBudgetScope:
        budget: Final = self._valid_token.team_member_model_max_budget
        requested: Final = self._request_data.get("model")
        return TeamMemberModelBudgetScope(
            models=self._models,
            rewritable_model=requested if isinstance(requested, str) else None,
            has_model_budgets=self._valid_token.user_id is not None
            and self._valid_token.team_id is not None
            and isinstance(budget, Mapping)
            and bool(budget),
            can_fall_back=self._llm_router is not None and not fallbacks_disabled(self._request_data),
        )

    async def check_team_member_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        user_id: Final = self._valid_token.user_id
        team_id: Final = self._valid_token.team_id
        budget: Final = self._valid_token.team_member_model_max_budget
        if user_id is None or team_id is None or not isinstance(budget, Mapping):
            return None
        return await _model_budget_verdict(
            self._limiter.is_team_member_within_model_budget(
                user_id=user_id, team_id=team_id, team_member_model_max_budget=budget, model=model
            )
        )

    async def read_team_member_fallbacks(self, model: str) -> tuple[FallbackTarget, ...]:
        llm_router: Final = self._llm_router
        if (
            llm_router is None
            or self._request.headers.get("content-type", "").lower().startswith("multipart/form-data")
            or references_provider_scoped_resource(self._request_data)
        ):
            return ()
        from litellm.proxy.proxy_server import proxy_config

        prisma_client, _, proxy_logging_obj = self._resolved_dependencies()
        try:
            router_settings: Final[dict[str, object] | None] = (
                cast(  # cast-ok: proxy settings return an unparameterized dict
                    dict[str, object] | None,
                    await proxy_config.get_hierarchical_router_settings(
                        user_api_key_dict=self._valid_token,
                        prisma_client=prisma_client,
                        proxy_logging_obj=proxy_logging_obj,
                    ),
                )
            )
        except Exception:  # noqa: BLE001  # Settings lookup failures must not authorize a fallback.
            return ()

        configured_fallbacks: Final[object] = router_settings.get("fallbacks") if router_settings is not None else None
        router_fallbacks: Final[Sequence[object] | None] = (
            cast(Sequence[object], configured_fallbacks)  # cast-ok: guarded by the list check below
            if isinstance(configured_fallbacks, list)
            else _router_fallbacks(llm_router)
        )
        if router_fallbacks is None:
            return ()
        fallback_models, _ = request_fallback_chain(list(router_fallbacks), self._request_data, model)
        model_group_aliases: Final[object] = (
            router_settings.get("model_group_alias") if router_settings is not None else None
        )
        key_aliases: Final[object] = cast(  # cast-ok: auth aliases are dynamically typed at runtime
            object, self._valid_token.aliases
        )
        resolved: Final[tuple[tuple[str, str], ...]] = tuple(
            (
                candidate,
                target
                if (target := resolve_model_group_alias(model_group_aliases, candidate)) is not None
                else candidate,
            )
            for candidate in _budget_fallback_candidates(fallback_models or ())
        )
        return tuple(
            FallbackTarget(
                requested=candidate,
                model=fallback_model,
                served_model=resolve_served_model(
                    fallback_model, litellm.model_alias_map, key_aliases, model_group_aliases
                ),
            )
            for candidate, fallback_model in resolved
        )

    async def vet_team_member_fallback(self, target: FallbackTarget) -> bool:
        llm_router: Final = self._llm_router
        user_id: Final = self._valid_token.user_id
        team_id: Final = self._valid_token.team_id
        if llm_router is None or user_id is None or team_id is None:
            return False
        prisma_client, user_api_key_cache, proxy_logging_obj = self._resolved_dependencies()
        try:
            team_membership: Final = await get_team_membership(
                user_id=user_id,
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception:  # noqa: BLE001  # membership lookup failures must not authorize a fallback
            return False
        try:
            return await self._is_usable_fallback(
                target, llm_router, team_membership, prisma_client, user_api_key_cache, proxy_logging_obj
            )
        except Exception:  # noqa: BLE001  # A candidate check failure must not escape the fail-closed walk.
            verbose_proxy_logger.warning(
                "Skipping fallback candidate %s after a budget or authorization check error",
                target.model.replace("\r", "").replace("\n", ""),
                exc_info=True,
            )
            return False

    async def _is_usable_fallback(
        self,
        target: FallbackTarget,
        llm_router: litellm.Router,
        team_membership: LiteLLM_TeamMembership | None,
        prisma_client: PrismaClient | None,
        user_api_key_cache: UserApiKeyCache,
        proxy_logging_obj: ProxyLogging,
    ) -> bool:
        if (
            litellm.block_requests_for_models_without_pricing
            and RouteChecks.is_llm_api_route(route=self._route)
            and _unpriced_models_in_request(model=target.served_model, llm_router=llm_router)
        ):
            return False
        if self._end_user_object is not None and self._end_user_object.models:
            can_customer_access_model(
                model=target.model,
                end_user_object=self._end_user_object,
                llm_router=llm_router,
                valid_token=self._valid_token,
            )
        if not is_model_cost_zero(
            model=target.served_model, llm_router=llm_router
        ) and not await _is_within_all_fallback_model_budgets(
            valid_token=self._valid_token,
            model_max_budget_limiter=self._limiter,
            models=tuple(
                dict.fromkeys(
                    (
                        target.requested,
                        target.model,
                        target.served_model,
                        llm_router.registry().resolve(target.served_model),
                    )
                )
            ),
        ):
            return False
        if not await _is_fallback_model_authorized(
            model=target.model,
            valid_token=self._valid_token,
            llm_model_list=self._llm_model_list,
            llm_router=llm_router,
            prisma_client=self._prisma_client,
            team_object=self._team_object,
            team_membership=team_membership,
            project_object=self._project_object,
            refuse_agent_keys=True,
        ):
            return False
        self._valid_token.matched_model_access_groups = []
        matched_groups: Final = await stamp_matched_model_access_groups(
            model=target.model,
            valid_token=self._valid_token,
            team_object=self._team_object,
            project_object=self._project_object,
            llm_router=llm_router,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            team_membership=team_membership,
            team_membership_loaded=True,
        )
        self._valid_token.matched_model_access_groups = list(matched_groups)
        if matched_groups:
            await _model_access_group_max_budget_check(
                matched_model_access_groups=list(matched_groups),
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
            )
        return True

    async def reroute_team_member_request(self, model: str) -> None:
        _reroute_request(self._request_data, self._request, model)


async def _check_team_member_model_budget_with_fallback(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _EveryModelBudgetLimiter,
    models: Sequence[str],
    request_data: dict[str, object],  # mutable-ok: fallback rewrites the request body
    request: Request,
    llm_model_list: Sequence[object] | None,
    llm_router: litellm.Router | None,
    project_object: LiteLLM_ProjectTableCachedObj | None = None,
    team_object: LiteLLM_TeamTable | None = None,
    end_user_object: LiteLLM_EndUserTable | None = None,
    route: str = "",
    prisma_client: PrismaClient | None = None,
    user_api_key_cache: UserApiKeyCache | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
) -> None:
    host: Final = _TeamMemberModelBudgetHost(
        valid_token=valid_token,
        limiter=model_max_budget_limiter,
        models=models,
        request_data=request_data,
        request=request,
        llm_model_list=llm_model_list,
        llm_router=llm_router,
        project_object=project_object,
        team_object=team_object,
        end_user_object=end_user_object,
        route=route,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    match await _AUTH_CORE.model_budgets().check_team_member(host):
        case ModelBudgetExceeded() as exceeded:
            raise denial_exception(
                exceeded, token=valid_token.token, route=route, jwt_auth_enabled=False, key=valid_token
            )
        case WithinModelBudgets():
            return


def _selectors(value: str | list[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(selector) for selector in value) if isinstance(value, list) else (str(value),)


def _routing_override(override: JWTRoutingOverride) -> RoutingOverride:
    return RoutingOverride(
        iss=_selectors(override.iss),
        client_id=_selectors(override.client_id),
        scope=_selectors(override.scope),
        aud=_selectors(override.aud),
    )


def _oauth2_routing_overrides(jwt_handler: JWTHandler) -> tuple[RoutingOverride, ...]:
    overrides: Final = jwt_handler.litellm_jwtauth.routing_overrides or ()
    return tuple(_routing_override(override) for override in overrides if override.path == "oauth2")


def _unauthenticated_pass_through_routes(endpoints: object) -> frozenset[str]:
    if not isinstance(endpoints, list):
        return frozenset()
    return frozenset(
        str(endpoint.get("path", ""))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # raw config dicts
        for endpoint in endpoints  # pyright: ignore[reportUnknownVariableType]  # raw config list
        if isinstance(endpoint, dict) and endpoint.get("auth", True) is not True  # pyright: ignore[reportUnknownMemberType]  # raw config dicts
    )


def _additional_public_routes(general_settings: Mapping[str, object]) -> tuple[str, ...]:
    routes: Final = general_settings.get("public_routes") or []
    return tuple(route for route in routes if isinstance(route, str)) if isinstance(routes, list) else ()  # pyright: ignore[reportUnknownVariableType]  # raw config list


def _plan_settings(
    general_settings: Mapping[str, object], jwt_handler: JWTHandler, *, premium_user: bool, custom_auth: bool
) -> PlanSettings:
    jwt_enabled: Final = general_settings.get("enable_jwt_auth", False) is True
    return PlanSettings(
        custom_auth=custom_auth,
        premium_user=premium_user,
        public_routes=frozenset(LiteLLMRoutes.public_routes.value),
        additional_public_routes=_additional_public_routes(general_settings),
        metrics_require_auth=litellm.require_auth_for_metrics_endpoint is not False,
        oauth2_introspection=general_settings.get("enable_oauth2_auth", False) is True,
        oauth2_proxy=general_settings.get("enable_oauth2_proxy_auth", False) is True,
        jwt_enabled=jwt_enabled,
        oauth2_routing_overrides=_oauth2_routing_overrides(jwt_handler) if jwt_enabled else (),
        unauthenticated_pass_through_routes=_unauthenticated_pass_through_routes(
            general_settings.get("pass_through_endpoints")
        ),
    )


def _request_facts(route: str, api_key: str) -> RequestFacts:
    return RequestFacts(route=route, token=api_key, is_llm_api_route=RouteChecks.is_llm_api_route(route=route))


def _plan_facts(route: str, api_key: str, settings: PlanSettings, jwt_handler: JWTHandler) -> PlanFacts:
    return PlanFacts(
        request=_request_facts(route, api_key),
        is_info_route=RouteChecks.is_info_route(route=route),
        unverified_claims=(
            jwt_handler.get_unverified_claims(token=api_key) if settings.oauth2_routing_overrides else None
        ),
    )


def oauth2_introspection_applies(
    api_key: str, route: str, *, enable_oauth2_auth: bool, enable_jwt_auth: bool, jwt_handler: JWTHandler
) -> bool:
    """Whether LiteLLM validates ``api_key`` on ``route`` by OAuth2 token introspection, on LLM API, info and
    lens routes only: a non-JWT token when OAuth2 auth is on globally, or a JWT matched by a routing override."""
    settings: Final = _plan_settings(
        {"enable_oauth2_auth": enable_oauth2_auth, "enable_jwt_auth": enable_jwt_auth},
        jwt_handler,
        premium_user=False,
        custom_auth=False,
    )
    return _AUTH_CORE.plan().oauth2_introspection_applies(_plan_facts(route, api_key, settings, jwt_handler), settings)


@dataclass(frozen=True, slots=True)
class _Planned:
    plan: CredentialPlan
    facts: PlanFacts
    api_key: str
    issued_key: bool


class _CustomAuthHost:
    def __init__(self, request: Request, api_key: str, user_custom_auth: Callable | None) -> None:
        self._request: Final = request
        self._api_key: Final = api_key
        self._user_custom_auth: Final = user_custom_auth
        self.resolved: UserAPIKeyAuth | None = None

    def _resolve(self, response: object) -> CustomIdentity:
        self.resolved = _custom_auth_token_for_this_request(response)
        return CustomIdentity(
            user_id=self.resolved.user_id, team_id=self.resolved.team_id, hashed_token=self.resolved.token
        )

    async def custom_auth(self, facts: RequestFacts) -> CustomAuthResult:
        if enterprise_custom_auth is None:
            assert self._user_custom_auth is not None
            return self._resolve(await self._user_custom_auth(request=self._request, api_key=self._api_key))
        with tracer.trace("litellm.proxy.auth.enterprise_custom_auth"):
            response: Final = await enterprise_custom_auth(
                request=self._request, api_key=self._api_key, user_custom_auth=self._user_custom_auth
            )
        if isinstance(response, UserAPIKeyAuth):
            return self._resolve(response)
        if isinstance(response, str):
            return ReplacedKey(token=response)
        return NoCustomIdentity()


async def _custom_auth_user(
    validated: UserAPIKeyAuth,
    request: Request,
    request_data: dict[str, object],
    route: str,
    parent_otel_span: Span | None,
) -> UserAPIKeyAuth:
    checked: Final = (
        await _run_post_custom_auth_checks(
            valid_token=validated,
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=parent_otel_span,
        )
        if getattr(litellm, "enable_post_custom_auth_checks", False)
        else validated
    )
    checked.authenticated_by_custom_auth = True
    return checked


async def _plan_request(
    request: Request,
    request_data: dict[str, object],
    route: str,
    api_key: str,
    settings: PlanSettings,
    jwt_handler: JWTHandler,
    user_custom_auth: Callable | None,
    parent_otel_span: Span | None,
) -> UserAPIKeyAuth | _Planned:
    facts: Final = _plan_facts(route, api_key, settings, jwt_handler)
    first: Final = _AUTH_CORE.plan().plan(facts, settings)
    if not isinstance(first, CustomValidator):
        return _Planned(plan=first, facts=facts, api_key=api_key, issued_key=False)
    host: Final = _CustomAuthHost(request=request, api_key=api_key, user_custom_auth=user_custom_auth)
    result: Final = await _AUTH_CORE.custom().authenticate(host, facts.request)
    match result:
        case CustomIdentity():
            assert host.resolved is not None
            return await _custom_auth_user(host.resolved, request, request_data, route, parent_otel_span)
        case ReplacedKey(token=token):
            issued: Final = _plan_facts(route, token, settings, jwt_handler)
            return _Planned(
                plan=_AUTH_CORE.plan().after_custom_auth(issued, settings, issued_key=True),
                facts=issued,
                api_key=token,
                issued_key=True,
            )
        case NoCustomIdentity():
            return _Planned(
                plan=_AUTH_CORE.plan().after_custom_auth(facts, settings, issued_key=False),
                facts=facts,
                api_key=api_key,
                issued_key=False,
            )


def _anonymous_auth(anonymous: Anonymous) -> UserAPIKeyAuth:
    match anonymous.reason:
        case AnonymousReason.PUBLIC_ROUTE:
            return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
        case AnonymousReason.PASS_THROUGH_AUTH_OFF:
            return UserAPIKeyAuth()


def _premium_message(feature: PremiumFeature) -> str:
    match feature:
        case PremiumFeature.OAUTH2:
            return (
                "Oauth2 token validation is only available for premium users. "
                + CommonProxyErrors.not_premium_user.value
            )
        case PremiumFeature.JWT:
            return f"JWT Auth is an enterprise only feature. {CommonProxyErrors.not_premium_user.value}"


def _premium_required(required: PremiumRequired) -> ProxyException:
    return ProxyException(
        message=_premium_message(required.feature),
        type=ProxyErrorTypes.auth_error,
        param="premium_user",
        code=status.HTTP_403_FORBIDDEN,
    )


async def _oauth2_auth(mode: OAuth2Mode, *, api_key: str, request: Request) -> UserAPIKeyAuth:
    match mode:
        case OAuth2Mode.INTROSPECTION:
            return await Oauth2Handler.check_oauth2_token(token=api_key)
        case OAuth2Mode.PROXY:
            return await handle_oauth2_proxy_request(request=request)


async def _authenticate_without_key(
    plan: AnonymousValidator | OAuth2Validator | PremiumRequired, *, api_key: str, request: Request
) -> UserAPIKeyAuth:
    match plan:
        case AnonymousValidator(reason=reason):
            return _anonymous_auth(_AUTH_CORE.anonymous().authenticate(reason))
        case PremiumRequired():
            raise _premium_required(plan)
        case OAuth2Validator(mode=mode):
            return await _oauth2_auth(mode, api_key=api_key, request=request)


def get_bearer_token(
    api_key: str,
) -> str:
    return scheme_credential(api_key) or ""


_get_bearer_token: Final = get_bearer_token


def _apply_budget_limits_to_end_user_params(
    end_user_params: dict,
    budget_info: LiteLLM_BudgetTable,
    end_user_id: str | None,
) -> None:
    """
    Helper function to apply budget limits to end user parameters.

    Args:
        end_user_params: Dictionary to update with budget parameters
        budget_info: Budget table object containing limits
        end_user_id: ID of the end user for logging
    """
    if budget_info.tpm_limit is not None:
        end_user_params["end_user_tpm_limit"] = budget_info.tpm_limit

    if budget_info.rpm_limit is not None:
        end_user_params["end_user_rpm_limit"] = budget_info.rpm_limit

    if budget_info.tpd_limit is not None:
        end_user_params["end_user_tpd_limit"] = budget_info.tpd_limit

    if budget_info.max_budget is not None:
        end_user_params["end_user_max_budget"] = budget_info.max_budget

    if budget_info.model_max_budget is not None:
        end_user_params["end_user_model_max_budget"] = budget_info.model_max_budget

    if budget_info.model_tpm_limit is not None:
        end_user_params["end_user_model_tpm_limit"] = budget_info.model_tpm_limit  # rebind-ok: in-place like siblings

    if budget_info.model_rpm_limit is not None:
        end_user_params["end_user_model_rpm_limit"] = budget_info.model_rpm_limit  # rebind-ok: in-place like siblings

    verbose_proxy_logger.debug("Applied budget limits to end user %s", end_user_id)


_WEBSOCKET_DENIAL_DETAIL: Final = MappingProxyType(
    {
        WebSocketDenial.NO_API_KEY: "No API key provided",
        WebSocketDenial.INVALID_AUTHORIZATION: "Invalid Authorization header format",
    }
)


async def user_api_key_auth_websocket(websocket: WebSocket) -> UserAPIKeyAuth:
    return await user_api_key_auth_websocket_for_model(websocket, model=websocket.query_params.get("model", ""))


async def user_api_key_auth_realtime_websocket(websocket: WebSocket) -> UserAPIKeyAuth:
    route_path: Final[object] = websocket.scope.get("path")
    model: Final = resolve_realtime_route_model(
        model=websocket.query_params.get("model"),
        intent=websocket.query_params.get("intent"),
        is_translation=isinstance(route_path, str) and route_path.endswith("/realtime/translations"),
    )
    return await user_api_key_auth_websocket_for_model(websocket, model=model or "")


async def user_api_key_auth_live_websocket(websocket: WebSocket) -> UserAPIKeyAuth:
    return await user_api_key_auth_websocket_for_model(websocket, model="")


def _has_required_baggage_tag_mapping() -> bool:
    from litellm.proxy import proxy_server
    from litellm.proxy.common_utils.request_tag_mappings import REQUEST_TAG_MAPPINGS_ADAPTER

    raw_mappings: Final = _GENERAL_SETTINGS_ADAPTER.validate_python(proxy_server.general_settings).get(
        "request_tag_mappings"
    )
    return bool(raw_mappings) and any(
        rule.required for rule in REQUEST_TAG_MAPPINGS_ADAPTER.validate_python(raw_mappings)
    )


async def user_api_key_auth_websocket_for_model(
    websocket: WebSocket, model: str, *, close_on_auth_error: bool = True
) -> UserAPIKeyAuth:
    ws_scope: Final = websocket.scope or {}
    scope_headers: Final = list(ws_scope.get("headers") or [])
    # ``get_request_route`` falls back to ``request.url.path`` when
    # ``scope["path"]`` is absent. On WebSockets that fallback reads
    # ``websocket.url``, which Starlette reconstructs from the (poisonable)
    # Host header. Carry the ASGI scope's path / root_path so the lookup
    # never reaches the fallback.
    synthetic_scope: Final[dict[str, object]] = {
        "type": "http",
        "method": "GET",
        "query_string": ws_scope.get("query_string", b""),
        "headers": scope_headers,
        "path": ws_scope.get("path", ""),
        "state": ws_scope.setdefault("state", {}),
    }
    for key in ("root_path", "app_root_path"):
        if key in ws_scope:
            synthetic_scope[key] = ws_scope[key]
    request: Final = Request(scope=synthetic_scope)

    request._url = websocket.url  # pyright: ignore[reportPrivateUsage]  # Starlette WebSocket URL storage

    async def return_body():
        return realtime_request_body(model)

    request.body = return_body

    selected: Final = select_websocket_api_key(
        WebSocketKeys(
            authorization=websocket.headers.get("authorization"),
            api_key=websocket.headers.get("api-key"),
            subprotocols=websocket.headers.get("sec-websocket-protocol"),
        )
    )
    if isinstance(selected, WebSocketDenial):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(status_code=403, detail=_WEBSOCKET_DENIAL_DETAIL[selected])
    api_key: Final = selected

    # Call user_api_key_auth with the extracted API key
    # Note: You'll need to modify this to work with WebSocket context if needed
    try:
        authorized: Final = await user_api_key_auth(request=request, api_key=f"Bearer {api_key}")
        if _has_required_baggage_tag_mapping():
            raise HTTPException(
                403, "Required baggage tag mappings need HTTP inference, WebSocket inference is disabled"
            )
    except Exception as e:
        if not close_on_auth_error:
            raise
        if is_invalid_virtual_key_error(e):
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        log_model_access_denial(e)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(status_code=403, detail=str(e))
    else:
        return authorized


async def user_api_key_auth_accepted_websocket_for_model(websocket: WebSocket, model: str) -> UserAPIKeyAuth:
    """Re-authorize an already accepted WebSocket for a model it named after the upgrade.

    Runs the same key, team, member, user, project and agent checks, plus budget reservation, as
    the upgrade does for ``?model=``, but raises auth errors without closing the socket so the
    caller can reply with an error frame.
    """
    return await user_api_key_auth_websocket_for_model(websocket, model, close_on_auth_error=False)


def update_valid_token_with_end_user_params(valid_token: UserAPIKeyAuth, end_user_params: dict) -> UserAPIKeyAuth:
    valid_token.end_user_id = end_user_params.get("end_user_id")
    # Only overwrite token fields when the DB-derived value is not None.
    # This prevents DB lookups (where the budget table has no value set)
    # from silently clearing values that a custom auth function may have
    # already set on the token.
    if end_user_params.get("end_user_tpm_limit") is not None:
        valid_token.end_user_tpm_limit = end_user_params["end_user_tpm_limit"]
    if end_user_params.get("end_user_rpm_limit") is not None:
        valid_token.end_user_rpm_limit = end_user_params["end_user_rpm_limit"]
    if end_user_params.get("end_user_tpd_limit") is not None:
        valid_token.end_user_tpd_limit = end_user_params["end_user_tpd_limit"]
    if end_user_params.get("allowed_model_region") is not None:
        valid_token.allowed_model_region = end_user_params["allowed_model_region"]
    if end_user_params.get("end_user_model_max_budget") is not None:
        valid_token.end_user_model_max_budget = end_user_params["end_user_model_max_budget"]
    if "end_user_model_tpm_limit" in end_user_params:
        valid_token.end_user_model_tpm_limit = end_user_params["end_user_model_tpm_limit"]  # rebind-ok: in-place
    if "end_user_model_rpm_limit" in end_user_params:
        valid_token.end_user_model_rpm_limit = end_user_params["end_user_model_rpm_limit"]  # rebind-ok: in-place
    return valid_token


# Reusable coordinator for global spend to prevent cache stampede
_global_spend_coordinator: Final = EventDrivenCacheCoordinator(log_prefix="[GLOBAL SPEND]")


async def fetch_global_spend_with_event_coordination(
    cache_key: str,
    user_api_key_cache: UserApiKeyCache,
    prisma_client: PrismaClient,
) -> float | None:
    """
    Fetch global spend with event-driven coordination to prevent cache stampede.
    Uses EventDrivenCacheCoordinator: first request queries DB and signals others when done.

    Reads the proxy budget aggregate user row, which accrues proxy-wide spend
    per request and is zeroed by ResetBudgetJob every ``litellm.budget_duration``.
    """

    async def _load_global_spend() -> float | None:
        proxy_budget_row: Final = await bounded_db_lookup(
            UserRepository(prisma_client).table.find_unique(where={"user_id": LITELLM_PROXY_BUDGET_NAME}),
            name="proxy_budget",
        )
        return float(proxy_budget_row.spend) if proxy_budget_row is not None else None

    return await _global_spend_coordinator.get_or_load(
        cache_key=cache_key,
        cache=user_api_key_cache,  # pyright: ignore[reportArgumentType]
        load_fn=_load_global_spend,
    )


_fetch_global_spend_with_event_coordination: Final = fetch_global_spend_with_event_coordination


async def get_global_proxy_spend(
    litellm_proxy_admin_name: str,
    user_api_key_cache: UserApiKeyCache,
    prisma_client: PrismaClient | None,
    token: str,
    proxy_logging_obj: ProxyLogging,
) -> float | None:
    global_proxy_spend = None
    if litellm.max_budget > 0 and prisma_client is not None:  # user set proxy max budget
        # Use event-driven coordination to prevent cache stampede
        cache_key: Final = GLOBAL_PROXY_SPEND_CACHE_KEY
        global_proxy_spend = (  # rebind-ok: pre-existing rebinding on a rename-only line
            await fetch_global_spend_with_event_coordination(
                cache_key=cache_key,
                user_api_key_cache=user_api_key_cache,
                prisma_client=prisma_client,
            )
        )
        if global_proxy_spend is not None:
            user_info: Final = CallInfo(
                user_id=litellm_proxy_admin_name,
                max_budget=litellm.max_budget,
                spend=global_proxy_spend,
                token=token,
                event_group=Litellm_EntityType.PROXY,
            )
            asyncio.create_task(
                proxy_logging_obj.budget_alerts(
                    type="proxy_budget",
                    user_info=user_info,
                )
            )
    return global_proxy_spend


def get_rbac_role(jwt_handler: JWTHandler, scopes: list[str]) -> str:
    is_admin: Final = jwt_handler.is_admin(scopes=scopes)
    if is_admin:
        return LitellmUserRoles.PROXY_ADMIN
    else:
        return LitellmUserRoles.TEAM


def _query_api_key(route: str, request: Request) -> str | None:
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.common_utils.http_parsing_utils import (
        safe_get_request_query_params,
    )

    if not RouteChecks.is_generate_content_route(route=route):
        return None
    return safe_get_request_query_params(request).get("key")


def _pass_through_header_api_key(endpoint: Mapping[str, object], route: str, request: Request) -> str | None:
    if endpoint.get("path", "") != route:
        return None
    headers: Final = _PASS_THROUGH_HEADERS_ADAPTER.validate_python(endpoint.get("headers"))
    header_name: Final = headers.get("litellm_user_api_key", "") if headers is not None else None
    return request.headers.get(header_name) if isinstance(header_name, str) else None


def _pass_through_api_key(
    pass_through_endpoints: Sequence[Mapping[str, object]] | None, route: str, request: Request
) -> str | None:
    header_values: Final = tuple(
        value
        for value in (
            _pass_through_header_api_key(endpoint, route, request) for endpoint in pass_through_endpoints or ()
        )
        if value is not None
    )
    return header_values[-1] if header_values else None


def _header_or_none(value: object) -> str | None:
    """Direct callers of ``user_api_key_auth`` leave unpassed headers as their ``fastapi.Security`` defaults."""
    return value if isinstance(value, str) else None


def get_api_key(
    custom_litellm_key_header: str | None,
    api_key: str,
    azure_api_key_header: str | None,
    anthropic_api_key_header: str | None,
    google_ai_studio_api_key_header: str | None,
    azure_apim_header: str | None,
    pass_through_endpoints: Sequence[Mapping[str, object]] | None,
    route: str,
    request: Request,
    *,
    custom_header: str | None = None,
) -> tuple[str, str | None]:
    selected: Final = select_api_key(
        PresentedKeys(
            custom_header=custom_header,
            litellm_header=_header_or_none(custom_litellm_key_header),
            authorization=_header_or_none(api_key),
            api_key=_header_or_none(azure_api_key_header),
            x_api_key=_header_or_none(anthropic_api_key_header),
            google=_header_or_none(google_ai_studio_api_key_header),
            apim=_header_or_none(azure_apim_header),
            query=_query_api_key(route, request),
            pass_through=_pass_through_api_key(pass_through_endpoints, route, request),
        )
    )
    if selected is None:
        return api_key, None
    verbose_proxy_logger.debug("Selected API key from %s", selected.source.value)
    return selected.token, selected.presented


async def check_api_key_for_custom_headers_or_pass_through_endpoints(
    request: Request,
    route: str,
    pass_through_endpoints: list[dict] | None,
    api_key: str,
) -> str:
    is_mapped_pass_through_route: bool = False
    normalized_route: Final = normalize_route_for_root_path(route)
    if normalized_route is not None:
        for mapped_route in LiteLLMRoutes.mapped_pass_through_routes.value:
            if normalized_route.startswith(mapped_route):
                is_mapped_pass_through_route = True
                break
    if is_mapped_pass_through_route:
        if request.headers.get("litellm_user_api_key") is not None:
            api_key = request.headers.get("litellm_user_api_key") or ""
    if pass_through_endpoints is not None:
        for endpoint in pass_through_endpoints:
            if isinstance(endpoint, dict) and endpoint.get("path", "") == route:
                if endpoint.get("custom_auth_parser") is not None and endpoint.get("custom_auth_parser") == "langfuse":
                    # langfuse returns {'Authorization': 'Basic <base64(username:password)>'}
                    # check the langfuse public key if it contains the litellm api key
                    import base64

                    api_key = api_key.replace("Basic ", "").strip()
                    decoded_bytes = base64.b64decode(api_key)
                    decoded_str = decoded_bytes.decode("utf-8")
                    api_key = decoded_str.split(":")[0]
                else:
                    headers = endpoint.get("headers", None)
                    if headers is not None:
                        header_key = headers.get("litellm_user_api_key", "")
                        if isinstance(request.headers, dict) and request.headers.get(key=header_key) is not None:
                            api_key = request.headers.get(key=header_key)
    return api_key


# Cache sentinel written when a JWT under AUTO_REGISTER resolved to a proxy
# admin via auth_builder. Proxy admins don't need a mapped virtual key (they
# have full access via auth_builder anyway), but without a cache entry every
# subsequent request from the same JWT identity would re-query the DB for a
# non-existent mapping. Sentinel tells _resolve_jwt_to_virtual_key to skip
# the lookup and return None (caller proceeds to auth_builder).
_JWT_PROXY_ADMIN_SENTINEL: Final = "__JWT_PROXY_ADMIN__"


class _JWTKeyMappingFailure(Enum):
    NO_ELIGIBLE_KEY = "JWT Key Mapping: No eligible existing virtual key for this identity. Access denied."


class _PendingAutoRegister(NamedTuple):
    """
    Signal returned by ``_resolve_jwt_to_virtual_key`` when the JWT's claim is
    unmapped and ``unregistered_jwt_client_behavior`` is AUTO_REGISTER or LINK_ONLY.

    The caller MUST run standard ``JWTAuthManager.auth_builder`` to apply RBAC,
    scope mappings, ``custom_validate``, and ``user_allowed_email_domain``
    policy BEFORE calling ``_auto_register_jwt_mapping`` with the validated
    ``team_id`` / ``user_id`` from the auth_builder result. Auto-registering
    purely on a signature-valid JWT (the old behavior) bypassed every JWT
    policy beyond signature verification.
    """

    claim_field: str
    claim_value: str
    cache_key: str
    jwt_issuer: str | None = None
    behavior: UnregisteredJWTClientBehavior = UnregisteredJWTClientBehavior.AUTO_REGISTER


def _claim_identifies_user(jwt_handler: JWTHandler, claim_field: str, jwt_issuer: str | None) -> bool:
    if jwt_handler.litellm_jwtauth.is_user_identity_claim(claim_field, jwt_issuer):
        return True
    verbose_proxy_logger.warning(
        "JWT Key Mapping: claim '%s' is not the user_id or user_email JWT field "
        "and may be shared by several users, so existing user keys cannot be linked automatically.",
        claim_field,
    )
    return False


async def _reusable_key_hash_for_user(prisma_client: PrismaClient, user_id: str, team_id: str | None) -> str | None:
    key: Final = await VerificationTokenRepository(prisma_client).find_newest_reusable_key(user_id, team_id)
    return None if key is None else key.token


async def _auto_register_jwt_mapping(
    virtual_key_claim_field: str,
    claim_value: str,
    jwt_handler: JWTHandler,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
    cache_key: str,
    jwt_issuer: str | None = None,
    team_id: str | None = None,
    user_id: str | None = None,
    org_id: str | None = None,
    end_user_id: str | None = None,
    agent_id: str | None = None,
    behavior: UnregisteredJWTClientBehavior = UnregisteredJWTClientBehavior.AUTO_REGISTER,
) -> UserAPIKeyAuth | _JWTKeyMappingFailure | None:
    """
    Auto-register: create a new virtual key + mapping for an unrecognised JWT
    claim value, or point the mapping at a key the resolved user already owns
    when ``auto_register_map_existing_key`` is set. LINK_ONLY always searches
    for an existing key and rejects when none is eligible. ``team_id`` and ``user_id``
    must come from a successful ``JWTAuthManager.auth_builder`` run — they
    encode the JWT identity AFTER
    RBAC/scope/custom_validate/email-domain policy has been enforced. The key
    is stamped with those values so the cached future-request path inherits
    the same team/user/org limits the auth_builder path would have applied.

    Race safety: if two concurrent requests both reach here simultaneously (both
    saw no mapping in the DB), one will win the unique-constraint race on
    litellm_jwtkeymapping. The loser catches the conflict, deletes its orphaned
    key, fetches the winner's mapping, and proceeds — no error surfaced.
    """
    # Inline import required: key_management_endpoints imports user_api_key_auth
    # (line 51) so a module-level import here would create a circular dependency.
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )

    link_only: Final = behavior == UnregisteredJWTClientBehavior.LINK_ONLY
    reuse_existing: Final = link_only or jwt_handler.litellm_jwtauth.auto_register_map_existing_key
    existing_token_hash: Final = (
        await _reusable_key_hash_for_user(
            prisma_client,
            user_id,
            team_id,
        )
        if reuse_existing
        and user_id is not None
        and _claim_identifies_user(jwt_handler, virtual_key_claim_field, jwt_issuer)
        else None
    )
    if link_only and existing_token_hash is None:
        return _JWTKeyMappingFailure.NO_ELIGIBLE_KEY
    minted: Final = existing_token_hash is None
    if existing_token_hash is not None:
        token_hash = existing_token_hash
    else:
        # ``table_name="key"`` is required: without it, generate_key_helper_fn
        # falls into the user-upsert branch (`table_name is None or "user"`) and
        # attempts to insert into LiteLLM_UserTable with user_id=None, which fails
        # the NOT NULL @id constraint. Every successful key-creation caller (e.g.
        # /key/generate) passes table_name="key" explicitly.
        key_data: Final = await generate_key_helper_fn(
            llm_router=None,
            request_type="key",
            table_name="key",
            team_id=team_id,
            user_id=user_id,
            organization_id=org_id,
            agent_id=agent_id,
            metadata={
                "auto_registered": True,
                "jwt_claim_field": virtual_key_claim_field,
                "jwt_claim_value": claim_value,
            },
        )
        # generate_key_helper_fn returns the plaintext key in "token"; the persisted
        # row in LiteLLM_VerificationToken uses its hash, so hash here to get the FK
        # value referenced by LiteLLM_JWTKeyMapping.token.
        token_hash = hash_token(key_data["token"])

    try:
        async with db_span("auto_register_jwt_mapping", "LiteLLM_JWTKeyMapping"):
            await JWTKeyMappingRepository(prisma_client).table.create(
                data={
                    "jwt_issuer": jwt_issuer or "",
                    "jwt_claim_name": virtual_key_claim_field,
                    "jwt_claim_value": claim_value,
                    "token": token_hash,
                    "created_by": behavior.value,
                    "updated_by": behavior.value,
                }
            )
    except Exception as e:
        error_str: Final = str(e).lower()
        if "unique" in error_str or "p2002" in error_str:
            # A concurrent request won the race. The key generate_key_helper_fn
            # just persisted to LiteLLM_VerificationToken is orphaned — nothing
            # maps to it, but it's a fully valid unrestricted API key sitting in
            # the DB and the cleartext is in memory on this request. Delete it
            # so orphans don't accumulate under sustained concurrency.
            verbose_proxy_logger.debug(
                "JWT Key Mapping (%s): unique conflict on create, fetching winner's mapping for %s='%s'.",
                behavior.value,
                virtual_key_claim_field,
                claim_value,
            )
            if minted:
                try:
                    async with db_span("delete_orphaned_jwt_key", "LiteLLM_VerificationToken"):
                        await VerificationTokenRepository(prisma_client).table.delete(where={"token": token_hash})
                except Exception as delete_err:
                    # Don't fail the request if cleanup fails — the orphan is
                    # unmapped and inert. Log so an operator can prune it later.
                    verbose_proxy_logger.warning(
                        "JWT Key Mapping (auto_register): failed to delete orphaned key after race: %s",
                        delete_err,
                    )
            token_hash = await get_jwt_key_mapping_object(
                jwt_claim_name=virtual_key_claim_field,
                jwt_claim_value=claim_value,
                prisma_client=prisma_client,
                jwt_issuer=jwt_issuer,
            )
            if token_hash is None:
                # The winner's mapping vanished between the unique-constraint
                # conflict and our re-fetch (concurrent delete). Returning None
                # here would silently fall through to team-based JWT auth —
                # a less-restrictive path than the operator configured. Raise
                # 503 so the caller retries against a stable state instead.
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"JWT Key Mapping: {behavior.value.upper()} race resolution failed — "
                        "winner's mapping was concurrently removed. Retry the request."
                    ),
                )
        else:
            raise

    await user_api_key_cache.async_set_cache(
        key=cache_key,
        value=token_hash,
        ttl=jwt_handler.litellm_jwtauth.virtual_key_mapping_cache_ttl,
    )

    verbose_proxy_logger.info(
        "JWT Key Mapping (%s): %s virtual key for %s='%s'.",
        behavior.value,
        "created new" if minted else "mapped existing",
        virtual_key_claim_field,
        claim_value,
    )

    auto_registered_key: Final = IdentityStore.key_from_principal(
        await IdentityStore(
            prisma_client,
            user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        ).resolve(hashed_token=token_hash)
    )
    if auto_registered_key is not None:
        if minted:
            auto_registered_key.org_id = org_id
        auto_registered_key.end_user_id = end_user_id
        auto_registered_key.api_key = auto_registered_key.token
    return auto_registered_key


async def _lookup_jwt_mapping_token_hash(
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    virtual_key_claim_field: str,
    claim_value: str,
    normalized_issuer: str | None,
    cache_key: str,
    ttl: float,
) -> str | None:
    issuer_scoped: Final = await get_jwt_key_mapping_object(
        jwt_claim_name=virtual_key_claim_field,
        jwt_claim_value=claim_value,
        prisma_client=prisma_client,
        jwt_issuer=normalized_issuer,
    )
    if issuer_scoped is not None:
        await user_api_key_cache.async_set_cache(key=cache_key, value=issuer_scoped, ttl=ttl)
        return issuer_scoped
    if normalized_issuer is None:
        return None
    # Another issuer may have already resolved (and cached) this same
    # global mapping -- check its cache entry before re-querying the DB.
    global_cache_key: Final = jwt_key_mapping_cache_key(virtual_key_claim_field, claim_value)
    cached_global: Final = await _raw_cache(user_api_key_cache).async_get_cache(key=global_cache_key)
    if isinstance(cached_global, str) and cached_global != "__NO_MAPPING__":
        return cached_global
    global_row: Final = await get_jwt_key_mapping_object(
        jwt_claim_name=virtual_key_claim_field,
        jwt_claim_value=claim_value,
        prisma_client=prisma_client,
        jwt_issuer=None,
    )
    if global_row is not None:
        await user_api_key_cache.async_set_cache(key=global_cache_key, value=global_row, ttl=ttl)
    return global_row


async def _resolve_jwt_to_virtual_key(
    jwt_claims: dict,
    jwt_handler: JWTHandler,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> Union[UserAPIKeyAuth | None, "_PendingAutoRegister"]:
    """
    Returns:
      - ``UserAPIKeyAuth``: a resolved virtual key (cache hit or DB hit). The
        caller may use this directly; JWT policy has been enforced previously
        (at key-creation time or, for cached results, before caching).
      - ``_PendingAutoRegister``: claim is unmapped and behavior is AUTO_REGISTER or LINK_ONLY.
        The caller MUST run ``JWTAuthManager.auth_builder`` to enforce JWT
        policy (RBAC, scope, custom_validate, email-domain), then invoke
        ``_auto_register_jwt_mapping`` with the validated team_id/user_id.
      - ``None``: claim is unmapped and behavior is FALLBACK_TEAM_MAPPING.
        The caller falls through to standard team-based JWT auth (which itself
        enforces full JWT policy via auth_builder).
      - Raises HTTPException: REJECT policy hit, missing claim under
        REJECT/AUTO_REGISTER/LINK_ONLY, or other policy violations.
    """
    raw_issuer: Final = jwt_claims.get(JWTHandler.LITELLM_JWT_ISSUER_CLAIM)
    normalized_issuer: Final = raw_issuer if isinstance(raw_issuer, str) else None
    virtual_key_claim_field: Final = jwt_handler.litellm_jwtauth.get_virtual_key_claim_field(normalized_issuer)
    if virtual_key_claim_field is None:
        return None
    behavior: Final = jwt_handler.litellm_jwtauth.get_unregistered_jwt_client_behavior(normalized_issuer)

    claim_value: Final = get_nested_value(
        data=jwt_claims,
        key_path=virtual_key_claim_field,
        default=None,
    )

    if claim_value is None:
        verbose_proxy_logger.debug(
            "JWT Key Mapping: Claim field '%s' not found in JWT claims.", virtual_key_claim_field
        )
        # A missing claim is an unmapped client — apply the no-match policy
        # rather than returning early. Otherwise a caller can bypass REJECT
        # simply by presenting a JWT that omits the configured field. For
        # AUTO_REGISTER there is no stable identity to map without a claim
        # value, so we deny rather than create a sentinel-keyed record.
        if behavior in (
            UnregisteredJWTClientBehavior.REJECT,
            UnregisteredJWTClientBehavior.AUTO_REGISTER,
            UnregisteredJWTClientBehavior.LINK_ONLY,
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"JWT Key Mapping: Required claim '{virtual_key_claim_field}' "
                    "is missing from the JWT. Access denied."
                ),
            )
        return None

    cache_key: Final = jwt_key_mapping_cache_key(virtual_key_claim_field, str(claim_value), normalized_issuer)
    raw_cached_mapping: Final = await user_api_key_cache.async_get_cache(cache_key)
    sentinel_written_by_this_policy: Final = behavior == UnregisteredJWTClientBehavior.AUTO_REGISTER
    cached_mapping: Final = (
        None
        if raw_cached_mapping == _JWT_PROXY_ADMIN_SENTINEL and not sentinel_written_by_this_policy
        else raw_cached_mapping
    )

    if cached_mapping == _JWT_PROXY_ADMIN_SENTINEL:
        # Previously resolved to a proxy admin via auth_builder; skip the
        # mapping lookup and let the caller re-run auth_builder. Avoids a
        # repeated DB hit on every proxy-admin request under AUTO_REGISTER.
        return None

    if cached_mapping == "__NO_MAPPING__":
        if behavior == UnregisteredJWTClientBehavior.REJECT:
            raise HTTPException(
                status_code=403,
                detail=f"JWT Key Mapping: No registered mapping for {virtual_key_claim_field}='{claim_value}'. Access denied.",
            )
        if behavior in (UnregisteredJWTClientBehavior.AUTO_REGISTER, UnregisteredJWTClientBehavior.LINK_ONLY):
            # Stale sentinel written under a prior fallback_team_mapping config —
            # evict it and defer auto-register to after auth_builder runs. Raise
            # the same 500 as the fresh-path AUTO_REGISTER branch when there is
            # no DB, so behavior is consistent regardless of whether the cache
            # happens to hold the sentinel.
            if prisma_client is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"JWT Key Mapping: {behavior.value.upper()} requires a database connection. "
                        "Configure a database or change unregistered_jwt_client_behavior."
                    ),
                )
            await user_api_key_cache.async_delete_cache(cache_key)
            return _PendingAutoRegister(
                claim_field=virtual_key_claim_field,
                claim_value=str(claim_value),
                cache_key=cache_key,
                jwt_issuer=normalized_issuer,
                behavior=behavior,
            )
        return None
    elif cached_mapping is not None:
        return IdentityStore.key_from_principal(
            await IdentityStore(
                prisma_client,
                user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            ).resolve(hashed_token=cached_mapping)
        )

    # Resolve the mapping from DB, or treat prisma_client=None as a definitive
    # miss (no DB → no mapping can exist → apply no-match policy below). An
    # issuer-scoped row wins; falling back to the global (no-issuer) row keeps
    # mappings created before issuer scoping existed working for every issuer.
    # Each tier is cached under ITS OWN key (the global tier under the
    # issuer-less cache key, not under `cache_key`/this issuer's key) so that
    # updating or deleting either row invalidates exactly the cache entries it
    # can affect. Caching a global-row hit under the requesting issuer's key
    # would leave every OTHER issuer that had fallen back to that same global
    # mapping serving its stale token until TTL after the row changes.
    token_hash: Final = (
        await _lookup_jwt_mapping_token_hash(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            virtual_key_claim_field=virtual_key_claim_field,
            claim_value=str(claim_value),
            normalized_issuer=normalized_issuer,
            cache_key=cache_key,
            ttl=jwt_handler.litellm_jwtauth.virtual_key_mapping_cache_ttl,
        )
        if prisma_client is not None
        else None
    )

    if token_hash is not None:
        return IdentityStore.key_from_principal(
            await IdentityStore(
                prisma_client,
                user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            ).resolve(hashed_token=token_hash)
        )

    # No mapping found (DB miss or no DB) — apply no-match policy.
    if behavior == UnregisteredJWTClientBehavior.REJECT:
        # Cache the miss before raising so repeated rejections are served from
        # cache and don't re-query the DB on every request.
        await user_api_key_cache.async_set_cache(
            key=cache_key,
            value="__NO_MAPPING__",
            ttl=jwt_handler.litellm_jwtauth.virtual_key_mapping_cache_ttl,
        )
        raise HTTPException(
            status_code=403,
            detail=f"JWT Key Mapping: No registered mapping for {virtual_key_claim_field}='{claim_value}'. Access denied.",
        )

    if behavior in (UnregisteredJWTClientBehavior.AUTO_REGISTER, UnregisteredJWTClientBehavior.LINK_ONLY):
        if prisma_client is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"JWT Key Mapping: {behavior.value.upper()} requires a database connection. "
                    "Configure a database or change unregistered_jwt_client_behavior."
                ),
            )
        # Defer: caller runs JWTAuthManager.auth_builder to enforce RBAC, scope,
        # custom_validate, and email-domain policy, then auto-registers using
        # the validated identity. Auto-registering here on a signature-only
        # JWT would bypass every JWT policy beyond signature verification.
        return _PendingAutoRegister(
            claim_field=virtual_key_claim_field,
            claim_value=str(claim_value),
            cache_key=cache_key,
            jwt_issuer=normalized_issuer,
            behavior=behavior,
        )

    # FALLBACK_TEAM_MAPPING (default): cache the miss and return None so the
    # caller falls through to standard team-based JWT auth.
    await user_api_key_cache.async_set_cache(
        key=cache_key,
        value="__NO_MAPPING__",
        ttl=jwt_handler.litellm_jwtauth.virtual_key_mapping_cache_ttl,
    )
    return None


def _ensure_litellm_received_at_on_request_state(request: Request) -> datetime:
    """Idempotently stamp ``request.state.litellm_received_at`` with the moment
    litellm's own code started handling this request -- the first line of
    ``user_api_key_auth``, before any auth/pre-call work runs. This is the
    basis for the request-latency Prometheus metrics (see
    ``litellm/integrations/prometheus.py``), and unlike the OTEL SERVER span
    below, it is set unconditionally so those metrics don't depend on OTEL
    being configured.
    """
    existing_received_at: Final[datetime | None] = getattr(request.state, "litellm_received_at", None)
    if existing_received_at is not None:
        return existing_received_at
    received_at: Final = datetime.now(timezone.utc)
    try:
        request.state.litellm_received_at = received_at
    except Exception:
        pass
    return received_at


def _ensure_parent_otel_span_on_request_state(request: Request) -> None:
    """Idempotently create the OTEL SERVER span and stash it on
    ``request.state.parent_otel_span``. Safe to call multiple times.

    Called both at the top of ``user_api_key_auth`` (so body-parse failures
    have a span to close) and inside ``_user_api_key_auth_builder`` (for
    callers that bypass ``user_api_key_auth``, e.g. MCP).
    """
    from litellm.proxy.proxy_server import open_telemetry_logger

    start_time: Final = _ensure_litellm_received_at_on_request_state(request)

    if open_telemetry_logger is None:
        return
    if (existing_span := getattr(request.state, "parent_otel_span", None)) is not None:
        set_request_parent_otel_span(existing_span)
        return
    parent_otel_span: Final = open_telemetry_logger.create_litellm_proxy_request_started_span(
        start_time=start_time,
        headers=safe_get_request_headers(request),
    )
    # Under V2 the FastAPI instrumentor stamps http.route / url.path on the server
    # span; only the legacy logger needs these set explicitly.
    set_route_attrs: Final = getattr(open_telemetry_logger, "set_proxy_request_route_attributes", None)
    if not is_otel_v2_enabled() and set_route_attrs is not None:
        set_route_attrs(
            parent_otel_span,
            url_path=get_request_route(request=request),
            http_route=get_request_route_template(request),
        )
    request.state.parent_otel_span = parent_otel_span
    set_request_parent_otel_span(parent_otel_span)


async def _read_request_body_deferring_parse_failure(
    request: Request,
) -> tuple[dict, ProxyException | None]:
    """Parse the body, returning a parse failure instead of raising it.

    A body that fails to parse is still a request from a known caller, so auth
    must run (resolving identity onto the request's trace) before the 400 goes
    out; the caller re-raises the returned exception once identity is seeded.
    """
    if is_opaque_pass_through_request(
        route=get_request_route(request=request),
        content_type=safe_get_request_headers(request=request).get("content-type", ""),
    ):
        safe_set_request_parsed_body(request=request, parsed_body={})
        return {}, None
    try:
        parsed_body: Final = await read_request_body(request=request)
    except ProxyException as parse_exception:
        return {}, parse_exception
    return populate_request_with_path_params(request_data=parsed_body, request=request), None


async def _record_unparsable_body_failure(
    user_api_key_dict: UserAPIKeyAuth,
    body_parse_exception: ProxyException,
    route: str,
) -> None:
    """Record the 400 an unparsable body earns as a failed request log.

    The endpoint never runs for these, so no downstream failure hook writes the
    spend log row the Admin UI reads. Logging must not change what the caller
    sees, so a failure here is swallowed and the 400 is raised either way.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj

    try:
        await proxy_logging_obj.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # bare dict in sig
            request_data={},
            original_exception=body_parse_exception,
            user_api_key_dict=user_api_key_dict,
            error_type=ProxyErrorTypes.bad_request_error,
            route=route,
        )
    except Exception as e:  # noqa: BLE001  # any logging failure must leave the caller's 400 untouched
        verbose_proxy_logger.exception("Failed to log the request rejected for an unparsable body: %s", e)


async def _refresh_session_token_grants(
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> UserAPIKeyAuth:
    """Rebuild a ``lite login`` session token's grants from the live user and team rows.

    The blob only proves who logged in and which team they picked. Team models, aliases, the user's own model
    list, and their role are re-read every request, so a `/team/update` or a demotion shows up without a
    re-login, and a user removed from the team or deleted outright is refused. When a row cannot be read for
    a reason unrelated to the caller, the minted grants stand in exactly as they did before this refresh.
    """
    outcome: Final = await GrantResolver(
        prisma_client,
        user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
        load_user=get_user_object,
        load_team=get_team_object,
        load_membership=get_team_membership,
    ).resolve(UserLookup(user_id=valid_token.user_id), team_id=valid_token.team_id)
    match outcome:
        case ResolvedGrants(
            user_object=LiteLLM_UserTable() as user_object, team_object=team_object, team_membership=team_membership
        ):
            return UserAPIKeyAuth.model_validate(
                MappingProxyType(
                    {
                        **valid_token.model_dump(exclude_none=True),
                        **team_grants(team_object, team_membership, user_object.user_id),
                        "user_role": get_user_role(user_object),
                        "models": () if team_object is not None else user_models(user_object),
                    }
                )
            )
        case ResolvedGrants():
            return valid_token
        case LookupDegraded(error=error):
            verbose_proxy_logger.debug("Session token grants not refreshed, keeping minted grants: %s", error)
            return valid_token
        case _:
            raise_public(outcome)


async def _resolve_object_permission_for_unresolvable_team(
    object_permission_id: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> LiteLLM_ObjectPermissionTable | None:
    """Re-resolve a team's object permission by id when the team row itself is unreadable, so the
    token-derived fallback doesn't silently drop it."""
    if object_permission_id is None or prisma_client is None:
        return None
    return await get_object_permission(
        object_permission_id=object_permission_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )


def _custom_auth_token_for_this_request(response: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth.model_validate(response).model_copy()


_AUTH_CORE: Final = AUTH_CORE


class _ResolvedEndUser(NamedTuple):
    end_user_id: str | None
    end_user_object: LiteLLM_EndUserTable | None
    budget: LiteLLM_BudgetTable | None

    @property
    def params(self) -> dict[str, object]:
        if not self.end_user_id:
            return {}
        region: Final = (
            {} if self.end_user_object is None else {"allowed_model_region": self.end_user_object.allowed_model_region}
        )
        params: Final[dict[str, object]] = {"end_user_id": self.end_user_id, **region}
        if self.budget is not None:
            _apply_budget_limits_to_end_user_params(
                end_user_params=params, budget_info=self.budget, end_user_id=self.end_user_id
            )
        return params


async def _resolve_end_user(
    request: Request,
    request_data: Mapping[str, object],
    token_end_user_id: str | None,
    api_key: str,
    key_is_resolved: bool,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
    route: str,
) -> _ResolvedEndUser:
    raw_end_user_id: Final = token_end_user_id or get_end_user_id_from_request_body(
        request_data, safe_get_request_headers(request)
    )
    end_user_id: Final = await resolve_and_validate_end_user_id(
        raw_end_user_id=raw_end_user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
        route=route,
    )
    if prisma_client is not None:
        await prefetch_identity_keys(
            _identity_cache_keys(api_key, end_user_id=end_user_id, key_is_resolved=key_is_resolved),
            user_api_key_cache=user_api_key_cache,
            before_read=arm_routing_cooldowns,
        )
    if not end_user_id:
        return _ResolvedEndUser(end_user_id=end_user_id, end_user_object=None, budget=None)
    try:
        with tracer.trace("litellm.proxy.auth.get_end_user_object"):
            end_user_object: Final = await get_end_user_object(
                end_user_id=end_user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
                route=route,
            )
        if end_user_object is not None:
            return _ResolvedEndUser(
                end_user_id=end_user_id, end_user_object=end_user_object, budget=end_user_object.litellm_budget_table
            )
        if litellm.max_end_user_budget_id is None:
            return _ResolvedEndUser(end_user_id=end_user_id, end_user_object=None, budget=None)
        default_budget: Final = await get_default_end_user_budget(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
        )
        return _ResolvedEndUser(end_user_id=end_user_id, end_user_object=None, budget=default_budget)
    except litellm.BudgetExceededError:
        raise
    except Exception as e:
        verbose_proxy_logger.debug("Unable to find user in db. Error - %s", e)
        return _ResolvedEndUser(end_user_id=end_user_id, end_user_object=None, budget=None)


def _proxy_model_list() -> list[object] | None:
    from litellm.proxy.proxy_server import llm_model_list

    return llm_model_list


_MASTER_KEY_ONLY_ROUTES: Final = frozenset(LiteLLMRoutes.master_key_only_routes.value)


def _core_settings(master_key: str | None, allow_user_auth: bool, prisma_client: PrismaClient | None) -> KeySettings:
    return KeySettings(
        master_key=master_key,
        allow_user_auth=allow_user_auth,
        has_db=prisma_client is not None,
        master_key_only_routes=_MASTER_KEY_ONLY_ROUTES,
        proxy_max_budget=litellm.max_budget,
    )


def _aware_expiry(expires: datetime | str) -> datetime:
    expiry: Final = expires if isinstance(expires, datetime) else datetime.fromisoformat(expires)
    return (
        expiry if expiry.tzinfo is not None and expiry.utcoffset() is not None else expiry.replace(tzinfo=timezone.utc)
    )


def _presented_key(key: UserAPIKeyAuth) -> PresentedKey:
    if key.user_role != LitellmUserRoles.PROXY_ADMIN:
        return NonAdminKey()
    return AdminKey(expires=None if key.expires is None else _aware_expiry(key.expires))


def _with_end_user(key: UserAPIKeyAuth, end_user: _ResolvedEndUser, parent_otel_span: Span | None) -> UserAPIKeyAuth:
    with_end_user: Final = update_valid_token_with_end_user_params(valid_token=key, end_user_params=end_user.params)
    with_end_user.parent_otel_span = parent_otel_span
    if end_user.end_user_object is not None:
        with_end_user.end_user_object_permission = end_user.end_user_object.object_permission
    return with_end_user


def _reset_end_user_fields(key: UserAPIKeyAuth, end_user: _ResolvedEndUser) -> UserAPIKeyAuth:
    key.end_user_id = end_user.end_user_id or None
    key.end_user_tpm_limit = end_user.budget.tpm_limit if end_user.budget is not None else None
    key.end_user_rpm_limit = end_user.budget.rpm_limit if end_user.budget is not None else None
    key.end_user_tpd_limit = end_user.budget.tpd_limit if end_user.budget is not None else None
    if end_user.budget is not None and end_user.budget.model_tpm_limit is not None:
        key.end_user_model_tpm_limit = end_user.budget.model_tpm_limit
    if end_user.budget is not None and end_user.budget.model_rpm_limit is not None:
        key.end_user_model_rpm_limit = end_user.budget.model_rpm_limit
    key.allowed_model_region = (
        end_user.end_user_object.allowed_model_region if end_user.end_user_object is not None else None
    )
    return key


async def _refresh_team_fields_from_cache(
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> None:
    if valid_token.team_id is None or valid_token.team_id == UI_TEAM_ID:
        return
    ## UPDATE TEAM VALUES BASED ON CACHED TEAM OBJECT - allows `/team/update` values to work for cached token
    try:
        team_obj: Final[LiteLLM_TeamTableCachedObj] = await get_team_object(
            team_id=valid_token.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            check_cache_only=True,
        )

        if (
            team_obj.last_refreshed_at is not None
            and valid_token.last_refreshed_at is not None
            and team_obj.last_refreshed_at > valid_token.last_refreshed_at
        ):
            team_obj_dict: Final[Mapping[str, object]] = MappingProxyType(team_obj.__dict__)

            for k, v in team_obj_dict.items():
                field_name = f"team_{k}"
                if field_name in valid_token.__fields__:
                    setattr(valid_token, field_name, v)
    except Exception as e:
        verbose_logger.debug(e)  # moving from .warning to .debug as it spams logs when team missing from cache.


class _KeyHost:
    def __init__(
        self,
        *,
        request: Request,
        request_data: dict[str, object],  # mutable-ok: model access checks rewrite the request model in place
        token: str,
        preresolved_key: UserAPIKeyAuth | None,
        prisma_client: PrismaClient | None,
        user_api_key_cache: UserApiKeyCache,
        parent_otel_span: Span | None,
        proxy_logging_obj: ProxyLogging,
        route: str,
    ) -> None:
        self._request: Final = request
        self._request_data: Final = request_data
        self._token: Final = token
        self._preresolved_key: Final = preresolved_key
        self._prisma_client: Final = prisma_client
        self._user_api_key_cache: Final = user_api_key_cache
        self._parent_otel_span: Final = parent_otel_span
        self._proxy_logging_obj: Final = proxy_logging_obj
        self._route: Final = route
        self.end_user: _ResolvedEndUser = _ResolvedEndUser(end_user_id=None, end_user_object=None, budget=None)
        self.resolved_key: UserAPIKeyAuth | None = preresolved_key
        self.owner: LiteLLM_UserTable | None = None

    @property
    def token(self) -> str:
        return self._token

    @property
    def parent_otel_span(self) -> Span | None:
        return self._parent_otel_span

    @property
    def user_api_key_cache(self) -> UserApiKeyCache:
        return self._user_api_key_cache

    @property
    def proxy_logging_obj(self) -> ProxyLogging:
        return self._proxy_logging_obj

    @property
    def key(self) -> UserAPIKeyAuth:
        if self.resolved_key is None:
            raise RuntimeError("the auth core granted a key it never resolved")
        return self.resolved_key

    def _identity_store(self, *, check_cache_only: bool) -> IdentityStore:
        return IdentityStore(
            self._prisma_client,
            self._user_api_key_cache,
            parent_otel_span=self._parent_otel_span,
            proxy_logging_obj=self._proxy_logging_obj,
            check_cache_only=check_cache_only,
        )

    async def resolve_end_user(self, prefetch_key: bool) -> None:
        self.end_user = await _resolve_end_user(
            request=self._request,
            request_data=self._request_data,
            token_end_user_id=self._preresolved_key.end_user_id if self._preresolved_key is not None else None,
            api_key=self._token,
            key_is_resolved=not prefetch_key or self._preresolved_key is not None,
            prisma_client=self._prisma_client,
            user_api_key_cache=self._user_api_key_cache,
            parent_otel_span=self._parent_otel_span,
            proxy_logging_obj=self._proxy_logging_obj,
            route=self._route,
        )

    async def _cached_key(self, hashed_token: str) -> UserAPIKeyAuth | None:
        try:
            with tracer.trace("litellm.proxy.auth.get_key_object_check_cache"):
                cached: Final = IdentityStore.key_from_principal(
                    await self._identity_store(check_cache_only=True).resolve(hashed_token=hashed_token)
                )
        except Exception:
            verbose_logger.debug("api key not found in cache.")
            return None
        # Key-cache entries are written only after the proxy validated a virtual key or the master key, but
        # via_virtual_key is exclude=True so serialization drops it. Restore it at this trusted boundary.
        cached.via_virtual_key = True
        return cached

    def _ui_login_key(self) -> UserAPIKeyAuth | None:
        # Decryption fails closed for anything that is not a genuine UI login blob.
        if self._token.startswith("sk-") or get_secret_bool("EXPERIMENTAL_UI_LOGIN") is False:
            return None
        return ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(self._token)

    async def resolve_presented_key(self, hashed_token: str) -> PresentedKey | None:
        found: Final = self._preresolved_key or await self._cached_key(hashed_token) or self._ui_login_key()
        if found is None:
            return None
        self.resolved_key = (
            await _refresh_session_token_grants(
                valid_token=found,
                prisma_client=self._prisma_client,
                user_api_key_cache=self._user_api_key_cache,
                parent_otel_span=self._parent_otel_span,
                proxy_logging_obj=self._proxy_logging_obj,
            )
            if found.is_session_token and self._prisma_client is not None
            else found
        )
        return _presented_key(self.resolved_key)

    async def evict_presented_key(self, hashed_token: str) -> None:
        await delete_cache_key_object(
            hashed_token=hashed_token,
            user_api_key_cache=self._user_api_key_cache,
            proxy_logging_obj=self._proxy_logging_obj,
        )

    async def load_key(self, hashed_token: str) -> None:
        with tracer.trace("litellm.proxy.auth.get_key_object_from_db"):
            self.resolved_key = IdentityStore.key_from_principal(
                await self._identity_store(check_cache_only=False).resolve(hashed_token=hashed_token)
            )

    async def prepare_key(self, loaded_from_db: bool) -> None:
        if loaded_from_db:
            _reset_end_user_fields(self.key, self.end_user)
            return
        await _refresh_team_fields_from_cache(
            valid_token=self.key,
            prisma_client=self._prisma_client,
            user_api_key_cache=self._user_api_key_cache,
            parent_otel_span=self._parent_otel_span,
            proxy_logging_obj=self._proxy_logging_obj,
        )

    async def inspect_key(self) -> KeyFacts:
        key: Final = _update_key_budget_with_temp_budget_increase(self.key)
        self.resolved_key = key
        return KeyFacts(
            has_token=key.token is not None,
            blocked=key.blocked is True,
            expires=None if key.expires is None else _aware_expiry(key.expires),
            spend=key.spend,
            soft_budget=key.soft_budget,
            max_budget=key.max_budget,
            user_id=key.user_id,
            team_member=TeamMember(user_id=key.user_id, team_id=key.team_id)
            if key.team_member_spend is not None and key.user_id is not None and key.team_id is not None
            else None,
            budget_throttle_pct=budget_throttle_percentage() if should_throttle_budget_exceeded(key) else None,
        )

    async def enforce_model_access(self) -> None:
        await _normalize_model_for_auth(self._request_data, self.key, self._request, self._route)
        self._request.state.requires_key_model_check = True
        await _prefetch_referenced_auth_objects(
            self.key,
            end_user_id=self.end_user.end_user_id,
            user_api_key_cache=self._user_api_key_cache,
            prisma_client=self._prisma_client,
        )

    async def load_key_owner(self, user_id: str) -> OwnerFacts | None:
        try:
            with tracer.trace("litellm.proxy.auth.get_user_object"):
                owner: Final = await get_user_object(
                    user_id=user_id,
                    prisma_client=self._prisma_client,
                    user_api_key_cache=self._user_api_key_cache,
                    user_id_upsert=False,
                    parent_otel_span=self._parent_otel_span,
                    proxy_logging_obj=self._proxy_logging_obj,
                )
        except Exception as e:
            verbose_logger.debug("Unable to get user %s from db/cache, continuing without it: %s", user_id, e)
            return None
        self.owner = owner
        if owner is None:
            return None
        key: Final = self.key
        # The joint verification-token view carries the key's columns only, so the user's own per-model budget
        # reaches enforcement and the post-call increment through the row fetched here.
        key.user_model_max_budget = owner.model_max_budget
        return OwnerFacts(deactivated=isinstance(owner.metadata, dict) and owner.metadata.get("scim_active") is False)

    async def check_budget_exemption(self) -> bool:
        from litellm.proxy.proxy_server import llm_router

        model: Final = _get_model_from_request_context(
            request_data=self._request_data,
            route=self._route,
            request=self._request,
            llm_router=llm_router,
            team_id=self.key.team_id,
        )
        return request_skips_budget_checks(
            route=self._route, model=model, llm_router=llm_router, request_body=self._request_data
        )

    async def read_team_member_budget(self, member: TeamMember) -> TeamMemberBudget | None:
        from litellm.proxy.proxy_server import get_current_spend

        user_id: Final = member.user_id
        team_id: Final = member.team_id

        if self._prisma_client is None:
            return None
        cache_key: Final = team_membership_auth_cache_key(team_id=team_id, user_id=user_id)
        cached: Final = await self._user_api_key_cache.async_get_cache(key=cache_key, model_type=LiteLLM_TeamMembership)
        membership: Final = cached if cached is not None else await self._load_team_membership(user_id, team_id)
        if membership is None or membership.litellm_budget_table is None:
            return None
        max_budget: Final = membership.litellm_budget_table.effective_max_budget(now=datetime.now(timezone.utc))
        if max_budget is None or max_budget <= 0:
            return None
        spend: Final = await get_current_spend(
            counter_key=team_member_counter(user_id, team_id),
            fallback_spend=self.key.team_member_spend or 0.0,
            max_budget=max_budget,
        )
        return TeamMemberBudget(spend=spend, max_budget=max_budget)

    async def _load_team_membership(self, user_id: str, team_id: str) -> LiteLLM_TeamMembership | None:
        if self._prisma_client is None:
            return None
        row: Final = await TeamMembershipRepository(self._prisma_client).table.find_first(
            where={"user_id": user_id, "team_id": team_id},
            include={"litellm_budget_table": True},
        )
        if row is None:
            return None
        membership: Final = LiteLLM_TeamMembership.model_validate(row.model_dump())
        await self._user_api_key_cache.async_set_cache(
            key=team_membership_auth_cache_key(team_id=team_id, user_id=user_id),
            value=membership,
            model_type=LiteLLM_TeamMembership,
            ttl=5,
        )
        return membership

    async def alert_team_member_budget(self, member: TeamMember, budget: TeamMemberBudget) -> None:
        # common_checks sends this alert on requests that get past here, so only the rejected request sends it here.
        team_member_max_budget_alert_check(
            team_id=member.team_id,
            team_alias=self.key.team_alias,
            team_metadata=self.key.team_metadata,
            organization_id=self.key.org_id,
            user_id=member.user_id,
            user_email=self.owner.user_email if self.owner is not None else None,
            proxy_logging_obj=self._proxy_logging_obj,
            spend=budget.spend,
            max_budget=budget.max_budget,
        )

    async def send_key_budget_alerts(self) -> None:
        await virtual_key_max_budget_alert_check(
            valid_token=self.key, proxy_logging_obj=self._proxy_logging_obj, user_obj=self.owner
        )

    async def read_key_spend(self, max_budget: float) -> float:
        from litellm.proxy.proxy_server import get_current_spend

        # Redis-first cross-pod counter, falling back to the key object's recorded spend.
        return await get_current_spend(
            counter_key=key_counter(str(self.key.token)),
            fallback_spend=self.key.spend or 0.0,
            max_budget=max_budget,
        )

    async def alert_key_budget(self, spend: float) -> None:
        send_token_budget_alert(self.key, spend, self._proxy_logging_obj, self.owner)

    async def alert_key_soft_budget(self) -> None:
        send_soft_budget_alert(self.key, self._proxy_logging_obj, self.owner)

    def model_budgets(self) -> _ModelBudgetHost:
        from litellm.proxy.proxy_server import llm_router, model_max_budget_limiter

        return _ModelBudgetHost(
            request=self._request,
            request_data=self._request_data,
            valid_token=self.key,
            route=self._route,
            limiter=model_max_budget_limiter,
            llm_router=llm_router,
            llm_model_list=_proxy_model_list(),
        )

    async def load_key_context(self) -> None:
        await _load_team_and_project(
            valid_token=self.key,
            prisma_client=self._prisma_client,
            user_api_key_cache=self._user_api_key_cache,
            parent_otel_span=self._parent_otel_span,
            proxy_logging_obj=self._proxy_logging_obj,
        )

    async def read_proxy_spend(self) -> float | None:
        if self._prisma_client is None:
            return None
        with tracer.trace("litellm.proxy.auth.get_global_proxy_spend"):
            return await fetch_global_spend_with_event_coordination(
                cache_key=GLOBAL_PROXY_SPEND_CACHE_KEY,
                user_api_key_cache=self._user_api_key_cache,
                prisma_client=self._prisma_client,
            )

    async def alert_proxy_budget(self, spend: float) -> None:
        from litellm.proxy.proxy_server import litellm_proxy_admin_name

        call_info: Final = CallInfo(
            token=self.key.token,
            spend=spend,
            max_budget=litellm.max_budget,
            user_id=litellm_proxy_admin_name,
            team_id=self.key.team_id,
            event_group=Litellm_EntityType.PROXY,
        )
        asyncio.create_task(self._proxy_logging_obj.budget_alerts(type="proxy_budget", user_info=call_info))


async def user_api_key_auth_builder(
    request: Request,
    api_key: str,
    azure_api_key_header: str | None,
    anthropic_api_key_header: str | None,
    google_ai_studio_api_key_header: str | None,
    azure_apim_header: str | None,
    request_data: dict[str, object],
    custom_litellm_key_header: str | None = None,
) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import (
        general_settings,
        jwt_handler,
        litellm_proxy_admin_name,
        llm_router,
        master_key,
        model_max_budget_limiter,
        open_telemetry_logger,
        premium_user,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
        user_custom_auth,
    )

    parent_otel_span: Span | None = None
    # Prefer the receive-instant stamped by the early helper in
    # user_api_key_auth (before body parse) — overwriting it would shorten
    # the preprocessing-duration measurement by the body-parse window.
    start_time: Final = getattr(request.state, "litellm_received_at", None) or datetime.now(timezone.utc)
    try:
        request.state.litellm_received_at = start_time
    except Exception:
        pass
    route: Final[str] = get_request_route(request=request)
    valid_token: UserAPIKeyAuth | None = None

    try:
        with tracer.trace("litellm.proxy.auth.pre_db_read_auth_checks"):
            await pre_db_read_auth_checks(
                request_data=request_data,
                request=request,
                route=route,
            )
        pass_through_endpoints: Final[list[dict] | None] = general_settings.get("pass_through_endpoints", None)
        custom_litellm_key_header_name: Final = general_settings.get("litellm_key_header_name")
        api_key, _ = get_api_key(
            custom_header=(
                request.headers.get(custom_litellm_key_header_name)
                if custom_litellm_key_header_name is not None
                else None
            ),
            custom_litellm_key_header=custom_litellm_key_header,
            api_key=api_key,
            azure_api_key_header=azure_api_key_header,
            anthropic_api_key_header=anthropic_api_key_header,
            google_ai_studio_api_key_header=google_ai_studio_api_key_header,
            azure_apim_header=azure_apim_header,
            pass_through_endpoints=pass_through_endpoints,
            route=route,
            request=request,
        )

        if open_telemetry_logger is not None:
            # Reuse the span created by user_api_key_auth (before body parse)
            # so it survives _read_request_body failures. For callers that
            # bypass user_api_key_auth (e.g. MCP), create it lazily.
            _ensure_parent_otel_span_on_request_state(request)
            parent_otel_span = getattr(request.state, "parent_otel_span", None)

        plan_settings: Final = _plan_settings(
            general_settings,
            jwt_handler,
            premium_user=premium_user is True,
            custom_auth=enterprise_custom_auth is not None or user_custom_auth is not None,
        )
        planned: Final = await _plan_request(
            request=request,
            request_data=request_data,
            route=route,
            api_key=api_key,
            settings=plan_settings,
            jwt_handler=jwt_handler,
            user_custom_auth=user_custom_auth,
            parent_otel_span=parent_otel_span,
        )
        if isinstance(planned, UserAPIKeyAuth):
            return planned
        api_key = planned.api_key
        enable_jwt_auth: Final = plan_settings.jwt_enabled
        credential_plan: Final = planned.plan
        if not isinstance(credential_plan, (KeyValidator, JwtValidator)):
            return await _authenticate_without_key(credential_plan, api_key=api_key, request=request)
        if isinstance(credential_plan, JwtValidator):
            # Try JWT-to-Virtual-Key mapping first to avoid
            # unnecessary DB queries in auth_builder
            do_standard_jwt_auth = True
            pending_auto_register: _PendingAutoRegister | None = None
            if jwt_handler.litellm_jwtauth.is_virtual_key_mapping_configured():
                # Decode JWT to get claims without running full auth_builder
                jwt_claims: dict | None = await jwt_handler.auth_jwt(token=api_key)

                from litellm.proxy.agent_endpoints.identity_store import resolve_managed_agent

                if (
                    jwt_claims
                    and await resolve_managed_agent(jwt_claims, prisma_client, cache=user_api_key_cache) is not None
                ):
                    raise HTTPException(
                        403, "Managed agents require direct JWT authentication without virtual-key mapping"
                    )

                resolve_result: Final = await _resolve_jwt_to_virtual_key(
                    jwt_claims=jwt_claims,
                    jwt_handler=jwt_handler,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                )
                if isinstance(resolve_result, UserAPIKeyAuth):
                    mapped_end_user_id: Final = (
                        jwt_handler.get_end_user_id(  # pyright: ignore[reportUnknownMemberType]  # untyped dict token
                            token=jwt_claims or {}, default_value=None
                        )
                        or None
                    )
                    valid_token = resolve_result.model_copy(update={"end_user_id": mapped_end_user_id})
                    api_key = valid_token.token or ""
                    valid_token.jwt_claims = jwt_claims
                    do_standard_jwt_auth = False
                    # Fall through to virtual key checks
                    if valid_token.user_id is not None and valid_token.user_email is None:
                        mapped_claims = jwt_claims or {}
                        mapped_user_email = jwt_handler.get_user_email(token=mapped_claims, default_value=None)
                        mapped_jwt_user_id: Final = jwt_handler.get_user_id(token=mapped_claims, default_value=None)
                        if mapped_user_email is not None and mapped_jwt_user_id == valid_token.user_id:
                            try:
                                mapped_user_obj: Final = await get_user_object(
                                    user_id=valid_token.user_id,
                                    prisma_client=prisma_client,
                                    user_api_key_cache=user_api_key_cache,
                                    user_id_upsert=False,
                                    parent_otel_span=parent_otel_span,
                                    proxy_logging_obj=proxy_logging_obj,
                                    user_email=mapped_user_email,
                                )
                            except Exception as e:
                                verbose_proxy_logger.debug("JWT mapped-key user_email backfill skipped: %s", e)
                            else:
                                if mapped_user_obj is not None:
                                    valid_token.user_email = mapped_user_obj.user_email
                elif isinstance(resolve_result, _PendingAutoRegister):
                    # Run full JWT policy (RBAC, scope, custom_validate,
                    # email-domain) via auth_builder, then create the key
                    # from the validated identity below.
                    pending_auto_register = resolve_result
                # else: None → FALLBACK_TEAM_MAPPING, falls through to
                # standard JWT auth_builder below

            if do_standard_jwt_auth:
                with tracer.trace("litellm.proxy.auth.jwt_auth_builder"):
                    result: Final = await JWTAuthManager.auth_builder(
                        request_data=request_data,
                        general_settings=general_settings,
                        api_key=api_key,
                        jwt_handler=jwt_handler,
                        route=route,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        proxy_logging_obj=proxy_logging_obj,
                        parent_otel_span=parent_otel_span,
                        request_headers=safe_get_request_headers(request),
                        request_method=RouteChecks.get_request_method(request=request),
                    )

                is_proxy_admin: Final = result["is_proxy_admin"]
                team_id: Final = result["team_id"]
                user_id: Final = result["user_id"]
                user_email: Final = result["user_email"]
                user_object: Final = result["user_object"]
                end_user_id = result["end_user_id"]
                org_id: Final = result["org_id"]
                jwt_claims = result.get("jwt_claims", None)
                agent_id: Final[str | None] = result.get("agent_id")

                if (
                    user_object is not None
                    and isinstance(user_object.metadata, dict)
                    and user_object.metadata.get("scim_active") is False
                ):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"User={user_id} has been deactivated via SCIM. Keys owned by this user cannot be used.",
                    )

                if is_proxy_admin:
                    # Proxy admins authenticate via auth_builder (full
                    # access), not via a mapped virtual key. If
                    # AUTO_REGISTER was pending, cache a sentinel so
                    # future requests from this JWT identity skip the
                    # DB mapping lookup in _resolve_jwt_to_virtual_key.
                    # Without this, every proxy-admin request under
                    # AUTO_REGISTER re-hits get_jwt_key_mapping_object.
                    if pending_auto_register is not None:
                        await user_api_key_cache.async_set_cache(
                            key=pending_auto_register.cache_key,
                            value=_JWT_PROXY_ADMIN_SENTINEL,
                            ttl=jwt_handler.litellm_jwtauth.virtual_key_mapping_cache_ttl,
                        )
                    return JWTAuthManager.user_api_key_auth_from_result(result, parent_otel_span)

                valid_token = JWTAuthManager.user_api_key_auth_from_result(result, parent_otel_span)

                # AUTO_REGISTER deferred from _resolve_jwt_to_virtual_key.
                # JWT policy (RBAC, scope, custom_validate, email-domain)
                # has now been enforced by auth_builder above. Create the
                # mapping + virtual key from the *validated* identity, then
                # replace valid_token with the new key so downstream checks
                # use the key-scoped path.
                auto_registered: Final = (
                    await _auto_register_jwt_mapping(
                        virtual_key_claim_field=pending_auto_register.claim_field,
                        claim_value=pending_auto_register.claim_value,
                        jwt_handler=jwt_handler,
                        prisma_client=prisma_client,
                        user_api_key_cache=user_api_key_cache,
                        parent_otel_span=parent_otel_span,
                        proxy_logging_obj=proxy_logging_obj,
                        cache_key=pending_auto_register.cache_key,
                        jwt_issuer=pending_auto_register.jwt_issuer,
                        team_id=team_id,
                        user_id=user_id,
                        org_id=org_id,
                        end_user_id=end_user_id,
                        agent_id=agent_id,
                        behavior=pending_auto_register.behavior,
                    )
                    if pending_auto_register is not None and prisma_client is not None
                    else None
                )
                if isinstance(auto_registered, _JWTKeyMappingFailure):
                    raise HTTPException(status_code=403, detail=auto_registered.value)
                if auto_registered is not None:
                    auto_registered.jwt_claims = jwt_claims
                    auto_registered.user_email = user_email
                    # The auto-registered token is built from the new key's
                    # columns, which carry no user budget. Carry over the
                    # already-loaded user row rather than re-reading it, or
                    # the budget check below has nothing to enforce.
                    auto_registered.user_model_max_budget = (
                        user_object.model_max_budget if user_object is not None else None
                    )
                    valid_token = auto_registered
                    api_key = valid_token.token or ""

                falls_through_to_key_checks: Final = (
                    auto_registered is not None
                    and (
                        jwt_handler.litellm_jwtauth.auto_register_map_existing_key
                        or (
                            pending_auto_register is not None
                            and pending_auto_register.behavior == UnregisteredJWTClientBehavior.LINK_ONLY
                        )
                    )
                    and master_key is not None
                )
                if not falls_through_to_key_checks:
                    # Check if model has zero cost - if so, skip all budget checks
                    model = _get_model_from_request_context(
                        request_data=request_data,
                        route=route,
                        request=request,
                        llm_router=llm_router,
                        team_id=valid_token.team_id,
                    )
                    skip_budget_checks = request_skips_budget_checks(
                        route=route, model=model, llm_router=llm_router, request_body=request_data
                    )

                    # Fetch project object for JWT path if project_id is set
                    _jwt_project_obj = None
                    if valid_token.project_id is not None:
                        _jwt_project_obj = await get_project_object(
                            project_id=valid_token.project_id,
                            prisma_client=prisma_client,
                            user_api_key_cache=user_api_key_cache,
                            proxy_logging_obj=proxy_logging_obj,
                        )
                        if _jwt_project_obj is not None:
                            valid_token.project_metadata = _jwt_project_obj.metadata
                            valid_token.project_alias = _jwt_project_obj.project_alias

                    # JWT auth returns here rather than falling through to the
                    # virtual-key checks below, so the user's per-model budget
                    # has to be enforced on this path too. Without it the
                    # post-call increment still charges the counter and nothing
                    # ever reads it, which is worse than not tracking at all.
                    # Guarded by the same flag the virtual-key path uses, or a
                    # zero-cost model would be refused here and allowed there,
                    # while the log above claims all budget checks were skipped.
                    if not skip_budget_checks:
                        await _enforce_model_budgets(
                            _AUTH_CORE.model_budgets().check_user,
                            _ModelBudgetHost(
                                request=request,
                                request_data=request_data,
                                valid_token=cast(UserAPIKeyAuth, valid_token),
                                route=route,
                                limiter=model_max_budget_limiter,
                                llm_router=llm_router,
                                llm_model_list=_proxy_model_list(),
                            ),
                        )

                    return cast(UserAPIKeyAuth, valid_token)
        key_plan: Final = (
            credential_plan
            if isinstance(credential_plan, KeyValidator)
            else _AUTH_CORE.plan().after_jwt(planned.facts, plan_settings, issued_key=planned.issued_key)
        )
        if isinstance(key_plan, AnonymousValidator):
            return _anonymous_auth(_AUTH_CORE.anonymous().authenticate(key_plan.reason))
        if key_plan.pass_through_key:
            api_key = await check_api_key_for_custom_headers_or_pass_through_endpoints(
                request=request,
                route=route,
                pass_through_endpoints=pass_through_endpoints,
                api_key=api_key,
            )
        if master_key is not None and not isinstance(master_key, str):
            raise HTTPException(
                status_code=500,
                detail={f"Master key must be a valid string. Current type={type(master_key)}"},
            )
        host: Final = _KeyHost(
            request=request,
            request_data=request_data,
            token=api_key,
            preresolved_key=valid_token,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            route=route,
        )
        try:
            outcome: Final = await _AUTH_CORE.keys().authenticate(
                host,
                RequestFacts(route=route, token=api_key, is_llm_api_route=RouteChecks.is_llm_api_route(route=route)),
                _core_settings(
                    master_key=master_key,
                    allow_user_auth=general_settings.get("allow_user_auth", False) is True,
                    prisma_client=prisma_client,
                ),
                datetime.now(timezone.utc),
            )
        finally:
            valid_token = host.resolved_key  # rebind-ok: the exception handler below reports the resolved identity
        return await _authenticated(
            outcome,
            host,
            admin_user_id=litellm_proxy_admin_name,
            route=route,
            start_time=start_time,
            jwt_auth_enabled=enable_jwt_auth,
        )
    except Exception as e:
        return await UserAPIKeyAuthExceptionHandler.handle_authentication_error(
            e=e,
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=parent_otel_span,
            api_key=api_key,
            resolved_identity=valid_token,
        )


_user_api_key_auth_builder: Final = user_api_key_auth_builder


async def _load_team_and_project(
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> None:
    team_obj: Final = (
        None
        if valid_token.team_id is None
        else await _key_team_object(
            valid_token, valid_token.team_id, prisma_client, user_api_key_cache, parent_otel_span, proxy_logging_obj
        )
    )
    valid_token.team_object_permission = team_obj.object_permission if team_obj is not None else None
    if team_obj is not None:
        # Guardrails or other metadata added after the key was cached apply without a cache eviction.
        valid_token.team_metadata = team_obj.metadata
    if valid_token.project_id is None:
        return
    project_obj: Final = await get_project_object(
        project_id=valid_token.project_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    if project_obj is not None:
        valid_token.project_metadata = project_obj.metadata
        valid_token.project_alias = project_obj.project_alias


async def _key_team_object(
    valid_token: UserAPIKeyAuth,
    team_id: str,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> LiteLLM_TeamTableCachedObj:
    fetched: Final = (
        None
        if team_id == UI_TEAM_ID
        else await _team_object_or_none(team_id, prisma_client, user_api_key_cache, parent_otel_span, proxy_logging_obj)
    )
    if fetched is not None:
        return fetched
    return LiteLLM_TeamTableCachedObj(
        team_id=team_id,
        max_budget=valid_token.team_max_budget,
        soft_budget=valid_token.team_soft_budget,
        model_max_budget=valid_token.team_model_max_budget,
        spend=valid_token.team_spend,
        tpm_limit=valid_token.team_tpm_limit,
        rpm_limit=valid_token.team_rpm_limit,
        tpd_limit=valid_token.team_tpd_limit,
        blocked=valid_token.team_blocked,
        models=_token_team_models(valid_token),
        metadata=valid_token.team_metadata,
        object_permission_id=valid_token.team_object_permission_id,
        object_permission=await _resolve_object_permission_for_unresolvable_team(
            object_permission_id=valid_token.team_object_permission_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        ),
    )


async def _team_object_or_none(
    team_id: str,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging,
) -> LiteLLM_TeamTableCachedObj | None:
    try:
        with tracer.trace("litellm.proxy.auth.get_team_object"):
            return await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
    except HTTPException:
        return None


async def _master_key_auth(host: _KeyHost, admin_user_id: str, route: str, start_time: datetime) -> UserAPIKeyAuth:
    # A stable alias replaces the raw master key so neither the key nor its hash reaches spend logs,
    # metrics labels, audit trails or rate-limit buckets.
    master_key_auth: Final = await _return_user_api_key_auth_obj(
        user_obj=None,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key=LITELLM_PROXY_MASTER_KEY_ALIAS,
        parent_otel_span=host.parent_otel_span,
        valid_token_dict={**host.end_user.params, "user_id": admin_user_id},
        route=route,
        start_time=start_time,
    )
    asyncio.create_task(
        cache_key_object(
            hashed_token=hash_token(host.token),
            user_api_key_obj=master_key_auth,
            user_api_key_cache=host.user_api_key_cache,
            proxy_logging_obj=host.proxy_logging_obj,
        )
    )
    with_end_user: Final = _with_end_user(master_key_auth, host.end_user, host.parent_otel_span)
    with_end_user.via_virtual_key = True
    return with_end_user


async def _authenticated(
    outcome: Outcome,
    host: _KeyHost,
    admin_user_id: str,
    route: str,
    start_time: datetime,
    jwt_auth_enabled: bool,
) -> UserAPIKeyAuth:
    match outcome:
        case Deny(denial=denial):
            raise denial_exception(
                denial, token=host.token, route=route, jwt_auth_enabled=jwt_auth_enabled, key=host.resolved_key
            )
        case Allow(grant=grant):
            return await _granted(grant, host, admin_user_id, route, start_time)


async def _granted(
    grant: Grant, host: _KeyHost, admin_user_id: str, route: str, start_time: datetime
) -> UserAPIKeyAuth:
    match grant:
        case NoMasterKey(token=token):
            return UserAPIKeyAuth(
                api_key=token, user_role=LitellmUserRoles.INTERNAL_USER, parent_otel_span=host.parent_otel_span
            )
        case UserAuthRoute():
            return UserAPIKeyAuth()
        case MasterKey():
            return await _master_key_auth(host, admin_user_id, route, start_time)
        case CachedAdminKey():
            return _with_end_user(host.key, host.end_user, host.parent_otel_span)
        case VirtualKey(budget_throttle_pct=budget_throttle_pct):
            return await _virtual_key_auth(host, budget_throttle_pct, route, start_time)
        case _:
            assert_never(grant)


async def _virtual_key_auth(
    host: _KeyHost, budget_throttle_pct: float | None, route: str, start_time: datetime
) -> UserAPIKeyAuth:
    key: Final = host.key
    token: Final = key.token
    if token is None:
        raise RuntimeError("the auth core granted a virtual key without a token")
    end_user: Final = host.end_user
    # budget_throttle_pct is excluded from model_dump so it never leaks into serialized responses.
    throttle: Final = {} if budget_throttle_pct is None else {"budget_throttle_pct": budget_throttle_pct}
    end_user_params: Final = (
        end_user.params if end_user.end_user_object is not None or litellm.max_end_user_budget_id is not None else {}
    )
    end_user_permission: Final = (
        {}
        if end_user.end_user_object is None
        else {"end_user_object_permission": end_user.end_user_object.object_permission}
    )
    virtual_key_auth_obj: Final = await _return_user_api_key_auth_obj(
        user_obj=host.owner,
        api_key=token,
        parent_otel_span=host.parent_otel_span,
        valid_token_dict={
            **{k: v for k, v in key.model_dump(exclude_none=True).items() if k != "token"},
            **throttle,
            **end_user_params,
            **end_user_permission,
        },
        route=route,
        start_time=start_time,
    )
    virtual_key_auth_obj.via_virtual_key = True
    return virtual_key_auth_obj


async def _safe_fetch(label: str, awaitable):
    """Run an awaitable and return its result. Re-raises authentication /
    authorization failures (HTTPException, ProxyException,
    BudgetExceededError) so they propagate to the caller.
    Other exceptions (e.g. transient DB errors fetching context) are
    swallowed with a debug log and ``None`` is returned so
    ``common_checks`` can still run against whatever limits are recorded
    directly on the token.
    """
    try:
        return await awaitable
    except (HTTPException, ProxyException, litellm.BudgetExceededError) as e:
        verbose_proxy_logger.debug(
            "centralized auth: %s fetch failed (%s: %s)",
            label,
            type(e).__name__,
            e,
        )
        raise
    except Exception as e:
        verbose_proxy_logger.debug(
            "centralized auth: %s fetch swallowed (%s: %s)",
            label,
            type(e).__name__,
            e,
        )
        return None


def _team_obj_from_token(valid_token: UserAPIKeyAuth) -> LiteLLM_TeamTableCachedObj:
    """Reconstruct a cached team object from the fields already on the
    UserAPIKeyAuth. Only called when valid_token.team_id is known to be
    non-None (the caller gates on it)."""
    assert valid_token.team_id is not None
    token_team_models: Final = _token_team_models(valid_token)
    return LiteLLM_TeamTableCachedObj(
        team_id=valid_token.team_id,
        max_budget=valid_token.team_max_budget,
        soft_budget=valid_token.team_soft_budget,
        model_max_budget=valid_token.team_model_max_budget,
        spend=valid_token.team_spend,
        tpm_limit=valid_token.team_tpm_limit,
        rpm_limit=valid_token.team_rpm_limit,
        tpd_limit=valid_token.team_tpd_limit,
        blocked=valid_token.team_blocked,
        models=token_team_models,
        metadata=valid_token.team_metadata,
        object_permission_id=valid_token.team_object_permission_id,
    )


def _token_can_vouch_for_team(valid_token: UserAPIKeyAuth, lookup_error: BaseException) -> bool:
    """Whether the token's own team fields may stand in for a team that failed to
    resolve, without widening access.

    The UI dashboard mints every session key against the ``UI_TEAM_ID`` sentinel,
    which by design never has a team row, so a failed lookup for it is not a
    degraded read to be treated with suspicion; it always vouches, exactly as it
    always safely has (these keys are restricted elsewhere to UI-only routes).

    For every other team, a team that is provably gone is a definitive answer,
    not a degraded read, so nothing may stand in for it and no setting may
    override that.

    Otherwise the team's grant is merely unknown. A token carrying one may vouch,
    since replaying a recorded grant cannot widen it and denying every team key
    while the row is briefly unreadable would trade the widening for an outage. A
    token carrying none may not: ``team_models=[]`` reads as every model and
    ``team_blocked=False`` as unblocked. ``allow_requests_on_db_unavailable`` opts
    back out, and is only consulted here because the failure is known by this
    point to be a degraded read.
    """
    if valid_token.team_id == UI_TEAM_ID:
        return True
    if isinstance(lookup_error, TeamNotFoundError):
        return False
    if valid_token.team_models:
        return True
    return PrismaDBExceptionHandler.should_allow_request_on_db_unavailable()


async def _inherit_org_identity(
    user_api_key_auth_obj: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTableCachedObj | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    proxy_logging_obj: ProxyLogging | None,
) -> None:
    if user_api_key_auth_obj.org_id is None and team_object is not None and team_object.organization_id is not None:
        user_api_key_auth_obj.org_id = team_object.organization_id
    already_populated: Final = any(
        value is not None
        for value in (
            user_api_key_auth_obj.organization_alias,
            user_api_key_auth_obj.organization_max_budget,
            user_api_key_auth_obj.organization_tpm_limit,
            user_api_key_auth_obj.organization_rpm_limit,
            user_api_key_auth_obj.organization_metadata,
        )
    )
    if user_api_key_auth_obj.org_id is None or already_populated or prisma_client is None:
        return
    org_object: Final = await get_org_object_for_request(
        org_id=user_api_key_auth_obj.org_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if org_object is None:
        return
    user_api_key_auth_obj.organization_alias = org_object.organization_alias
    user_api_key_auth_obj.organization_metadata = org_object.metadata
    budget: Final = org_object.litellm_budget_table
    if budget is None:
        return
    user_api_key_auth_obj.organization_max_budget = budget.max_budget
    user_api_key_auth_obj.organization_tpm_limit = budget.tpm_limit
    user_api_key_auth_obj.organization_rpm_limit = budget.rpm_limit


def is_no_auth_dev_mode(master_key: str | None, general_settings: Mapping[str, object]) -> bool:
    return master_key is None and not any(
        general_settings.get(flag, False)
        for flag in ("enable_jwt_auth", "enable_oauth2_auth", "enable_oauth2_proxy_auth")
    )


async def _load_team_member_default_model_budget(
    user_api_key_auth_obj: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    skip_budget_checks: bool,
    proxy_logging_obj: ProxyLogging | None,
) -> None:
    team_metadata_value: Final[object] = getattr(team_object, "metadata", None) if team_object is not None else None
    team_metadata: Final[Mapping[str, object] | None] = (
        cast(  # cast-ok: team metadata is a string-keyed JSON object
            Mapping[str, object], team_metadata_value
        )
        if isinstance(team_metadata_value, Mapping)
        else None
    )
    team_member_budget_id: Final[object | None] = (
        team_metadata.get("team_member_budget_id") if isinstance(team_metadata, Mapping) else None
    )
    if (
        not isinstance(user_api_key_auth_obj.user_id, str)
        or not isinstance(user_api_key_auth_obj.team_id, str)
        or prisma_client is None
    ):
        return

    try:
        membership: Final = await get_team_membership(
            user_id=user_api_key_auth_obj.user_id,
            team_id=user_api_key_auth_obj.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        member_budget_row: Final = membership.litellm_budget_table if membership is not None else None
        member_budget_value: Final[object] = (
            getattr(member_budget_row, "model_max_budget", None) if member_budget_row is not None else None
        )
        if (
            membership is not None
            and membership.budget_id is not None
            and membership.budget_id != team_member_budget_id
            and isinstance(member_budget_value, Mapping)
            and member_budget_value
        ):
            user_api_key_auth_obj.team_member_model_max_budget = (  # rebind-ok: auth context carries member budget downstream
                member_budget_value
            )
            return
        if not isinstance(team_member_budget_id, str):
            return
        team_member_default_budget: Final = await get_team_member_default_budget(
            budget_id=team_member_budget_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            raise_on_lookup_error=True,
        )
    except Exception:
        if not skip_budget_checks:
            raise
        verbose_proxy_logger.warning("Skipping team-member model budget lookup after a lookup error", exc_info=True)
        return

    default_budget_value: Final[object] = (
        getattr(team_member_default_budget, "model_max_budget", None)
        if team_member_default_budget is not None
        else None
    )
    user_api_key_auth_obj.team_member_model_max_budget = (  # rebind-ok: auth context carries default budget downstream
        default_budget_value if isinstance(default_budget_value, Mapping) else None
    )


@tracer.wrap()
async def run_centralized_common_checks(
    user_api_key_auth_obj: UserAPIKeyAuth,
    request: Request,
    request_data: dict[str, object],
    route: str,
    *,
    force_virtual_key_checks: bool = False,
    sub_call: bool = False,
) -> None:
    """Run ``common_checks`` once at the ``user_api_key_auth`` wrapper
    boundary, regardless of which ``_user_api_key_auth_builder`` path
    returned. This is the single invariant enforcement point for key
    model-access, budgets, guardrails, org, and vector-store checks.

    Invariants:
    - ``user_custom_auth`` with ``custom_auth_run_common_checks`` unset
      skips the gate — matches the existing custom-auth RPS guarantee.
      Custom-auth deployments don't use OAuth2 / DB-fallback paths, so
      the skip does not re-open any bypass.
    - ``PROXY_ADMIN`` tokens still run through ``common_checks`` so
      team-blocked / team-budget / end-user-budget / tag-budget /
      vector-store / tool-allowlist enforcement applies to admin keys
      too. Admin status is honored where the underlying check exempts it
      (``_is_api_route_allowed``, ``organization_role_based_access_check``).
    - ``sub_call`` marks a proxy-made call inside an already admitted
      request. Its model's budgets are checked, the key's own
      ``max_budget`` included, because the builder checked that only for
      the parent's model, which may be free. Nothing is reserved, because
      the sub-call is charged once it runs.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        litellm_proxy_admin_name,
        llm_model_list,  # pyright: ignore[reportUnknownVariableType]  # proxy global has untyped deployment entries
        llm_router,
        master_key,
        model_max_budget_limiter,
        prisma_client,
        proxy_logging_obj,
        spend_counter_store,
        user_api_key_cache,
        user_custom_auth,
    )

    # Public routes (e.g. /health/liveness) are exempt from
    # auth in the builder — the wrapper must not retroactively apply
    # authz on top, or k8s readiness probes and other unauthenticated
    # callers get 401.
    if not force_virtual_key_checks and (
        route in LiteLLMRoutes.public_routes.value or route_in_additonal_public_routes(current_route=route)
    ):
        return

    # User-configured pass-through endpoints with ``auth: false`` are
    # explicitly unauthenticated — the builder returns an empty
    # UserAPIKeyAuth() and the request is forwarded as-is. Running
    # common_checks on the empty token would reject the request as
    # admin-only. The "auth" flag on the endpoint config is the
    # contract; honor it.
    pass_through_endpoints: Final = general_settings.get("pass_through_endpoints", None)
    if not force_virtual_key_checks and pass_through_endpoints is not None:
        for endpoint in pass_through_endpoints:
            if isinstance(endpoint, dict) and endpoint.get("path", "") == route and endpoint.get("auth") is not True:
                return

    # No-auth dev mode: master_key unset AND no JWT/OAuth2 auth
    # configured. The builder returns an INTERNAL_USER token for any
    # api_key; the proxy is unauthenticated by configuration.
    # Running common_checks would block every admin route on these
    # deployments where that was previously not the contract. If any
    # authn is enabled (JWT, OAuth2, OAuth2-proxy), authz must run.
    if not force_virtual_key_checks and is_no_auth_dev_mode(master_key, general_settings):
        return

    if (
        not force_virtual_key_checks
        and user_custom_auth is not None
        and not general_settings.get("custom_auth_run_common_checks", False)
    ):
        return

    parent_otel_span: Final = user_api_key_auth_obj.parent_otel_span
    # In the integrated auth flow ``_user_api_key_auth_builder`` has already
    # resolved the end-user id and attached it here. Reuse that to avoid a
    # second extraction pass; fall back to extracting locally when the
    # function is invoked in isolation (e.g. in direct unit tests).
    key_end_user_budget_id: Final = get_key_end_user_budget_id(user_api_key_auth_obj.metadata)
    end_user_id = user_api_key_auth_obj.end_user_id
    if end_user_id is None:
        raw_end_user_id: Final = get_end_user_id_from_request_body(request_data, safe_get_request_headers(request))
        end_user_id = await resolve_and_validate_end_user_id(
            raw_end_user_id=raw_end_user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            route=route,
            key_end_user_budget_id=key_end_user_budget_id,
        )
        if end_user_id is not None and key_end_user_budget_id is not None:
            user_api_key_auth_obj.end_user_id = end_user_id

    fetch_coros: Final = []
    if user_api_key_auth_obj.team_id is not None and user_api_key_auth_obj.team_id != UI_TEAM_ID:
        fetch_coros.append(
            _safe_fetch(
                "team",
                get_team_object(
                    team_id=user_api_key_auth_obj.team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                ),
            )
        )
    else:
        fetch_coros.append(_safe_fetch("team", _noop_none()))

    if user_api_key_auth_obj.user_id is not None:
        fetch_coros.append(
            _safe_fetch(
                "user",
                get_user_object(
                    user_id=user_api_key_auth_obj.user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    user_id_upsert=False,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                ),
            )
        )
    else:
        fetch_coros.append(_safe_fetch("user", _noop_none()))

    if user_api_key_auth_obj.project_id is not None:
        fetch_coros.append(
            _safe_fetch(
                "project",
                get_project_object(
                    project_id=user_api_key_auth_obj.project_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                ),
            )
        )
    else:
        fetch_coros.append(_safe_fetch("project", _noop_none()))

    if end_user_id:
        fetch_coros.append(
            _safe_fetch(
                "end_user",
                get_end_user_object(
                    end_user_id=end_user_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    parent_otel_span=parent_otel_span,
                    proxy_logging_obj=proxy_logging_obj,
                    route=route,
                    token_end_user_max_budget=user_api_key_auth_obj.end_user_max_budget,
                    key_end_user_budget_id=key_end_user_budget_id,
                ),
            )
        )
    else:
        fetch_coros.append(_safe_fetch("end_user", _noop_none()))

    fetch_coros.append(
        _safe_fetch(
            "global_spend",
            get_global_proxy_spend(
                litellm_proxy_admin_name=litellm_proxy_admin_name,
                user_api_key_cache=user_api_key_cache,
                prisma_client=prisma_client,
                token=user_api_key_auth_obj.token or "",
                proxy_logging_obj=proxy_logging_obj,
            ),
        )
    )

    # Per-fetch error isolation. ``_safe_fetch`` lets HTTPException,
    # ProxyException, and BudgetExceededError escape (everything else is
    # already swallowed to None). A bare ``except`` over ``gather`` would
    # let one fetch's HTTPException null out every other context — e.g.
    # a 404 from ``get_team_object`` (token references a deleted team)
    # would silently skip the user, end-user, project, and global-spend
    # checks. Use ``return_exceptions=True`` and apply per-fetch fallback
    # so a missing team only zeros out the team object.
    (
        team_result,
        user_result,
        project_result,
        end_user_result,
        global_spend_result,
    ) = await asyncio.gather(*fetch_coros, return_exceptions=True)

    # ProxyException / BudgetExceededError are authorization failures —
    # propagate so the wrapper renders them. HTTPException is fallback
    # material (404 from get_team_object is the only known producer).
    for r in (
        team_result,
        user_result,
        project_result,
        end_user_result,
        global_spend_result,
    ):
        if isinstance(r, (ProxyException, litellm.BudgetExceededError)):
            raise r

    # Use BaseException (not HTTPException) in the narrowing checks so
    # mypy can narrow ``Any | BaseException`` to the typed object in the
    # else branch. After the for-loop above, the only BaseException that
    # can still appear here is HTTPException (other listed re-raises were
    # propagated; non-listed exceptions were already swallowed to None).
    team_object: LiteLLM_TeamTableCachedObj | None
    if isinstance(team_result, BaseException):
        # Token-derived fallback only valid when a team_id is set;
        # _team_obj_from_token asserts that precondition.
        if user_api_key_auth_obj.team_id is None:
            team_object = None
        elif _token_can_vouch_for_team(user_api_key_auth_obj, team_result):
            team_object = _team_obj_from_token(user_api_key_auth_obj)
        else:
            raise team_result
    else:
        team_object = (
            _team_obj_from_token(user_api_key_auth_obj) if user_api_key_auth_obj.team_id == UI_TEAM_ID else team_result
        )

    user_object: LiteLLM_UserTable | None = user_result if isinstance(user_result, LiteLLM_UserTable) else None
    project_object: Final[LiteLLM_ProjectTableCachedObj | None] = (
        None if isinstance(project_result, BaseException) else project_result
    )
    end_user_object: Final = end_user_result if isinstance(end_user_result, LiteLLM_EndUserTable) else None
    global_proxy_spend: float | None = None if isinstance(global_spend_result, BaseException) else global_spend_result
    carry_team_and_user_budget_state(
        valid_token=user_api_key_auth_obj,
        team_object=team_object,
        user_object=user_object,
    )

    await _inherit_org_identity(
        user_api_key_auth_obj=user_api_key_auth_obj,
        team_object=team_object,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )

    # common_checks identifies admin via user_object, not the token
    # (non_proxy_admin_allowed_routes_check). JWT admin shortcut and
    # master_key tokens get admin from the token; the DB row for the
    # same user_id (e.g. litellm_proxy_admin_name = "default_user_id")
    # may have a non-admin user_role and would otherwise demote the
    # caller. The token is the source of truth for these paths — force
    # the admin user_object whenever the token says PROXY_ADMIN, even
    # if a DB row was fetched. Every admin token except the built-in admin
    # identity (litellm_proxy_admin_name: the master key and the UI
    # username/password login) keeps the row's personal budget (max_budget,
    # and user_email for budget alerts) so common_checks enforces it; every
    # other field keeps its default.
    if user_api_key_auth_obj.user_role == LitellmUserRoles.PROXY_ADMIN:
        budget_row: Final = (
            user_object
            if isinstance(user_object, LiteLLM_UserTable) and user_api_key_auth_obj.user_id != litellm_proxy_admin_name
            else None
        )
        user_object = LiteLLM_UserTable(
            user_id=user_api_key_auth_obj.user_id or litellm_proxy_admin_name,
            user_role=LitellmUserRoles.PROXY_ADMIN,
            spend=user_object.spend if user_object is not None else 0.0,
            max_budget=budget_row.max_budget if budget_row is not None else None,
            user_email=budget_row.user_email if budget_row is not None else None,
            object_permission_id=(
                user_object.object_permission_id if isinstance(user_object, LiteLLM_UserTable) else None
            ),
        )

    if project_object is not None:
        user_api_key_auth_obj.project_metadata = project_object.metadata
        user_api_key_auth_obj.project_alias = project_object.project_alias

    if end_user_id and key_end_user_budget_id is not None and prisma_client is not None:
        await _apply_key_end_user_default_budget_to_token(
            valid_token=user_api_key_auth_obj,
            end_user_object=end_user_object,
            key_end_user_budget_id=key_end_user_budget_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            keep_token_limits=user_custom_auth is not None,
        )
    end_user_model_max_budget: Final = await _end_user_model_max_budget(
        end_user_id=end_user_id,
        end_user_object=end_user_object,
        key_end_user_budget_id=key_end_user_budget_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
    )
    if end_user_id and (
        not user_api_key_auth_obj.authenticated_by_custom_auth
        or user_api_key_auth_obj.end_user_model_max_budget is None
    ):
        user_api_key_auth_obj.end_user_model_max_budget = (  # rebind-ok: the spend hook charges from this token
            dict(end_user_model_max_budget) if end_user_model_max_budget else None
        )

    skip_budget_checks: Final = _should_skip_budget_checks(
        request_data=request_data,
        route=route,
        request=request,
        llm_router=llm_router,
        team_id=user_api_key_auth_obj.team_id,
    )
    typed_team_object: Final[LiteLLM_TeamTable | None] = (
        team_object if isinstance(team_object, LiteLLM_TeamTable) else None
    )
    await _load_team_member_default_model_budget(
        user_api_key_auth_obj=user_api_key_auth_obj,
        team_object=typed_team_object,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        skip_budget_checks=skip_budget_checks or route_skips_budget_checks(route=route),
        proxy_logging_obj=proxy_logging_obj,
    )

    # Pin the metadata variable name (litellm_metadata vs metadata) before
    # any tag merge runs. Without this, header tags from
    # apply_client_tag_policy_pre_auth would land in `metadata` while the
    # later seed in common_checks pushes key tags and the
    # _tag_max_budget_check read into `litellm_metadata`, hiding header
    # tags from per-tag budget enforcement on LITELLM_METADATA_ROUTES.
    # Merge x-litellm-tags into request_data BEFORE common_checks runs.
    # _tag_max_budget_check inside common_checks only inspects request_data;
    # without this pre-merge, header-supplied tags bypass tag-budget
    # enforcement.
    for tag_target in LiteLLMProxyRequestSetup.pre_auth_tag_merge_targets(request, request_data):
        LiteLLMProxyRequestSetup.pre_seed_litellm_metadata_for_route(request_data=tag_target, route=route)
        LiteLLMProxyRequestSetup.apply_client_tag_policy_pre_auth(
            request=request,
            request_data=tag_target,
            user_api_key_dict=user_api_key_auth_obj,
        )

    bind_admission_counter_keys(user_api_key_auth_obj, end_user_id=end_user_id)
    bind_committed_admission_counters(user_api_key_auth_obj, end_user_id=end_user_id)
    try:
        _ = await common_checks(
            request=request,
            request_body=request_data,
            team_object=team_object,
            user_object=user_object,
            end_user_object=end_user_object,
            general_settings=general_settings,
            global_proxy_spend=global_proxy_spend,
            route=route,
            llm_router=llm_router,
            proxy_logging_obj=proxy_logging_obj,
            valid_token=user_api_key_auth_obj,
            skip_budget_checks=skip_budget_checks,
            project_object=project_object,
        )
        if not skip_budget_checks:
            models: Final = _get_model_names_for_budget_checks(
                model=_get_model_from_request_context(
                    request_data=request_data,
                    route=route,
                    request=request,
                    llm_router=llm_router,
                    team_id=user_api_key_auth_obj.team_id,
                )
            )
            await _check_team_model_budget(
                valid_token=user_api_key_auth_obj,
                model_max_budget_limiter=model_max_budget_limiter,
                models=models,
            )
            typed_project_object: Final[LiteLLM_ProjectTableCachedObj | None] = (
                project_object if isinstance(project_object, LiteLLM_ProjectTableCachedObj) else None
            )
            if not route_skips_budget_checks(route=route):
                await _check_team_member_model_budget_with_fallback(
                    valid_token=user_api_key_auth_obj,
                    model_max_budget_limiter=model_max_budget_limiter,
                    models=models,
                    request_data=request_data,
                    request=request,
                    llm_model_list=cast(  # cast-ok: router model lists are deployment sequences
                        Sequence[object] | None, llm_model_list
                    ),
                    llm_router=llm_router,
                    project_object=typed_project_object,
                    team_object=typed_team_object,
                    end_user_object=end_user_object if isinstance(end_user_object, LiteLLM_EndUserTable) else None,
                    route=route,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                )
                await _check_end_user_model_budget(
                    end_user_id=end_user_id,
                    end_user_model_max_budget=user_api_key_auth_obj.end_user_model_max_budget,
                    model_max_budget_limiter=model_max_budget_limiter,
                    team_id=user_api_key_auth_obj.team_id,
                    models=_get_model_names_for_budget_checks(
                        model=_get_model_from_request_context(
                            request_data=request_data,
                            route=route,
                            request=request,
                            llm_router=llm_router,
                            team_id=user_api_key_auth_obj.team_id,
                        )
                    ),
                )
            if sub_call and RouteChecks.is_llm_api_route(route=route):
                await virtual_key_max_budget_check(
                    valid_token=user_api_key_auth_obj,
                    proxy_logging_obj=proxy_logging_obj,
                    user_obj=user_object,
                )

        if committed_spend_tracked():
            await carry_committed_budget_spend(
                valid_token=user_api_key_auth_obj,
                team_object=typed_team_object,
                user_object=user_object,
                end_user_object=end_user_object,
                read_committed=(
                    None
                    if route_skips_budget_checks(route=route)
                    else partial(read_admission_committed_spend, spend_counter_store)
                ),
            )
        await _reserve_budget_after_common_checks(
            user_api_key_auth_obj=user_api_key_auth_obj,
            request=request,
            request_data=request_data,
            route=route,
            llm_router=llm_router,
            team_object=cast(  # cast-ok: preserve plain team-table objects through reservation
                LiteLLM_TeamTableCachedObj | None,
                team_object,
            ),
            user_object=user_object,
            end_user_id=end_user_id,
            end_user_object=end_user_object,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            skip_budget_checks=skip_budget_checks or sub_call,
            general_settings=general_settings,
        )
    finally:
        release_spend_counter_batch()


_run_centralized_common_checks: Final = run_centralized_common_checks


async def _noop_none() -> None:
    """Sentinel coroutine for asyncio.gather when a fetch is unnecessary
    (e.g. token has no team_id). Keeps the result tuple positional."""
    return


async def _apply_key_end_user_default_budget_to_token(
    valid_token: UserAPIKeyAuth,
    end_user_object: LiteLLM_EndUserTable | None,
    key_end_user_budget_id: str,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    parent_otel_span: Span | None,
    keep_token_limits: bool,
) -> None:
    """The builder's end-user pass runs before the key is resolved, so only here can the key's
    ``end_user_budget_id`` win over the proxy-wide default on the token that reservation reads.
    On the virtual-key path the token's end-user limits are the builder's proxy-wide defaults and
    the key budget replaces them wholesale. With ``keep_token_limits`` (custom auth) the token's
    limits are caps the custom auth callable set, so the key budget only fills the ones it left
    unset."""
    default_budget: Final = (
        end_user_object.litellm_budget_table
        if end_user_object is not None
        else await resolve_default_end_user_budget(
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            key_end_user_budget_id=key_end_user_budget_id,
            parent_otel_span=parent_otel_span,
        )
    )
    if default_budget is None:
        return

    if not keep_token_limits or valid_token.end_user_max_budget is None:
        valid_token.end_user_max_budget = default_budget.max_budget
    if not keep_token_limits or valid_token.end_user_tpm_limit is None:
        valid_token.end_user_tpm_limit = default_budget.tpm_limit
    if not keep_token_limits or valid_token.end_user_rpm_limit is None:
        valid_token.end_user_rpm_limit = default_budget.rpm_limit
    if not keep_token_limits or valid_token.end_user_tpd_limit is None:
        valid_token.end_user_tpd_limit = default_budget.tpd_limit
    if not keep_token_limits or valid_token.end_user_model_max_budget is None:
        valid_token.end_user_model_max_budget = default_budget.model_max_budget
    if not keep_token_limits or valid_token.end_user_model_tpm_limit is None:
        valid_token.end_user_model_tpm_limit = default_budget.model_tpm_limit  # rebind-ok: in-place like siblings
    if not keep_token_limits or valid_token.end_user_model_rpm_limit is None:
        valid_token.end_user_model_rpm_limit = default_budget.model_rpm_limit  # rebind-ok: in-place like siblings


async def _reserve_budget_after_common_checks(
    user_api_key_auth_obj: UserAPIKeyAuth,
    request_data: dict,
    route: str,
    llm_router: litellm.Router | None,
    team_object: LiteLLM_TeamTableCachedObj | None,
    user_object: LiteLLM_UserTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    skip_budget_checks: bool,
    general_settings: dict,
    end_user_id: str | None = None,
    end_user_object: LiteLLM_EndUserTable | None = None,
    request: Request | None = None,
) -> None:
    user_api_key_auth_obj.budget_reservation = None
    if not skip_budget_checks and general_settings.get("disable_budget_reservation") is not True:
        from litellm.proxy.spend_tracking.budget_reservation import (
            reserve_budget_for_request,
        )

        user_api_key_auth_obj.budget_reservation = await reserve_budget_for_request(
            request_body=request_data,
            route=route,
            llm_router=llm_router,
            valid_token=user_api_key_auth_obj,
            team_object=team_object,
            user_object=user_object,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            end_user_id=end_user_id,
            end_user_object=end_user_object,
            apply_user_budget_to_team_keys=general_settings.get("apply_user_budget_to_team_keys") is True,
            fail_closed_budget_enforcement=(
                general_settings.get("fail_closed_budget_enforcement") is True
                or user_api_key_auth_obj.billing_agent_policy is not None
            ),
            raw_body=await read_raw_json_body(request=request),
        )
    if request is not None:
        reservation: Final = user_api_key_auth_obj.budget_reservation
        request.state.budget_reservation = reservation  # rebind-ok: read by the release middleware


def _should_skip_budget_checks(
    request_data: dict[str, object],
    route: str,
    request: Request | None,
    llm_router: litellm.Router | None,
    team_id: str | None = None,
) -> bool:
    model: Final = _get_model_from_request_context(
        request_data=request_data,
        route=route,
        request=request,
        llm_router=llm_router,
        team_id=team_id,
    )
    return request_skips_budget_checks(route=route, model=model, llm_router=llm_router, request_body=request_data)


def _resolve_request_principal(request: Request, valid_token: UserAPIKeyAuth) -> Principal:
    """Project the resolved identity into one per-request Principal, off the key
    object the builder already fetched, and stamp the request network context
    onto it once. X-Forwarded-For is only trusted when the operator configured
    ``trusted_proxy_ranges``; otherwise the direct peer is authoritative.

    credential_ref and a stable subject fallback are always set off the token so
    the Principal can never be anonymous, even for a keyless service-account key
    with no user or alias."""
    cidrs: Final = get_trusted_proxy_cidrs()
    network: Final = resolve_network_context(
        request,
        TrustedProxyConfig(use_forwarded_for=bool(cidrs), trusted_proxy_cidrs=cidrs),
    )
    auth_method: Final = AuthMethod.BEARER_JWT if valid_token.jwt_claims else AuthMethod.API_KEY
    return IdentityStore.principal_from_key(
        valid_token,
        auth_method=auth_method,
        network=network,
        subject_fallback=valid_token.token,
        credential_ref=CredentialRef(token_id=valid_token.token),
    )


async def _authorize_authenticated_request(
    user_api_key_auth_obj: UserAPIKeyAuth,
    request: Request,
    request_data: dict[str, object],
    route: str,
    api_key: str,
    *,
    force_virtual_key_checks: bool = False,
) -> UserAPIKeyAuth | None:
    """Authorize an already-authenticated request: disabled-route check, the single
    ``common_checks`` gate (which also reserves budget), and end-user fallback
    resolution. Returns the auth object the exception handler recovered when a check
    failed but the request may proceed anyway, else ``None``.
    """
    ## ENSURE DISABLE ROUTE WORKS ACROSS ALL USER AUTH FLOWS ##
    RouteChecks.should_call_route(route=route, valid_token=user_api_key_auth_obj, request=request)
    await _normalize_claude_model(request_data, user_api_key_auth_obj, request, route)
    await _resolve_router_settings_model_group_alias(request_data, user_api_key_auth_obj, request, route)
    _resolve_codex_native_chatgpt_model(request_data, user_api_key_auth_obj, request, route)

    # Single authorization point. Builder paths MUST NOT call common_checks.
    # Route through the same exception handler the builder uses so
    # authorization failures (ProxyException, or plain Exception from
    # admin-only-route / model-access / budget checks) surface as
    # ProxyException consistently with pre-refactor behavior.
    try:
        from litellm.proxy.agent_endpoints.auth.managed_authorization import (
            admit_managed_actor,
            invocation_target,
            managed_agent_route_allowed,
            managed_inference_request,
            prepare_agent_invocation,
        )
        from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
        from litellm.proxy.proxy_server import (
            general_settings,
            llm_model_list,  # pyright: ignore[reportUnknownVariableType]  # proxy global has untyped deployment entries
            llm_router,
            prisma_client,
            user_model,
        )

        grants: Final = await access_group_ids_for_identity(user_api_key_auth_obj)
        user_api_key_auth_obj.apply_access_group_rule_grants(grants)
        key_check_required: Final = (
            getattr(request.state, "requires_key_model_check", False) or force_virtual_key_checks
        )
        if key_check_required or grants:
            model_identity: Final = (
                user_api_key_auth_obj
                if key_check_required
                else user_api_key_auth_obj.model_copy(update={"models": list(user_api_key_auth_obj.team_models or ())})
            )
            await enforce_key_and_fallback_model_access(
                valid_token=model_identity,
                request_data=request_data,
                route=route,
                request=request,
                llm_model_list=llm_model_list,
                llm_router=llm_router,
            )

        store: Final = AgentIdentityStore.from_client(prisma_client) if prisma_client is not None else None
        if user_api_key_auth_obj.agent_id is not None:
            await admit_managed_actor(user_api_key_auth_obj, store)
        if user_api_key_auth_obj.managed_agent_policy is not None and not managed_agent_route_allowed(
            route, request.method
        ):
            raise HTTPException(403, "Agent identities can only access inference and agent discovery routes")
        authorized_data: Final = (
            managed_inference_request(
                route,
                request_data,
                general_settings,
                user_model,
                request.path_params.get("model") or request.path_params.get("model_name"),
                request.query_params.get("model"),
            )
            if user_api_key_auth_obj.managed_agent_policy is not None
            else request_data
        )
        target_name: Final = invocation_target(route, authorized_data)
        if target_name is not None:
            await prepare_agent_invocation(
                user_api_key_auth_obj,
                target_name,
                store,
                billable=request_data.get("method")
                in (None, "message/send", "message/stream", "SendMessage", "SendStreamingMessage"),
            )
        if "request_tag_mappings" in general_settings and request.method == "POST":
            from litellm.proxy.common_utils.request_tag_mappings import is_baggage_mapped_route
            from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

            if is_baggage_mapped_route(route):
                LiteLLMProxyRequestSetup.pre_seed_litellm_metadata_for_route(request_data=authorized_data, route=route)
                await LiteLLMProxyRequestSetup.apply_baggage_tag_mappings_pre_auth(
                    request=request,
                    request_data=authorized_data,
                    general_settings=_GENERAL_SETTINGS_ADAPTER.validate_python(general_settings),
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                )
        await run_centralized_common_checks(
            user_api_key_auth_obj=user_api_key_auth_obj,
            request=request,
            request_data=authorized_data,
            route=route,
            force_virtual_key_checks=force_virtual_key_checks,
        )
    except Exception as e:
        return await UserAPIKeyAuthExceptionHandler.handle_authentication_error(
            e=e,
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=user_api_key_auth_obj.parent_otel_span,
            api_key=api_key,
            resolved_identity=user_api_key_auth_obj,
        )

    # Defense-in-depth: ``_user_api_key_auth_builder`` has multiple early-return
    # paths (no master key, /user/auth route, JWT short-circuits) that bypass
    # the end-user resolution block. If those paths produced an auth obj
    # without an ``end_user_id`` set, fall back to extracting from the request
    # body so spend logs are still attributed correctly. Validation honours
    # ``litellm.validate_end_user_id_in_db``.
    if user_api_key_auth_obj.end_user_id is None:
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        raw_end_user_id: Final = get_end_user_id_from_request_body(request_data, safe_get_request_headers(request))
        if raw_end_user_id is not None:
            resolved_end_user_id: Final = await resolve_and_validate_end_user_id(
                raw_end_user_id=raw_end_user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth_obj.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
                route=route,
                key_end_user_budget_id=get_key_end_user_budget_id(user_api_key_auth_obj.metadata),
            )
            if resolved_end_user_id is not None:
                user_api_key_auth_obj.end_user_id = resolved_end_user_id
    return None


def _spend_counter_store() -> SpendCounterStore:
    from litellm.proxy.proxy_server import spend_counter_store

    return spend_counter_store


def _identity_cache_keys(api_key: str, *, end_user_id: str | None, key_is_resolved: bool) -> tuple[str, ...]:
    """Cache keys auth reads before it knows the key's owners, all known from the request alone. A key object is
    cached under the hash of the bearer, so the bearer itself never reaches Redis."""
    return tuple(
        key
        for key in (
            None if key_is_resolved else hash_token(api_key),
            None if not end_user_id else end_user_cache_key(end_user_id),
            None if not end_user_id else end_user_restricted_registry_cache_key(),
            model_access_group_registry_cache_key(),
        )
        if key is not None
    )


async def _prefetch_referenced_auth_objects(
    valid_token: UserAPIKeyAuth,
    end_user_id: str | None,
    user_api_key_cache: UserApiKeyCache,
    prisma_client: PrismaClient | None,
) -> None:
    """Warm every object and spend counter the checks below will read, in one MGET each (one DB query when cold)."""
    bind_admission_counter_keys(valid_token, end_user_id=end_user_id or None)
    bind_committed_admission_counters(valid_token, end_user_id=end_user_id or None)
    await prefetch_auth_objects(
        refs=AuthObjectRefs.from_token(valid_token),
        user_api_key_cache=user_api_key_cache,
        prisma_client=prisma_client,
        before_read=arm_routing_cooldowns,
    )


def arm_routing_cooldowns() -> None:
    """Declares the cooldown read routing takes later into the auth prefetch's native round trip, so a request whose
    auth objects come from Redis pays one round trip for both. Armed only when a prefetch reads Redis, so a request
    served from memory, or denied, reads no cooldowns early."""
    from litellm.proxy.proxy_server import llm_router  # noqa: PLC0415  # runtime global

    if llm_router is not None:
        RoutingContext(llm_router).arm_cooldown_prefetch()


def _seed_request_destinations(user_api_key_dict: UserAPIKeyAuth, request: Request | None = None) -> None:
    """Anchor the OTLP destinations this key or team overrides its traces to.

    Called inside the ``auth`` phase span so that span reaches the tenant's account
    as well, and on the request task so the ``ContextVar`` is inherited by the logging
    tasks that close the LLM span. Best-effort: trace routing must never fail auth.

    ``request`` carries the headers, so a backend this request disabled with
    ``x-litellm-disable-callbacks`` resolves to no destination.

    Only destinations the published fan-out can build are anchored. Anchoring one is
    what tells the operator's exporter to hold that backend's spans back under
    ``override``, so an unbuildable one would leave the span with nowhere to go.

    The ``postgres`` spans under ``auth`` close before this runs, because they are the
    reads that resolve the identity being read here. They never reach the tenant's
    account, and they are never withheld from the operator's backend, whichever mode
    is set.
    """
    try:
        from litellm.integrations.otel.logger import fan_out_provider
        from litellm.integrations.otel.plumbing.context import set_request_destinations
        from litellm.integrations.otel.plumbing.providers import deliverable_destinations
        from litellm.proxy.litellm_pre_call_utils import (
            resolve_tenant_otel_destinations,
        )

        set_request_destinations(
            deliverable_destinations(
                resolve_tenant_otel_destinations(user_api_key_dict, safe_get_request_headers(request)),
                fan_out_provider(),
            )
        )
    except Exception as exc:  # noqa: BLE001  # telemetry routing is best-effort and must never break authentication
        verbose_proxy_logger.debug("OTel V2: tenant destination resolution failed: %s", exc)


@tracer.wrap()
async def user_api_key_auth(
    request: Request,
    api_key: str = fastapi.Security(api_key_header),
    azure_api_key_header: str | None = fastapi.Security(azure_api_key_header),
    anthropic_api_key_header: str | None = fastapi.Security(anthropic_api_key_header),
    google_ai_studio_api_key_header: str | None = fastapi.Security(google_ai_studio_api_key_header),
    azure_apim_header: str | None = fastapi.Security(azure_apim_header),
    custom_litellm_key_header: str | None = fastapi.Security(custom_litellm_key_header),
) -> UserAPIKeyAuth:
    """
    Parent function to authenticate user api key / jwt token.
    """

    # Create the SERVER span and stash it on request.state BEFORE reading the
    # body. _read_request_body can raise ProxyException for malformed JSON;
    # without this, that path leaves no span for the exception handler to
    # close, and the trace never reaches the backend.
    _ensure_parent_otel_span_on_request_state(request)

    request_data, body_parse_exception = await _read_request_body_deferring_parse_failure(request=request)
    phase_event("litellm.request.body_parsed")
    route: Final[str] = get_request_route(request=request)
    ## CHECK IF ROUTE IS ALLOWED

    # Run the whole auth phase inside a live ``auth`` span so the DB lookups it
    # triggers (key/user/team object reads) nest under it instead of flattening
    # onto the server span, and name every cache read in it an auth-object read.
    # No-op when OTel V2 isn't active.
    with (
        phase_span(f"auth {route}"),
        service_target(AUTH_OBJECTS_TARGET),
        spend_counter_batch_scope(_spend_counter_store()),
    ):
        try:
            authenticated: Final = await user_api_key_auth_builder(
                request=request,
                api_key=api_key,
                azure_api_key_header=azure_api_key_header,
                anthropic_api_key_header=anthropic_api_key_header,
                google_ai_studio_api_key_header=google_ai_studio_api_key_header,
                azure_apim_header=azure_apim_header,
                request_data=request_data,
                custom_litellm_key_header=custom_litellm_key_header,
            )
        except Exception:
            # The body was read first, so a caller who sent both a malformed body and
            # a rejected key used to get the 400; the response is unchanged, and the
            # auth failure is still recorded on the trace by the handler that ran.
            if body_parse_exception is not None:
                raise body_parse_exception
            raise
        from litellm.proxy.proxy_server import (
            general_settings,  # pyright: ignore[reportUnknownVariableType]  # proxy global is validated below
        )

        settings: Final = _GENERAL_SETTINGS_ADAPTER.validate_python(general_settings)
        user_api_key_auth_obj: Final = (
            authenticated.model_copy() if settings.get("access_group_rules") else authenticated
        )
        user_api_key_auth_obj.budget_reservation = None
        user_api_key_auth_obj.agent_caller = agent_caller_from_headers(
            safe_get_request_headers(request), user_api_key_auth_obj
        )
        _seed_request_destinations(user_api_key_auth_obj, request)

        # A body that never parsed is authenticated (so the trace carries identity
        # and this ``auth`` span) but not authorized: there is no model to check it
        # against, and budget reservation would increment live spend counters that
        # only the endpoint's post-call path releases; the endpoint never runs, since
        # the parse failure is raised below.
        if body_parse_exception is None:
            recovered_auth_obj: Final = await _authorize_authenticated_request(
                user_api_key_auth_obj=user_api_key_auth_obj,
                request=request,
                request_data=request_data,
                route=route,
                api_key=api_key,
            )
            if recovered_auth_obj is not None:
                return recovered_auth_obj

    # Identity is now resolved. Seed it AFTER the auth span closes so the Baggage
    # persists on the request task (detaching the span's context token inside the
    # ``with`` would unwind a Baggage attach made within it) and every post-auth
    # span — pre-call, LLM call, guardrail, spend write — inherits team/key/user.
    seed_request_identity(
        user_api_key_auth_obj,
        model=request_data.get("model") if isinstance(request_data, dict) else None,
    )
    user_api_key_auth_obj.request_route = normalize_request_route(route)

    if body_parse_exception is not None:
        await _record_unparsable_body_failure(
            user_api_key_dict=user_api_key_auth_obj,
            body_parse_exception=body_parse_exception,
            route=route,
        )
        raise body_parse_exception

    # Resolve caller identity once, here at the seam, into a single per-request
    # Principal projected off the key object the builder already fetched (no
    # second lookup). Downstream consumers read identity off this instead of
    # re-resolving it. Additive and defensive: a projection failure must never
    # reject an already-authenticated request, so it is left unset on failure;
    # any future consumer must treat a missing principal as deny, not allow.
    try:
        request.state.principal = _resolve_request_principal(request, user_api_key_auth_obj)
    except Exception as e:
        verbose_proxy_logger.warning("Principal projection at auth seam failed (non-fatal): %s", e)

    return user_api_key_auth_obj


async def _return_user_api_key_auth_obj(
    user_obj: LiteLLM_UserTable | None,
    api_key: str,
    parent_otel_span: Span | None,
    valid_token_dict: dict,
    route: str,
    start_time: datetime,
    user_role: LitellmUserRoles | None = None,
) -> UserAPIKeyAuth:
    end_time: Final = datetime.now(timezone.utc)

    asyncio.create_task(
        user_api_key_service_logger_obj.async_service_success_hook(
            service=ServiceTypes.AUTH,
            call_type=route,
            start_time=start_time,
            end_time=end_time,
            duration=end_time.timestamp() - start_time.timestamp(),
            parent_otel_span=parent_otel_span,
        )
    )

    retrieved_user_role: Final = user_role or get_user_role(user_obj=user_obj) or LitellmUserRoles.INTERNAL_USER

    user_api_key_kwargs: Final = {
        "api_key": api_key,
        "parent_otel_span": parent_otel_span,
        "user_role": retrieved_user_role,
        **valid_token_dict,
    }
    if user_obj is not None:
        user_api_key_kwargs.update(
            user_tpm_limit=user_obj.tpm_limit,
            user_rpm_limit=user_obj.rpm_limit,
            user_email=user_obj.user_email,
            user_spend=getattr(user_obj, "spend", None),
            user_max_budget=getattr(user_obj, "max_budget", None),
            user_model_max_budget=getattr(user_obj, "model_max_budget", None),
        )
    if user_obj is not None and is_user_proxy_admin(user_obj=user_obj):
        user_api_key_kwargs.update(
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        return UserAPIKeyAuth.model_validate(user_api_key_kwargs)
    else:
        return UserAPIKeyAuth.model_validate(user_api_key_kwargs)


def _get_temp_budget_increase(valid_token: UserAPIKeyAuth):
    valid_token_metadata: Final = valid_token.metadata
    if "temp_budget_increase" in valid_token_metadata and "temp_budget_expiry" in valid_token_metadata:
        expiry = datetime.fromisoformat(valid_token_metadata["temp_budget_expiry"])
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry > datetime.now(timezone.utc):
            return valid_token_metadata["temp_budget_increase"]
    return None


def _update_key_budget_with_temp_budget_increase(
    valid_token: UserAPIKeyAuth,
) -> UserAPIKeyAuth:
    if valid_token.max_budget is None:
        return valid_token
    temp_budget_increase: Final = _get_temp_budget_increase(valid_token)
    if not temp_budget_increase:
        return valid_token
    return valid_token.model_copy(update={"max_budget": valid_token.max_budget + temp_budget_increase})


async def _lookup_end_user_and_apply_budget(
    valid_token: UserAPIKeyAuth,
    route: str,
    parent_otel_span: Span | None,
    prisma_client,
    user_api_key_cache,
    proxy_logging_obj,
):
    """Look up end_user from DB and apply budget limits to valid_token."""
    end_user_object = None
    key_end_user_budget_id: Final = get_key_end_user_budget_id(valid_token.metadata)
    try:
        end_user_object = await get_end_user_object(
            end_user_id=valid_token.end_user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            route=route,
            token_end_user_max_budget=valid_token.end_user_max_budget,
            key_end_user_budget_id=key_end_user_budget_id,
        )
        if end_user_object is not None:
            end_user_params = {
                "end_user_id": valid_token.end_user_id,
                "allowed_model_region": end_user_object.allowed_model_region,
            }
            if end_user_object.litellm_budget_table is not None:
                _apply_budget_limits_to_end_user_params(
                    end_user_params=end_user_params,
                    budget_info=end_user_object.litellm_budget_table,
                    end_user_id=valid_token.end_user_id or "",
                )
            valid_token = update_valid_token_with_end_user_params(
                valid_token=valid_token, end_user_params=end_user_params
            )
        elif key_end_user_budget_id is not None or litellm.max_end_user_budget_id is not None:
            default_budget: Final = await resolve_default_end_user_budget(
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                key_end_user_budget_id=key_end_user_budget_id,
                parent_otel_span=parent_otel_span,
            )
            if default_budget is not None:
                end_user_params = {"end_user_id": valid_token.end_user_id}
                _apply_budget_limits_to_end_user_params(
                    end_user_params=end_user_params,
                    budget_info=default_budget,
                    end_user_id=valid_token.end_user_id or "",
                )
                valid_token = update_valid_token_with_end_user_params(
                    valid_token=valid_token, end_user_params=end_user_params
                )
                if valid_token.end_user_max_budget is None:
                    valid_token.end_user_max_budget = default_budget.max_budget
    except Exception as e:
        if isinstance(e, litellm.BudgetExceededError):
            raise e
        verbose_proxy_logger.debug("Unable to find user in db. Error - %s", e)
    return valid_token, end_user_object


async def enforce_key_and_fallback_model_access(
    *,
    valid_token: UserAPIKeyAuth,
    request_data: dict,
    route: str,
    request: Request | None,
    llm_model_list: list | None,
    llm_router: litellm.Router | None,
) -> None:
    """
    Key-level model allowlist and client fallbacks (same as standard auth).
    Not included in common_checks — common_checks enforces team/user/project model access only.
    """
    await _normalize_claude_model(request_data, valid_token, request, route)
    await _resolve_router_settings_model_group_alias(request_data, valid_token, request, route)
    _resolve_codex_native_chatgpt_model(request_data, valid_token, request, route)
    config: Final = valid_token.config

    if config != {}:
        model_list: Final = config.get("model_list", [])
        new_model_list: Final = model_list
        verbose_proxy_logger.debug("\n new llm router model list %s", new_model_list)
    elif isinstance(valid_token.models, list) and "all-team-models" in valid_token.models:
        pass
    else:
        model: Final = _get_model_from_request_context(
            request_data=request_data,
            route=route,
            request=request,
            llm_router=llm_router,
            team_id=valid_token.team_id,
        )

        if model is not None:
            await can_key_call_model(
                model=model,
                llm_model_list=llm_model_list,
                valid_token=valid_token,
                llm_router=llm_router,
            )

        fallback_names: Final = tuple(
            name
            for target in iter_request_fallback_targets(request_data)
            if (name := fallback_target_model_name(target)) is not None
        )

        for _name in dict.fromkeys(fallback_names):  # dedupe, preserve order
            await can_key_call_model(
                model=_name,
                llm_model_list=llm_model_list,
                valid_token=valid_token,
                llm_router=llm_router,
            )
            await is_valid_fallback_model(
                model=_name,
                llm_router=llm_router,
                user_model=None,
            )


_enforce_key_and_fallback_model_access: Final = enforce_key_and_fallback_model_access  # pyright: ignore[reportUnknownVariableType]  # existing tests use this alias


async def _run_post_custom_auth_checks(
    valid_token: UserAPIKeyAuth,
    request: Request,
    request_data: dict[str, object],
    route: str,
    parent_otel_span: Span | None,
) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_model_list,
        llm_router,
        model_max_budget_limiter,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    # 1. Look up end_user object from DB if end_user_id is set
    end_user_object = None
    if valid_token.end_user_id is not None:
        valid_token, end_user_object = await _lookup_end_user_and_apply_budget(
            valid_token=valid_token,
            route=route,
            parent_otel_span=parent_otel_span,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        # common_checks() enforces the end-user budget, but the centralized
        # gate skips it for custom-auth deployments unless
        # custom_auth_run_common_checks is set. Enforce it here on that path
        # so an over-budget end user can't keep making requests.
        if end_user_object is not None and not general_settings.get("custom_auth_run_common_checks", False):
            await check_end_user_budget(end_user_obj=end_user_object, route=route)

    # 2. Check token expiry
    if valid_token.expires is not None:
        current_time: Final = datetime.now(timezone.utc)
        if isinstance(valid_token.expires, datetime):
            expiry_time = valid_token.expires
        else:
            expiry_time = datetime.fromisoformat(valid_token.expires)
        if expiry_time.tzinfo is None or expiry_time.tzinfo.utcoffset(expiry_time) is None:
            expiry_time = expiry_time.replace(tzinfo=timezone.utc)
        if expiry_time < current_time:
            raise ProxyException(
                message=f"Authentication Error - Expired Key. Key Expiry time {expiry_time} and current time {current_time}",
                type=ProxyErrorTypes.expired_key,
                code=status.HTTP_401_UNAUTHORIZED,
                param=(abbreviate_api_key(api_key=valid_token.token) if valid_token.token else ""),
            )

    if general_settings.get("custom_auth_run_common_checks", False):
        await _normalize_model_for_auth(request_data, valid_token, request, route)
        request.state.requires_key_model_check = True  # rebind-ok: defer until central grants

    current_model: Final = _get_model_from_request_context(
        request_data=request_data,
        route=route,
        request=request,
        llm_router=llm_router,
        team_id=valid_token.team_id,
    )

    # A zero-cost model cannot move any counter, so refusing it means refusing on
    # spend some other model accrued. The JWT and virtual-key paths already skip
    # every budget check for these; this path did not, so the same request could
    # be refused under custom auth and served under the other two.
    skip_budget_checks: Final = (
        is_model_cost_zero(model=current_model, llm_router=llm_router)
        if current_model is not None and llm_router is not None
        else False
    )

    # Custom auth builds its own token, so unlike the main path nothing has
    # loaded the user row yet. The attach is unconditional because the post-call
    # spend hook reads this field off the token: gating it on the same condition
    # as enforcement would leave the user's counter uncharged whenever this
    # request was not itself enforceable, so its spend would go untracked.
    user_budget: Final = await _read_user_model_max_budget(
        user_id=valid_token.user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    valid_token.user_model_max_budget = user_budget  # rebind-ok: the spend hook reads it off this token
    if not skip_budget_checks:
        await _enforce_model_budgets(
            _AUTH_CORE.model_budgets().check_key_and_user,
            _ModelBudgetHost(
                request=request,
                request_data=request_data,
                valid_token=valid_token,
                route=route,
                limiter=model_max_budget_limiter,
                llm_router=llm_router,
                llm_model_list=cast(  # cast-ok: router model lists are deployment sequences
                    Sequence[object] | None, llm_model_list
                ),
            ),
        )

    # Read after the budget checks, which may have rerouted the request to a fallback model.
    current_models: Final = _get_model_names_for_budget_checks(
        model=_get_model_from_request_context(
            request_data=request_data,
            route=route,
            request=request,
            llm_router=llm_router,
            team_id=valid_token.team_id,
        )
    )

    end_user_mmb: Final = valid_token.end_user_model_max_budget
    if (
        not skip_budget_checks
        and end_user_mmb is not None
        and isinstance(end_user_mmb, dict)
        and len(end_user_mmb) > 0
        and current_models
        and valid_token.end_user_id is not None
    ):
        for model_name in current_models:
            await model_max_budget_limiter.is_end_user_within_model_budget(
                end_user_id=valid_token.end_user_id,
                end_user_model_max_budget=end_user_mmb,
                model=model_name,
                team_id=valid_token.team_id,
            )

    # team / user / end_user / project context objects are fetched by
    # the centralized common_checks gate in user_api_key_auth after
    # this helper returns. Keep only the project fetch here because it
    # mutates the token (project_metadata / project_alias).
    if valid_token.project_id is not None:
        _project_obj: Final = await get_project_object(
            project_id=valid_token.project_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if _project_obj is not None:
            valid_token.project_metadata = _project_obj.metadata
            valid_token.project_alias = _project_obj.project_alias

    return valid_token


async def _authorize_resolved_key(
    identity: UserAPIKeyAuth,
    key_hash: str,
    request: Request,
    request_data: dict[str, object],  # mutable-ok: model access checks rewrite the request model in place
    route: str,
) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import (
        litellm_proxy_admin_name,
        master_key,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    start_time: Final = datetime.now(timezone.utc)
    host: Final = _KeyHost(
        request=request,
        request_data=request_data,
        token=key_hash,
        preresolved_key=identity,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
        route=route,
    )
    outcome: Final = await _AUTH_CORE.keys().check(
        host,
        RequestFacts(route=route, token=key_hash, is_llm_api_route=RouteChecks.is_llm_api_route(route=route)),
        _core_settings(master_key=master_key, allow_user_auth=False, prisma_client=prisma_client),
        start_time,
    )
    return await _authenticated(
        outcome,
        host,
        admin_user_id=litellm_proxy_admin_name,
        route=route,
        start_time=start_time,
        jwt_auth_enabled=False,
    )


async def authorize_internal_virtual_key(
    key_hash: str, request: Request, request_data: dict[str, object]
) -> UserAPIKeyAuth:
    """Authorize a server-owned job against its persisted virtual-key assignment, never a client-supplied bearer hash."""
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    identity: Final = IdentityStore.key_from_principal(
        await IdentityStore(prisma_client, user_api_key_cache, proxy_logging_obj=proxy_logging_obj).resolve(
            hashed_token=key_hash
        )
    )
    route: Final = get_request_route(request=request)
    await pre_db_read_auth_checks(request_data=request_data, request=request, route=route)
    auth: Final = await _authorize_resolved_key(
        identity=identity, key_hash=key_hash, request=request, request_data=request_data, route=route
    )
    auth.budget_reservation = None
    recovered: Final = await _authorize_authenticated_request(
        user_api_key_auth_obj=auth,
        request=request,
        request_data=request_data,
        route=route,
        api_key=key_hash,
        force_virtual_key_checks=True,
    )
    if recovered is not None:
        return recovered
    _seed_request_destinations(auth, request)
    auth.request_route = route
    request.state.principal = _resolve_request_principal(request, auth)
    return auth
