"""
This file handles authentication for the LiteLLM Proxy.

it checks if the user passed a valid API Key to the LiteLLM Proxy

Returns a UserAPIKeyAuth object if the API key is valid

"""

import asyncio
import base64
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial, reduce
from typing import TYPE_CHECKING, Final, Protocol, cast

import anyio
import fastapi
from fastapi import HTTPException, Request, WebSocket, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import TypeAdapter, ValidationError
from starlette.exceptions import WebSocketException
from starlette.requests import ClientDisconnect

import litellm
from litellm._internal_context import service_target, set_request_parent_otel_span
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm._service_logger import ServiceLogging
from litellm.constants import (
    CLIENT_REQUESTED_MODEL_SCOPE_KEY,
    GLOBAL_PROXY_SPEND_CACHE_KEY,
    LITELLM_PROXY_BUDGET_NAME,
    MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY,
)
from litellm.integrations.otel.model.config import is_otel_v2_enabled
from litellm.integrations.otel.runtime import phase_event, phase_span, seed_request_identity
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.team import LiteLLM_TeamRebuiltFromToken
from litellm.proxy._types import *
from litellm.proxy.agent_endpoints.auth.agent_caller import agent_caller_from_headers
from litellm.proxy.auth import core
from litellm.proxy.auth import denial as auth_denial
from litellm.proxy.auth.access_group_rules import access_group_ids_for_identity
from litellm.proxy.auth.auth_checks import (  # noqa: F401  # legacy module exports
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
    cache_key_object,  # pyright: ignore[reportUnusedImport]  # backwards-compatible package export
    can_customer_access_model,
    can_key_call_model,
    can_object_call_model,
    can_project_access_model,
    can_team_access_model,
    check_end_user_budget,
    common_checks,
    delete_cache_key_object,  # pyright: ignore[reportUnusedImport]  # backwards-compatible package export
    get_end_user_object,
    get_key_end_user_budget_id,
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
    is_video_retrieval_request,
    key_access_group_grants_model,
    key_model_aliases_for_auth_check,
    model_max_budget_of,
    request_skips_budget_checks,
    resolve_and_validate_end_user_id,
    resolve_default_end_user_budget,
    route_skips_budget_checks,
    stamp_matched_model_access_groups,
    team_member_max_budget_alert_check,  # pyright: ignore[reportUnusedImport]  # backwards-compatible package export
    team_model_aliases_for_auth_check,
    virtual_key_max_budget_alert_check,  # pyright: ignore[reportUnusedImport]  # backwards-compatible package export
    virtual_key_max_budget_check,
    virtual_key_soft_budget_check,  # pyright: ignore[reportUnusedImport]  # backwards-compatible package export
)
from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler
from litellm.proxy.auth.auth_method import AuthMethod
from litellm.proxy.auth.auth_object_prefetch import (
    AUTH_OBJECTS_TARGET,
    AuthObjectRefs,
    prefetch_auth_objects,
)
from litellm.proxy.auth.auth_utils import (
    abbreviate_api_key,
    checks_team_s3_buckets_in_handler,
    fallback_target_model_name,
    get_end_user_id_from_request_body,
    get_model_from_request,
    get_request_route,
    get_request_route_template,
    is_invalid_virtual_key_error,
    is_request_body_safe,
    iter_request_fallback_targets,
    log_model_access_denial,
    normalize_request_route,
    pre_db_read_auth_checks,
    request_dispatched_to_pass_through_endpoint,
    request_dispatched_to_provider_pass_through,
    route_in_additonal_public_routes,
)
from litellm.proxy.auth.host import identity as auth_identity
from litellm.proxy.auth.network import TrustedProxyConfig, resolve_network_context
from litellm.proxy.auth.oauth2_check import Oauth2Handler
from litellm.proxy.auth.oauth2_proxy_hook import handle_oauth2_proxy_request
from litellm.proxy.auth.resolvers import CredentialRef, Principal
from litellm.proxy.auth.resolvers.store import IdentityStore
from litellm.proxy.auth.route_checks import RouteChecks
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
    read_request_path_params,
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
)
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.litellm_pre_call_utils import (
    LiteLLMProxyRequestSetup,
    pre_call_served_model,
    request_served_model,
)
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
)
from litellm.repositories.user_repository import UserRepository
from litellm.router_api import (
    RoutingContext,
    fallbacks_disabled,
    references_provider_scoped_resource,
    request_fallback_chain,
)
from litellm.rust_bridge._native import RustAuth, RustDenial, RustPlan
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    from litellm.proxy.auth.host.python import PythonAuthHost
    from litellm.rust_bridge._native import RustAuthOutcome, RustAuthSettings

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


def get_model_from_request_context(
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


async def _normalize_claude_model(
    request_data: dict, valid_token: UserAPIKeyAuth, request: Request | None, route: str
) -> None:
    from litellm.proxy.proxy_server import llm_router, prisma_client, proxy_config, proxy_logging_obj

    if route not in _CLAUDE_MODEL_ROUTES or llm_router is None:
        return
    if request is not None and request.scope.get(_CLAUDE_MODEL_NORMALIZED) is True:
        return
    requested: Final = get_model_from_request_context(request_data, route, request, llm_router, valid_token.team_id)
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


_SERVED_MODEL_BUDGET_WAIVER_ROUTES: Final = frozenset(
    {
        "/chat/completions",
        "/v1/chat/completions",
        "/completions",
        "/v1/completions",
        "/responses",
        "/v1/responses",
        "/messages",
        "/v1/messages",
    }
)


def _inference_model_overridden() -> bool:
    from litellm.proxy import proxy_server

    cli_model: Final[object] = proxy_server.user_model  # pyright: ignore[reportUnknownMemberType]  # legacy untyped CLI global
    completion_model: Final = proxy_server.general_settings_view().get("completion_model")
    return (isinstance(completion_model, str) and completion_model != "") or (
        isinstance(cli_model, str) and cli_model != ""
    )


async def budget_served_model(
    model: str | Sequence[str] | None,
    request_data: Mapping[str, object],
    request: Request | None,
    llm_router: litellm.Router | None,
    valid_token: UserAPIKeyAuth,
) -> str | Sequence[str] | None:
    """The model the zero-cost budget waiver judges. For a client's own HTTP request to a chat, completions,
    responses or messages route that names a string body `model`, it is the model request preparation serves it
    with through the caller's team, key, global and router aliases. Everywhere else it stays the requested model:
    requests built inside the proxy, a path or query `model`, a configured `completion_model` or CLI `--model`,
    and requests whose body auth has not yet passed through the router aliases."""
    if request is None or llm_router is None or not isinstance(model, str):
        return model
    if get_request_route_template(request) not in _SERVED_MODEL_BUDGET_WAIVER_ROUTES:
        return model
    if request.scope.get(MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY) is not True or request_data.get("model") != model:
        return model
    if "model" in safe_get_request_query_params(request=request) or _inference_model_overridden():
        return model
    if not (
        key_model_aliases_for_auth_check(valid_token)
        or team_model_aliases_for_auth_check(valid_token)
        or model in litellm.model_alias_map
    ):
        return model
    from litellm.proxy.proxy_server import prisma_client, proxy_config, proxy_logging_obj

    settings: Final = await proxy_config.get_hierarchical_router_settings(
        user_api_key_dict=valid_token, prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
    )
    group_alias: Final[object] = settings.get("model_group_alias") if isinstance(settings, Mapping) else None
    return request_served_model(model, valid_token, group_alias, llm_router)


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


def get_model_names_for_budget_checks(
    model: str | list[str] | None,
) -> list[str]:
    if model is None:
        return []
    if isinstance(model, str):
        return [model]
    return model


async def normalize_model_for_auth(
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


class _KeyModelBudgetLimiter(Protocol):
    async def is_key_within_model_budget(self, user_api_key_dict: UserAPIKeyAuth, model: str) -> bool: ...

    async def get_fallback_model_within_budget(self, user_api_key_dict: UserAPIKeyAuth, model: str) -> str | None: ...


class _UserModelBudgetLimiter(Protocol):
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


async def check_user_model_budget(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _UserModelBudgetLimiter,
    models: Sequence[str],
) -> None:
    """Enforce the internal user's own `model_max_budget` across the request's models.

    Separate from the key check: a user's per-model budget caps every key they
    own, so a caller cannot escape it by minting another key.
    """
    user_model_max_budget: Final = valid_token.user_model_max_budget
    if valid_token.user_id is None or not isinstance(user_model_max_budget, Mapping) or not user_model_max_budget:
        return
    for model_name in models:
        await model_max_budget_limiter.is_user_within_model_budget(
            user_id=valid_token.user_id,
            user_model_max_budget=user_model_max_budget,
            model=model_name,
            team_id=valid_token.team_id,
        )


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
    team_member_model_max_budget: Final = valid_token.team_member_model_max_budget
    if (
        valid_token.user_id is None
        or valid_token.team_id is None
        or not isinstance(team_member_model_max_budget, Mapping)
        or not team_member_model_max_budget
    ):
        return

    user_id: Final = valid_token.user_id
    team_id: Final = valid_token.team_id
    for model_name in models:
        await _check_team_member_model_budget_for_model(
            user_id=user_id,
            team_id=team_id,
            team_member_model_max_budget=team_member_model_max_budget,
            model_name=model_name,
            valid_token=valid_token,
            model_max_budget_limiter=model_max_budget_limiter,
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


async def _check_team_member_model_budget_for_model(
    user_id: str,
    team_id: str,
    team_member_model_max_budget: Mapping[str, object],
    model_name: str,
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _EveryModelBudgetLimiter,
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
    try:
        await model_max_budget_limiter.is_team_member_within_model_budget(
            user_id=user_id,
            team_id=team_id,
            team_member_model_max_budget=team_member_model_max_budget,
            model=model_name,
        )
    except litellm.BudgetExceededError:
        if request_data.get("model") != model_name or llm_router is None or fallbacks_disabled(request_data):
            raise
        fallback_succeeded: Final = await _try_team_member_model_budget_fallback(
            user_id=user_id,
            team_id=team_id,
            model_name=model_name,
            valid_token=valid_token,
            model_max_budget_limiter=model_max_budget_limiter,
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
        if not fallback_succeeded:
            raise


async def _try_team_member_model_budget_fallback(
    user_id: str,
    team_id: str,
    model_name: str,
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _EveryModelBudgetLimiter,
    request_data: dict[str, object],  # mutable-ok: fallback rewrites the request body
    request: Request,
    llm_model_list: Sequence[object] | None,
    llm_router: litellm.Router,
    project_object: LiteLLM_ProjectTableCachedObj | None = None,
    team_object: LiteLLM_TeamTable | None = None,
    end_user_object: LiteLLM_EndUserTable | None = None,
    route: str = "",
    prisma_client: PrismaClient | None = None,
    user_api_key_cache: UserApiKeyCache | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
) -> bool:
    if request.headers.get("content-type", "").lower().startswith(
        "multipart/form-data"
    ) or references_provider_scoped_resource(request_data):
        return False

    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import proxy_config

    resolved_prisma_client: Final = prisma_client if prisma_client is not None else proxy_server.prisma_client
    resolved_user_api_key_cache: Final = (
        user_api_key_cache if user_api_key_cache is not None else proxy_server.user_api_key_cache
    )
    resolved_proxy_logging_obj: Final = (
        proxy_logging_obj if proxy_logging_obj is not None else proxy_server.proxy_logging_obj
    )

    try:
        router_settings: Final[dict[str, object] | None] = (
            cast(  # cast-ok: proxy settings return an unparameterized dict
                dict[str, object] | None,
                await proxy_config.get_hierarchical_router_settings(
                    user_api_key_dict=valid_token,
                    prisma_client=resolved_prisma_client,
                    proxy_logging_obj=resolved_proxy_logging_obj,
                ),
            )
        )
    except Exception:  # noqa: BLE001  # Settings lookup failures must not authorize a fallback.
        return False

    configured_fallbacks: Final[object] = router_settings.get("fallbacks") if router_settings is not None else None
    router_fallbacks: Final[Sequence[object] | None] = (
        cast(Sequence[object], configured_fallbacks)  # cast-ok: guarded by the list check below
        if isinstance(configured_fallbacks, list)
        else _router_fallbacks(llm_router)
    )
    if router_fallbacks is None:
        return False

    fallback_models, _ = request_fallback_chain(list(router_fallbacks), request_data, model_name)
    fallback_targets: Final[Sequence[object]] = fallback_models or ()
    fallback_candidates: Final[tuple[str, ...]] = tuple(
        model for target in fallback_targets if (model := _budget_fallback_target_model(target)) is not None
    )
    if not fallback_candidates:
        return False

    model_group_aliases: Final[object] = (
        router_settings.get("model_group_alias") if router_settings is not None else None
    )
    resolved_fallback_candidates: Final[tuple[tuple[str, str], ...]] = tuple(
        (
            candidate,
            target if (target := resolve_model_group_alias(model_group_aliases, candidate)) is not None else candidate,
        )
        for candidate in fallback_candidates
    )
    fallback_candidates_with_served_models: Final = (
        (
            fallback_candidate,
            fallback_model,
            pre_call_served_model(fallback_model, valid_token, model_group_aliases),
        )
        for fallback_candidate, fallback_model in resolved_fallback_candidates
    )

    try:
        team_membership: Final = await get_team_membership(
            user_id=user_id,
            team_id=team_id,
            prisma_client=resolved_prisma_client,
            user_api_key_cache=resolved_user_api_key_cache,
            proxy_logging_obj=resolved_proxy_logging_obj,
        )
    except Exception:  # noqa: BLE001  # membership lookup failures must not authorize a fallback
        return False

    for fallback_candidate, fallback_model, served_model in fallback_candidates_with_served_models:
        if fallback_model == model_name:
            continue
        try:
            if (
                litellm.block_requests_for_models_without_pricing
                and RouteChecks.is_llm_api_route(route=route)
                and _unpriced_models_in_request(model=served_model, llm_router=llm_router)
            ):
                continue
            if end_user_object is not None and end_user_object.models:
                can_customer_access_model(
                    model=fallback_model,
                    end_user_object=end_user_object,
                    llm_router=llm_router,
                    valid_token=valid_token,
                )
            if not is_model_cost_zero(
                model=served_model, llm_router=llm_router
            ) and not await _is_within_all_fallback_model_budgets(
                valid_token=valid_token,
                model_max_budget_limiter=model_max_budget_limiter,
                models=tuple(
                    dict.fromkeys(
                        (
                            fallback_candidate,
                            fallback_model,
                            served_model,
                            llm_router.registry().resolve(served_model),
                        )
                    )
                ),
            ):
                continue
            if not await _is_fallback_model_authorized(
                model=fallback_model,
                valid_token=valid_token,
                llm_model_list=llm_model_list,
                llm_router=llm_router,
                prisma_client=prisma_client,
                team_object=team_object,
                team_membership=team_membership,
                project_object=project_object,
                refuse_agent_keys=True,
            ):
                continue
            valid_token.matched_model_access_groups = []  # rebind-ok: clear prior groups before evaluating this candidate
            matched_groups = await stamp_matched_model_access_groups(
                model=fallback_model,
                valid_token=valid_token,
                team_object=team_object,
                project_object=project_object,
                llm_router=llm_router,
                prisma_client=resolved_prisma_client,
                user_api_key_cache=resolved_user_api_key_cache,
                proxy_logging_obj=resolved_proxy_logging_obj,
                team_membership=team_membership,
                team_membership_loaded=True,
            )
            valid_token.matched_model_access_groups = list(matched_groups)  # rebind-ok: spend uses the candidate groups
            if matched_groups:
                await _model_access_group_max_budget_check(
                    matched_model_access_groups=list(matched_groups),
                    prisma_client=resolved_prisma_client,
                    user_api_key_cache=resolved_user_api_key_cache,
                )
        except Exception:  # noqa: BLE001  # A candidate check failure must not escape the fail-closed walk.
            verbose_proxy_logger.warning(
                "Skipping fallback candidate %s after a budget or authorization check error",
                fallback_model.replace("\r", "").replace("\n", ""),
                exc_info=True,
            )
            continue
        rewrite_request_model(request_data=request_data, request=request, model=fallback_model)
        if "path_params" in request.scope:
            request.scope["path_params"]["model"] = fallback_model  # rebind-ok: route handlers use the fallback model
        return True
    return False


async def check_key_model_budget_with_fallback(
    valid_token: UserAPIKeyAuth,
    model_max_budget_limiter: _KeyModelBudgetLimiter,
    model_name: str,
    request_data: dict[str, object],  # mutable-ok: fallback rewrites the request body
    request: Request,
    llm_model_list: Sequence[object] | None = None,
    llm_router: litellm.Router | None = None,
) -> None:
    """
    Enforce the key's per-model budget for `model_name`. If exceeded and the
    key has a `budget_fallbacks` chain configured for `model_name`, reroute
    the request to the first fallback model still within its own budget
    instead of rejecting the request.

    The selected fallback is validated against the key's model-access
    allowlist and the team's model restrictions so that budget_fallbacks
    cannot bypass model authorization. The rewrite is persisted to the
    parsed-body cache, Starlette's JSON cache (``request._json``), and
    path parameters so that downstream handlers see the final model
    regardless of whether they consume ``_read_request_body()``,
    ``request.json()``, or the path ``model`` parameter.

    Fallback is only attempted when ``model_name`` matches the top-level
    ``request_data["model"]``; models extracted from nested fields
    (``session.model``, ``completion.model``, etc.) are not rewritable
    and raise immediately.

    Raises:
        BudgetExceededError: if `model_name` is over budget and no configured
            fallback is within budget either (or the fallback is not authorized).
    """
    try:
        await model_max_budget_limiter.is_key_within_model_budget(
            user_api_key_dict=valid_token,
            model=model_name,
        )
    except litellm.BudgetExceededError as e:
        if request_data.get("model") != model_name:
            raise e
        fallback_model: Final = await model_max_budget_limiter.get_fallback_model_within_budget(
            user_api_key_dict=valid_token,
            model=model_name,
        )
        if fallback_model is None:
            raise e
        if not await _is_fallback_model_authorized(
            model=fallback_model,
            valid_token=valid_token,
            llm_model_list=llm_model_list,
            llm_router=llm_router,
        ):
            raise
        rewrite_request_model(request_data=request_data, request=request, model=fallback_model)
        if "path_params" in request.scope:
            request.scope["path_params"]["model"] = fallback_model  # rebind-ok: route handlers use the fallback model


def _get_bearer_token_or_received_api_key(api_key: str) -> str:
    if api_key.startswith("Bearer "):  # ensure Bearer token passed in
        api_key = api_key.replace("Bearer ", "")  # extract the token
    elif api_key.startswith("Basic "):
        api_key = api_key.replace("Basic ", "")  # handle langfuse input
    elif api_key.startswith("bearer "):
        api_key = api_key.replace("bearer ", "")
    elif api_key.startswith("AWS4-HMAC-SHA256"):
        # Handle AWS Signature V4 format from LangChain
        # Format: AWS4-HMAC-SHA256 Credential=Bearer $LITELLM_MASTER_KEY/date/region/service/aws4_request, SignedHeaders=..., Signature=...
        # Extract the Bearer token from the Credential field
        match = re.search(r"Credential=Bearer\s+([^/\s,]+)", api_key)
        if match:
            api_key = match.group(1)
        else:
            # If no Bearer token found in Credential, try to extract just the credential value
            match = re.search(r"Credential=([^/\s,]+)", api_key)
            if match:
                api_key = match.group(1)

    return api_key


def _auth_request(
    api_key: str | None,
    route: str,
    request: Request,
    request_data: Mapping[str, object],
    *,
    pass_through_open: bool,
    from_custom_auth: bool,
) -> core.AuthRequest:
    from litellm.proxy.proxy_server import llm_router  # noqa: PLC0415  # runtime globals

    model: Final = get_model_from_request_context(
        request_data=dict(request_data), route=route, request=request, llm_router=llm_router
    )
    headers: Final = safe_get_request_headers(request)
    return core.request(
        api_key,
        route,
        RouteChecks.get_request_method(request=request),
        headers=headers,
        model=model if isinstance(model, str) else None,
        end_user_id=get_end_user_id_from_request_body(request_data, headers),
        pass_through_open=pass_through_open,
        from_custom_auth=from_custom_auth,
    )


def _python_host(
    request: Request,
    request_data: dict[str, object],  # mutable-ok: the host rewrites the request body like the checks it runs
    route: str,
    parent_otel_span: Span | None,
    start_time: datetime,
    api_key: str | None,
) -> "PythonAuthHost":
    from litellm.proxy.auth.host.python import (  # noqa: PLC0415  # host.python imports this module's helpers
        HttpCall,
        proxy_host,
    )

    return proxy_host(
        route,
        api_key,
        parent_otel_span=parent_otel_span,
        http=HttpCall(request=request, request_data=request_data, start_time=start_time),
    )


def get_bearer_token(
    api_key: str,
) -> str:
    if api_key.startswith("Bearer "):  # ensure Bearer token passed in
        api_key = api_key.replace("Bearer ", "")  # extract the token
    elif api_key.startswith("Basic "):
        api_key = api_key.replace("Basic ", "")  # handle langfuse input
    elif api_key.startswith("bearer "):
        api_key = api_key.replace("bearer ", "")
    elif api_key.startswith("AWS4-HMAC-SHA256"):
        # Handle AWS Signature V4 format from LangChain
        # Format: AWS4-HMAC-SHA256 Credential=Bearer $LITELLM_MASTER_KEY/date/region/service/aws4_request, SignedHeaders=..., Signature=...
        # Extract the Bearer token from the Credential field
        match = re.search(r"Credential=Bearer\s+([^/\s,]+)", api_key)
        if match:
            api_key = match.group(1)
        else:
            # If no Bearer token found in Credential, try to extract just the credential value
            match = re.search(r"Credential=([^/\s,]+)", api_key)
            if match:
                api_key = match.group(1)
            else:
                api_key = ""
    else:
        api_key = ""
    return api_key


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

    authorization: Final = websocket.headers.get("authorization")
    # If no Authorization header, try the api-key header
    if not authorization:
        api_key = websocket.headers.get("api-key")
        if not api_key:
            # Try extracting from WebSocket subprotocol (browser clients)
            for protocol in websocket.headers.get("sec-websocket-protocol", "").split(","):
                protocol = protocol.strip()
                if protocol.startswith("openai-insecure-api-key."):
                    api_key = protocol[len("openai-insecure-api-key.") :]
                    break
        if not api_key:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(status_code=403, detail="No API key provided")
    else:
        # Extract the API key from the Bearer token
        if not authorization.startswith("Bearer "):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            raise HTTPException(status_code=403, detail="Invalid Authorization header format")

        api_key = authorization[len("Bearer ") :].strip()

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


def get_api_key(
    custom_litellm_key_header: str | None,
    api_key: str,
    azure_api_key_header: str | None,
    anthropic_api_key_header: str | None,
    google_ai_studio_api_key_header: str | None,
    azure_apim_header: str | None,
    pass_through_endpoints: list[dict] | None,
    route: str,
    request: Request,
) -> tuple[str, str | None]:
    """
    Returns:
        Tuple[Optional[str], Optional[str]]: Tuple of the api_key and the passed_in_key
    """
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.common_utils.http_parsing_utils import (
        safe_get_request_query_params,
    )

    api_key = api_key
    passed_in_key: str | None = None
    if isinstance(custom_litellm_key_header, str):
        passed_in_key = custom_litellm_key_header
        api_key = _get_bearer_token_or_received_api_key(custom_litellm_key_header)
    elif isinstance(api_key, str) and len(api_key) > 0:
        passed_in_key = api_key
        api_key = get_bearer_token(api_key=api_key)
    elif isinstance(azure_api_key_header, str):
        passed_in_key = azure_api_key_header
        api_key = azure_api_key_header
    elif isinstance(anthropic_api_key_header, str):
        passed_in_key = anthropic_api_key_header
        api_key = anthropic_api_key_header
    elif isinstance(google_ai_studio_api_key_header, str):
        passed_in_key = google_ai_studio_api_key_header
        api_key = google_ai_studio_api_key_header
    elif isinstance(azure_apim_header, str):
        passed_in_key = azure_apim_header
        api_key = azure_apim_header
    elif (
        RouteChecks.is_generate_content_route(route=route)
        and request is not None
        and safe_get_request_query_params(request).get("key")
    ):
        google_auth_key: Final[str] = safe_get_request_query_params(request).get("key") or ""
        passed_in_key = google_auth_key
        api_key = google_auth_key
    elif pass_through_endpoints is not None:
        for endpoint in pass_through_endpoints:
            if endpoint.get("path", "") == route:
                headers: dict | None = endpoint.get("headers", None)
                if headers is not None:
                    header_key: str = headers.get("litellm_user_api_key", "")
                    if request.headers.get(header_key) is not None:
                        api_key = request.headers.get(header_key) or ""
                        passed_in_key = api_key
    return api_key, passed_in_key


def _route_pass_through_endpoints(
    route: str, pass_through_endpoints: list[dict] | None
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        endpoint
        for endpoint in pass_through_endpoints or ()
        if isinstance(endpoint, dict) and endpoint.get("path", "") == route
    )


def pass_through_auth_off(route: str, pass_through_endpoints: list[dict] | None) -> bool:
    """A configured pass-through endpoint on ``route`` turns auth off. ``auth`` defaults to on: a raw config
    dict without the key must never become an unauthenticated forwarder."""
    return any(
        endpoint.get("auth", True) is not True
        for endpoint in _route_pass_through_endpoints(route, pass_through_endpoints)
    )


def _endpoint_key(key: str, endpoint: Mapping[str, object]) -> str:
    if endpoint.get("custom_auth_parser") != "langfuse":
        return key
    # langfuse sends 'Basic <base64(public_key:secret_key)>', and the public key is the litellm key
    return base64.b64decode(key.replace("Basic ", "").strip()).decode("utf-8").split(":")[0]


def pass_through_key(
    request: Request,
    route: str,
    pass_through_endpoints: list[dict] | None,
    api_key: str,
) -> str:
    """The key a pass-through route authenticates: a ``litellm_user_api_key`` header on a mapped provider
    route, then each configured Langfuse endpoint's public key, else ``api_key``."""
    normalized_route: Final = normalize_route_for_root_path(route)
    mapped_header: Final = (
        request.headers.get("litellm_user_api_key")
        if normalized_route is not None
        and any(normalized_route.startswith(prefix) for prefix in LiteLLMRoutes.mapped_pass_through_routes.value)
        else None
    )
    return reduce(
        _endpoint_key,
        _route_pass_through_endpoints(route, pass_through_endpoints),
        api_key if mapped_header is None else mapped_header,
    )


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
    have a span to close) and inside ``user_api_key_auth_builder`` (for
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


PREFLIGHT_FAILURE_JSON_BODY_MAX_BYTES: Final = 64 * 1024
PREFLIGHT_FAILURE_JSON_BODY_TIMEOUT_SECONDS: Final = 1.0


def _preflight_failure_model(
    body: Mapping[str, object], route: str, request: Request, team_id: str | None
) -> str | Sequence[str] | None:
    from litellm.proxy.proxy_server import llm_router

    try:
        return get_model_from_request_context(dict(body), route, request, llm_router, team_id)
    except Exception:  # noqa: BLE001  # logging enrichment must never replace the existing preflight rejection
        return None


async def _preflight_failure_request_data(
    request: Request, route: str, cached_key: UserAPIKeyAuth
) -> Mapping[str, object]:
    from litellm.proxy.common_request_processing import resolve_litellm_call_id
    from litellm.proxy.litellm_pre_call_utils import key_and_team_logging_fields
    from litellm.proxy.proxy_server import proxy_config
    from litellm.proxy.utils import (
        _call_type_for_route,  # pyright: ignore[reportPrivateUsage]  # share proxy failure route classification
    )

    body: Final = await _small_preflight_failure_json(request)
    model: Final = _preflight_failure_model(body, route, request, cached_key.team_id)
    call_type: Final = _call_type_for_route(route, method=request.method)
    _seed_request_destinations(cached_key, request)
    seed_request_identity(cached_key, model=model, request_data=body)
    return {
        **{key: value for key, value in body.items() if key != "model"},
        **({"model": model} if model is not None else {}),
        "litellm_call_id": resolve_litellm_call_id(request.headers.get("x-litellm-call-id")),
        "call_type": call_type,
        **key_and_team_logging_fields(user_api_key_dict=cached_key, proxy_config=proxy_config),
    }


async def _small_preflight_failure_json(request: Request) -> Mapping[str, object]:
    declared: Final = request.headers.get("content-length", "")
    content_type: Final = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if (
        not declared.isdecimal()
        or len(declared) > 10
        or not 0 < int(declared) <= PREFLIGHT_FAILURE_JSON_BODY_MAX_BYTES
        or request.headers.get("transfer-encoding") is not None
        or content_type != "application/json"
    ):
        return {}
    buffered: Final = bytearray()
    try:
        with anyio.fail_after(PREFLIGHT_FAILURE_JSON_BODY_TIMEOUT_SECONDS):
            async for chunk in request.stream():
                if len(buffered) + len(chunk) > PREFLIGHT_FAILURE_JSON_BODY_MAX_BYTES:
                    return {}
                buffered.extend(chunk)
        parsed: Final = _GENERAL_SETTINGS_ADAPTER.validate_json(buffered)
    except (ValidationError, RuntimeError, ClientDisconnect, TimeoutError):
        return {}
    return {
        **({"model": parsed["model"]} if isinstance(parsed.get("model"), str) else {}),
        **{
            slot: {
                name: value
                for name, value in _GENERAL_SETTINGS_ADAPTER.validate_python(metadata).items()
                if name in ("tags", "session_id", "trace_user_id", "trace_id", "trace_name")
            }
            for slot in ("metadata", "litellm_metadata")
            if isinstance(metadata := parsed.get(slot), dict)
        },
    }


async def _reject_from_headers_before_reading_body(
    request: Request, api_key: str | None, custom_litellm_key_header: str | None, route: str
) -> None:
    """Refuse a request its headers already condemn, a blocked key or an exhausted key limit, before reading its body.

    Only a key already in the cache is considered, so a miss changes nothing and the full auth path still runs.
    A retry flood of large payloads against a limited key used to buffer every payload just to answer 429.
    """
    if not RouteChecks.is_llm_api_route(route=route):
        return
    from litellm.proxy import proxy_server
    from litellm.proxy.common_request_processing import proxy_exception_from_http_exception
    from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    configured_header: Final = proxy_server.general_settings_view().get("litellm_key_header_name")
    bearer_token: Final = (
        get_api_key_from_custom_header(request=request, custom_litellm_key_header_name=configured_header)
        if isinstance(configured_header, str) and configured_header in request.headers
        else _get_bearer_token_or_received_api_key(custom_litellm_key_header)
        if isinstance(custom_litellm_key_header, str)
        else get_bearer_token(api_key=api_key)
        if isinstance(api_key, str)
        else ""
    )
    if not bearer_token:
        return
    hashed_token: Final = hash_token(bearer_token)
    try:
        cached_key: Final = IdentityStore.key_from_principal(
            await IdentityStore(prisma_client, user_api_key_cache, check_cache_only=True).resolve(
                hashed_token=hashed_token
            )
        )
    except Exception:  # noqa: BLE001  # any cache miss defers to the full auth path below
        return
    parent_otel_span: Final[Span | None] = getattr(request.state, "parent_otel_span", None)
    if cached_key.blocked is True:
        request_data: Final = dict(await _preflight_failure_request_data(request, route, cached_key))
        await UserAPIKeyAuthExceptionHandler.handle_authentication_error(
            e=ProxyException(
                message="Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin.",
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            ),
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=parent_otel_span,
            api_key=bearer_token,
            resolved_identity=cached_key,
            dispatched_call_type=str(request_data["call_type"]) if request_data["call_type"] is not None else None,
        )
        return
    limiter: Final = proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
    if not isinstance(limiter, PROXY_MaxParallelRequestsHandler_v3):
        return
    rate_limited: Final = cached_key.model_copy(
        update={
            "api_key": cached_key.api_key or hashed_token,
            "parent_otel_span": parent_otel_span,
            "request_route": normalize_request_route(route),
        }
    )
    try:
        await limiter.raise_if_over_limit_before_body(rate_limited)
    except HTTPException as rejection:
        failure_data: Final = dict(await _preflight_failure_request_data(request, route, rate_limited))
        transformed: Final = await proxy_logging_obj.post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # bare dict in sig
            request_data=failure_data,
            original_exception=rejection,
            user_api_key_dict=rate_limited,
            route=route,
            dispatched_call_type=str(failure_data["call_type"]) if failure_data["call_type"] is not None else None,
        )
        failure: Final = transformed if transformed is not None else rejection
        if isinstance(failure, ProxyException):
            raise failure
        raise proxy_exception_from_http_exception(failure, dict(failure.headers or {}))


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


def _custom_auth_token_for_this_request(response: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth.model_validate(response).model_copy()


async def _custom_auth(
    request: Request,
    api_key: str,
    request_data: dict[str, object],
    route: str,
    parent_otel_span: Span | None,
    user_custom_auth: Callable[..., Awaitable[object]] | None,
) -> UserAPIKeyAuth | str | None:
    """The identity custom auth vouched for, a key it handed back to authenticate with instead, or ``None``
    when enterprise custom auth declined and the request keeps its own token."""
    if enterprise_custom_auth is not None:
        with tracer.trace("litellm.proxy.auth.enterprise_custom_auth"):
            response: Final = await enterprise_custom_auth(
                request=request, api_key=api_key, user_custom_auth=user_custom_auth
            )
        if isinstance(response, str):
            return response
        if not isinstance(response, UserAPIKeyAuth):
            return None
        return await _custom_auth_identity(response, request, request_data, route, parent_otel_span)
    if user_custom_auth is None:
        return None
    return await _custom_auth_identity(
        await user_custom_auth(request=request, api_key=api_key), request, request_data, route, parent_otel_span
    )


async def _custom_auth_identity(
    response: object,
    request: Request,
    request_data: dict[str, object],
    route: str,
    parent_otel_span: Span | None,
) -> UserAPIKeyAuth:
    token: Final = _custom_auth_token_for_this_request(response)
    validated: Final = (
        await _run_post_custom_auth_checks(
            valid_token=token,
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=parent_otel_span,
        )
        if getattr(litellm, "enable_post_custom_auth_checks", False)
        else token
    )
    validated.authenticated_by_custom_auth = True
    return validated


async def _core_outcome(
    path: "RustPlan",
    auth_request: core.AuthRequest,
    api_key: str,
    settings: "RustAuthSettings",
    host: "PythonAuthHost",
    request: Request,
    pass_through_endpoints: list[dict] | None,
) -> "RustAuthOutcome":
    match path:
        case RustPlan.Key:
            key: Final = (
                api_key
                if auth_request.from_custom_auth
                else pass_through_key(request, auth_request.route, pass_through_endpoints, api_key)
            )
            return await core.drive(RustAuth.authenticate_key(auth_request, key, settings), host)
        case RustPlan.Jwt | RustPlan.JwtWithOperatorCode:
            return await core.drive(RustAuth.authenticate_jwt(auth_request, settings), host)
        case _:
            raise ValueError(f"no credential path for auth plan {path!r}")


async def user_api_key_auth_builder(
    request: Request,
    api_key: str,
    azure_api_key_header: str,
    anthropic_api_key_header: str | None,
    google_ai_studio_api_key_header: str | None,
    azure_apim_header: str | None,
    request_data: dict[str, object],
    custom_litellm_key_header: str | None = None,
    defer_team_checks: bool = False,
) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import (
        general_settings,
        open_telemetry_logger,
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
    denied_key: UserAPIKeyAuth | None = None  # rebind-ok: the key a denied request had resolved, for the failure log

    try:
        with tracer.trace("litellm.proxy.auth.pre_db_read_auth_checks"):
            await pre_db_read_auth_checks(
                request_data=request_data,
                request=request,
                route=route,
                defer_team_checks=defer_team_checks,
            )
        pass_through_endpoints: Final[list[dict] | None] = general_settings.get("pass_through_endpoints", None)
        ## CHECK IF X-LITELM-API-KEY IS PASSED IN - supercedes Authorization header
        api_key, passed_in_key = get_api_key(
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
        custom_litellm_key_header_name: Final = general_settings.get("litellm_key_header_name")
        if custom_litellm_key_header_name is not None and custom_litellm_key_header_name in request.headers:
            api_key = get_api_key_from_custom_header(
                request=request,
                custom_litellm_key_header_name=custom_litellm_key_header_name,
            )

        if open_telemetry_logger is not None:
            # Reuse the span created by user_api_key_auth (before body parse)
            # so it survives _read_request_body failures. For callers that
            # bypass user_api_key_auth (e.g. MCP), create it lazily.
            _ensure_parent_otel_span_on_request_state(request)
            parent_otel_span = getattr(request.state, "parent_otel_span", None)

        settings: Final = core.proxy_settings()
        custom: Final = (
            await _custom_auth(request, api_key, request_data, route, parent_otel_span, user_custom_auth)
            if enterprise_custom_auth is not None or user_custom_auth is not None
            else None
        )
        if isinstance(custom, UserAPIKeyAuth):
            return custom
        handed_key: Final = isinstance(custom, str)
        if isinstance(custom, str):
            api_key = custom  # rebind-ok: custom auth handed back the key to authenticate with
        rust_host: Final = _python_host(request, request_data, route, parent_otel_span, start_time, api_key)
        auth_request: Final = _auth_request(
            api_key,
            route,
            request,
            request_data,
            pass_through_open=pass_through_auth_off(route, pass_through_endpoints),
            from_custom_auth=handed_key,
        )
        match RustAuth.plan(auth_request, settings):
            case RustDenial() as denial:
                raise auth_denial.denial_error(denial)
            case RustPlan.Public:
                return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
            case RustPlan.OAuth2Introspection:
                return await Oauth2Handler.check_oauth2_token(token=api_key)
            case RustPlan.OAuth2Proxy:
                return await handle_oauth2_proxy_request(request=request)
            case RustPlan.PassThroughOpen:
                return UserAPIKeyAuth()
            case plan:
                outcome: Final = await _core_outcome(
                    plan, auth_request, api_key, settings, rust_host, request, pass_through_endpoints
                )
        if outcome.denied:
            denied_key = outcome.denied_key  # rebind-ok: see above
            auth_denial.raise_denial(outcome)
        return await auth_identity.project(outcome, rust_host)

    except Exception as e:
        return await UserAPIKeyAuthExceptionHandler.handle_authentication_error(
            e=e,
            request=request,
            request_data=request_data,
            route=route,
            parent_otel_span=parent_otel_span,
            api_key=api_key,
            resolved_identity=denied_key,
        )


_user_api_key_auth_builder: Final = user_api_key_auth_builder


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


def _team_obj_from_token(valid_token: UserAPIKeyAuth) -> LiteLLM_TeamRebuiltFromToken:
    """Reconstruct a cached team object from the fields already on the
    UserAPIKeyAuth. Only called when valid_token.team_id is known to be
    non-None (the caller gates on it)."""
    assert valid_token.team_id is not None
    token_team_models: Final = _token_team_models(valid_token)
    return LiteLLM_TeamRebuiltFromToken(
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
    boundary, regardless of which ``user_api_key_auth_builder`` path
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

    if not sub_call and getattr(request.state, "deferred_model_body_check", False) is True:
        body_model: Final = request_data.get("model")
        is_request_body_safe(
            request_body=request_data,
            general_settings=_GENERAL_SETTINGS_ADAPTER.validate_python(general_settings),
            llm_router=llm_router,
            model=body_model if isinstance(body_model, str) else "",
            route=route,
            team_id=user_api_key_auth_obj.team_id,
            defer_s3_buckets=checks_team_s3_buckets_in_handler(request),
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
    # In the integrated auth flow ``user_api_key_auth_builder`` has already
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

    if (
        end_user_id
        and (key_end_user_budget_id is not None or user_api_key_auth_obj.jwt_claims is not None)
        and prisma_client is not None
    ):
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

    skip_budget_checks: Final = await _should_skip_budget_checks(
        request_data=request_data,
        route=route,
        request=request,
        llm_router=llm_router,
        valid_token=user_api_key_auth_obj,
        sub_call=sub_call,
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
            models: Final = get_model_names_for_budget_checks(
                model=get_model_from_request_context(
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
                    models=get_model_names_for_budget_checks(
                        model=get_model_from_request_context(
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
    key_end_user_budget_id: str | None,
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

        reservation_path_params: Final = read_request_path_params(request)
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
            path_params=reservation_path_params,
        )
    if request is not None:
        reservation: Final = user_api_key_auth_obj.budget_reservation
        request.state.budget_reservation = reservation  # rebind-ok: read by the release middleware


async def _should_skip_budget_checks(
    request_data: dict[str, object],
    route: str,
    request: Request | None,
    llm_router: litellm.Router | None,
    valid_token: UserAPIKeyAuth,
    sub_call: bool = False,
) -> bool:
    model: Final = get_model_from_request_context(
        request_data=request_data,
        route=route,
        request=request,
        llm_router=llm_router,
        team_id=valid_token.team_id,
    )
    return request_skips_budget_checks(
        route=route,
        model=(model if sub_call else await budget_served_model(model, request_data, request, llm_router, valid_token)),
        llm_router=llm_router,
        request_body=request_data,
        http_method=RouteChecks.get_request_method(request),
    )


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

    # Defense-in-depth: ``user_api_key_auth_builder`` has multiple early-return
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


def identity_cache_keys(api_key: str, *, end_user_id: str | None, key_is_resolved: bool) -> tuple[str, ...]:
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


async def prefetch_referenced_auth_objects(
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
    azure_api_key_header: str = fastapi.Security(azure_api_key_header),
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

    route: Final[str] = get_request_route(request=request)
    await _reject_from_headers_before_reading_body(
        request=request, api_key=api_key, custom_litellm_key_header=custom_litellm_key_header, route=route
    )
    request_data, body_parse_exception = await _read_request_body_deferring_parse_failure(request=request)
    phase_event("litellm.request.body_parsed")
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
                defer_team_checks=True,
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


async def return_user_api_key_auth_obj(
    user_obj: LiteLLM_UserTable | None,
    api_key: str,
    parent_otel_span: Span | None,
    valid_token_dict: Mapping[str, object],
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


def get_api_key_from_custom_header(request: Request, custom_litellm_key_header_name: str) -> str:
    """
    Get API key from custom header

    Args:
        request (Request): Request object
        custom_litellm_key_header_name (str): Custom header name

    Returns:
        Optional[str]: API key
    """
    api_key: str = ""
    # use this as the virtual key passed to litellm proxy
    custom_litellm_key_header_name = custom_litellm_key_header_name.lower()
    _headers: Final = {k.lower(): v for k, v in request.headers.items()}
    verbose_proxy_logger.debug(
        "searching for custom_litellm_key_header_name= %s, in headers=%s",
        custom_litellm_key_header_name,
        _headers,
    )
    custom_api_key: Final = _headers.get(custom_litellm_key_header_name)
    if custom_api_key:
        api_key = get_bearer_token(api_key=custom_api_key)  # rebind-ok: pre-existing rebinding on a rename-only line
        verbose_proxy_logger.debug(
            "Found custom API key using header: %s, setting api_key=%s",
            custom_litellm_key_header_name,
            abbreviate_api_key(api_key),
        )
    else:
        verbose_proxy_logger.exception(
            "No LiteLLM Virtual Key pass. Please set header=%s: Bearer <api_key>", custom_litellm_key_header_name
        )
    return api_key


def _get_temp_budget_increase(valid_token: UserAPIKeyAuth):
    valid_token_metadata: Final = valid_token.metadata
    if "temp_budget_increase" in valid_token_metadata and "temp_budget_expiry" in valid_token_metadata:
        expiry = datetime.fromisoformat(valid_token_metadata["temp_budget_expiry"])
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        if expiry > datetime.now(timezone.utc):
            return valid_token_metadata["temp_budget_increase"]
    return None


def update_key_budget_with_temp_budget_increase(
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
        model: Final = get_model_from_request_context(
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
        await normalize_model_for_auth(request_data, valid_token, request, route)
        request.state.requires_key_model_check = True  # rebind-ok: defer until central grants

    current_model = get_model_from_request_context(
        request_data=request_data,
        route=route,
        request=request,
        llm_router=llm_router,
        team_id=valid_token.team_id,
    )
    current_models = get_model_names_for_budget_checks(model=current_model)

    # A zero-cost model cannot move any counter, so refusing it means refusing on
    # spend some other model accrued. The JWT and virtual-key paths already skip
    # every budget check for these; this path did not, so the same request could
    # be refused under custom auth and served under the other two.
    skip_budget_checks: Final = is_video_retrieval_request(
        route=route, http_method=RouteChecks.get_request_method(request)
    ) or (
        is_model_cost_zero(
            model=await budget_served_model(current_model, request_data, request, llm_router, valid_token),
            llm_router=llm_router,
        )
        if current_model is not None and llm_router is not None
        else False
    )

    # 3. Check key-level model_max_budget
    max_budget_per_model: Final = valid_token.model_max_budget
    if (
        not skip_budget_checks
        and max_budget_per_model is not None
        and isinstance(max_budget_per_model, dict)
        and len(max_budget_per_model) > 0
        and current_models
        and valid_token.token is not None
    ):
        for model_name in current_models:
            await check_key_model_budget_with_fallback(
                valid_token=valid_token,
                model_max_budget_limiter=model_max_budget_limiter,
                model_name=model_name,
                request_data=request_data,
                request=request,
                llm_model_list=cast(  # cast-ok: router model lists are deployment sequences
                    Sequence[object] | None, llm_model_list
                ),
                llm_router=llm_router,
            )

        # Recompute after a potential budget-fallback rewrite so
        # the end-user check below validates the final model
        current_model = get_model_from_request_context(
            request_data=request_data,
            route=route,
            request=request,
            llm_router=llm_router,
            team_id=valid_token.team_id,
        )
        current_models = get_model_names_for_budget_checks(model=current_model)

    # 3b. Attach and check the internal user's model_max_budget.
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
    if not skip_budget_checks and current_models:
        await check_user_model_budget(
            valid_token=valid_token,
            model_max_budget_limiter=model_max_budget_limiter,
            models=current_models,
        )

    # 4. Check end-user model_max_budget
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
