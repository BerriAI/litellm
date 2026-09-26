import asyncio
import base64
import hashlib
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, nullcontext
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, JsonValue, TypeAdapter
from starlette.types import Message

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.chatgpt.live import LiveDeployment, LiveOperation, LiveTransport, live_session_path
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy._types import (
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamTableCachedObj,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _cache_team_object,  # pyright: ignore[reportPrivateUsage]  # same cache write the chat path performs
    _get_team_object_from_cache,  # pyright: ignore[reportPrivateUsage]  # same cache read the chat path performs
    can_key_call_resolved_model,  # pyright: ignore[reportUnknownVariableType]  # legacy authorization accepts untyped deployment lists
    can_org_access_model,
    can_user_call_model,
    collect_matched_model_access_groups,
    get_object_permission,
    get_org_object,
    get_project_object,
    get_team_membership,
    get_team_object,
    get_user_object,
)
from litellm.proxy.auth.user_api_key_auth import get_websocket_api_key, user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper
from litellm.proxy.common_utils.user_api_key_cache import (
    get_management_object_ttl,
    live_model_access_group_limits_cache_key,
)
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,  # pyright: ignore[reportPrivateUsage]  # limiter class is the existing hook identity
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # slot transfer is available on this existing hook
    isolated_request_stash,
)
from litellm.proxy.hooks.realtime_call_lease import RealtimeCallLease, realtime_call_attachment
from litellm.proxy.realtime_endpoints.call_sessions import process_codex_request
from litellm.proxy.realtime_endpoints.call_supervision import CALL_SUPERVISORS, CallSupervisor
from litellm.proxy.spend_tracking.budget_reservation import (
    release_or_invalidate_budget_reservation,  # pyright: ignore[reportUnknownVariableType]  # budget helper accepts legacy reservation dicts
)
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.table_repositories import ModelAccessGroupBudgetRepository
from litellm.repositories.team_repository import TeamRepository

_routes: Final = APIRouter()
_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_EMPTY: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_CACHEABLE_MODEL = TypeVar("_CACHEABLE_MODEL", bound=BaseModel)
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])
_DEPLOYMENT: Final = TypeAdapter(LiveDeployment)
_MODEL_NAMES: Final = TypeAdapter(tuple[str, ...])
_OBJECT_VALUE: Final = TypeAdapter(object)
_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_PREFIX: Final = "live_litellm_"


def _json_value(value: object) -> JsonValue:
    root: Final[list[JsonValue]] = [None]  # mutable-ok: iterative conversion fills JSON output containers
    pending: Final[  # mutable-ok: work stack carries mutable JSON output containers
        list[tuple[object, dict[str, JsonValue] | list[JsonValue], str | int, int]]
    ] = [(value, root, 0, 0)]  # mutable-ok: traversal adds pending nodes
    while pending:
        source, parent, key, depth = pending.pop()
        if depth > 256:
            raise ValueError("Live JSON nesting exceeds the supported depth")
        converted: JsonValue
        if isinstance(source, Mapping):
            entries: Mapping[str, object] = _MAPPING.validate_python(source)
            converted = {name: None for name in entries}  # mutable-ok: JSON wire objects require dicts
            pending.extend((item, converted, name, depth + 1) for name, item in entries.items())
        elif isinstance(source, (tuple, list)):
            items: tuple[object, ...] = TypeAdapter(tuple[object, ...]).validate_python(source)
            array: list[JsonValue] = [None] * len(items)
            pending.extend((item, array, index, depth + 1) for index, item in enumerate(items))
            converted = array
        else:
            converted = _JSON.validate_python(source)
        match parent, key:
            case dict(), str():
                parent[key] = converted
            case list(), int():
                parent[key] = converted
            case _:
                raise TypeError("Invalid Live JSON conversion target")
    return root[0]


def _object(value: object) -> Mapping[str, JsonValue]:
    return _OBJECT.validate_python(_json_value(value))


def _mutable(
    value: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: legacy proxy and ASGI contracts mutate inputs
    return dict(value)  # mutable-ok: make the mutable copy at the framework boundary


def _encode_json(value: object) -> str:
    return json.dumps(_json_value(value))


class _ConnectionState:
    def __init__(self) -> None:
        self.connection: ClientConnection | None = None


class LiveHandle(BaseModel):
    session_id: str
    alias: str
    deployment: Mapping[str, JsonValue]
    owner: str
    expires_at: float
    parallel_reserved: bool = False
    initialization_seconds: float = 0
    policy: Mapping[str, JsonValue] = Field(default_factory=lambda: _EMPTY)


def encode_session(handle: LiveHandle) -> str:
    encrypted: Final = encrypt_value_helper(handle.model_dump_json())
    return _PREFIX + base64.urlsafe_b64encode(encrypted.encode()).decode().rstrip("=")


def decode_session(token: str, owner: str) -> LiveHandle:
    try:
        if not token.startswith(_PREFIX):
            raise ValueError("Invalid prefix")
        encoded: Final = token[len(_PREFIX) :]
        encrypted: Final = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        handle: Final = LiveHandle.model_validate_json(
            decrypt_value_helper(encrypted.decode(), key="live_session") or ""
        )
        if handle.owner != owner or handle.expires_at <= time.time():
            raise ValueError("Invalid ownership or expiry")
        live_session_path(handle.session_id, "attach")
        return handle
    except (ValueError, TypeError, UnicodeError) as exc:
        raise HTTPException(403, "Invalid or expired Live session") from exc


def rewrite_session_ids(value: JsonValue | Mapping[str, JsonValue], raw_id: str, public_id: str) -> JsonValue:
    if not isinstance(value, Mapping):
        return _json_value(value)
    session: Final = value.get("session")
    return _json_value(
        MappingProxyType(
            {
                **value,
                **(MappingProxyType({"session_id": public_id}) if value.get("session_id") == raw_id else _EMPTY),
                **(
                    MappingProxyType({"session": MappingProxyType({**session, "id": public_id})})
                    if isinstance(session, Mapping) and session.get("id") == raw_id
                    else _EMPTY
                ),
            }
        )
    )


def _owner(auth: UserAPIKeyAuth) -> str:
    if not auth.api_key:
        raise HTTPException(403, "Live sessions require an authenticated API key")
    return hashlib.sha256(auth.api_key.encode(), usedforsecurity=False).hexdigest()


async def _auth(request: Request) -> UserAPIKeyAuth:
    return await user_api_key_auth(
        request=request,
        api_key=request.headers.get("authorization", ""),
        azure_api_key_header=request.headers.get("api-key", ""),
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=request.headers.get("x-litellm-api-key"),
    )


async def _body(request: Request) -> Mapping[str, JsonValue]:
    from litellm.proxy.realtime_endpoints.call_sessions import MAX_REALTIME_OFFER_BYTES

    chunks: Final = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > MAX_REALTIME_OFFER_BYTES:
            raise HTTPException(413, "Live request exceeds the 8 MiB limit")
        chunks.extend(chunk)
    try:
        return _OBJECT.validate_json(bytes(chunks)) if chunks else _EMPTY
    except ValueError as exc:
        raise HTTPException(400, "Expected a JSON object") from exc


def _request(source: Request | WebSocket, body: Mapping[str, JsonValue]) -> Request:
    async def receive() -> Message:
        return _mutable(
            MappingProxyType({"type": "http.request", "body": _encode_json(body).encode(), "more_body": False})
        )

    return Request(_mutable(MappingProxyType({**source.scope, "type": "http", "method": "POST"})), receive=receive)


def _session_model(body: Mapping[str, JsonValue], fallback: str | None = None) -> str:
    session: Final = body.get("session", _EMPTY)
    if not isinstance(session, Mapping):
        raise HTTPException(400, "session must be a JSON object")
    model: Final = session.get("model", fallback)
    if not isinstance(model, str) or not model:
        raise HTTPException(400, "session.model is required")
    if fallback is not None and "model" in session:
        raise HTTPException(400, "A session fork cannot change the authorized model")
    return model


async def _live_organization_id(auth: UserAPIKeyAuth) -> str | None:
    if auth.org_id is not None or auth.team_id is None:
        return auth.org_id
    from litellm.proxy import proxy_server as server

    try:
        team_object: Final = await get_team_object(
            team_id=auth.team_id,
            prisma_client=server.prisma_client,
            user_api_key_cache=server.user_api_key_cache,
            parent_otel_span=auth.parent_otel_span,
            proxy_logging_obj=server.proxy_logging_obj,
        )
    except Exception as exc:
        raise HTTPException(503, "Could not verify Live team organization model access") from exc
    return team_object.organization_id


async def _authorize(model: str, auth: UserAPIKeyAuth) -> None:
    from litellm.proxy import proxy_server as server

    await can_key_call_resolved_model(
        model=model,
        llm_model_list=list(  # mutable-ok: legacy authorization requires a concrete list
            TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(server.llm_model_list or ())  # pyright: ignore[reportUnknownMemberType]  # legacy registry is validated here
        ),
        valid_token=auth,
        llm_router=server.llm_router,
    )

    if auth.user_id is not None and auth.team_id is None and auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        try:
            user_object: Final = await get_user_object(
                user_id=auth.user_id,
                prisma_client=server.prisma_client,
                user_api_key_cache=server.user_api_key_cache,
                user_id_upsert=False,
                parent_otel_span=auth.parent_otel_span,
                proxy_logging_obj=server.proxy_logging_obj,
            )
        except Exception as exc:
            raise HTTPException(503, "Could not verify Live user model access") from exc
        if user_object is None:
            raise HTTPException(503, "Could not verify Live user model access")
        await can_user_call_model(model=model, llm_router=server.llm_router, user_object=user_object)

    organization_id: Final = await _live_organization_id(auth)
    if organization_id is not None:
        try:
            org_object: Final = await get_org_object(
                org_id=organization_id,
                prisma_client=server.prisma_client,
                user_api_key_cache=server.user_api_key_cache,
                parent_otel_span=auth.parent_otel_span,
                proxy_logging_obj=server.proxy_logging_obj,
            )
        except Exception as exc:
            raise HTTPException(503, "Could not verify Live organization model access") from exc
        if org_object is None:
            raise HTTPException(503, "Could not verify Live organization model access")
        can_org_access_model(model=model, org_object=org_object, llm_router=server.llm_router)


async def _deployment(model: str, processed: Mapping[str, object]) -> LiveDeployment:
    from litellm.proxy import proxy_server as server

    if server.llm_router is None:
        raise HTTPException(503, "Live requires a configured model deployment")
    selected: Final = _MAPPING.validate_python(
        await server.llm_router.async_get_available_deployment(  # pyright: ignore[reportUnknownMemberType]  # validate the legacy router result at this boundary
            model=model, request_kwargs=_mutable(processed)
        )
    )
    await server.llm_router.async_routing_strategy_pre_call_checks(_mutable(selected), None)  # pyright: ignore[reportUnknownMemberType]  # existing routing strategy has an untyped deployment contract
    params: Final = _object(selected["litellm_params"])
    qualified: Final = str(params["model"])
    prefix, _, suffix = qualified.partition("/")
    provider: Final = prefix if prefix in ("openai", "chatgpt") else "openai"
    upstream: Final = suffix if prefix in ("openai", "chatgpt") else qualified
    if provider == "chatgpt" and any(
        params.get(key) is not None for key in ("chatgpt_auth_profile", "chatgpt_token_dir", "chatgpt_auth_file")
    ):
        raise HTTPException(
            400, "ChatGPT Live uses the proxy OAuth credentials; deployment auth overrides are unsupported"
        )
    if prefix not in ("openai", "chatgpt"):
        if "/" in qualified:
            raise HTTPException(400, "Live requires an OpenAI or ChatGPT deployment")
    return _DEPLOYMENT.validate_python(
        _mutable(
            MappingProxyType(
                {
                    "model": upstream,
                    "provider": provider,
                    "model_id": str(_MAPPING.validate_python(selected["model_info"])["id"]),
                    "api_base": params.get("api_base"),
                    "api_key": params.get("api_key"),
                    "extra_headers": params.get("extra_headers") or _EMPTY,
                    "extra_query": params.get("extra_query") or _EMPTY,
                }
            )
        )
    )


def _pinned(handle: LiveHandle) -> LiveDeployment:
    return _DEPLOYMENT.validate_python(_mutable(handle.deployment))


def _validate_pinned_deployment(handle: LiveHandle) -> LiveDeployment:
    """Reject handles whose deployment was removed, blocked, or replaced."""
    from litellm.proxy import proxy_server as server

    deployment: Final = _pinned(handle)
    router = server.llm_router
    if router is None or deployment.model_id is None:
        raise HTTPException(410, "Live session deployment is no longer available")
    configured_raw = router.get_deployment(model_id=deployment.model_id)
    if configured_raw is None:
        raise HTTPException(410, "Live session deployment is no longer available")
    configured_data: Final = (
        configured_raw.model_dump()
        if hasattr(configured_raw, "model_dump")
        else vars(configured_raw)
        if not isinstance(configured_raw, Mapping)
        else configured_raw
    )
    configured: Final = _MAPPING.validate_python(configured_data)
    model_info: Final = _MAPPING.validate_python(configured.get("model_info", _EMPTY))
    if model_info.get("blocked") is True:
        raise HTTPException(410, "Live session deployment is no longer available")
    params: Final = _object(configured["litellm_params"])
    qualified: Final = str(params.get("model", ""))
    prefix, _, suffix = qualified.partition("/")
    provider: Final = prefix if prefix in ("openai", "chatgpt") else "openai"
    upstream: Final = suffix if prefix in ("openai", "chatgpt") else qualified
    if any(
        (
            deployment.model != upstream,
            deployment.provider != provider,
            str(model_info.get("id")) != deployment.model_id,
            params.get("api_base") != deployment.api_base,
            params.get("api_key") != deployment.api_key,
            (params.get("extra_headers") or _EMPTY) != deployment.extra_headers,
            (params.get("extra_query") or _EMPTY) != deployment.extra_query,
        )
    ):
        raise HTTPException(410, "Live session deployment is no longer available")
    return deployment


def _new_handle(
    session_id: str,
    alias: str,
    deployment: LiveDeployment,
    auth: UserAPIKeyAuth,
    lease: RealtimeCallLease | None,
    initialization_seconds: float = 0,
    policy: Mapping[str, JsonValue] | None = None,
) -> LiveHandle:
    live_session_path(session_id, "attach")
    routing: Final = _object(
        MappingProxyType(
            {
                "model": deployment.model,
                "provider": deployment.provider,
                "model_id": deployment.model_id,
                "api_base": deployment.api_base,
                "api_key": deployment.api_key,
                "extra_headers": deployment.extra_headers,
                "extra_query": deployment.extra_query,
            }
        )
    )
    return LiveHandle(
        session_id=session_id,
        alias=alias,
        deployment=routing,
        owner=_owner(auth),
        expires_at=time.time() + 30 * 86400,
        parallel_reserved=lease is not None,
        initialization_seconds=initialization_seconds,
        policy=policy or _EMPTY,
    )


def _session_id(payload: Mapping[str, JsonValue]) -> str:
    session: Final = payload.get("session")
    if isinstance(session, Mapping) and isinstance(session.get("id"), str):
        return TypeAdapter(str).validate_python(session["id"])
    raise HTTPException(502, "Upstream did not return a Live session ID")


class _BudgetOwnership:
    def __init__(self, auth: UserAPIKeyAuth) -> None:
        self.auth = auth
        self.transferred = False

    def replace_auth(self, auth: UserAPIKeyAuth) -> None:
        self.auth = auth


@asynccontextmanager
async def _budget_scope(auth: UserAPIKeyAuth) -> AsyncGenerator[_BudgetOwnership]:
    ownership: Final = _BudgetOwnership(auth)
    try:
        yield ownership
    finally:
        if not ownership.transferred:
            await release_or_invalidate_budget_reservation(budget_reservation=ownership.auth.budget_reservation)


async def _reauth(ownership: _BudgetOwnership, request: Request, body: Mapping[str, JsonValue], model: str) -> None:
    await release_or_invalidate_budget_reservation(budget_reservation=ownership.auth.budget_reservation)
    ownership.replace_auth(await _auth(_request(request, MappingProxyType({**body, "model": model}))))


def _policy_object(value: object) -> Mapping[str, JsonValue]:
    try:
        return _object(value)
    except ValueError as exc:
        raise HTTPException(400, "Live session delegation and responses must be JSON objects") from exc


def _policy_body(body: Mapping[str, JsonValue], source: LiveHandle | None) -> Mapping[str, JsonValue]:
    current: Final = _policy_object(body.get("session", _EMPTY))
    inherited: Final = source.policy if source is not None else _EMPTY
    parent_delegation: Final = _policy_object(inherited.get("delegation") or _EMPTY)
    child_delegation: Final = _policy_object(current.get("delegation") or _EMPTY)
    merged_delegation: Final = MappingProxyType(
        {
            **parent_delegation,
            **child_delegation,
            "responses": MappingProxyType(
                {
                    **_policy_object(parent_delegation.get("responses") or _EMPTY),
                    **_policy_object(child_delegation.get("responses") or _EMPTY),
                }
            ),
        }
    )
    return _policy_object(
        MappingProxyType(
            {
                **body,
                "session": MappingProxyType(
                    {
                        **inherited,
                        **current,
                        **(
                            MappingProxyType({"delegation": merged_delegation})
                            if parent_delegation or child_delegation
                            else _EMPTY
                        ),
                    }
                ),
            }
        )
    )


def _session_policy(body: Mapping[str, JsonValue], source: LiveHandle | None) -> Mapping[str, JsonValue]:
    session: Final = _object(_policy_body(body, source)["session"])
    delegation: Final = _object(session.get("delegation") or _EMPTY)
    responses: Final = _object(delegation.get("responses") or _EMPTY)
    return _object(
        MappingProxyType(
            {
                **(
                    MappingProxyType(
                        {
                            "delegation": MappingProxyType(
                                {
                                    "type": delegation.get("type"),
                                    "responses": MappingProxyType({"model": responses.get("model")}),
                                }
                            )
                        }
                    )
                    if delegation.get("type") == "responses" or responses.get("model")
                    else _EMPTY
                ),
                **(MappingProxyType({"client": session["client"]}) if "client" in session else _EMPTY),
            }
        )
    )


def _managed_constraints(auth: UserAPIKeyAuth) -> bool:
    from litellm.proxy import proxy_server as server

    if any(
        value is not None
        for value in (
            auth.rpm_limit,
            auth.tpm_limit,
            auth.team_rpm_limit,
            auth.team_tpm_limit,
            auth.user_rpm_limit,
            auth.user_tpm_limit,
            auth.organization_rpm_limit,
            auth.organization_tpm_limit,
            auth.team_member_rpm_limit,
            auth.team_member_tpm_limit,
            auth.end_user_rpm_limit,
            auth.end_user_tpm_limit,
            auth.max_parallel_requests,
            auth.max_budget,
            auth.team_max_budget,
            auth.user_max_budget,
            auth.end_user_max_budget,
            auth.organization_max_budget,
        )
    ):
        return True

    # An admin-configured proxy-wide concurrency cap admits every key through the limiter,
    # and delegated backend invocations never reach that admission, so it constrains the key
    # the same way a key-level `max_parallel_requests` does.
    if (
        _MAPPING.validate_python(getattr(server, "general_settings", None) or _EMPTY).get(
            "global_max_parallel_requests"
        )
        is not None
    ):
        return True

    direct_maps: Final[tuple[object, ...]] = (
        _OBJECT_VALUE.validate_python(getattr(auth, "model_max_budget", None)),
        _OBJECT_VALUE.validate_python(getattr(auth, "user_model_max_budget", None)),
        _OBJECT_VALUE.validate_python(getattr(auth, "end_user_model_max_budget", None)),
        _OBJECT_VALUE.validate_python(getattr(auth, "rpm_limit_per_model", None)),
        _OBJECT_VALUE.validate_python(getattr(auth, "tpm_limit_per_model", None)),
        _OBJECT_VALUE.validate_python(getattr(auth, "budget_limits", None)),
    )
    if any(_nonempty_limit_value(value) for value in direct_maps):
        return True

    pending: Final[list[object]] = [  # mutable-ok: explicit metadata traversal stack
        value
        for value in (
            _OBJECT_VALUE.validate_python(getattr(auth, "team_metadata", None)),
            _OBJECT_VALUE.validate_python(getattr(auth, "metadata", None)),
            _OBJECT_VALUE.validate_python(getattr(auth, "organization_metadata", None)),
            _OBJECT_VALUE.validate_python(getattr(auth, "project_metadata", None)),
        )
        if isinstance(value, (Mapping, list, tuple))
    ]
    visited: Final[set[int]] = set()
    while pending:
        current: object = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if len(visited) > 4096:
            return True
        if isinstance(current, Mapping):
            entries: Mapping[str, object] = _MAPPING.validate_python(current)
            for key, item in entries.items():
                if key in (
                    "rpm_limit",
                    "tpm_limit",
                    "max_budget",
                    "model_rpm_limit",
                    "model_tpm_limit",
                    "model_itpm_limit",
                    "model_otpm_limit",
                    "model_max_budget",
                    "budget_limits",
                ) and _nonempty_limit_value(item):
                    return True
                if isinstance(item, (Mapping, list, tuple)):
                    nested: object = _OBJECT_VALUE.validate_python(item)
                    pending.append(nested)
        elif isinstance(current, (list, tuple)):
            sequence: object = _OBJECT_VALUE.validate_python(current)
            pending.extend(_SEQUENCE.validate_python(sequence))
    return False


def _nonempty_limit_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        mapping: Final[object] = _OBJECT_VALUE.validate_python(value)
        return bool(_MAPPING.validate_python(mapping))
    if isinstance(value, (list, tuple)):
        sequence: Final[object] = _OBJECT_VALUE.validate_python(value)
        return bool(_SEQUENCE.validate_python(sequence))
    return True


def _restricted_model_list(models: object) -> bool:
    values: Final = _MODEL_NAMES.validate_python(models or ())
    return bool(values) and "*" not in values and "all-proxy-models" not in values


def _restricted_models(auth: UserAPIKeyAuth) -> bool:
    if _restricted_model_list(getattr(auth, "models", None)) or _restricted_model_list(
        getattr(auth, "team_models", None)
    ):
        return True
    if auth.user_id is not None and auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        return True
    return bool(
        auth.access_group_ids or auth.matched_model_access_groups or auth.project_id or auth.org_id or auth.team_id
    )


def _live_budget_configured(value: object, zero_is_limit: bool) -> bool:
    if value is None:
        return False
    value_object: Final[object] = value
    value_mapping: Final[Mapping[str, object] | None] = (
        _MAPPING.validate_python(value) if isinstance(value, Mapping) else None
    )
    max_budget: Final[object] = _OBJECT_VALUE.validate_python(
        value_mapping.get("max_budget") if value_mapping is not None else getattr(value_object, "max_budget", None)
    )
    if max_budget is not None:
        if zero_is_limit or (isinstance(max_budget, (int, float)) and max_budget > 0):
            return True
        if not isinstance(max_budget, (int, float)):
            return True
    fields: Final[tuple[object, ...]] = tuple(
        _OBJECT_VALUE.validate_python(
            value_mapping.get(field) if value_mapping is not None else getattr(value_object, field, None)
        )
        for field in ("rpm_limit", "tpm_limit", "model_max_budget", "max_parallel_requests")
    )
    return any(_nonempty_limit_value(field) for field in fields)


def _live_budget_scope_present(auth: UserAPIKeyAuth, model: str | None, llm_router: object | None) -> bool:
    explicit_model_scope: Final = _restricted_model_list(getattr(auth, "models", None)) or _restricted_model_list(
        getattr(auth, "team_models", None)
    )
    model_group_lookup_scope: Final = bool(
        model is not None and llm_router is not None and (explicit_model_scope or auth.org_id is not None)
    )
    return bool(
        auth.team_id
        or auth.project_id
        or auth.access_group_ids
        or auth.matched_model_access_groups
        or model_group_lookup_scope
    )


async def _live_team_membership(auth: UserAPIKeyAuth) -> object | None:
    from litellm.proxy import proxy_server as server

    if auth.team_id is None or auth.user_id is None:
        return None
    return await get_team_membership(
        user_id=auth.user_id,
        team_id=auth.team_id,
        prisma_client=server.prisma_client,
        user_api_key_cache=server.user_api_key_cache,
        proxy_logging_obj=server.proxy_logging_obj,
    )


async def _live_cached_object(
    *,
    key: str,
    model_type: type[_CACHEABLE_MODEL],
    load: Callable[[], Awaitable[_CACHEABLE_MODEL | None]],
) -> _CACHEABLE_MODEL | None:
    """Read one management object through the proxy cache, storing the row when the read misses.

    ``auth_checks`` already caches these rows for the chat path, but the two getters this gate
    would use are unusable there: ``get_team_object`` reports every failed database read as an
    HTTP 404, and ``get_team_member_default_budget`` returns ``None`` when its read raises. Both
    turn an outage into "no limit configured", and this gate answers that question by allowing
    managed delegation, so an unreadable limit has to stay an error. The cache key and TTL stay
    the shared ones, so the entry is still written, read, and invalidated like any other.
    """
    from litellm.proxy import proxy_server as server

    cached: Final = await server.user_api_key_cache.async_get_cache(key=key, model_type=model_type)
    if cached is not None:
        return cached
    loaded: Final = await load()
    if loaded is not None:
        await server.user_api_key_cache.async_set_cache(
            key=key,
            value=loaded,
            model_type=model_type,
            ttl=get_management_object_ttl(server.user_api_key_cache),
        )
    return loaded


async def _live_team(auth: UserAPIKeyAuth) -> LiteLLM_TeamTable | None:
    from litellm.proxy import proxy_server as server

    if auth.team_id is None:
        return None
    team_id: Final = auth.team_id
    cached: Final = await _get_team_object_from_cache(
        key=f"team_id:{team_id}",
        user_api_key_cache=server.user_api_key_cache,
        parent_otel_span=None,
    )
    if cached is not None:
        return cached

    row: Final = await TeamRepository(server.prisma_client).find_by_id(team_id, id_field="team_id")
    if row is None:
        return None
    team: Final = LiteLLM_TeamTableCachedObj.model_validate(row.model_dump())
    if team.object_permission_id and not team.object_permission:
        # The entry is written under the key the chat path reads, so it has to carry the same
        # permission relation the chat path caches; a cache hit elsewhere must not see a team
        # stripped of the permissions it was about to enforce.
        try:
            team.object_permission = await get_object_permission(
                object_permission_id=team.object_permission_id,
                prisma_client=server.prisma_client,
                user_api_key_cache=server.user_api_key_cache,
                parent_otel_span=None,
                proxy_logging_obj=server.proxy_logging_obj,
            )
        except Exception as exc:  # noqa: BLE001  # same degradation as the chat path: cache the team without permissions and log it
            verbose_proxy_logger.debug("Failed to load object_permission for Live team %s: %s", team_id, exc)
    await _cache_team_object(
        team_id=team_id,
        team_table=team,
        user_api_key_cache=server.user_api_key_cache,
        proxy_logging_obj=server.proxy_logging_obj,
    )
    return team


def _live_team_budget_configured(auth: UserAPIKeyAuth, team: LiteLLM_TeamTable | None) -> bool:
    if team is None:
        return False
    team_budget_limits: Final = getattr(team, "budget_limits", None)
    if _nonempty_limit_value(team_budget_limits):
        return True
    if any(getattr(team, field, None) is not None for field in ("rpm_limit", "tpm_limit", "max_budget")):
        return True
    if _nonempty_limit_value(getattr(team, "model_max_budget", None)):
        return True
    team_metadata_value: Final = getattr(team, "metadata", None)
    team_metadata: Final = (
        team_metadata_value if team_metadata_value is not None else getattr(auth, "team_metadata", None)
    )
    return _managed_constraints(
        auth.model_copy(update=MappingProxyType({"team_metadata": team_metadata, "budget_limits": team_budget_limits}))
    )


async def _live_default_budget(auth: UserAPIKeyAuth, team: LiteLLM_TeamTable | None) -> LiteLLM_BudgetTable | None:
    metadata_source: Final = (
        getattr(team, "metadata", None) if team is not None else getattr(auth, "team_metadata", None)
    )
    default_id: Final = _MAPPING.validate_python(metadata_source or _EMPTY).get("team_member_budget_id")
    if not isinstance(default_id, str) or auth.team_id is None or auth.user_id is None:
        return None
    from litellm.proxy import proxy_server as server

    async def load() -> LiteLLM_BudgetTable | None:
        return await BudgetRepository(server.prisma_client).find_by_id(default_id, id_field="budget_id")

    return await _live_cached_object(
        key=f"team_member_default_budget:{default_id}",
        model_type=LiteLLM_BudgetTable,
        load=load,
    )


async def _live_project(auth: UserAPIKeyAuth) -> LiteLLM_ProjectTableCachedObj | None:
    from litellm.proxy import proxy_server as server

    if auth.project_id is None:
        return None
    return await get_project_object(
        project_id=auth.project_id,
        prisma_client=server.prisma_client,
        user_api_key_cache=server.user_api_key_cache,
        proxy_logging_obj=server.proxy_logging_obj,
    )


async def _live_project_budget_configured(auth: UserAPIKeyAuth, project: LiteLLM_ProjectTableCachedObj | None) -> bool:
    if project is None:
        return False
    project_budget: Final = getattr(project, "litellm_budget_table", None)
    if _live_budget_configured(project_budget, zero_is_limit=True):
        return True
    if _nonempty_limit_value(getattr(project, "model_rpm_limit", None)) or _nonempty_limit_value(
        getattr(project, "model_tpm_limit", None)
    ):
        return True
    project_metadata_value: Final = getattr(project, "metadata", None)
    project_metadata: Final = (
        project_metadata_value if project_metadata_value is not None else getattr(auth, "project_metadata", None)
    )
    return _managed_constraints(auth.model_copy(update=MappingProxyType({"project_metadata": project_metadata})))


def _live_group_limits(row: object) -> LiteLLM_BudgetTable:
    """The limit fields of the budget linked to one model access group row.

    The row arrives as a Prisma join, so the limits are read by name. A group with no linked
    budget yields an empty budget table: it reads as no limit, which is what the gate needs, and
    it stays cacheable so the group is not re-read on every request.
    """
    budget: Final = getattr(row, "litellm_budget_table", None)
    if budget is None:
        return LiteLLM_BudgetTable()
    return LiteLLM_BudgetTable.model_validate(
        {  # mutable-ok: field values are read from the joined row into a fresh validation mapping
            field: getattr(budget, field, None)
            for field in ("max_budget", "rpm_limit", "tpm_limit", "model_max_budget", "max_parallel_requests")
        }
    )


async def _live_fetch_group_limits(groups: tuple[str, ...]) -> tuple[LiteLLM_BudgetTable, ...]:
    """Fetch the linked budget of each group in one query and cache one entry per group."""
    if not groups:
        return ()
    from litellm.proxy import proxy_server as server

    rows: Final = await ModelAccessGroupBudgetRepository(server.prisma_client).table.find_many(
        where={  # mutable-ok: Prisma serializes query filters from concrete dictionaries
            "access_group_name": {  # mutable-ok: Prisma serializes nested filters from concrete dictionaries
                "in": list(groups),
            }
        },
        include={"litellm_budget_table": True},  # mutable-ok: Prisma serializes concrete include dictionaries
    )
    linked: Final = MappingProxyType({getattr(row, "access_group_name", None): _live_group_limits(row) for row in rows})
    limits: Final = tuple(linked.get(group) or LiteLLM_BudgetTable() for group in groups)
    await asyncio.gather(
        *(
            server.user_api_key_cache.async_set_cache(
                key=live_model_access_group_limits_cache_key(group),
                value=limit,
                model_type=LiteLLM_BudgetTable,
                ttl=get_management_object_ttl(server.user_api_key_cache),
            )
            for group, limit in zip(groups, limits)
        )
    )
    return limits


async def _live_model_group_limits(groups: tuple[str, ...]) -> tuple[LiteLLM_BudgetTable, ...]:
    """One cached budget entry per group, served from a single row batch on a cold miss."""
    from litellm.proxy import proxy_server as server

    cached: Final = await asyncio.gather(
        *(
            server.user_api_key_cache.async_get_cache(
                key=live_model_access_group_limits_cache_key(group),
                model_type=LiteLLM_BudgetTable,
            )
            for group in groups
        )
    )
    uncached: Final = tuple(group for group, entry in zip(groups, cached) if entry is None)
    fetched: Final = MappingProxyType(dict(zip(uncached, await _live_fetch_group_limits(uncached))))
    return tuple(entry if entry is not None else fetched[group] for group, entry in zip(groups, cached))


async def _live_model_group_budget_configured(
    auth: UserAPIKeyAuth,
    model: str | None,
    team: LiteLLM_TeamTable | None,
    project: LiteLLM_ProjectTableCachedObj | None,
) -> bool:
    if model is None:
        return False
    from litellm.proxy import proxy_server as server

    matched_groups: Final = await collect_matched_model_access_groups(
        model=model,
        valid_token=auth,
        team_object=team,
        project_object=project,
        llm_router=server.llm_router,
        prisma_client=server.prisma_client,
        user_api_key_cache=server.user_api_key_cache,
        proxy_logging_obj=server.proxy_logging_obj,
        strict_grant_lookup=True,
    )
    if not matched_groups:
        return False
    # The shared group-budget helper flattens the row down to spend and max_budget, which would
    # drop the rpm and tpm limits this gate exists to refuse, so the linked row is read in full.
    limits: Final = await _live_model_group_limits(matched_groups)
    return any(_live_budget_configured(limit, zero_is_limit=False) for limit in limits)


async def _managed_member_budget(auth: UserAPIKeyAuth, model: str | None = None) -> bool:
    from litellm.proxy import proxy_server as server

    try:
        if not _live_budget_scope_present(auth, model, server.llm_router):
            return False
        if server.prisma_client is None:
            raise HTTPException(503, "Could not verify Live managed budgets")
        membership: Final = await _live_team_membership(auth)
        if _live_budget_configured(getattr(membership, "litellm_budget_table", None), zero_is_limit=True):
            return True
        team: Final = await _live_team(auth)
        if _live_team_budget_configured(auth, team):
            return True
        default: Final = await _live_default_budget(auth, team)
        if _live_budget_configured(default, zero_is_limit=False):
            return True
        project: Final = await _live_project(auth)
        if await _live_project_budget_configured(auth, project):
            return True
        return await _live_model_group_budget_configured(auth, model, team, project)
    except Exception as exc:
        raise HTTPException(503, "Could not verify Live managed budgets") from exc


async def _authorize_delegation(body: Mapping[str, JsonValue], auth: UserAPIKeyAuth) -> None:
    session: Final = body.get("session")
    if not isinstance(session, Mapping):
        return
    delegation: Final = session.get("delegation")
    if not isinstance(delegation, dict) or delegation.get("type") == "client":
        return
    responses: Final = delegation.get("responses")
    if delegation.get("type") != "responses" and not isinstance(responses, dict):
        return
    model: Final = responses.get("model") if isinstance(responses, dict) else None
    if _managed_constraints(auth) or await _managed_member_budget(
        auth,
        model if isinstance(model, str) else None,
    ):
        raise HTTPException(
            400,
            "Managed Live delegation cannot enforce configured budgets or rate limits; use client delegation",
        )
    if not isinstance(responses, dict):
        if body.get("type") != "session.update" and _restricted_models(auth):
            raise HTTPException(400, "Restricted keys require an explicit authorized delegation.responses.model")
        return
    if isinstance(model, str):
        await _authorize(model, auth)
    elif body.get("type") != "session.update" and _restricted_models(auth):
        raise HTTPException(400, "Restricted keys require an explicit authorized delegation.responses.model")

    transport: Final = body.get("transport")
    if isinstance(transport, dict) and transport.get("type") == "webrtc" and _restricted_models(auth):
        client: Final = session.get("client")
        channel: Final = client.get("data_channel") if isinstance(client, dict) else None
        events: Final = channel.get("allowed_client_events") if isinstance(channel, dict) else None
        if not isinstance(events, list) or any(
            not isinstance(event, str) or event.strip() == "session.update" or "*" in event for event in events
        ):
            raise HTTPException(
                400, "Restricted keys must explicitly exclude session.update from WebRTC allowed_client_events"
            )


async def _authorize_fork_policy(
    body: Mapping[str, JsonValue], source: LiveHandle | None, auth: UserAPIKeyAuth
) -> None:
    if (
        source is not None
        and source.policy.get("delegation")
        and (_restricted_models(auth) or _managed_constraints(auth))
    ):
        # Handles contain startup policy; later sideband or WebRTC updates can change the backend model.
        session: Final = _policy_object(body.get("session", _EMPTY))
        delegation: Final = _policy_object(session.get("delegation") or _EMPTY)
        responses: Final = _policy_object(delegation.get("responses") or _EMPTY)
        if delegation.get("type") != "client" and not (
            delegation.get("type") == "responses" and isinstance(responses.get("model"), str) and responses.get("model")
        ):
            raise HTTPException(
                400, "Constrained-key forks require explicit client delegation or an authorized responses model"
            )
    await _authorize_delegation(_policy_body(body, source), auth)


class _Prepared:
    def __init__(
        self,
        processed: Mapping[str, object],
        logger: Logging,
        lease: RealtimeCallLease | None,
        ownership: _BudgetOwnership | None = None,
    ) -> None:
        self.processed = processed
        self.logger = logger
        self.lease = lease
        self.transferred = False
        self.ownership = ownership

    def transfer(self) -> None:
        self.transferred = True
        if self.ownership is not None:
            self.ownership.transferred = True


class _PrecallState:
    def __init__(self) -> None:
        self.prepared: _Prepared | None = None
        self.lease: RealtimeCallLease | None = None


@asynccontextmanager
async def _precall(
    request: Request,
    auth: UserAPIKeyAuth,
    model: str,
    *,
    attachment: object | None = None,
    parallel_reserved: bool = False,
    ownership: _BudgetOwnership | None = None,
) -> AsyncGenerator[_Prepared]:
    from litellm.proxy import proxy_server as server

    limiter: Final = server.proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
    with isolated_request_stash():
        state: Final = _PrecallState()
        signaling_auth: Final = auth.model_copy(update=MappingProxyType({"budget_reservation": None}))
        try:
            await _authorize(model, auth)
            if isinstance(limiter, _PROXY_MaxParallelRequestsHandler) and (
                auth.max_parallel_requests is not None
                or _MAPPING.validate_python(server.general_settings).get("global_max_parallel_requests")  # pyright: ignore[reportUnknownMemberType]  # legacy settings are validated here is not None
            ):
                raise HTTPException(400, "Live requires the V3 rate limiter")
            payload: Final = _OBJECT.validate_json(await request.body())
            await _authorize_delegation(payload, auth)
            data: Final = MappingProxyType(
                {key: value for key, value in payload.items() if key in ("session", "transport")}
            )
            with (
                realtime_call_attachment(attachment) if attachment is not None and parallel_reserved else nullcontext()
            ):
                processed, logger = await process_codex_request(
                    request,
                    _mutable(
                        MappingProxyType(
                            {
                                **data,
                                "model": model,
                                **(MappingProxyType({"websocket": attachment}) if attachment is not None else _EMPTY),
                            }
                        )
                    ),
                    signaling_auth,
                    model,
                    "_arealtime" if attachment is not None else "arealtime_calls",
                )
            if attachment is None and isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3):
                state.lease = limiter.transfer_realtime_call_slot(processed)
                if state.lease is not None:
                    state.lease.start()
                    if not await state.lease.renew():
                        raise HTTPException(503, "Live quota reservation was lost")
            await _authorize_delegation(_processed_body(payload, processed), auth)
            state.prepared = _Prepared(processed, logger, state.lease, ownership)
            yield state.prepared
        finally:
            try:
                if state.prepared is None or not state.prepared.transferred:
                    if state.lease is not None:
                        await state.lease.close()
                    if ownership is None:
                        await release_or_invalidate_budget_reservation(budget_reservation=auth.budget_reservation)
            finally:
                if isinstance(limiter, _PROXY_MaxParallelRequestsHandler_v3):
                    await limiter.async_post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # hook accepts legacy mutable request data
                        request_data=_mutable(_EMPTY),
                        original_exception=Exception("Live signaling complete"),
                        user_api_key_dict=signaling_auth,
                    )


async def _supervise(
    request: Request, handle: LiveHandle, auth: UserAPIKeyAuth, logger: Logging, lease: RealtimeCallLease | None
) -> RealTimeStreaming:
    with isolated_request_stash():
        return await _start_supervisor(request, handle, auth, lease)


async def _start_supervisor(
    request: Request, handle: LiveHandle, auth: UserAPIKeyAuth, lease: RealtimeCallLease | None
) -> RealTimeStreaming:
    deployment: Final = _pinned(handle)
    transport: Final = LiveTransport(deployment, request.headers)
    state: Final = _ConnectionState()
    try:
        processed, logger = await process_codex_request(
            _request(request, MappingProxyType({"model": handle.alias})),
            _mutable(MappingProxyType({"model": handle.alias})),
            auth,
            handle.alias,
            "_arealtime",
            internal_realtime_observer=True,
        )
        import litellm

        metadata: Final = _mutable(
            MappingProxyType(
                {
                    **_MAPPING.validate_python(processed.get("litellm_metadata") or _EMPTY),
                    **(
                        MappingProxyType(
                            {
                                "model_info": _mutable(
                                    MappingProxyType(
                                        {
                                            **litellm.get_model_info(model=deployment.model_id),
                                            "id": deployment.model_id,
                                        }
                                    )
                                )
                            }
                        )
                        if deployment.model_id is not None
                        else _EMPTY
                    ),
                }
            )
        )
        pinned: Final = _mutable(MappingProxyType({**processed, "litellm_metadata": metadata}))
        logger.update_from_kwargs(  # pyright: ignore[reportUnknownMemberType]  # logging accepts legacy mutable provider parameters
            kwargs=pinned,
            model=deployment.model,
            user=None,
            optional_params=_mutable(_EMPTY),
            litellm_params=_mutable(
                MappingProxyType(
                    {
                        **_MAPPING.validate_python(logger.litellm_params),  # pyright: ignore[reportUnknownMemberType]  # legacy logging parameters are validated here
                        "litellm_metadata": metadata,
                        "arealtime": True,
                    }
                )
            ),
            custom_llm_provider=deployment.provider,
        )
        state.connection = await transport.connect(live_session_path(handle.session_id, "attach"))

        async def receive() -> Message:
            return _mutable(MappingProxyType({"type": "websocket.disconnect", "code": 1000}))

        async def send(message: Message) -> None:
            return None

        frontend: Final = WebSocket(
            _mutable(MappingProxyType({**request.scope, "type": "websocket"})), receive=receive, send=send
        )
        stream: Final = RealTimeStreaming(
            frontend,
            state.connection,
            logger,
            model=_pinned(handle).model,
            user_api_key_dict=auth,
            live_initialization_seconds=handle.initialization_seconds,
        )

        async def hangup() -> None:
            result: Final = await transport.request("POST", live_session_path(handle.session_id, "hangup"))
            result.raise_for_status()

        supervisor: Final = CallSupervisor(
            state.connection,
            stream,
            logger,
            auth,
            hangup,
            force_close_call=hangup,
            lease=lease,
            terminal_usage_required=True,
            connected_ready=True,
        )
        await CALL_SUPERVISORS.start(supervisor)
        return stream
    except BaseException:
        from litellm.proxy.spend_tracking.budget_reservation import (
            invalidate_budget_reservation_counters,  # pyright: ignore[reportUnknownVariableType]  # reservation helper has a legacy dict contract
        )

        try:
            result: Final = await transport.request("POST", live_session_path(handle.session_id, "hangup"))
            result.raise_for_status()
        except Exception:
            await invalidate_budget_reservation_counters(budget_reservation=auth.budget_reservation)
        finally:
            if state.connection is not None:
                await state.connection.close()
        raise


def _processed_body(body: Mapping[str, JsonValue], processed: Mapping[str, object]) -> Mapping[str, JsonValue]:
    return _object(
        MappingProxyType(
            {
                **body,
                **_object(
                    MappingProxyType(
                        {key: value for key, value in processed.items() if key in ("session", "transport")}
                    )
                ),
            }
        )
    )


def _provider_body(body: Mapping[str, JsonValue], model: str) -> Mapping[str, JsonValue]:
    session: Final = _object(body.get("session", _EMPTY))
    return _object(MappingProxyType({**body, "session": MappingProxyType({**session, "model": model})}))


def _response(response: httpx.Response, handle: LiveHandle | None = None) -> Response:
    if handle is None:
        return Response(
            response.content,
            status_code=response.status_code,
            headers=MappingProxyType(
                {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() in ("content-type", "content-disposition", "content-range", "accept-ranges")
                }
            ),
        )
    return Response(
        _encode_json(
            rewrite_session_ids(_OBJECT.validate_json(response.content), handle.session_id, encode_session(handle))
        ),
        status_code=response.status_code,
        media_type="application/json",
    )


async def _create(request: Request, token: str | None = None) -> Response:
    body: Final = await _body(request)
    auth: Final = await _auth(
        _request(request, _EMPTY if token else MappingProxyType({**body, "model": _session_model(body)}))
    )
    async with _budget_scope(auth) as ownership:
        source: Final = decode_session(token, _owner(auth)) if token else None
        model: Final = _session_model(body, source.alias if source else None)
        if source is not None:
            await _reauth(ownership, request, body, model)
        async with _precall(
            _request(request, MappingProxyType({**body, "model": model})), ownership.auth, model, ownership=ownership
        ) as prepared:
            await _authorize_fork_policy(_processed_body(body, prepared.processed), source, ownership.auth)
            deployment: Final = (
                _validate_pinned_deployment(source) if source else await _deployment(model, prepared.processed)
            )
            transport: Final = LiveTransport(deployment, request.headers)
            path: Final = live_session_path(source.session_id, "fork") if source else "live/sessions"
            response: Final = await transport.request(
                "POST",
                path,
                body=_processed_body(body, prepared.processed)
                if source
                else _provider_body(_processed_body(body, prepared.processed), deployment.model),
            )
            if response.is_error:
                return _response(response)
            handle: Final = _new_handle(
                _session_id(_OBJECT.validate_json(response.content)),
                model,
                deployment,
                ownership.auth,
                prepared.lease,
                initialization_seconds=15,
                policy=_session_policy(_processed_body(body, prepared.processed), source),
            )
            await _supervise(request, handle, ownership.auth, prepared.logger, prepared.lease)
            prepared.transfer()
            return _response(response, handle)


@_routes.post("/sessions")
async def create_live_session(request: Request) -> Response:
    return await _create(request)


@_routes.post("/sessions/{session_id}/fork")
async def fork_live_session(request: Request, session_id: str) -> Response:
    return await _create(request, session_id)


@_routes.get("/sessions/{session_id}/content")
@_routes.post("/sessions/{session_id}/accept")
@_routes.post("/sessions/{session_id}/reject")
@_routes.post("/sessions/{session_id}/refer")
@_routes.post("/sessions/{session_id}/hangup")
async def control_live_session(request: Request, session_id: str) -> Response:
    body: Final = await _body(request)
    auth: Final = await _auth(_request(request, _EMPTY))
    async with _budget_scope(auth) as ownership:
        operation: Final = TypeAdapter[LiveOperation](LiveOperation).validate_python(
            request.url.path.rsplit("/", 1)[-1]
        )
        if not session_id.startswith(_PREFIX):
            return await _incoming_sip(request, session_id, operation, body, auth, ownership)
        handle: Final = decode_session(session_id, _owner(auth))
        await _reauth(ownership, request, body, handle.alias)
        async with _precall(
            _request(request, MappingProxyType({"model": handle.alias})),
            ownership.auth,
            handle.alias,
            attachment=request,
            parallel_reserved=handle.parallel_reserved,
            ownership=ownership,
        ):
            response: Final = await LiveTransport(_pinned(handle), request.headers).request(
                request.method, live_session_path(handle.session_id, operation), body=body if body else None
            )
            return _response(response)


async def _incoming_sip(
    request: Request,
    session_id: str,
    operation: LiveOperation,
    body: Mapping[str, JsonValue],
    auth: UserAPIKeyAuth,
    ownership: _BudgetOwnership,
) -> Response:
    from litellm.proxy import proxy_server as server

    if auth.user_role != LitellmUserRoles.PROXY_ADMIN or operation not in ("accept", "reject"):
        raise HTTPException(403, "Incoming SIP enrollment requires a proxy administrator")
    alias: Final = request.headers.get("x-litellm-live-model")
    if not alias:
        raise HTTPException(400, "Incoming SIP requires x-litellm-live-model identifying one deployment")
    configured: Final = tuple(
        item
        for item in TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(
            getattr(server, "llm_model_list", ()) or ()
        )
        if item.get("model_name") == alias
    )
    if len(configured) != 1:
        raise HTTPException(400, "Incoming SIP requires a model alias with exactly one deployment")
    live_session_path(session_id, "attach")
    if operation == "accept" and _session_model(body) != alias:
        raise HTTPException(400, "session.model must match x-litellm-live-model")
    await _reauth(ownership, request, body, alias)
    async with _precall(
        _request(request, MappingProxyType({**body, "model": alias})), ownership.auth, alias, ownership=ownership
    ) as prepared:
        deployment: Final = await _deployment(alias, prepared.processed)
        response: Final = await LiveTransport(deployment, request.headers).request(
            "POST",
            live_session_path(session_id, operation),
            body=_provider_body(_processed_body(body, prepared.processed), deployment.model)
            if operation == "accept"
            else body,
        )
        if response.is_error or operation == "reject":
            return _response(response)
        handle: Final = _new_handle(
            session_id,
            alias,
            deployment,
            ownership.auth,
            prepared.lease,
            policy=_session_policy(_processed_body(body, prepared.processed), None),
        )
        await _supervise(request, handle, ownership.auth, prepared.logger, prepared.lease)
        prepared.transfer()
        return Response(
            response.content,
            status_code=response.status_code,
            headers=MappingProxyType({"x-litellm-live-session-id": encode_session(handle)}),
        )


class _PublicSocket:
    def __init__(
        self,
        websocket: WebSocket,
        handle: LiveHandle,
        public_id: str,
        auth: UserAPIKeyAuth,
        observer: RealTimeStreaming | None = None,
    ) -> None:
        self.websocket = websocket
        self.handle = handle
        self.public_id = public_id
        self.auth = auth
        self.observer = observer
        self.scope = websocket.scope
        self.headers = websocket.headers

    async def send_text(self, data: str) -> None:
        if self.observer is not None:
            self.observer.store_message(data)  # pyright: ignore[reportUnknownMemberType]  # stream also accepts legacy dict events
        await self.websocket.send_text(
            _encode_json(rewrite_session_ids(_OBJECT.validate_json(data), self.handle.session_id, self.public_id))
        )

    async def receive_text(self) -> str:
        data: Final = await self.websocket.receive_text()
        payload: Final = _OBJECT.validate_json(data)
        if payload.get("type") == "session.start":
            raise HTTPException(400, "Session has already started")
        session: Final = payload.get("session")
        if isinstance(session, Mapping) and "model" in session:
            raise HTTPException(400, "Session model cannot change")
        await _authorize_delegation(payload, self.auth)
        return _encode_json(rewrite_session_ids(payload, self.public_id, self.handle.session_id))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        await self.websocket.close(code=code, reason=reason)


class _StartupEvents:
    def __init__(self) -> None:
        self.messages: tuple[str, ...] = ()
        self.size = 0

    def store(self, event: Mapping[str, JsonValue]) -> None:
        message: Final = _encode_json(event)
        if len(self.messages) >= 128 or self.size + len(message) > 8 * 1024 * 1024:
            raise HTTPException(502, "Upstream exceeded the Live startup event limit")
        self.messages = (*self.messages, message)
        self.size += len(message)


async def _wait_started(
    connection: "ClientConnection", websocket: WebSocket, startup: _StartupEvents | None = None
) -> Mapping[str, JsonValue]:
    async def receive_started() -> Mapping[str, JsonValue]:
        while True:
            event: Mapping[str, JsonValue] = _OBJECT.validate_json(await connection.recv())
            if event.get("type") == "session.started":
                return event
            if startup is not None:
                startup.store(event)
            await websocket.send_json(event)
            if event.get("type") in ("error", "session.closed"):
                raise HTTPException(502, "Upstream did not start the session")

    return await asyncio.wait_for(receive_started(), 20)


@_routes.websocket("/sessions")
@_routes.websocket("/sessions/{session_id}/attach")
@_routes.websocket("/sessions/{session_id}/fork")
async def websocket_live_session(websocket: WebSocket, session_id: str | None = None) -> None:
    state: Final = _ConnectionState()
    try:
        api_key: Final = get_websocket_api_key(websocket)
        if not api_key:
            raise HTTPException(403, "API key required")
        auth_request: Final = _request(websocket, _EMPTY)
        inbound_headers: Final = TypeAdapter(tuple[tuple[bytes, bytes], ...]).validate_python(
            websocket.scope["headers"]
        )
        auth_request.scope["headers"] = (
            *(item for item in inbound_headers if item[0].lower() != b"authorization"),
            (b"authorization", f"Bearer {api_key}".encode()),
        )
        auth: Final = await _auth(auth_request)
        async with _budget_scope(auth) as ownership:
            source: Final = decode_session(session_id, _owner(auth)) if session_id else None
            attached: Final = source is not None and websocket.url.path.endswith("/attach")
            await websocket.accept()
            first: Final = (
                _EMPTY if attached else _OBJECT.validate_json(await asyncio.wait_for(websocket.receive_text(), 20))
            )
            if not attached and first.get("type") != "session.start":
                raise HTTPException(400, "First message must be session.start")
            model: Final = (
                source.alias
                if attached and source is not None
                else _session_model(first, source.alias if source else None)
            )
            await _reauth(ownership, auth_request, first, model)
            async with _precall(
                _request(websocket, MappingProxyType({**first, "model": model})),
                ownership.auth,
                model,
                attachment=websocket if attached else None,
                parallel_reserved=source.parallel_reserved if attached and source is not None else False,
                ownership=ownership,
            ) as prepared:
                if attached:
                    await _authorize_delegation(
                        _policy_body(_processed_body(first, prepared.processed), source), ownership.auth
                    )
                else:
                    await _authorize_fork_policy(_processed_body(first, prepared.processed), source, ownership.auth)
                deployment: Final = (
                    _validate_pinned_deployment(source) if source else await _deployment(model, prepared.processed)
                )
                path: Final = (
                    live_session_path(source.session_id, "attach" if attached else "fork")
                    if source
                    else "live/sessions"
                )
                state.connection = await LiveTransport(deployment, websocket.headers).connect(path)

                async def start_session() -> tuple[
                    LiveHandle, RealTimeStreaming | None, Mapping[str, JsonValue] | None
                ]:
                    if attached and source is not None:
                        return source, None, None
                    if state.connection is None:
                        raise RuntimeError("Live connection was not established")
                    await state.connection.send(
                        _encode_json(
                            _processed_body(first, prepared.processed)
                            if source
                            else _provider_body(_processed_body(first, prepared.processed), deployment.model)
                        )
                    )
                    startup: Final = _StartupEvents()
                    initial: Final = await _wait_started(state.connection, websocket, startup)
                    handle: Final = _new_handle(
                        _session_id(initial),
                        model,
                        deployment,
                        ownership.auth,
                        prepared.lease,
                        policy=_session_policy(_processed_body(first, prepared.processed), source),
                    )
                    observer: Final = await _supervise(
                        _request(websocket, MappingProxyType({"model": model})),
                        handle,
                        ownership.auth,
                        prepared.logger,
                        prepared.lease,
                    )
                    for buffered in startup.messages:
                        observer.store_message(buffered)  # pyright: ignore[reportUnknownMemberType]  # stream also accepts legacy dict events
                    prepared.transfer()
                    return handle, observer, initial

                handle, observer, initial = await start_session()
                public_id: Final = session_id if attached and session_id is not None else encode_session(handle)
                if initial is not None:
                    await websocket.send_json(rewrite_session_ids(initial, handle.session_id, public_id))
                frontend: Final = _PublicSocket(websocket, handle, public_id, ownership.auth, observer)
                stream: Final = RealTimeStreaming(
                    frontend,
                    state.connection,
                    prepared.logger,
                    model=deployment.model,
                    user_api_key_dict=auth,
                    request_data=_mutable(prepared.processed),
                    account_usage=False,
                )
                await stream.bidirectional_forward()
    except (HTTPException, ValueError, WebSocketDisconnect, asyncio.TimeoutError):
        try:
            await websocket.close(code=1008, reason="Live session rejected")
        except RuntimeError:
            # The peer may have closed the socket before the rejection response.
            return
    except Exception:
        try:
            await websocket.close(code=1011, reason="Live upstream connection failed")
        except RuntimeError:
            # The peer may have closed the socket before the failure response.
            return
    finally:
        if state.connection is not None:
            await state.connection.close()


router: Final = APIRouter()
for _prefix in ("/v1/live", "/live", "/openai/v1/live"):
    router.include_router(_routes, prefix=_prefix)
