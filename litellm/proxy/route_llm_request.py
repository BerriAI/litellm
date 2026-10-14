import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, repeat, takewhile
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.router_utils.common_utils import is_proxy_admin_request
from litellm.router_utils.fallback_event_handlers import (
    check_non_standard_fallback_format,
    fallback_lookup_groups,
    get_fallback_model_group_for_lookup_groups,
    has_unattempted_fallback_target,
)
from litellm.types.utils import COST_MAP_LOOKUP_KEY, echoed_cost_map_fields

# Client-supplied params that make the router or the call path fabricate a
# failure or a delay instead of calling the provider. The ``mock_testing_*``
# names are kept in sync with ``litellm.types.router.MockRouterTestingParams``
# by ``test_gated_mock_params_cover_mock_router_testing_params``. Hardcoding
# (rather than deriving via ``dataclasses.fields(MockRouterTestingParams)`` at
# import time) avoids a cyclic import: ``litellm.types.router`` imports
# back into proxy modules before this module finishes loading.
GATED_MOCK_PARAM_NAMES: Final[tuple[str, ...]] = (
    "mock_testing_fallbacks",
    "mock_testing_context_fallbacks",
    "mock_testing_content_policy_fallbacks",
    "mock_testing_rate_limit_error",
    "mock_timeout",
    "mock_delay",
)

MOCK_TESTING_CONFIG_KEY: Final = "dangerously_allow_mock_testing_request_params"

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def _route_user_config_request(data: dict, route_type: str):
    """Route a request using the user-provided router config."""
    router_config: Final = data.pop("user_config")

    # Filter router_config to only include valid Router.__init__ arguments
    # This prevents TypeError when invalid parameters are stored in the database
    valid_args: Final = litellm.Router.get_valid_args()
    filtered_config: Final = {k: v for k, v in router_config.items() if k in valid_args}

    user_router: Final = litellm.Router(**filtered_config)
    ret_val: Final = getattr(user_router, f"{route_type}")(**data)
    user_router.discard()
    return ret_val


def _is_a2a_agent_model(model_name: object) -> bool:
    """Check if the model name is for an A2A agent (a2a/ prefix)."""
    return isinstance(model_name, str) and model_name.startswith("a2a/")


def _raise_if_model_fully_blocked(llm_router: LitellmRouter, model_name: object, team_id: str | None) -> None:
    if not isinstance(model_name, str) or not model_name:
        return
    if not isinstance(llm_router, litellm.Router):
        return
    deployments: Final = llm_router.get_model_list(model_name=model_name, team_id=team_id) or []
    if llm_router.are_all_deployments_blocked(deployments):
        raise litellm.PermissionDeniedError(
            message="Model is blocked",
            model=model_name,
            llm_provider="",
            response=httpx.Response(
                status_code=403,
                request=httpx.Request(method="POST", url="https://github.com/BerriAI/litellm"),
            ),
        )


ROUTE_ENDPOINT_MAPPING: Final = {
    "acompletion": "/chat/completions",
    "atext_completion": "/completions",
    "aembedding": "/embeddings",
    "aimage_generation": "/image/generations",
    "aspeech": "/audio/speech",
    "atranscription": "/audio/transcriptions",
    "amoderation": "/moderations",
    "arerank": "/rerank",
    "aresponses": "/responses",
    "_aresponses_websocket": "/responses",
    "alist_input_items": "/responses/{response_id}/input_items",
    "aimage_edit": "/images/edits",
    "acancel_responses": "/responses/{response_id}/cancel",
    "acompact_responses": "/responses/compact",
    "aocr": "/ocr",
    "asearch": "/search",
    "adecisions": "/decisions",
    "asystemone": "/systemone",
    "avideo_generation": "/videos",
    "avideo_list": "/videos",
    "avideo_status": "/videos/{video_id}",
    "avideo_content": "/videos/{video_id}/content",
    "avideo_remix": "/videos/{video_id}/remix",
    "avideo_create_character": "/videos/characters",
    "avideo_get_character": "/videos/characters/{character_id}",
    "avideo_edit": "/videos/edits",
    "avideo_extension": "/videos/extensions",
    "acreate_realtime_client_secret": "/realtime/client_secrets",
    "arealtime_calls": "/realtime/calls",
    "acreate_realtime_transcription_session": "/realtime/transcription_sessions",
    "acreate_container": "/containers",
    "alist_containers": "/containers",
    "aretrieve_container": "/containers/{container_id}",
    "adelete_container": "/containers/{container_id}",
    # Auto-generated container file routes
    "aupload_container_file": "/containers/{container_id}/files",
    "alist_container_files": "/containers/{container_id}/files",
    "aretrieve_container_file": "/containers/{container_id}/files/{file_id}",
    "adelete_container_file": "/containers/{container_id}/files/{file_id}",
    "aretrieve_container_file_content": "/containers/{container_id}/files/{file_id}/content",
    "acreate_skill": "/skills",
    "alist_skills": "/skills",
    "aget_skill": "/skills/{skill_id}",
    "adelete_skill": "/skills/{skill_id}",
    "aingest": "/rag/ingest",
    # Google Interactions API routes
    "acreate_interaction": "/interactions",
    "aget_interaction": "/interactions/{interaction_id}",
    "adelete_interaction": "/interactions/{interaction_id}",
    "acancel_interaction": "/interactions/{interaction_id}/cancel",
    # Google Managed Agents API routes
    "acreate_agent": "/v1beta/agents",
    "alist_agents": "/v1beta/agents",
    "aget_agent": "/v1beta/agents/{name}",
    "adelete_agent": "/v1beta/agents/{name}",
    "alist_agent_versions": "/v1beta/agents/{name}/versions",
    # OpenAI Evals API routes
    "acreate_eval": "/evals",
    "alist_evals": "/evals",
    "aget_eval": "/evals/{eval_id}",
    "aupdate_eval": "/evals/{eval_id}",
    "adelete_eval": "/evals/{eval_id}",
    "acancel_eval": "/evals/{eval_id}/cancel",
    # OpenAI Evals Runs API routes
    "acreate_run": "/evals/{eval_id}/runs",
    "alist_runs": "/evals/{eval_id}/runs",
    "aget_run": "/evals/{eval_id}/runs/{run_id}",
    "acancel_run": "/evals/{eval_id}/runs/{run_id}/cancel",
    "adelete_run": "/evals/{eval_id}/runs/{run_id}",
    "acreate_batch": "/batches",
    "aretrieve_batch": "/batches",
}


_AVAILABLE_MODELS_HINT: Final = "Call `/v1/models` to view available models for your key."


class ProxyModelNotFoundError(HTTPException):
    def __init__(self, route: str, model_name: str, retryable_with_model_read_through: bool = True):
        self.retryable_with_model_read_through: Final = retryable_with_model_read_through
        self.spend_log_error_message: Final = f"{route}: Invalid model name passed in. {_AVAILABLE_MODELS_HINT}"
        detail: Final = {"error": f"{route}: Invalid model name passed in model={model_name}. {_AVAILABLE_MODELS_HINT}"}
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


REQUIRED_BODY_PARAMS_BY_ROUTE: Final[Mapping[str, tuple[str, ...]]] = {
    "acompletion": ("messages",),
    "aembedding": ("input",),
    "aresponses": ("input",),
    "acreate_batch": ("input_file_id", "endpoint", "completion_window"),
}

REQUIRED_PRESENT_BODY_PARAMS_BY_ROUTE: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "aspeech": ("input",),
        "amoderation": ("input",),
        "aimage_generation": ("prompt",),
        "asearch": ("query",),
        "atext_completion": ("prompt",),
        "atranscription": ("file",),
        "arerank": ("query", "documents"),
        "acompact_responses": ("input",),
        "anthropic_messages": ("messages", "max_tokens"),
        "agenerate_content": ("contents",),
        "aocr": ("document",),
        "avector_store_search": ("query",),
        "avector_store_file_create": ("file_id",),
        "avector_store_file_update": ("attributes",),
        "avideo_generation": ("prompt",),
        "avideo_remix": ("prompt",),
        "avideo_edit": ("prompt",),
        "avideo_extension": ("prompt", "seconds"),
        "avideo_create_character": ("name", "video"),
        "acreate_container": ("name",),
        "aupload_container_file": ("file",),
        "acreate_agent": ("name",),
        "acreate_interaction": ("input",),
        "acreate_eval": ("data_source_config", "testing_criteria"),
        "acreate_run": ("data_source",),
    }
)

REQUIRED_ONE_OF_BODY_PARAMS_BY_ROUTE: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {"acreate_interaction": ("model", "agent")}
)

JSON_OBJECT_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])


class ProxyMissingRequiredParamError(ProxyException):
    def __init__(self, route: str, param: str):
        super().__init__(
            message=f"{route}: Missing required parameter: '{param}'.",
            type="invalid_request_error",
            param=param,
            code=status.HTTP_400_BAD_REQUEST,
        )


class ProxyMissingParamWithoutLoadedModelError(ProxyMissingRequiredParamError):
    pass


@dataclass(frozen=True, slots=True)
class MissingBodyParam:
    name: str
    model_deployments_loaded: bool


def _find_missing_required_body_param(
    route_type: str,
    data: Mapping[str, object],
    llm_router: LitellmRouter | None,
) -> MissingBodyParam | None:
    one_of_params: Final = REQUIRED_ONE_OF_BODY_PARAMS_BY_ROUTE.get(route_type)
    if one_of_params is not None and all(data.get(param) is None for param in one_of_params):
        return MissingBodyParam(name=one_of_params[0], model_deployments_loaded=True)
    missing_merge_base_param: Final = next(
        (param for param in REQUIRED_BODY_PARAMS_BY_ROUTE.get(route_type, ()) if data.get(param) is None),
        None,
    )
    if missing_merge_base_param is not None:
        return MissingBodyParam(name=missing_merge_base_param, model_deployments_loaded=True)
    missing_present_params: Final = tuple(
        param for param in REQUIRED_PRESENT_BODY_PARAMS_BY_ROUTE.get(route_type, ()) if param not in data
    )
    if not missing_present_params:
        return None
    candidate_litellm_params: Final = _candidate_deployment_litellm_params(data, llm_router)
    router_default_litellm_params: Final = _router_default_litellm_params(
        route_type, data, llm_router, model_listed=bool(candidate_litellm_params)
    )
    missing_param: Final = next(
        (
            param
            for param in missing_present_params
            if router_default_litellm_params.get(param) is None
            and (
                route_type in _ROUTE_TYPES_WITH_POSITIONAL_REQUIRED_PARAM
                or not any(deployment_params.get(param) is not None for deployment_params in candidate_litellm_params)
            )
        ),
        None,
    )
    if missing_param is None:
        return None
    return MissingBodyParam(name=missing_param, model_deployments_loaded=bool(candidate_litellm_params))


# The Router method takes the route's required param positionally, so it binds before
# any default or deployment param is merged in, and only the request body can supply it.
_ROUTE_TYPES_WITH_POSITIONAL_REQUIRED_PARAM: Final[frozenset[str]] = frozenset(
    {"aimage_generation", "aspeech", "atext_completion", "atranscription"}
)

_ROUTE_TYPES_WITHOUT_ROUTER_DEFAULTS_MERGE: Final[frozenset[str]] = (
    frozenset({"asearch", "acreate_agent", "acreate_eval", "acreate_run"}) | _ROUTE_TYPES_WITH_POSITIONAL_REQUIRED_PARAM
)

# The Router merges defaults on these only when `get_model_list` lists the model.
_ROUTE_TYPES_MERGING_ONLY_LISTED_MODELS: Final[frozenset[str]] = frozenset({"amoderation"})


def _router_default_litellm_params(
    route_type: str,
    data: Mapping[str, object],
    llm_router: LitellmRouter | None,
    *,
    model_listed: bool,
) -> Mapping[str, object]:
    # Mirror exactly the defaults the dispatching router will merge at dispatch time:
    # the listed route types either never pass through the router's merge or take the
    # param positionally, which binds before the merge; user_config requests dispatch on
    # their own throwaway Router; model-less direct dispatch never merges; and moderation
    # merges only for a model `get_model_list` lists.
    if route_type in _ROUTE_TYPES_WITHOUT_ROUTER_DEFAULTS_MERGE:
        return {}
    user_config: Final[Mapping[str, object] | None] = (
        data.get("user_config") if isinstance(data.get("user_config"), Mapping) else None
    )
    if user_config is not None:
        defaults: Final[Mapping[str, object] | None] = user_config.get("default_litellm_params")
        return defaults if isinstance(defaults, Mapping) else {}
    model_name: Final = data.get("model")
    if (
        not isinstance(model_name, str)
        or not model_name
        or (route_type in _ROUTE_TYPES_MERGING_ONLY_LISTED_MODELS and not model_listed)
    ):
        return {}
    router_defaults: Final[Mapping[str, object] | None] = getattr(llm_router, "default_litellm_params", None)
    return router_defaults if isinstance(router_defaults, Mapping) else {}


def _candidate_deployment_litellm_params(
    data: Mapping[str, object],
    llm_router: LitellmRouter | None,
) -> tuple[dict[str, object], ...]:
    model_name: Final = data.get("model")
    if llm_router is None or not isinstance(model_name, str):
        return ()
    deployments: Final = (
        llm_router.get_model_list(
            model_name=model_name,
            team_id=get_team_id_from_data(dict(data)),
        )
        or ()
    )
    return tuple(
        params for deployment in deployments if (params := _validated_deployment_litellm_params(deployment)) is not None
    )


def _validated_deployment_litellm_params(deployment: Mapping[str, object]) -> dict[str, object] | None:
    try:
        return JSON_OBJECT_ADAPTER.validate_python(deployment.get("litellm_params"))
    except ValidationError:
        return None


def raise_if_required_body_param_missing(
    route_type: str,
    data: Mapping[str, object],
    llm_router: LitellmRouter | None,
) -> None:
    missing_param: Final = _find_missing_required_body_param(route_type, data, llm_router)
    if missing_param is None:
        return
    error_class: Final = (
        ProxyMissingRequiredParamError
        if missing_param.model_deployments_loaded
        else ProxyMissingParamWithoutLoadedModelError
    )
    raise error_class(
        route=ROUTE_ENDPOINT_MAPPING.get(route_type, route_type),
        param=missing_param.name,
    )


CHAT_SHAPED_ENDPOINT_BY_ROUTE_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "acompletion": "/v1/chat/completions",
        "atext_completion": "/v1/completions",
        "anthropic_messages": "/v1/messages",
        "aresponses": "/v1/responses",
        "agenerate_content": "generateContent",
        "agenerate_content_stream": "streamGenerateContent",
        "acreate_interaction": "/v1beta/interactions",
        "_aresponses_websocket": "/v1/responses",
        "acompact_responses": "/v1/responses/compact",
    }
)

CHAT_SHAPED_ENDPOINTS: Final[frozenset[str]] = frozenset(
    {
        "/chat/completions",
        "/completions",
        "/responses",
        *CHAT_SHAPED_ENDPOINT_BY_ROUTE_TYPE.values(),
    }
)

CHAT_SHAPED_MODES: Final[frozenset[str]] = frozenset({"chat", "completion", "responses"})


class _DeploymentEndpointDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: str | None = None
    supported_endpoints: tuple[str, ...] | None = None


class ProxyUnsupportedEndpointError(ProxyException):
    pass


_FALLBACK_LIST_ADAPTER: Final[TypeAdapter[list[object]]] = TypeAdapter(  # mutable-ok: router fallbacks are lists
    list[object], config=ConfigDict(strict=True)
)


class _RouterFallbacks(Protocol):
    @property
    def fallbacks(self) -> Sequence[object] | None: ...


def _loaded_catalog_entry(model_info: Mapping[str, object]) -> Mapping[str, object]:
    catalog_key: Final = model_info.get(COST_MAP_LOOKUP_KEY)
    if not isinstance(catalog_key, str):
        return MappingProxyType({})
    return GetModelCostMap.loaded_model_cost_map().get(catalog_key) or MappingProxyType({})


def _catalog_serves_chat(catalog_entry: Mapping[str, object]) -> bool:
    try:
        declared: Final = _DeploymentEndpointDeclaration.model_validate(dict(catalog_entry))
    except ValidationError:
        return False
    return (declared.mode or "").strip().lower() in CHAT_SHAPED_MODES or not CHAT_SHAPED_ENDPOINTS.isdisjoint(
        declared.supported_endpoints or ()
    )


def _declaration_overruled_by_catalog(model_info: Mapping[str, object]) -> bool:
    catalog_entry: Final = _loaded_catalog_entry(model_info)
    return _catalog_serves_chat(catalog_entry) or "supported_endpoints" in echoed_cost_map_fields(
        model_info, catalog_entry
    )


def _declared_non_chat_endpoints(deployment: Mapping[str, object]) -> frozenset[str] | None:
    try:
        model_info: Final = JSON_OBJECT_ADAPTER.validate_python(deployment.get("model_info"))
        declared: Final = _DeploymentEndpointDeclaration.model_validate(model_info)
    except ValidationError:
        return None
    mode: Final = (declared.mode or "").strip().lower()
    if not mode or mode in CHAT_SHAPED_MODES or declared.supported_endpoints is None:
        return None
    endpoints: Final = frozenset(declared.supported_endpoints)
    if not endpoints.isdisjoint(CHAT_SHAPED_ENDPOINTS):
        return None
    return None if _declaration_overruled_by_catalog(model_info) else endpoints


def _deployment_by_id(model_id: str, llm_router: LitellmRouter) -> tuple[Mapping[str, object], ...]:
    try:
        return (JSON_OBJECT_ADAPTER.validate_python(llm_router.get_model_info(id=model_id)),)
    except ValidationError:
        return ()


def _group_deployments(model: str, llm_router: LitellmRouter, team_id: str | None) -> tuple[Mapping[str, object], ...]:
    return tuple(llm_router.get_model_list(model_name=model, team_id=team_id) or ()) or _deployment_by_id(
        model, llm_router
    )


def _deployments_by_backend_name(model: str, llm_router: LitellmRouter) -> tuple[Mapping[str, object], ...]:
    if (
        model not in llm_router.deployment_names
        or llm_router.router_general_settings.pass_through_all_models
        or llm_router.pattern_router.patterns
    ):
        return ()
    return tuple(
        deployment
        for deployment in llm_router.get_model_list() or ()
        if deployment["litellm_params"].get("model") == model
    )


def _requested_deployments(
    model: str, llm_router: LitellmRouter, team_id: str | None
) -> tuple[Mapping[str, object], ...]:
    return _group_deployments(model, llm_router, team_id) or _deployments_by_backend_name(model, llm_router)


def _any_serves_chat(deployments: Sequence[Mapping[str, object]]) -> bool:
    return any(_declared_non_chat_endpoints(deployment) is None for deployment in deployments)


def _override_fallbacks(data: Mapping[str, object]) -> object:
    try:
        return JSON_OBJECT_ADAPTER.validate_python(data.get("router_settings_override")).get("fallbacks")
    except ValidationError:
        return None


def _fallbacks_for_request(data: Mapping[str, object], llm_router: LitellmRouter) -> object:
    if "fallbacks" in data:
        return data["fallbacks"]
    override_fallbacks: Final = _override_fallbacks(data)
    return override_fallbacks if override_fallbacks is not None else _router_fallbacks(llm_router)


def _router_fallbacks(llm_router: _RouterFallbacks) -> Sequence[object] | None:
    return llm_router.fallbacks


def _fallback_targets(
    model: str,
    data: Mapping[str, object],
    fallbacks: list[object],  # mutable-ok: mirrors get_fallback_model_group's contract
) -> Sequence[object]:
    if check_non_standard_fallback_format(fallbacks=fallbacks):
        return fallbacks
    resolved, _ = get_fallback_model_group_for_lookup_groups(
        fallbacks=fallbacks,
        lookup_groups=fallback_lookup_groups(data, model),
    )
    return resolved or ()


def _fallback_target_group(target: object) -> str | None:
    if isinstance(target, str):
        return target
    try:
        entry: Final = JSON_OBJECT_ADAPTER.validate_python(target)
    except ValidationError:
        return None
    named_model: Final = entry.get("model")
    return named_model if isinstance(named_model, str) else None


def _next_fallback_groups(
    model: str,
    data: Mapping[str, object],
    fallbacks: list[object],  # mutable-ok: mirrors get_fallback_model_group's contract
) -> tuple[str, ...]:
    return tuple(
        group
        for target in _fallback_targets(model, data, fallbacks)
        if has_unattempted_fallback_target((target,), data) and (group := _fallback_target_group(target)) is not None
    )


@dataclass(frozen=True, slots=True)
class _FallbackWalk:
    frontier: tuple[str, ...]
    seen: frozenset[str]
    reaches_chat: bool = False


def _walk_one_layer(
    walk: _FallbackWalk,
    data: Mapping[str, object],
    fallbacks: list[object],  # mutable-ok: mirrors get_fallback_model_group's contract
    llm_router: LitellmRouter,
    team_id: str | None,
) -> _FallbackWalk:
    reached: Final = dict.fromkeys(
        chain.from_iterable(_next_fallback_groups(m, data, fallbacks) for m in walk.frontier)
    )
    fresh: Final = tuple(group for group in reached if group not in walk.seen)
    deployments_by_group: Final = tuple((group, _group_deployments(group, llm_router, team_id)) for group in fresh)
    return _FallbackWalk(
        frontier=tuple(group for group, deployments in deployments_by_group if deployments),
        seen=walk.seen | frozenset(fresh),
        reaches_chat=any(deployments and _any_serves_chat(deployments) for _, deployments in deployments_by_group),
    )


def _fallbacks_reach_chat(
    model: str,
    data: Mapping[str, object],
    fallbacks: list[object],  # mutable-ok: mirrors get_fallback_model_group's contract
    llm_router: LitellmRouter,
    team_id: str | None,
) -> bool:
    walks: Final = accumulate(
        repeat(None),
        lambda walk, _: _walk_one_layer(walk, data, fallbacks, llm_router, team_id),
        initial=_FallbackWalk(frontier=(model,), seen=frozenset({model})),
    )
    return any(walk.reaches_chat for walk in takewhile(lambda walk: walk.frontier or walk.reaches_chat, walks))


def _fallback_would_answer(
    model: str,
    data: Mapping[str, object],
    llm_router: LitellmRouter,
    team_id: str | None,
) -> bool:
    if data.get("disable_fallbacks") is True:
        return False
    raw_fallbacks: Final = _fallbacks_for_request(data, llm_router)
    if raw_fallbacks is None:
        return False
    try:
        fallbacks: Final = _FALLBACK_LIST_ADAPTER.validate_python(raw_fallbacks)
    except ValidationError:
        return True
    return _fallbacks_reach_chat(model, data, fallbacks, llm_router, team_id)


def endpoint_unsupported_by_model(
    route_type: str,
    data: Mapping[str, object],
    llm_router: LitellmRouter | None,
    team_id: str | None,
) -> ProxyUnsupportedEndpointError | None:
    endpoint: Final = CHAT_SHAPED_ENDPOINT_BY_ROUTE_TYPE.get(route_type)
    model: Final = data.get("model")
    if endpoint is None or not isinstance(model, str) or not model or llm_router is None:
        return None
    deployments: Final = _requested_deployments(model, llm_router, team_id)
    if not deployments or _any_serves_chat(deployments):
        return None
    if _fallback_would_answer(model, data, llm_router, team_id):
        return None
    supported: Final = ", ".join(
        sorted(frozenset[str]().union(*(_declared_non_chat_endpoints(d) or frozenset() for d in deployments)))
    )
    return ProxyUnsupportedEndpointError(
        message=f"{endpoint} is not supported by model={model}. Supported endpoints: {supported or 'none'}",
        type="invalid_request_error",
        param="model",
        code=status.HTTP_400_BAD_REQUEST,
    )


class MockTestingParamsDisabledError(HTTPException):
    def __init__(self, params: tuple[str, ...]):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": (
                    f"Mock testing request params are disabled on this proxy: {', '.join(params)}. "
                    f"An admin can enable them by setting `general_settings.{MOCK_TESTING_CONFIG_KEY}: true` "
                    "in config.yaml. This setting cannot be changed from the Admin UI or the API."
                )
            },
        )


def raise_if_mock_testing_params_disallowed(data: Mapping[str, object], *, allowed: bool) -> None:
    """Reject client-supplied mock testing params unless an admin opted in.

    Rejecting (rather than silently dropping) keeps a request that asked for a
    synthetic failure from returning a normal success, which reads as a passing
    fallback test that never ran.
    """
    if allowed:
        return
    present: Final = tuple(name for name in GATED_MOCK_PARAM_NAMES if name in data)
    if present:
        raise MockTestingParamsDisabledError(params=present)


def mock_testing_params_allowed() -> bool:
    """Read the opt-in from the running proxy's ``general_settings``."""
    from litellm.proxy import proxy_server

    return proxy_server.general_settings.get(MOCK_TESTING_CONFIG_KEY, False) is True


def get_team_id_from_data(data: dict) -> str | None:
    """
    Get the team id from the data's metadata or litellm_metadata params.
    """
    if "metadata" in data and data["metadata"] is not None and "user_api_key_team_id" in data["metadata"]:
        return data["metadata"].get("user_api_key_team_id")
    elif (
        "litellm_metadata" in data
        and data["litellm_metadata"] is not None
        and "user_api_key_team_id" in data["litellm_metadata"]
    ):
        return data["litellm_metadata"].get("user_api_key_team_id")
    return None


_shared_session_lock: asyncio.Lock | None = None


def _get_shared_session_lock() -> asyncio.Lock:
    """Lazily create the shared session lock (must be called within a running event loop).

    WARNING: Do not reset _shared_session_lock to None while any coroutine may be
    executing the session-recovery path; doing so breaks the double-checked locking
    guarantee and can cause duplicate session creation.
    """
    global _shared_session_lock
    if _shared_session_lock is None:
        _shared_session_lock = asyncio.Lock()
    return _shared_session_lock


async def add_shared_session_to_data(data: dict) -> None:
    """
    Add shared aiohttp session for connection reuse (prevents cold starts).
    If the session was closed (e.g. due to network interruption or idle timeout),
    automatically recreates it so connection pooling is restored.
    Uses an asyncio.Lock to prevent race conditions where multiple concurrent
    requests could each create a new session, leaking intermediate ones.
    Silently continues without session reuse if import fails or session is unavailable.

    Args:
        data: Dictionary to add the shared session to
    """
    try:
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy import proxy_server

        session = proxy_server.shared_aiohttp_session

        if session is not None and not session.closed:
            data["shared_session"] = session
            verbose_proxy_logger.info("SESSION REUSE: Attached shared aiohttp session to request (ID: %s)", id(session))
        elif session is not None and session.closed:
            # Session was created at startup but has since closed — recreate it
            # Use lock to prevent concurrent recreation (avoids session/connector leak)
            lock: Final = _get_shared_session_lock()
            async with lock:
                # Double-check under lock — another coroutine may have already recreated it
                session = proxy_server.shared_aiohttp_session
                if session is not None and not session.closed:
                    data["shared_session"] = session
                    return

                # session could be None here (if another coroutine set it to None)
                # or closed — either way we need to recreate
                if session is not None:
                    verbose_proxy_logger.warning(
                        "SESSION REUSE: Shared aiohttp session is closed (ID: %s), recreating...", id(session)
                    )
                else:
                    verbose_proxy_logger.warning(
                        "SESSION REUSE: Shared aiohttp session is None after re-check, recreating..."
                    )
                try:
                    new_session = (  # rebind-ok: pre-existing rebinding on a rename-only line
                        await proxy_server.initialize_shared_aiohttp_session()
                    )
                except Exception:
                    verbose_proxy_logger.exception("SESSION REUSE: Exception during shared session recreation")
                    new_session = None
                if new_session is not None:
                    proxy_server.shared_aiohttp_session = new_session
                    data["shared_session"] = new_session
                else:
                    verbose_proxy_logger.info(
                        "SESSION REUSE: Failed to recreate shared session, continuing without session reuse"
                    )
        else:
            verbose_proxy_logger.info("SESSION REUSE: No shared session available for this request")
    except Exception:
        # Continue without session reuse — this outer handler covers import failures
        # and other unexpected errors to avoid breaking the request path.
        # Inner recovery logic has its own specific exception handling.
        try:
            from litellm._logging import verbose_proxy_logger

            verbose_proxy_logger.debug(
                "SESSION REUSE: Unexpected error in session setup, continuing without reuse",
                exc_info=True,
            )
        except Exception:
            pass


RouteType = Literal[
    "acompletion",
    "atext_completion",
    "aembedding",
    "aimage_generation",
    "aspeech",
    "atranscription",
    "amoderation",
    "arerank",
    "aresponses",
    "aget_responses",
    "adelete_responses",
    "acancel_responses",
    "acompact_responses",
    "acreate_response_reply",
    "alist_input_items",
    "_arealtime",  # private function for realtime API
    "acreate_realtime_client_secret",
    "arealtime_calls",
    "acreate_realtime_transcription_session",
    "_aresponses_websocket",  # private function for responses WebSocket mode
    "aimage_edit",
    "agenerate_content",
    "agenerate_content_stream",
    "allm_passthrough_route",
    "acreate_batch",
    "aretrieve_batch",
    "alist_batches",
    "afile_content",
    "afile_retrieve",
    "acreate_fine_tuning_job",
    "acancel_fine_tuning_job",
    "alist_fine_tuning_jobs",
    "aretrieve_fine_tuning_job",
    "avector_store_search",
    "avector_store_create",
    "avector_store_retrieve",
    "avector_store_list",
    "avector_store_update",
    "avector_store_delete",
    "avector_store_file_create",
    "avector_store_file_list",
    "avector_store_file_retrieve",
    "avector_store_file_content",
    "avector_store_file_update",
    "avector_store_file_delete",
    "aocr",
    "asearch",
    "adecisions",
    "asystemone",
    "avideo_generation",
    "avideo_list",
    "avideo_status",
    "avideo_content",
    "avideo_remix",
    "avideo_create_character",
    "avideo_get_character",
    "avideo_edit",
    "avideo_extension",
    "acreate_container",
    "alist_containers",
    "aretrieve_container",
    "adelete_container",
    "aupload_container_file",
    "alist_container_files",
    "aretrieve_container_file",
    "adelete_container_file",
    "aretrieve_container_file_content",
    "acreate_skill",
    "alist_skills",
    "aget_skill",
    "adelete_skill",
    "aingest",
    "anthropic_messages",
    "acreate_interaction",
    "aget_interaction",
    "adelete_interaction",
    "acancel_interaction",
    "acreate_agent",
    "alist_agents",
    "aget_agent",
    "adelete_agent",
    "alist_agent_versions",
    "asend_message",
    "call_mcp_tool",
    "acancel_batch",
    "afile_delete",
    "acreate_eval",
    "alist_evals",
    "aget_eval",
    "aupdate_eval",
    "adelete_eval",
    "acancel_eval",
    "acreate_run",
    "alist_runs",
    "aget_run",
    "acancel_run",
    "adelete_run",
]


async def route_request(
    data: dict,
    llm_router: LitellmRouter | None,
    user_model: str | None,
    route_type: RouteType,
    user_api_key_dict: UserAPIKeyAuth | None = None,
):
    """
    Common helper to route the request
    """
    try:
        return await _route_request_single_attempt(
            data=data,
            llm_router=llm_router,
            user_model=user_model,
            route_type=route_type,
            user_api_key_dict=user_api_key_dict,
        )
    except (ProxyModelNotFoundError, ProxyMissingParamWithoutLoadedModelError) as e:
        requested_model: Final = data.get("model", "")
        if (
            (isinstance(e, ProxyModelNotFoundError) and not e.retryable_with_model_read_through)
            or not isinstance(requested_model, str)
            or not requested_model
        ):
            raise
        from litellm.proxy import proxy_server
        from litellm.proxy.common_utils.registry_read_through import (
            model_registry_read_through,
        )

        if not await model_registry_read_through.attempt(requested_model):
            raise
        unsupported_endpoint: Final = endpoint_unsupported_by_model(
            route_type=route_type,
            data=JSON_OBJECT_ADAPTER.validate_python(data),
            llm_router=proxy_server.llm_router,
            team_id=user_api_key_dict.team_id if user_api_key_dict is not None else None,
        )
        if unsupported_endpoint is not None:
            raise unsupported_endpoint
        return await _route_request_single_attempt(
            data=data,
            llm_router=proxy_server.llm_router,
            user_model=user_model,
            route_type=route_type,
            user_api_key_dict=user_api_key_dict,
        )


async def _route_request_single_attempt(  # noqa: ANN202  # returns unawaited provider coroutines; the inferred union keeps route_request's callers typed
    data: dict,  # mutable-ok: request body is the proxy-wide mutable dict contract shared with route_request
    llm_router: LitellmRouter | None,
    user_model: str | None,
    route_type: RouteType,
    user_api_key_dict: UserAPIKeyAuth | None = None,
):
    raise_if_required_body_param_missing(route_type=route_type, data=data, llm_router=llm_router)

    await add_shared_session_to_data(data)

    raise_if_mock_testing_params_disallowed(data, allowed=mock_testing_params_allowed())

    data.pop("enable_tag_filtering", None)
    if "user_config" in data and data["user_config"] is None:
        data.pop("user_config")  # pyright: ignore[reportUnknownMemberType]  # route_request takes an untyped dict body

    team_id: Final = get_team_id_from_data(data)
    router_model_names: Final = llm_router.model_names if llm_router is not None else []
    is_proxy_admin_without_team: Final = team_id is None and is_proxy_admin_request(data)

    # Preprocess Google GenAI generate content requests
    if route_type in ["agenerate_content", "agenerate_content_stream"]:
        # Map generationConfig to config parameter for Google GenAI compatibility
        if "generationConfig" in data and "config" not in data:
            data["config"] = data.pop("generationConfig")
    if "api_key" in data or "api_base" in data:
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**data)
        else:
            return getattr(litellm, f"{route_type}")(**data)

    elif (
        route_type == "acompletion"
        and data.get("model", "") is not None
        and "," in data.get("model", "")
        and llm_router is not None
    ):
        # Handle batch completions with comma-separated models BEFORE user_config check
        # This ensures batch completion logic is applied even when user_config is set
        if data.get("fastest_response", False):
            return llm_router.abatch_completion_fastest_response(**data)
        else:
            models: Final = [model.strip() for model in data.pop("model").split(",")]
            return llm_router.abatch_completion(models=models, **data)

    elif "user_config" in data:
        return _route_user_config_request(data, route_type)

    elif "router_settings_override" in data:
        # Apply per-request router settings overrides from key/team config
        # Instead of creating a new Router (expensive), merge settings into kwargs
        # The Router already supports per-request overrides for these settings
        override_settings: Final = data.pop("router_settings_override")

        # Settings that the Router accepts as per-request kwargs
        # These override the global router settings for this specific request
        per_request_settings: Final = [
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
            "num_retries",
            "timeout",
            "model_group_retry_policy",
            "routing_strategy",
            "enable_tag_filtering",
        ]

        for key in per_request_settings:
            if override_settings.get(key) is not None and key not in data:
                data[key] = override_settings[key]

        # Use main router with overridden kwargs
        if llm_router is not None:
            return getattr(llm_router, f"{route_type}")(**data)
        else:
            return getattr(litellm, f"{route_type}")(**data)
    elif llm_router is not None:
        _raise_if_model_fully_blocked(llm_router=llm_router, model_name=data.get("model"), team_id=team_id)
        # Evals API: always route to litellm directly (not through router)
        # But extract model credentials if a model is provided
        if route_type in [
            "acreate_eval",
            "alist_evals",
            "aget_eval",
            "aupdate_eval",
            "adelete_eval",
            "acancel_eval",
            "acreate_run",
            "alist_runs",
            "aget_run",
            "acancel_run",
            "adelete_run",
        ]:
            # If a model is provided, get its credentials from the router
            model: Final = data.get("model")
            if model and llm_router:
                try:
                    # Try to get deployment credentials for this model
                    deployment_creds = llm_router.get_deployment_credentials(model_id=model)
                    if not deployment_creds:
                        # Try by model group name
                        deployment: Final = llm_router.get_deployment_by_model_group_name(model_group_name=model)
                        if (
                            deployment
                            and deployment.litellm_params
                            and not llm_router.is_deployment_blocked(deployment)
                        ):
                            deployment_creds = deployment.litellm_params.model_dump(exclude_none=True)

                    # If we found credentials, merge them into data (but don't override user-provided values)
                    if deployment_creds:
                        data.update(deployment_creds)
                except Exception:
                    # If we can't get deployment creds, continue without them
                    pass

            return getattr(litellm, f"{route_type}")(**data)
        # Skip model-based routing for container operations
        if route_type in [
            "acreate_container",
            "alist_containers",
            "aretrieve_container",
            "adelete_container",
            "aupload_container_file",
            "alist_container_files",
            "aretrieve_container_file",
            "adelete_container_file",
            "aretrieve_container_file_content",
        ]:
            return getattr(llm_router, f"{route_type}")(**data)
        # Interactions API: create with agent, get/delete/cancel don't need model routing
        if route_type in [
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
        ]:
            return getattr(llm_router, f"{route_type}")(**data)
        # Managed Agents API: these don't need model routing
        if route_type in [
            "acreate_agent",
            "alist_agents",
            "aget_agent",
            "adelete_agent",
            "alist_agent_versions",
        ]:
            return getattr(llm_router, f"{route_type}")(**data)
        if route_type in [
            "avideo_list",
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "avideo_create_character",
            "avideo_get_character",
            "avideo_edit",
            "avideo_extension",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_delete",
            "acreate_skill",
            "alist_skills",
            "aget_skill",
            "adelete_skill",
            "aingest",
        ] and (data.get("model") is None or data.get("model") == ""):
            # These endpoints don't need a model, use custom_llm_provider directly
            return getattr(litellm, f"{route_type}")(**data)

        if "model" not in data:
            raise ProxyMissingRequiredParamError(
                route=ROUTE_ENDPOINT_MAPPING.get(route_type, route_type),
                param="model",
            )
        team_model_name: Final = llm_router.map_team_model(data["model"], team_id) if team_id is not None else None
        if team_model_name is not None:
            data["model"] = team_model_name
            return getattr(llm_router, f"{route_type}")(**data)

        elif (
            is_proxy_admin_without_team
            and data["model"] not in router_model_names
            and data["model"] in llm_router.team_public_model_names
        ) or llm_router.is_recognized_model(data["model"]):
            return getattr(llm_router, f"{route_type}")(**data)

        elif data["model"] not in router_model_names:
            # Check wildcards before checking deployment_names
            # Priority: 1. Exact model_name match, 2. Wildcard match, 3. deployment_names match
            if llm_router.router_general_settings.pass_through_all_models:
                return getattr(litellm, f"{route_type}")(**data)
            elif llm_router.default_deployment is not None or len(llm_router.pattern_router.patterns) > 0:
                return getattr(llm_router, f"{route_type}")(**data)
            elif data["model"] in llm_router.deployment_names:
                # Only match deployment_names if no wildcard matched
                return getattr(llm_router, f"{route_type}")(**data, specific_deployment=True)
            elif route_type in [
                "amoderation",
                "aget_responses",
                "adelete_responses",
                "acancel_responses",
                "alist_input_items",
                "avector_store_create",
                "avector_store_search",
                "avector_store_retrieve",
                "avector_store_list",
                "avector_store_update",
                "avector_store_delete",
                "avector_store_file_create",
                "avector_store_file_list",
                "avector_store_file_retrieve",
                "avector_store_file_content",
                "avector_store_file_update",
                "avector_store_file_delete",
                "asearch",
                "acreate_container",
                "alist_containers",
                "aretrieve_container",
                "adelete_container",
                "aupload_container_file",
                "alist_container_files",
                "aretrieve_container_file",
                "adelete_container_file",
                "aretrieve_container_file_content",
            ]:
                # These endpoints can work with or without model parameter
                return getattr(llm_router, f"{route_type}")(**data)
            elif route_type in [
                "avideo_status",
                "avideo_content",
                "avideo_remix",
                "avideo_create_character",
                "avideo_get_character",
                "avideo_edit",
                "avideo_extension",
            ]:
                # Video endpoints: If model is provided (e.g., from decoded video_id or target_model_names),
                # try router first to allow for multi-deployment load balancing
                try:
                    return getattr(llm_router, f"{route_type}")(**data)
                except Exception:
                    # If router fails (e.g., model not found in router), fall back to direct call
                    return getattr(litellm, f"{route_type}")(**data)
            elif _is_a2a_agent_model(data.get("model", "")):
                from litellm.proxy.agent_endpoints.a2a_routing import (
                    route_a2a_agent_request,
                )

                result: Final = await route_a2a_agent_request(data, route_type, user_api_key_dict=user_api_key_dict)
                if result is not None:
                    return result
                # Fall through to raise exception below if result is None

    elif user_model is not None or route_type == "allm_passthrough_route":
        return getattr(litellm, f"{route_type}")(**data)

    # if no route found then it's a bad request
    route_name: Final = ROUTE_ENDPOINT_MAPPING.get(route_type, route_type)
    raise ProxyModelNotFoundError(
        route=route_name,
        model_name=data.get("model", ""),
    )
