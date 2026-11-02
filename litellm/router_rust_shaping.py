"""
The keyword arguments one router attempt sends to one deployment, as the Python router builds them.

The Rust router core picks the deployment; the host shapes the request for it here
(``_update_kwargs_with_deployment``, ``_update_kwargs_with_default_litellm_params``,
``_merge_tools_from_deployment``, the timeout resolvers and the reasoning-effort override), with
the same precedence and the same oddities. Nothing here reads a Python router.
"""

import dataclasses
from collections.abc import Mapping, MutableMapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from litellm import exceptions as litellm_exceptions
from litellm.constants import (
    BAGGAGE_MAPPED_TAGS_METADATA_KEY,
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    KUBERNETES_POD_ROUTING_KEY,
    ROUTING_REQUEST_TAGS_METADATA_KEY,
)
from litellm.litellm_core_utils.batch_utils import get_router_metadata_variable_name
from litellm.litellm_core_utils.clientside_credential_handler import (
    get_dynamic_litellm_params,  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
    is_clientside_credential,  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
)
from litellm.litellm_core_utils.completion_timeout import CompletionTimeout
from litellm.litellm_core_utils.get_litellm_params import SUPPLEMENTAL_KWARGS_KEYS
from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider, inferred_provider
from litellm.litellm_core_utils.request_timeout_resolver import get_service_tier_timeout
from litellm.litellm_core_utils.routing_common_utils import deployment_model_id_pin
from litellm.litellm_core_utils.tool_fields import TOOL_FIELDS, drop_policy
from litellm.responses.retrieve_stream_query import requests_streamed_retrieval, without_retrieve_stream_defaults
from litellm.router_api import deployment_id as generate_deployment_id
from litellm.router_strategy.tag_based_routing import (
    baggage_stamp_for_serving_deployment,
    get_tags_from_request_kwargs,
    tags_with_serving_deployment_tags,
)
from litellm.router_utils.pre_call_checks.io_token_rate_limit_check import (
    deployment_has_io_token_limits,  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
    refund_stale_reservation_before_retry,
    set_io_token_rate_limit_request_kwargs,
)
from litellm.router_utils.stream_usage_injection import router_config_requests_stream_usage
from litellm.types.litellm_params import STREAM_USAGE_INJECTED_FIELD
from litellm.types.router import Deployment, GenericLiteLLMParams, LiteLLM_Params, validate_litellm_params_strictly
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.router_python_host import PythonHost

# The SDK-client and max_parallel_requests reads of the routing cache: process-local, so the family
# only names a Redis span when a read misses through to Redis.
ROUTER_CLIENTS_TARGET: Final = "router_clients"
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_FORWARDED_CREDENTIAL_ISOLATION_PROVIDERS: Final = frozenset({"chatgpt", "anthropic"})
_ALIAS_MARKER_FORWARDED_PARAMS_KWARG: Final = "_alias_marker_forwarded_params"
# Generic handlers whose stream timeout is split from the request timeout.
STREAM_TIMEOUT_AWARE_GENERIC_HANDLERS: Final = frozenset({"anthropic_messages", "aresponses", "responses"})


@dataclasses.dataclass(frozen=True, slots=True)
class Shaping:
    """What the request shaping reads from the router, resolved once when the core is built.

    ``timeout`` is the router's ``timeout`` (its own setting, else the process ``request_timeout``),
    ``request_timeout`` the explicitly configured process ``request_timeout``, kept only when the
    router has a timeout of its own, as the Python router keeps them."""

    default_litellm_params: Mapping[str, object]
    timeout: object
    explicit_timeout: object
    request_timeout: float | None
    stream_timeout: object
    cache_responses: bool


def as_dict(value: object) -> dict[str, object]:  # mutable-ok: a fresh dict of a request mapping
    try:
        return dict(_MAPPING.validate_python(value))
    except ValidationError:
        return {}


def _mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING.validate_python(value)
    except ValidationError:
        return None


def metadata_bucket(
    kwargs: MutableMapping[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    name: str,
) -> dict[str, object]:  # mutable-ok: the request's own bucket
    """``kwargs.setdefault(name, {})``: the request's own metadata dict, created when absent."""
    bucket: Final = kwargs.get(name)
    if isinstance(bucket, dict):
        return bucket  # pyright: ignore[reportUnknownVariableType]  # the request's untyped metadata dict
    if name not in kwargs:
        fresh: Final[dict[str, object]] = {}  # mutable-ok: the request kwargs the Python router mutates in place
        kwargs[name] = fresh  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
        return fresh
    # A non-dict value under the key: `setdefault` returns it as is, and `.update` would raise.
    raise AttributeError(f"'{type(bucket).__name__}' object has no attribute 'update'")


# --- request params the deployment sets --------------------------------------------------------


def _drops_malformed_request_tools(
    deployment_params: Mapping[str, object], defaults: Mapping[str, object], request: Mapping[str, object]
) -> bool:
    if isinstance(request.get("tools") or [], (list, tuple)) or not deployment_params.get("tools"):
        return False
    set_defaults: Final = {name: value for name, value in defaults.items() if value is not None}
    return drop_policy({**deployment_params, **set_defaults, **request}).drops("tools")


def _malformed_request_params_the_deployment_sets(
    deployment_params: Mapping[str, object], defaults: Mapping[str, object], request: Mapping[str, object]
) -> frozenset[str]:
    overrides: Final = {
        key: value
        for key, value in request.items()
        if key in SUPPLEMENTAL_KWARGS_KEYS and deployment_params.get(key) is not None
    }
    if not overrides:
        return frozenset()
    try:
        validate_litellm_params_strictly(GenericLiteLLMParams, overrides)
    except ValidationError as error:
        set_defaults: Final = {name: value for name, value in defaults.items() if value is not None}
        policy: Final = drop_policy({**deployment_params, **set_defaults, **request})
        malformed: Final = (str(detail["loc"][0]) for detail in error.errors() if detail["loc"])
        return frozenset(key for key in malformed if policy.drops(key))
    return frozenset()


def _merged_tools(
    deployment_tools: Sequence[object], request_tools: object
) -> list[object] | None:  # mutable-ok: sent as the request's tools
    if not isinstance(request_tools, (list, tuple)):
        return None
    items: Final[Sequence[object]] = request_tools  # pyright: ignore[reportUnknownVariableType]  # a request's untyped tools
    return [*deployment_tools, *items] if deployment_tools or items else None


def merge_tools_from_deployment(
    deployment_params: Mapping[str, object],
    kwargs: MutableMapping[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    defaults: Mapping[str, object],
) -> None:
    """``_merge_tools_from_deployment``: deployment tools first, then the request's; the request's
    ``tool_choice`` else the deployment's."""
    if _drops_malformed_request_tools(deployment_params, defaults, kwargs):
        for name in TOOL_FIELDS:
            kwargs.pop(name, None)
    deployment_tools: Final = deployment_params.get("tools") or []  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
    merged: Final = _merged_tools(
        deployment_tools if isinstance(deployment_tools, list) else [],  # pyright: ignore[reportUnknownArgumentType]  # the deployment's untyped tools
        kwargs.get("tools") or [],
    )
    if merged is not None:
        kwargs["tools"] = merged  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
    if "tool_choice" not in kwargs and deployment_params.get("tool_choice") is not None:
        kwargs["tool_choice"] = (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
            deployment_params["tool_choice"]
        )


def _deployment_sets_litellm_param(deployment_params: Mapping[str, object], key: str) -> bool:
    value: Final = deployment_params.get(key)
    if value is None:
        return False
    field: Final = LiteLLM_Params.model_fields.get(key)
    return field is None or value != field.default  # pyright: ignore[reportAny]  # litellm's untyped helpers and request values


def _forwarded_alias_marker_keys_the_deployment_sets(
    deployment_params: Mapping[str, object], forwarded_keys: object
) -> tuple[str, ...]:
    if not isinstance(forwarded_keys, tuple):
        return ()
    keys: Final[tuple[object, ...]] = forwarded_keys  # pyright: ignore[reportUnknownVariableType]  # the router's own marker
    return tuple(key for key in keys if isinstance(key, str) and _deployment_sets_litellm_param(deployment_params, key))


# --- reasoning effort -------------------------------------------------------------------------


def _pop_effort_from_nested_carrier(params: dict[str, object], carrier: str) -> None:  # mutable-ok: copy-on-write
    nested: Final = _mapping(params.get(carrier)) if isinstance(params.get(carrier), Mapping) else None
    if nested is None:
        return
    sanitized: Final = {key: value for key, value in nested.items() if key != "effort"}
    if sanitized:
        params[carrier] = (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
            sanitized
        )
    else:
        params.pop(carrier, None)


def deployment_params_with_request_reasoning_override(
    deployment_params: Mapping[str, object], request_kwargs: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: the call's own copy of the deployment params
    """``_deployment_params_with_request_reasoning_override``: a request ``reasoning_effort`` drops
    the deployment's ``thinking`` and nested ``effort`` carriers."""
    sanitized: Final = dict(deployment_params)
    if request_kwargs.get("reasoning_effort") is None:
        return sanitized
    sanitized.pop("thinking", None)
    _pop_effort_from_nested_carrier(sanitized, "output_config")
    _pop_effort_from_nested_carrier(sanitized, "reasoning")
    extra_body: Final = sanitized.get("extra_body")
    if isinstance(extra_body, Mapping):
        sanitized_extra_body: Final = as_dict(extra_body)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
        sanitized_extra_body.pop("reasoning_effort", None)
        sanitized_extra_body.pop("thinking", None)
        _pop_effort_from_nested_carrier(sanitized_extra_body, "output_config")
        _pop_effort_from_nested_carrier(sanitized_extra_body, "reasoning")
        if sanitized_extra_body:
            sanitized["extra_body"] = sanitized_extra_body
        else:
            sanitized.pop("extra_body", None)
    return sanitized


def _is_classifier_internal_call(kwargs: Mapping[str, object]) -> bool:
    return any(
        isinstance(candidate, Mapping)
        and (_mapping(candidate) or {}).get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == AUTOROUTER_CLASSIFIER_CALL_ORIGIN  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
        for candidate in (kwargs.get("metadata"), kwargs.get("litellm_metadata"))
    )


def _declared_param_allowlist(params: Mapping[str, object]) -> frozenset[str]:
    declared: Final = params.get("allowed_openai_params")
    if not isinstance(declared, (list, tuple, set, frozenset)):
        return frozenset()
    entries: Final[Sequence[object]] = list(declared)  # pyright: ignore[reportUnknownArgumentType]  # the deployment's untyped allowlist
    return frozenset(entry for entry in entries if isinstance(entry, str))


def _deployment_supports_param(deployment: Mapping[str, object], group: str, param: str, host: "PythonHost") -> bool:
    params: Final = as_dict(deployment.get("litellm_params"))
    if not params:
        return True
    model_info: Final = as_dict(deployment.get("model_info"))
    base_model: Final = model_info.get("base_model") or params.get("base_model")
    try:
        model_label: Final = str(params.get("model") or group)
        provider_param: Final = params.get("custom_llm_provider")
        authenticating: Final = declared_authenticating_provider(
            model_label, provider_param if isinstance(provider_param, str) else None
        )
        if authenticating is not None:
            model, custom_llm_provider = model_label.removeprefix(f"{authenticating}/"), authenticating
        else:
            model, custom_llm_provider, _, _ = host.invoke(  # rebind-ok: rebound as the Python router rebinds it
                "get_llm_provider"
            )(
                model=model_label,
                custom_llm_provider=provider_param,
                litellm_params=LiteLLM_Params.model_validate(params),
            )
        supported: Final = host.invoke("get_supported_openai_params")(
            model=model,
            custom_llm_provider=custom_llm_provider,
            base_model=base_model if isinstance(base_model, str) else None,
        )
    except Exception:  # noqa: BLE001  # best-effort filter: an unresolvable provider must not narrow the request
        return True
    return supported is None or param in supported


def deployment_accepts_param(deployment: Mapping[str, object], group: str, param: str, host: "PythonHost") -> bool:
    params: Final = as_dict(deployment.get("litellm_params"))
    if not params:
        return True
    if param in _declared_param_allowlist(params):
        return True
    provider_param: Final = params.get("custom_llm_provider")
    if declared_authenticating_provider(
        str(params.get("model") or ""), provider_param if isinstance(provider_param, str) else None
    ):
        return True
    return _deployment_supports_param(deployment, group, param, host)


def drop_unsupported_classifier_reasoning_effort(
    deployment: Mapping[str, object],
    model: str,
    kwargs: MutableMapping[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    host: "PythonHost",
) -> None:
    """``_drop_unsupported_classifier_reasoning_effort``."""
    if kwargs.get("reasoning_effort") is None or not _is_classifier_internal_call(kwargs):
        return
    if deployment_accepts_param(deployment, model, "reasoning_effort", host):
        return
    kwargs.pop("reasoning_effort", None)
    proxy_server_request: Final = kwargs.get("proxy_server_request")
    if not isinstance(proxy_server_request, dict):
        return
    body: Final[object] = proxy_server_request.get("body")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the proxy's untyped request
    if isinstance(body, dict):
        body.pop("reasoning_effort", None)  # pyright: ignore[reportUnknownMemberType]  # same


# --- timeouts ---------------------------------------------------------------------------------


def stream_timeout(shaping: Shaping, kwargs: Mapping[str, object], data: Mapping[str, object]) -> object:
    """``_get_stream_timeout``."""
    return (
        kwargs.get("stream_timeout", None)
        or get_service_tier_timeout(kwargs.get("service_tier"), data, "stream_timeout_by_service_tier")
        or data.get("stream_timeout", None)
        or shaping.stream_timeout
        or shaping.request_timeout
        or shaping.default_litellm_params.get("stream_timeout", None)
    )


def non_stream_timeout(shaping: Shaping, kwargs: Mapping[str, object], data: Mapping[str, object]) -> object:
    """``_get_non_stream_timeout``."""
    return (
        kwargs.get("timeout", None)
        or kwargs.get("request_timeout", None)
        or get_service_tier_timeout(kwargs.get("service_tier"), data, "timeout_by_service_tier")
        or data.get("timeout", None)
        or data.get("request_timeout", None)
        or shaping.request_timeout
        or shaping.timeout
        or shaping.default_litellm_params.get("timeout", None)
    )


def call_timeout(shaping: Shaping, kwargs: Mapping[str, object], data: Mapping[str, object]) -> object:
    """``_get_timeout``."""
    stream: Final = stream_timeout(shaping, kwargs, data) if kwargs.get("stream") else None
    return stream if stream is not None else non_stream_timeout(shaping, kwargs, data)


# --- default litellm params -------------------------------------------------------------------


def update_kwargs_with_default_litellm_params(
    shaping: Shaping,
    kwargs: MutableMapping[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    metadata_variable_name: str,
) -> None:
    """``_update_kwargs_with_default_litellm_params``: unset non-None defaults fill the request;
    the default metadata is merged over the request's."""
    defaults: Final = dict(without_retrieve_stream_defaults(shaping.default_litellm_params, kwargs))
    metadata_defaults: Final = defaults.pop("metadata", {}) or {}  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
    for key, value in defaults.items():
        if value is None:
            continue
        kwargs.setdefault(key, value)
    metadata_bucket(kwargs, metadata_variable_name).update(as_dict(metadata_defaults))  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values


# --- clientside credentials -------------------------------------------------------------------


def _clientside_deployment(
    deployment: Mapping[str, object], kwargs: Mapping[str, object], function_name: str | None
) -> tuple[Mapping[str, object], str, object]:
    """``_handle_clientside_credential``: the request-scoped model_info, model and api_base."""
    model_info: Final = as_dict(deployment.get("model_info"))
    litellm_params: Final = as_dict(deployment.get("litellm_params"))
    dynamic_litellm_params: Final = get_dynamic_litellm_params(  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        litellm_params=litellm_params, request_kwargs=dict(kwargs)
    )
    metadata_variable_name: Final = get_router_metadata_variable_name(function_name=function_name)
    model_group: Final = as_dict(kwargs.get(metadata_variable_name)).get("model_group")
    model_id: Final = generate_deployment_id(model_group, dynamic_litellm_params)  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]  # the group as the request carries it
    original_model_id: Final = model_info.get("id")
    model_info["id"] = model_id
    model_info["original_model_id"] = original_model_id
    built: Final = Deployment(
        model_name=model_group,  # pyright: ignore[reportArgumentType]  # the group as the request carries it
        litellm_params=LiteLLM_Params.model_validate(dynamic_litellm_params),
        model_info=model_info,  # pyright: ignore[reportArgumentType]  # model_info as configured
    )
    return built.model_info.model_dump(), built.litellm_params.model, built.litellm_params.api_base


# --- the deployment --------------------------------------------------------------------------


def update_kwargs_with_deployment(
    shaping: Shaping,
    deployment: Mapping[str, object],
    kwargs: MutableMapping[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    *,
    cache: "DualCache | None",
    function_name: str | None = None,
    splits_stream_timeout: bool = False,
    opens_responses_socket: bool = False,
) -> None:
    """``_update_kwargs_with_deployment``, in place on the attempt's kwargs."""
    deployment_params: Final = as_dict(deployment.get("litellm_params"))
    for key in _forwarded_alias_marker_keys_the_deployment_sets(
        deployment_params, kwargs.pop(_ALIAS_MARKER_FORWARDED_PARAMS_KWARG, ())
    ):
        kwargs.pop(key, None)
    kwargs.pop("daily_reset_time", None)
    defaults: Final = shaping.default_litellm_params
    merge_tools_from_deployment(deployment_params, kwargs, defaults)

    model_info: Mapping[str, object] = as_dict(  # rebind-ok: rebound as the Python router rebinds it
        deployment.get("model_info")
    )
    deployment_model: object = deployment_params.get("model")  # rebind-ok: same
    deployment_api_base: object = deployment_params.get("api_base")  # rebind-ok: same
    deployment_model_name: Final = deployment.get("model_name")
    from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils

    for key in _malformed_request_params_the_deployment_sets(deployment_params, defaults, kwargs):
        kwargs.pop(key)
    declared_provider: Final = deployment_params.get("custom_llm_provider")
    declared_model: Final = deployment_params.get("model")
    deployment_provider: Final = (
        declared_provider
        if isinstance(declared_provider, str)
        else inferred_provider(declared_model if isinstance(declared_model, str) else None)
    )
    forwarded_authorization: Final = (
        ProviderSpecificHeaderUtils.forwarded_authorization(kwargs.get("provider_specific_header"), deployment_provider)
        if deployment_provider in _FORWARDED_CREDENTIAL_ISOLATION_PROVIDERS
        else None
    )
    if is_clientside_credential(request_kwargs=dict(kwargs)) or forwarded_authorization is not None:
        model_info, deployment_model, deployment_api_base = (  # rebind-ok: rebound as the Python router rebinds it
            _clientside_deployment(
                deployment,
                {**kwargs, **({"api_key": forwarded_authorization} if forwarded_authorization is not None else {})},
                function_name,
            )
        )

    metadata_variable_name: Final = get_router_metadata_variable_name(function_name=function_name)
    routing: Final = deployment.get(KUBERNETES_POD_ROUTING_KEY)
    metadata: Final = metadata_bucket(kwargs, metadata_variable_name)
    metadata.update(
        {
            "deployment": deployment_model,
            "model_info": model_info,
            "api_base": deployment_api_base,
            "deployment_model_name": deployment_model_name,
            KUBERNETES_POD_ROUTING_KEY: routing,
        }
    )
    other_bucket: Final = "metadata" if metadata_variable_name == "litellm_metadata" else "litellm_metadata"
    other_bucket_metadata: Final = kwargs.get(other_bucket)
    if isinstance(other_bucket_metadata, Mapping) and KUBERNETES_POD_ROUTING_KEY in other_bucket_metadata:
        kwargs[other_bucket] = {**as_dict(other_bucket_metadata), KUBERNETES_POD_ROUTING_KEY: routing}  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them

    if cache is not None:
        refund_stale_reservation_before_retry(cache, kwargs)  # pyright: ignore[reportArgumentType]  # the request's kwargs dict
    set_io_token_rate_limit_request_kwargs(
        kwargs,  # pyright: ignore[reportArgumentType]  # the request's kwargs dict
        store_in_context=deployment_has_io_token_limits(deployment),  # pyright: ignore[reportArgumentType]  # a deployment record
        opens_responses_socket=opens_responses_socket,
        metadata_channel=metadata_variable_name,
    )
    metadata.setdefault(
        ROUTING_REQUEST_TAGS_METADATA_KEY,
        tuple(get_tags_from_request_kwargs(kwargs, metadata_variable_name=metadata_variable_name)),  # pyright: ignore[reportArgumentType]  # the request's kwargs dict
    )
    deployment_tags: Final = deployment_params.get("tags")
    merged_tags: Final = tags_with_serving_deployment_tags(metadata, deployment_tags)
    if merged_tags is not None:
        metadata["tags"] = list(merged_tags)
    baggage_stamp: Final = baggage_stamp_for_serving_deployment(metadata, deployment_tags)
    if baggage_stamp is not None:
        metadata[BAGGAGE_MAPPED_TAGS_METADATA_KEY] = baggage_stamp

    credential_name: Final = deployment_params.get("litellm_credential_name")
    if credential_name:
        credential_tag: Final = f"Credential: {credential_name}"
        existing_tags: Final = metadata.get("tags") or []  # pyright: ignore[reportUnknownVariableType]  # litellm's untyped helpers and request values
        if isinstance(existing_tags, list) and credential_tag not in existing_tags:
            existing_tags.append(credential_tag)  # pyright: ignore[reportUnknownMemberType]  # the request's own tags list
        metadata["tags"] = existing_tags

    from litellm.llms.github_copilot.per_user_auth import github_copilot_per_user_credential_name

    configured_per_user_name: Final = github_copilot_per_user_credential_name(deployment_params)
    caller_credential_name: Final = kwargs.get("litellm_credential_name")
    if (
        configured_per_user_name is not None
        and isinstance(caller_credential_name, str)
        and caller_credential_name != configured_per_user_name
    ):
        raise litellm_exceptions.BadRequestError(
            message="litellm_credential_name cannot be overridden on a deployment that uses per-user GitHub OAuth",
            model=str(deployment_model_name),
            llm_provider="",
        )

    kwargs["model_info"] = (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
        model_info
    )
    _set_timeouts(shaping, deployment_params, kwargs, function_name, splits_stream_timeout)

    if router_config_requests_stream_usage(kwargs, (deployment_params, defaults)):
        kwargs[  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
            STREAM_USAGE_INJECTED_FIELD
        ] = True
    update_kwargs_with_default_litellm_params(shaping, kwargs, metadata_variable_name)
    pinned_model_id: Final = deployment_model_id_pin(deployment_params)
    if pinned_model_id is not None:
        kwargs["model_id"] = (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
            pinned_model_id
        )


def _set_timeouts(
    shaping: Shaping,
    deployment_params: Mapping[str, object],
    kwargs: MutableMapping[str, object],  # mutable-ok: the request kwargs the Python router mutates in place
    function_name: str | None,
    splits_stream_timeout: bool,
) -> None:
    if function_name == "_ageneric_api_call_with_fallbacks":
        from litellm.passthrough.timeout_utils import resolve_llm_passthrough_timeout

        router_timeout: Final = (
            shaping.request_timeout
            if shaping.request_timeout is not None
            else float(shaping.explicit_timeout)
            if isinstance(shaping.explicit_timeout, (int, float))
            else None
        )
        router_stream_timeout: Final = (
            shaping.stream_timeout
            if shaping.stream_timeout is not None
            else shaping.request_timeout
            if shaping.request_timeout is not None
            else shaping.default_litellm_params.get("stream_timeout")
        )
        passthrough_timeout: Final = partial(
            resolve_llm_passthrough_timeout,
            litellm_params=deployment_params,
            router_timeout=router_timeout,
            router_stream_timeout=router_stream_timeout,  # pyright: ignore[reportArgumentType]  # a configured default, passed as the Python router passes it
        )
        timeout_request: Final = {**kwargs, "stream": True} if requests_streamed_retrieval(kwargs) else dict(kwargs)
        kwargs.update(
            CompletionTimeout.stream_request_kwargs(
                passthrough_timeout(kwargs={**kwargs, "stream": False}), passthrough_timeout(kwargs=timeout_request)
            )
            if splits_stream_timeout and timeout_request.get("stream")
            else {"timeout": passthrough_timeout(kwargs=timeout_request)}
        )
    elif splits_stream_timeout:
        kwargs.update(
            CompletionTimeout.stream_request_kwargs(
                non_stream_timeout(shaping, kwargs, deployment_params),
                stream_timeout(shaping, kwargs, deployment_params) if kwargs.get("stream") else None,
            )
        )
    else:
        kwargs["timeout"] = (  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
            call_timeout(shaping, kwargs, deployment_params)
        )


def stamp_deployment_num_retries(error: BaseException, deployment: Mapping[str, object]) -> None:
    """``_set_deployment_num_retries_on_exception``."""
    if getattr(error, "num_retries", None) is not None:
        return
    value: Final = as_dict(deployment.get("litellm_params")).get("num_retries")
    if value is None:
        return
    try:
        error.num_retries = int(str(value)) if not isinstance(value, int) else value  # pyright: ignore[reportAttributeAccessIssue]  # the router stamps the same attribute  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
    except (ValueError, TypeError):
        pass


def stamp_failed_deployment_id(error: BaseException, model_info: object) -> None:
    """``_set_failed_deployment_id_on_exception`` with the effective model_info."""
    if getattr(error, "failed_deployment_id", None):
        return
    deployment_id: Final = as_dict(model_info).get("id")
    if deployment_id:
        try:
            error.failed_deployment_id = deployment_id  # pyright: ignore[reportAttributeAccessIssue]  # the router stamps the same attribute  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them
        except Exception:  # noqa: BLE001, S110  # as the Python router: a read-only error keeps its attributes
            pass


def stamp_retry_skip_deployment_id(error: BaseException, kwargs: Mapping[str, object]) -> None:
    """``_stamp_retry_skip_deployment_id``."""
    model_info: Final = kwargs.get("model_info")
    if not isinstance(model_info, Mapping):
        return
    info: Final = as_dict(model_info)  # pyright: ignore[reportUnknownArgumentType]  # litellm's untyped helpers and request values
    configured: Final = (
        info.get("original_model_id") if isinstance(error, litellm_exceptions.UnsupportedEndpointError) else None
    )
    deployment_id: Final = configured or info.get("id")
    if isinstance(deployment_id, str) and deployment_id:
        error.retry_skip_deployment_id = deployment_id  # pyright: ignore[reportAttributeAccessIssue]  # dynamic stamp, read on retry  # rebind-ok: the request kwargs are shaped in place, as the Python router shapes them


NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})

__all__ = [
    "STREAM_TIMEOUT_AWARE_GENERIC_HANDLERS",
    "Shaping",
    "as_dict",
    "call_timeout",
    "deployment_params_with_request_reasoning_override",
    "drop_unsupported_classifier_reasoning_effort",
    "merge_tools_from_deployment",
    "metadata_bucket",
    "stamp_deployment_num_retries",
    "stamp_failed_deployment_id",
    "stamp_retry_skip_deployment_id",
    "update_kwargs_with_default_litellm_params",
    "update_kwargs_with_deployment",
]
