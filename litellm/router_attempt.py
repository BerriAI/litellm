"""
One router attempt, as the host runs it: the request for the deployment the router core picked.

The Rust router core picks a deployment and hands the attempt to its host; ``PythonHost`` builds
the keyword arguments for that one deployment here and calls ``litellm.<op>`` with them. Nothing
here selects, retries, falls back or records cooldowns: the core does all of that. Nothing here
reads a Python router either, so this module outlives ``router_python_core.py``.
"""

import re
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.batch_utils import get_router_metadata_variable_name
from litellm.litellm_core_utils.get_llm_provider_logic import inferred_provider
from litellm.litellm_core_utils.pattern_match_deployments import PatternMatchRouter
from litellm.litellm_core_utils.routing_common_utils import without_router_only_kwargs

# Request keys the router reads to route the call; the deployment call never receives them.
ROUTING_KEYS: Final = frozenset(
    {
        "model",
        "num_retries",
        "disable_fallbacks",
        "specific_deployment",
        "_router_weights",
        "_retry_skipped_deployment_ids",
        "_target_order",
        "model_group_retry_policy",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "mock_testing_fallbacks",
        "mock_testing_context_fallbacks",
        "mock_testing_content_policy_fallbacks",
        "mock_testing_rate_limit_error",
    }
)

# The positional parameters of each endpoint, in litellm's order: a call that passes them
# positionally is read back into keywords before it is routed.
POSITIONAL: Final = MappingProxyType(
    {
        "acompletion": ("model", "messages"),
        "completion": ("model", "messages"),
        "atext_completion": ("model", "prompt"),
        "text_completion": ("model", "prompt"),
        "aembedding": ("model", "input"),
        "embedding": ("model", "input"),
        "aimage_generation": ("prompt", "model"),
        "image_generation": ("prompt", "model"),
        "atranscription": ("model", "file"),
        "aspeech": ("model", "input", "voice"),
        "arerank": ("model", "query", "documents"),
        "amoderation": ("input", "model"),
        "aresponses": ("input", "model"),
        "aanthropic_messages": (),
    }
)


def request_from_call(op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """The call's arguments as one keyword request. Raises ``TypeError`` like the endpoint would."""
    names: Final = POSITIONAL.get(op, ())
    if len(args) > len(names):
        raise TypeError(f"{op}() takes {len(names)} positional arguments but {len(args)} were given")
    request: Final = dict(kwargs)
    for name, value in zip(names, args, strict=False):
        if name in request:
            raise TypeError(f"{op}() got multiple values for argument {name!r}")
        request[name] = value
    return request


# Endpoints the Python router serves through ``_ageneric_api_call_with_fallbacks``, which writes
# the routing metadata under ``litellm_metadata``.
_GENERIC_API_OPS: Final = frozenset({"aanthropic_messages", "aresponses"})


def metadata_field(op: str) -> str:
    """``metadata`` or ``litellm_metadata``: where the router writes what it routed the call as."""
    return "litellm_metadata" if op in _GENERIC_API_OPS else get_router_metadata_variable_name(function_name=op)


def prepare_call(op: str, request: Mapping[str, object], model_group_alias: str | None) -> Mapping[str, object]:
    """What the router adds to a request once, before its first attempt
    (``_update_kwargs_before_fallbacks``): a trace id and the model group it was asked for."""
    field: Final = metadata_field(op)
    metadata: Final = dict(_as_mapping(request.get(field)))
    metadata.update({"model_group": request.get("model"), "model_group_alias": model_group_alias})
    return {**request, "litellm_trace_id": request.get("litellm_trace_id") or str(uuid.uuid4()), field: metadata}


def for_requested_model(deployment: Mapping[str, object], requested: str | None) -> Mapping[str, object]:
    """A wildcard deployment as the router serves ``requested`` through it: its ``model`` with
    the wildcards filled in from the requested name (``PatternMatchRouter.route``). Any other
    deployment comes back unchanged."""
    model_name: Final = deployment.get("model_name")
    params: Final = _as_mapping(deployment.get("litellm_params"))
    model: Final = params.get("model")
    if requested is None or not isinstance(model_name, str) or "*" not in model_name or not isinstance(model, str):
        return deployment
    regex: Final = PatternMatchRouter().pattern_to_regex(model_name)
    match: Final = re.match(regex, requested) or re.match(regex, f"{inferred_provider(requested)}/{requested}")
    if match is None:
        return deployment
    return {
        **deployment,
        "litellm_params": {**params, "model": PatternMatchRouter.set_deployment_model_name(match, model)},
    }


def attempt_kwargs(
    op: str,
    request: Mapping[str, object],
    deployment: Mapping[str, object],
    *,
    model_group: str,
    default_litellm_params: Mapping[str, object],
    timeout: float | None,
) -> Mapping[str, object]:
    """The keyword arguments for ``litellm.<op>`` against one deployment
    (``_update_kwargs_with_deployment`` and the call sites that merge ``litellm_params``)."""
    params: Final = _as_mapping(deployment.get("litellm_params"))
    model_info: Final = _model_info_for(request, deployment, model_group)
    field: Final = metadata_field(op)
    metadata: Final = dict(_as_mapping(request.get(field)))
    metadata.update(
        {
            "model_group": model_group,
            "deployment": params.get("model"),
            "model_info": model_info,
            "api_base": params.get("api_base"),
            "deployment_model_name": deployment.get("model_name"),
        }
    )
    tags: Final = _as_list(params.get("tags"))
    if tags:
        merged: Final = list(_as_list(metadata.get("tags")))
        merged.extend(tag for tag in tags if tag not in merged)
        metadata["tags"] = merged
    call: Final = {key: value for key, value in request.items() if key not in ROUTING_KEYS}
    for key, value in default_litellm_params.items():
        call.setdefault(key, value)
    call[field] = metadata
    call["model_info"] = model_info
    if "timeout" not in call:
        deployment_timeout: Final = params.get("timeout")
        chosen: Final = deployment_timeout if deployment_timeout is not None else timeout
        if chosen is not None:
            call["timeout"] = chosen
    return without_router_only_kwargs({**params, **call, "model": params.get("model")})


def _model_info_for(
    request: Mapping[str, object], deployment: Mapping[str, object], model_group: str
) -> dict[str, object]:  # mutable-ok: a fresh dict for the call
    """The deployment's ``model_info``, or, for a request carrying its own credentials, the
    request-scoped deployment's (``_handle_clientside_credential``): its own stable id, and the
    configured deployment's id as ``original_model_id``."""
    from litellm.litellm_core_utils import clientside_credential_handler as clientside
    from litellm.router_api import deployment_id

    model_info: Final = dict(_as_mapping(deployment.get("model_info")))
    if not clientside.is_clientside_credential(request_kwargs=dict(request)):  # pyright: ignore[reportUnknownMemberType]  # an untyped helper
        return model_info
    dynamic: Final = _as_mapping(
        clientside.get_dynamic_litellm_params(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # an untyped helper
            litellm_params=dict(_as_mapping(deployment.get("litellm_params"))), request_kwargs=dict(request)
        )
    )
    original: Final = model_info.get("id")
    model_info["id"] = deployment_id(model_group, dynamic)
    model_info["original_model_id"] = original
    return model_info


def stamp_attempt_error(error: BaseException, deployment: Mapping[str, object]) -> None:
    """What the router stamps on a failed attempt's error: the deployment's own ``num_retries``
    (``_set_deployment_num_retries_on_exception``) and its id (``failed_deployment_id``)."""
    params: Final = _as_mapping(deployment.get("litellm_params"))
    if getattr(error, "num_retries", None) is None and params.get("num_retries") is not None:
        try:
            error.num_retries = int(str(params["num_retries"]))  # pyright: ignore[reportAttributeAccessIssue]  # the router stamps the same attribute  # rebind-ok: the router stamps the error in place
        except ValueError:
            pass
    deployment_id: Final = _as_mapping(deployment.get("model_info")).get("id")
    if deployment_id and not getattr(error, "failed_deployment_id", None):
        try:
            error.failed_deployment_id = deployment_id  # pyright: ignore[reportAttributeAccessIssue]  # the router stamps the same attribute  # rebind-ok: see above
        except AttributeError:
            pass


def _as_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        entries: Final[Mapping[object, object]] = value  # pyright: ignore[reportUnknownVariableType]  # a request's untyped mapping
        return {str(key): item for key, item in entries.items()}
    return MappingProxyType({})


def _as_list(value: object) -> list[object]:  # mutable-ok: a fresh list
    if isinstance(value, list):
        items: Final[list[object]] = value  # pyright: ignore[reportUnknownVariableType]  # a request's untyped list  # mutable-ok: read, then copied
        return list(items)
    return []
