"""The one identity every deployment-scoped Prometheus series is keyed by.

The cooldown path (`litellm.router_utils.cooldown_callbacks.router_cooldown_event_callback`) and the call paths
(`PrometheusLogger.set_llm_deployment_success_metrics` / `set_llm_deployment_failure_metrics`) both go through
this module, so setting a deployment to complete outage and clearing it on recovery write the same series.

Labels are read from the deployment record, never from the request. The router stamps its attribution into
`metadata` for some endpoints and `litellm_metadata` for others, and a caller can write both dicts, so the only
values a series may carry are the ones the configured deployment itself holds. A request naming an id the
registry does not know keeps its deployment labels empty, which bounds cardinality to the configured deployments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.types.router import Deployment


@dataclass(frozen=True, slots=True)
class DeploymentStateIdentity:
    litellm_model_name: str
    model_id: str
    api_base: str
    api_provider: str


EMPTY_IDENTITY: Final = DeploymentStateIdentity("", "", "", "")

_METADATA_FIELDS: Final = ("metadata", "litellm_metadata")


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _metadata_buckets(request_kwargs: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    litellm_params: Final = _mapping(request_kwargs.get("litellm_params"))
    logged: Final = _mapping(request_kwargs.get("standard_logging_object"))
    named: Final = (
        *(_mapping(litellm_params.get(field)) for field in _METADATA_FIELDS),
        *(_mapping(request_kwargs.get(field)) for field in _METADATA_FIELDS),
        _mapping(logged.get("metadata")),
        _mapping(request_kwargs.get("model_info")),
    )
    return tuple(bucket for bucket in named if bucket)


def _model_infos(request_kwargs: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    found: Final = tuple(_mapping(bucket.get("model_info")) for bucket in _metadata_buckets(request_kwargs))
    return tuple(info for info in (*found, _mapping(request_kwargs.get("model_info"))) if info)


def _candidate_model_ids(request_kwargs: Mapping[str, object]) -> tuple[str, ...]:
    """Every id a request claims, a request-scoped id's configured original first."""
    infos: Final = _model_infos(request_kwargs)
    logged: Final = _mapping(request_kwargs.get("standard_logging_object"))
    claimed: Final = (
        *(_text(info.get("original_model_id")) for info in infos),
        *(_text(info.get("id")) for info in infos),
        _text(logged.get("model_id")),
    )
    return tuple(dict.fromkeys(candidate for candidate in claimed if candidate))


def resolve_deployment(
    request_kwargs: Mapping[str, object], lookup: Callable[[str], Deployment | None]
) -> Deployment | None:
    """The configured deployment a request names, or None when it names none the registry knows."""
    return next(
        (found for candidate in _candidate_model_ids(request_kwargs) if (found := lookup(candidate)) is not None),
        None,
    )


def normalize_api_base(api_base: str | None) -> str:
    from litellm.litellm_core_utils.litellm_logging import (
        StandardLoggingPayloadSetup,
        mask_api_base_credentials,
    )

    if not api_base:
        return ""
    return StandardLoggingPayloadSetup.strip_trailing_slash(mask_api_base_credentials(api_base)) or ""


def resolve_api_provider(model: str, declared_provider: object) -> str:
    declared: Final = _text(declared_provider)
    if declared is not None:
        return declared
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    try:
        _, provider, _, _ = get_llm_provider(model=model, custom_llm_provider=None, api_base=None, api_key=None)
    except Exception:
        return ""
    return provider or ""


def identity_from_request_kwargs(
    request_kwargs: Mapping[str, object], lookup: Callable[[str], Deployment | None] | None = None
) -> DeploymentStateIdentity:
    """Identity of the deployment a logged call was routed to, matching `identity_from_deployment`.

    `lookup` is the router registry, the only attribution a caller cannot write. Without it, as in-process
    `litellm.Router` callbacks, the request's own stamped metadata is read, which no proxy caller can reach.
    """
    if lookup is not None:
        deployment: Final = resolve_deployment(request_kwargs, lookup)
        return EMPTY_IDENTITY if deployment is None else identity_from_deployment(deployment)
    return _stamped_identity(request_kwargs)


def _stamped_identity(request_kwargs: Mapping[str, object]) -> DeploymentStateIdentity:
    stamped: Final = next((bucket for bucket in _metadata_buckets(request_kwargs) if "model_info" in bucket), {})
    litellm_params: Final = _mapping(request_kwargs.get("litellm_params"))
    logged: Final = _mapping(request_kwargs.get("standard_logging_object"))
    model_info: Final = _mapping(stamped.get("model_info"))
    litellm_model_name: Final = _text(stamped.get("deployment")) or _text(request_kwargs.get("model")) or ""
    api_base: Final = (
        normalize_api_base(_text(stamped.get("api_base")))
        if "api_base" in stamped
        else normalize_api_base(_text(litellm_params.get("api_base")) or _text(logged.get("api_base")))
    )
    return DeploymentStateIdentity(
        litellm_model_name=litellm_model_name,
        model_id=(
            _text(model_info.get("original_model_id"))
            or _text(logged.get("model_id"))
            or _text(model_info.get("id"))
            or ""
        ),
        api_base=api_base,
        api_provider=resolve_api_provider(
            litellm_model_name,
            _text(litellm_params.get("custom_llm_provider")) or _text(logged.get("custom_llm_provider")),
        ),
    )


def identity_from_deployment(deployment: Deployment) -> DeploymentStateIdentity:
    """Identity of a deployment record, matching `identity_from_request_kwargs` for a call it serves."""
    litellm_params: Final = deployment.litellm_params
    return DeploymentStateIdentity(
        litellm_model_name=litellm_params.model or "",
        model_id="" if deployment.model_info.id is None else str(deployment.model_info.id),
        api_base=normalize_api_base(litellm_params.api_base),
        api_provider=resolve_api_provider(litellm_params.model or "", litellm_params.get("custom_llm_provider")),
    )
