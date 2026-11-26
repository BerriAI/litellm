"""The one identity every deployment-scoped Prometheus series is keyed by.

The cooldown path (`litellm.router_utils.cooldown_callbacks.router_cooldown_event_callback`) and the call paths
(`PrometheusLogger.set_llm_deployment_success_metrics` / `set_llm_deployment_failure_metrics`) both go through
this module, so setting a deployment to complete outage and clearing it on recovery write the same series.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


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


def _router_stamped_metadata(request_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """The metadata dict `update_kwargs_with_deployment` stamped the deployment attribution into.

    That is `litellm_metadata` for batch, file and generic API calls, and `metadata` for everything else, so a
    caller that sends the other dict cannot shadow the router's own values.
    """
    from litellm.litellm_core_utils.batch_utils import get_router_metadata_variable_name

    bucket_name: Final = get_router_metadata_variable_name(
        _text(_mapping(request_kwargs.get("standard_logging_object")).get("call_type"))
    )
    other_name: Final = "metadata" if bucket_name == "litellm_metadata" else "litellm_metadata"
    litellm_params: Final = _mapping(request_kwargs.get("litellm_params"))
    stamped: Final = _mapping(litellm_params.get(bucket_name)) or _mapping(request_kwargs.get(bucket_name))
    if stamped:
        return stamped
    return _mapping(litellm_params.get(other_name)) or _mapping(request_kwargs.get(other_name))


def identity_from_request_kwargs(request_kwargs: Mapping[str, object]) -> DeploymentStateIdentity:
    """Identity of the deployment a logged call was routed to, matching `identity_from_deployment`.

    The router stamps `deployment`, `api_base` and `model_info` onto the call's metadata when it picks a
    deployment, and those are the values the cooldown path derives from the deployment record itself. A stamped
    empty `api_base` means the deployment configures none, which is not the same as metadata being absent, so the
    key's presence decides whether the call-time value is a fallback.
    """
    litellm_params: Final = _mapping(request_kwargs.get("litellm_params"))
    metadata: Final = _router_stamped_metadata(request_kwargs)
    logged: Final = _mapping(request_kwargs.get("standard_logging_object"))
    litellm_model_name: Final = (
        _text(metadata.get("deployment")) or _text(request_kwargs.get("model")) or _text(logged.get("model")) or ""
    )
    api_base: Final = (
        normalize_api_base(_text(metadata.get("api_base")))
        if "api_base" in metadata
        else normalize_api_base(_text(litellm_params.get("api_base")) or _text(logged.get("api_base")))
    )
    model_info: Final = _mapping(metadata.get("model_info"))
    return DeploymentStateIdentity(
        litellm_model_name=litellm_model_name,
        model_id=(
            _text(model_info.get("original_model_id"))
            or _text(logged.get("model_id"))
            or _text(model_info.get("id"))
            or _text(_mapping(request_kwargs.get("model_info")).get("id"))
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
    litellm_model_name: Final = litellm_params.model or ""
    return _identity(
        litellm_model_name=litellm_model_name,
        model_id=deployment.model_info.id,
        api_base=litellm_params.api_base,
        declared_provider=litellm_params.get("custom_llm_provider"),
    )


def _identity(
    *, litellm_model_name: str, model_id: object, api_base: str | None, declared_provider: object
) -> DeploymentStateIdentity:
    return DeploymentStateIdentity(
        litellm_model_name=litellm_model_name,
        model_id="" if model_id is None else str(model_id),
        api_base=normalize_api_base(api_base),
        api_provider=resolve_api_provider(litellm_model_name, declared_provider),
    )
