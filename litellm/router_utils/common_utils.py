import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.types.llms.openai import OpenAIFileObject

from litellm._logging import verbose_logger
from litellm.constants import ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS
from litellm.exceptions import BadRequestError
from litellm.types.router import CredentialLiteLLMParams


def _is_proxy_admin_request(request_kwargs: Mapping[str, object] | None) -> bool:
    if request_kwargs is None:
        return False
    metadata_value: Final = request_kwargs.get("metadata")
    litellm_metadata_value: Final = request_kwargs.get("litellm_metadata")
    metadata: Final = metadata_value if isinstance(metadata_value, Mapping) else {}
    litellm_metadata: Final = litellm_metadata_value if isinstance(litellm_metadata_value, Mapping) else {}
    user_api_key_auth: Final = metadata.get("user_api_key_auth") or litellm_metadata.get("user_api_key_auth")
    return getattr(user_api_key_auth, "user_role", None) == "proxy_admin"


def resolve_model_group_alias(model_group_alias: object, model: str) -> str | None:
    """
    Resolve ``model`` through a ``model_group_alias`` map.

    Handles both supported entry shapes, the plain string form
    ``{"alias": "target"}`` and the item form
    ``{"alias": {"model": "target", "hidden": true}}``, and tolerates malformed
    entries: the map can come from a key or team row rather than from validated
    config, so a bad value must not raise mid-request.

    Returns the target model group, or None when the map does not rewrite ``model``.
    """
    if not isinstance(model_group_alias, Mapping):
        return None
    entry: Final = model_group_alias.get(model)
    target: Final = entry.get("model") if isinstance(entry, Mapping) else entry
    if not isinstance(target, str) or not target:
        return None
    return target


def truncate_fallback_error_detail(detail: str) -> str:
    """
    Bound a fallback failure detail before it is logged or appended to an exception message.

    Each level of the fallback walk records the failure of the level below it, so an
    untruncated detail carries every nested failure with it and grows superlinearly with
    the number of attempted model groups. One deterministic pre-network failure walked
    through a small fallback graph is enough to turn that into hundreds of megabytes of
    output on the event-loop thread, which starves the process that produced it.
    """
    if len(detail) <= ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS:
        return detail
    dropped: Final = len(detail) - ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS
    return f"{detail[:ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS]}... [truncated {dropped} characters]"


def get_litellm_params_sensitive_credential_hash(litellm_params: dict) -> str:
    """
    Hash of the credential params, used for mapping the file id to the right model
    """
    sensitive_params: Final = CredentialLiteLLMParams(**litellm_params)
    return hashlib.sha256(json.dumps(sensitive_params.model_dump()).encode()).hexdigest()


def add_model_file_id_mappings(healthy_deployments: list[dict] | dict, responses: list["OpenAIFileObject"]) -> dict:
    """
    Create a mapping of model id to file id
    {
        "model_id": "file_id",
        "model_id": "file_id",
    }

    `healthy_deployments` may be either a list of deployment dicts (multiple
    matched deployments) or a single deployment dict (when the router resolved
    a specific deployment, e.g. because the requested model matched a
    `model_info.id`). Both shapes must be handled by extracting
    `model_info.id` from each deployment.
    """
    model_file_id_mapping: Final[dict[str, str]] = {}
    deployments_list: Final[list[dict]] = (
        healthy_deployments if isinstance(healthy_deployments, list) else [healthy_deployments]
    )
    for deployment, response in zip(deployments_list, responses):
        model_id = deployment.get("model_info", {}).get("id")
        if model_id is not None:
            model_file_id_mapping[model_id] = response.id
    return model_file_id_mapping


def filter_team_based_models(
    healthy_deployments: list[dict] | dict,
    request_kwargs: dict | None = None,
) -> list[dict] | dict:
    """
    If a model has a team_id

    Only use if request is from that team
    """
    if request_kwargs is None:
        return healthy_deployments

    metadata: Final = request_kwargs.get("metadata") or {}
    litellm_metadata: Final = request_kwargs.get("litellm_metadata") or {}
    request_team_id: Final = metadata.get("user_api_key_team_id") or litellm_metadata.get("user_api_key_team_id")
    if request_team_id is None and _is_proxy_admin_request(request_kwargs) and isinstance(healthy_deployments, list):
        requested_model: Final = (
            request_kwargs.get("model") or metadata.get("model_group") or litellm_metadata.get("model_group")
        )
        candidate_deployments: Final = tuple(
            (deployment.get("model_name"), deployment.get("model_info") or {}) for deployment in healthy_deployments
        )
        team_ids: Final = frozenset(
            team_id
            for _, model_info in candidate_deployments
            for team_id in [model_info.get("team_id")]
            if team_id is not None
        )
        matches_requested_model: Final = (
            isinstance(requested_model, str)
            and bool(candidate_deployments)
            and all(
                model_info.get("team_id") is not None
                and (model_name == requested_model or model_info.get("team_public_model_name") == requested_model)
                for model_name, model_info in candidate_deployments
            )
        )
        if matches_requested_model and len(team_ids) > 1:
            raise BadRequestError(
                message=(
                    f"Model name '{requested_model}' matches deployments from multiple teams. "
                    "Specify the deployment ID directly to disambiguate."
                ),
                model=requested_model,
                llm_provider="",
            )
        if matches_requested_model:
            return healthy_deployments

    ids_to_remove: Final = set()
    if isinstance(healthy_deployments, dict):
        return healthy_deployments
    for deployment in healthy_deployments:
        _model_info = deployment.get("model_info") or {}
        model_team_id = _model_info.get("team_id")
        if model_team_id is None:
            continue
        if model_team_id != request_team_id:
            ids_to_remove.add(_model_info.get("id"))

    return [
        deployment
        for deployment in healthy_deployments
        if deployment.get("model_info", {}).get("id") not in ids_to_remove
    ]


def _deployment_supports_web_search(deployment: dict) -> bool:
    """
    Check if a deployment supports web search.

    Priority:
    1. Check config-level override in model_info.supports_web_search
    2. Default to True (assume supported unless explicitly disabled)

    Note: Ideally we'd fall back to litellm.supports_web_search() but
    model_prices_and_context_window.json doesn't have supports_web_search
    tags on all models yet. TODO: backfill and add fallback.
    """
    model_info: Final = deployment.get("model_info", {})

    if "supports_web_search" in model_info:
        return model_info["supports_web_search"]

    return True


def filter_web_search_deployments(
    healthy_deployments: list[dict] | dict,
    request_kwargs: dict | None = None,
) -> list[dict] | dict:
    """
    If the request is websearch, filter out deployments that don't support web search
    """
    if request_kwargs is None:
        return healthy_deployments
    # When a specific deployment was already chosen, it's returned as a dict
    # rather than a list - nothing to filter, just pass through
    if isinstance(healthy_deployments, dict):
        return healthy_deployments

    is_web_search_request = False
    tools: Final = request_kwargs.get("tools") or []
    for tool in tools:
        # These are the two websearch tools for OpenAI / Azure.
        if tool.get("type") == "web_search" or tool.get("type") == "web_search_preview":
            is_web_search_request = True
            break

    if not is_web_search_request:
        return healthy_deployments

    # Filter out deployments that don't support web search
    final_deployments: Final = [d for d in healthy_deployments if _deployment_supports_web_search(d)]
    if len(healthy_deployments) > 0 and len(final_deployments) == 0:
        verbose_logger.warning("No deployments support web search for request")
    return final_deployments
