import hashlib
import json
from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    from litellm.types.llms.openai import OpenAIFileObject

from litellm.types.router import CredentialLiteLLMParams
from litellm._logging import verbose_logger


def get_litellm_params_sensitive_credential_hash(litellm_params: dict) -> str:
    """
    Hash of the credential params, used for mapping the file id to the right model
    """
    sensitive_params = CredentialLiteLLMParams(**litellm_params)
    return hashlib.sha256(
        json.dumps(sensitive_params.model_dump()).encode()
    ).hexdigest()


def add_model_file_id_mappings(
    healthy_deployments: Union[List[Dict], Dict], responses: List["OpenAIFileObject"]
) -> dict:
    """
    Create a mapping of model name to file id
    {
        "model_id": "file_id",
        "model_id": "file_id",
    }
    """
    model_file_id_mapping = {}
    if isinstance(healthy_deployments, list):
        for deployment, response in zip(healthy_deployments, responses):
            model_file_id_mapping[deployment.get("model_info", {}).get("id")] = (
                response.id
            )
    elif isinstance(healthy_deployments, dict):
        for model_id, file_id in healthy_deployments.items():
            model_file_id_mapping[model_id] = file_id
    return model_file_id_mapping


def filter_team_based_models(
    healthy_deployments: Union[List[Dict], Dict],
    request_kwargs: Optional[Dict] = None,
) -> Union[List[Dict], Dict]:
    """
    If a model has a team_id

    Only use if request is from that team
    """
    if request_kwargs is None:
        return healthy_deployments

    metadata = request_kwargs.get("metadata") or {}
    litellm_metadata = request_kwargs.get("litellm_metadata") or {}
    request_team_id = metadata.get("user_api_key_team_id") or litellm_metadata.get(
        "user_api_key_team_id"
    )
    ids_to_remove = []
    if isinstance(healthy_deployments, dict):
        return healthy_deployments
    for deployment in healthy_deployments:
        _model_info = deployment.get("model_info") or {}
        model_team_id = _model_info.get("team_id")
        if model_team_id is None:
            continue
        if model_team_id != request_team_id:
            ids_to_remove.append(deployment.get("model_info", {}).get("id"))

    return [
        deployment
        for deployment in healthy_deployments
        if deployment.get("model_info", {}).get("id") not in ids_to_remove
    ]

def _deployment_supports_web_search(deployment: Dict) -> bool:
    """
    Check if a deployment supports web search.

    Priority:
    1. Check config-level override in model_info.supports_web_search
    2. Default to True (assume supported unless explicitly disabled)

    Note: Ideally we'd fall back to litellm.supports_web_search() but
    model_prices_and_context_window.json doesn't have supports_web_search
    tags on all models yet. TODO: backfill and add fallback.
    """
    model_info = deployment.get("model_info", {})

    if "supports_web_search" in model_info:
        return model_info["supports_web_search"]

    return True


def filter_web_search_deployments(
    healthy_deployments: Union[List[Dict], Dict],
    request_kwargs: Optional[Dict] = None,
) -> Union[List[Dict], Dict]:
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
    tools = request_kwargs.get("tools") or []
    for tool in tools:
        # These are the two websearch tools for OpenAI / Azure. 
        if tool.get("type") == "web_search" or tool.get("type") == "web_search_preview":
            is_web_search_request = True
            break

    if not is_web_search_request:
        return healthy_deployments

    # Filter out deployments that don't support web search
    final_deployments = [d for d in healthy_deployments if _deployment_supports_web_search(d)]
    if len(healthy_deployments) > 0 and len(final_deployments) == 0:
        verbose_logger.warning("No deployments support web search for request")
    return final_deployments

