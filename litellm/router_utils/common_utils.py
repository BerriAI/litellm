import hashlib
import json
from typing import TYPE_CHECKING, Dict, List, Optional, Union

if TYPE_CHECKING:
    from litellm.types.llms.openai import OpenAIFileObject

from litellm.types.router import CredentialLiteLLMParams


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
