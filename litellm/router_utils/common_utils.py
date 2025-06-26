import hashlib
import json
from typing import TYPE_CHECKING, Dict, List, Union

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
