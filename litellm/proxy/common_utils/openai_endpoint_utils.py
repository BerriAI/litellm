"""
Contains utils used by OpenAI compatible endpoints 
"""

from typing import Optional, Set

from fastapi import Request

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

SENSITIVE_DATA_MASKER = SensitiveDataMasker()


def remove_sensitive_info_from_deployment(
    deployment_dict: dict,
    excluded_keys: Optional[Set[str]] = None,
) -> dict:
    """
    Removes sensitive information from a deployment dictionary.

    Args:
        deployment_dict (dict): The deployment dictionary to remove sensitive information from.
        excluded_keys (Optional[Set[str]]): Set of keys that should not be masked (exact match).

    Returns:
        dict: The modified deployment dictionary with sensitive information removed.
    """
    deployment_dict["litellm_params"].pop("api_key", None)
    deployment_dict["litellm_params"].pop("client_secret", None)
    deployment_dict["litellm_params"].pop("vertex_credentials", None)
    deployment_dict["litellm_params"].pop("aws_access_key_id", None)
    deployment_dict["litellm_params"].pop("aws_secret_access_key", None)

    deployment_dict["litellm_params"] = SENSITIVE_DATA_MASKER.mask_dict(
        deployment_dict["litellm_params"], excluded_keys=excluded_keys
    )

    return deployment_dict


async def get_custom_llm_provider_from_request_body(request: Request) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request body

    Safely reads the request body
    """
    request_body: dict = await _read_request_body(request=request) or {}
    if "custom_llm_provider" in request_body:
        return request_body["custom_llm_provider"]
    return None


def get_custom_llm_provider_from_request_query(request: Request) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request query parameters

    Safely reads the request query parameters
    """
    if "custom_llm_provider" in request.query_params:
        return request.query_params["custom_llm_provider"]
    return None

def get_custom_llm_provider_from_request_headers(request: Request) -> Optional[str]:
    """
    Get the `custom_llm_provider` from the request header `custom-llm-provider`
    """
    if "custom-llm-provider" in request.headers:
        return request.headers["custom-llm-provider"]
    return None
