"""
Contains utils used by OpenAI compatible endpoints
"""

from typing import Final

from fastapi import Request

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body

SENSITIVE_DATA_MASKER: Final = SensitiveDataMasker()


def remove_sensitive_info_from_deployment(
    deployment_dict: dict,
    excluded_keys: set[str] | None = None,
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
    deployment_dict["litellm_params"].pop("vertex_ai_credentials", None)
    deployment_dict["litellm_params"].pop("aws_access_key_id", None)
    deployment_dict["litellm_params"].pop("aws_secret_access_key", None)

    # Rate-limit config fields must never be masked — they are integers, not credentials.
    # The field names contain "key" which matches the masker's sensitive pattern, so we
    # explicitly exclude them here rather than widening the global non_sensitive_overrides.
    _rate_limit_config_keys: Final = {
        "default_api_key_tpm_limit",
        "default_api_key_rpm_limit",
    }
    _excluded: Final = (excluded_keys or set()) | _rate_limit_config_keys

    deployment_dict["litellm_params"] = SENSITIVE_DATA_MASKER.mask_dict(
        deployment_dict["litellm_params"], excluded_keys=_excluded
    )

    return deployment_dict


async def get_custom_llm_provider_from_request_body(request: Request) -> str | None:
    """
    Get the `custom_llm_provider` from the request body

    Safely reads the request body
    """
    request_body: Final[dict] = await _read_request_body(request=request) or {}
    if "custom_llm_provider" in request_body:
        return request_body["custom_llm_provider"]
    return None


def get_custom_llm_provider_from_request_query(request: Request) -> str | None:
    """
    Get the `custom_llm_provider` from the request query parameters

    Safely reads the request query parameters
    """
    if "custom_llm_provider" in request.query_params:
        return request.query_params["custom_llm_provider"]
    return None


def get_custom_llm_provider_from_request_headers(request: Request) -> str | None:
    """
    Get the `custom_llm_provider` from the request header `custom-llm-provider`
    """
    if "custom-llm-provider" in request.headers:
        return request.headers["custom-llm-provider"]
    return None


def apply_openai_project_to_data(
    data: dict,
    request: Request,
    general_settings: dict | None = None,
) -> None:
    """
    Resolve the OpenAI `project` for a files/batches request and store it in `data`.

    A caller can ask for a project with the `OpenAI-Project` header or a `project`
    field in the request body / multipart form. The proxy holds one OpenAI credential
    that reaches every project under it, so a caller-supplied project is a credential
    selector: it is only forwarded when the admin opted in with
    `general_settings: forward_openai_project: true`, mirroring `forward_openai_org_id`.
    Without the opt-in both inputs are dropped and the deployment credential or the
    server-side `OPENAI_PROJECT` keeps control.

    Every files/batches handler calls this once so a file or batch created in a project
    can still be retrieved, listed, cancelled or deleted through the proxy afterwards.
    """
    if not isinstance(general_settings, dict) or general_settings.get("forward_openai_project") is not True:
        data.pop("project", None)
        return
    header_project: Final[str | None] = request.headers.get("OpenAI-Project")
    body_value: Final[object] = data.get("project")
    body_project: Final[str | None] = body_value if isinstance(body_value, str) else None
    resolved_project: Final[str | None] = header_project or body_project or None
    if resolved_project is None:
        data.pop("project", None)
        return
    data["project"] = resolved_project
