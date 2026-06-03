from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

DEFAULT_COMETAPI_API_BASE = "https://api.cometapi.com/v1"


def get_cometapi_api_key(api_key: Optional[str] = None) -> Optional[str]:
    return (
        api_key or get_secret_str("COMETAPI_KEY") or get_secret_str("COMETAPI_API_KEY")
    )


def get_cometapi_api_base(api_base: Optional[str] = None) -> str:
    return (
        api_base
        or get_secret_str("COMETAPI_BASE_URL")
        or get_secret_str("COMETAPI_API_BASE")
        or DEFAULT_COMETAPI_API_BASE
    )


def get_cometapi_complete_url(api_base: Optional[str], endpoint: str) -> str:
    base_url = get_cometapi_api_base(api_base).rstrip("/")
    normalized_endpoint = endpoint.strip("/")
    parsed_base_url = urlsplit(base_url)
    path_segments = [segment for segment in parsed_base_url.path.split("/") if segment]
    endpoint_segments = normalized_endpoint.split("/")
    invalid_version_segments = [
        segment
        for segment in path_segments
        if segment.startswith("v") and segment[1:2].isdigit() and segment != "v1"
    ]
    if "v1" not in path_segments and invalid_version_segments:
        raise ValueError("CometAPI OpenAI-compatible endpoints require a /v1 api_base")

    if path_segments[-len(endpoint_segments) :] == endpoint_segments:
        if "v1" not in path_segments:
            raise ValueError(
                "CometAPI OpenAI-compatible endpoints require a /v1 api_base"
            )
        return base_url

    if "v1" in path_segments:
        complete_path_segments = path_segments + endpoint_segments
    else:
        complete_path_segments = path_segments + ["v1"] + endpoint_segments

    complete_path = "/" + "/".join(complete_path_segments)
    return urlunsplit(
        (
            parsed_base_url.scheme,
            parsed_base_url.netloc,
            complete_path,
            parsed_base_url.query,
            parsed_base_url.fragment,
        )
    )


def require_cometapi_api_key(api_key: Optional[str] = None) -> str:
    final_api_key = get_cometapi_api_key(api_key)
    if not final_api_key:
        raise ValueError("COMETAPI_KEY or COMETAPI_API_KEY is not set")
    return final_api_key


class CometAPIException(BaseLLMException):
    """CometAPI exception handling class"""

    pass
