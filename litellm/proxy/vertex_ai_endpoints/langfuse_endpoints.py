"""
What is this? 

Logging Pass-Through Endpoints
"""

"""
1. Create pass-through endpoints for any LITELLM_BASE_URL/langfuse/<endpoint> map to LANGFUSE_BASE_URL/<endpoint>
"""

import base64
import os
from base64 import b64encode
from typing import Optional
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, status

import litellm
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers
from litellm.proxy.litellm_pre_call_utils import _get_dynamic_logging_metadata
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    create_pass_through_route,
)

router = APIRouter()
default_vertex_config = None
_DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"


def create_request_copy(request: Request):
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": _safe_get_request_headers(request).copy(),
        "cookies": request.cookies,
        "query_params": dict(request.query_params),
    }


def _decode_to_convergence(value: str) -> str:
    previous = value
    while True:
        decoded = unquote(previous)
        if decoded == previous:
            return decoded
        previous = decoded


def _normalize_langfuse_base_url(base_target_url: str) -> str:
    if not (
        base_target_url.startswith("http://") or base_target_url.startswith("https://")
    ):
        # Existing behavior allows host-only Langfuse settings.
        base_target_url = "http://" + base_target_url

    try:
        base_url = httpx.URL(base_target_url)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Invalid Langfuse host: {str(e)}"},
        )

    if base_url.scheme not in ("http", "https") or not base_url.host:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid Langfuse host"},
        )

    if base_url.userinfo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Langfuse host must not include credentials"},
        )

    return str(base_url)


def _validate_langfuse_proxy_path(endpoint: str) -> str:
    decoded_endpoint = _decode_to_convergence(endpoint)
    if any(ord(char) < 32 for char in decoded_endpoint):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid Langfuse endpoint path"},
        )
    if "\\" in decoded_endpoint or decoded_endpoint.startswith("//"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid Langfuse endpoint path"},
        )

    endpoint_path = "/" + decoded_endpoint.lstrip("/")
    if any(segment in (".", "..") for segment in endpoint_path.split("/")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid Langfuse endpoint path"},
        )
    return endpoint_path


def _get_langfuse_proxy_credentials(
    *,
    dynamic_host_supplied: bool,
    dynamic_langfuse_public_key: Optional[str],
    dynamic_langfuse_secret_key: Optional[str],
):
    if dynamic_host_supplied:
        if not dynamic_langfuse_public_key or not dynamic_langfuse_secret_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Dynamic Langfuse hosts must include dynamic Langfuse credentials"
                },
            )
        return dynamic_langfuse_public_key, dynamic_langfuse_secret_key

    return (
        dynamic_langfuse_public_key
        or litellm.utils.get_secret(secret_name="LANGFUSE_PUBLIC_KEY"),
        dynamic_langfuse_secret_key
        or litellm.utils.get_secret(secret_name="LANGFUSE_SECRET_KEY"),
    )


def _build_langfuse_proxy_target(
    *,
    endpoint: str,
    base_target_url: str,
    dynamic_host_supplied: bool,
):
    endpoint_path = _validate_langfuse_proxy_path(endpoint)
    base_url = httpx.URL(_normalize_langfuse_base_url(base_target_url))
    updated_url = base_url.copy_with(path=endpoint_path)
    custom_headers = {}

    if dynamic_host_supplied and getattr(litellm, "user_url_validation", True):
        try:
            target_url, host_header = validate_url(str(updated_url))
        except SSRFError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": f"Invalid Langfuse host: {str(e)}"},
            )
        custom_headers["Host"] = host_header
        return target_url, custom_headers

    return str(updated_url), custom_headers


@router.api_route(
    "/langfuse/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Langfuse Pass-through", "pass-through"],
)
async def langfuse_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    Call Langfuse via LiteLLM proxy. Works with Langfuse SDK.

    [Docs](https://docs.litellm.ai/docs/pass_through/langfuse)
    """
    from litellm.proxy.proxy_server import proxy_config

    ## CHECK FOR LITELLM API KEY IN THE QUERY PARAMS - ?..key=LITELLM_API_KEY
    api_key = request.headers.get("Authorization") or ""

    ## decrypt base64 hash
    api_key = api_key.replace("Basic ", "")

    decoded_bytes = base64.b64decode(api_key)
    decoded_str = decoded_bytes.decode("utf-8")
    api_key = decoded_str.split(":")[1]  # assume api key is passed in as secret key

    user_api_key_dict = await user_api_key_auth(
        request=request, api_key="Bearer {}".format(api_key)
    )

    callback_settings_obj: Optional[TeamCallbackMetadata] = (
        _get_dynamic_logging_metadata(
            user_api_key_dict=user_api_key_dict, proxy_config=proxy_config
        )
    )

    dynamic_langfuse_public_key: Optional[str] = None
    dynamic_langfuse_secret_key: Optional[str] = None
    dynamic_langfuse_host: Optional[str] = None
    if (
        callback_settings_obj is not None
        and callback_settings_obj.callback_vars is not None
    ):
        for k, v in callback_settings_obj.callback_vars.items():
            if k == "langfuse_public_key":
                dynamic_langfuse_public_key = v
            elif k == "langfuse_secret_key":
                dynamic_langfuse_secret_key = v
            elif k == "langfuse_host":
                dynamic_langfuse_host = v

    dynamic_host_supplied = dynamic_langfuse_host is not None
    base_target_url: str = (
        dynamic_langfuse_host
        or os.getenv("LANGFUSE_HOST", _DEFAULT_LANGFUSE_HOST)
        or _DEFAULT_LANGFUSE_HOST
    )
    langfuse_public_key, langfuse_secret_key = _get_langfuse_proxy_credentials(
        dynamic_host_supplied=dynamic_host_supplied,
        dynamic_langfuse_public_key=dynamic_langfuse_public_key,
        dynamic_langfuse_secret_key=dynamic_langfuse_secret_key,
    )
    target_url, target_headers = _build_langfuse_proxy_target(
        endpoint=endpoint,
        base_target_url=base_target_url,
        dynamic_host_supplied=dynamic_host_supplied,
    )

    langfuse_combined_key = "Basic " + b64encode(
        f"{langfuse_public_key}:{langfuse_secret_key}".encode("utf-8")
    ).decode("ascii")
    target_headers["Authorization"] = langfuse_combined_key

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=target_url,
        custom_headers=target_headers,
        query_params=dict(request.query_params),  # type: ignore
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value
