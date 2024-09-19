import ast
import asyncio
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import fastapi
import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.main import FileObject
from litellm.fine_tuning.main import vertex_fine_tuning_apis_instance
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    create_pass_through_route,
)

router = APIRouter()
default_vertex_config = None


def set_default_vertex_config(config):
    global default_vertex_config
    if config is None:
        return

    if not isinstance(config, dict):
        raise ValueError("invalid config, vertex default config must be a dictionary")

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                config[key] = litellm.get_secret(value)

    default_vertex_config = config


def exception_handler(e: Exception):
    verbose_proxy_logger.error(
        "litellm.proxy.proxy_server.v1/projects/tuningJobs(): Exception occurred - {}".format(
            str(e)
        )
    )
    verbose_proxy_logger.debug(traceback.format_exc())
    if isinstance(e, HTTPException):
        return ProxyException(
            message=getattr(e, "message", str(e.detail)),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
        )
    else:
        error_msg = f"{str(e)}"
        return ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )


def construct_target_url(
    base_url: str,
    requested_route: str,
    default_vertex_location: Optional[str],
    default_vertex_project: Optional[str],
) -> httpx.URL:
    """
    Allow user to specify their own project id / location.

    If missing, use defaults

    Handle cachedContent scenario - https://github.com/BerriAI/litellm/issues/5460

    Constructed Url:
    POST https://LOCATION-aiplatform.googleapis.com/{version}/projects/PROJECT_ID/locations/LOCATION/cachedContents
    """
    new_base_url = httpx.URL(base_url)
    if "locations" in requested_route:  # contains the target project id + location
        updated_url = new_base_url.copy_with(path=requested_route)
        return updated_url
    """
    - Add endpoint version (e.g. v1beta for cachedContent, v1 for rest)
    - Add default project id
    - Add default location
    """
    vertex_version: Literal["v1", "v1beta1"] = "v1"
    if "cachedContent" in requested_route:
        vertex_version = "v1beta1"

    base_requested_route = "{}/projects/{}/locations/{}".format(
        vertex_version, default_vertex_project, default_vertex_location
    )

    updated_requested_route = "/" + base_requested_route + requested_route

    updated_url = new_base_url.copy_with(path=updated_requested_route)
    return updated_url


@router.api_route(
    "/vertex-ai/{endpoint:path}", methods=["GET", "POST", "PUT", "DELETE"]
)
async def vertex_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    encoded_endpoint = httpx.URL(endpoint).path

    import re

    verbose_proxy_logger.debug("requested endpoint %s", endpoint)
    headers: dict = {}

    vertex_project = None
    vertex_location = None
    # Use headers from the incoming request if default_vertex_config is not set
    if default_vertex_config is None:
        headers = dict(request.headers) or {}
        verbose_proxy_logger.debug(
            "default_vertex_config  not set, incoming request headers %s", headers
        )
        # extract location from endpoint, endpoint
        # "v1beta1/projects/adroit-crow-413218/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"
        match = re.search(r"/locations/([^/]+)", endpoint)
        vertex_location = match.group(1) if match else None
        base_target_url = f"https://{vertex_location}-aiplatform.googleapis.com/"
        headers.pop("content-length", None)
        headers.pop("host", None)
    else:
        vertex_project = default_vertex_config.get("vertex_project")
        vertex_location = default_vertex_config.get("vertex_location")
        vertex_credentials = default_vertex_config.get("vertex_credentials")

        base_target_url = f"https://{vertex_location}-aiplatform.googleapis.com/"

        _auth_header, vertex_project = (
            await vertex_fine_tuning_apis_instance._ensure_access_token_async(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai_beta",
            )
        )

        auth_header, _ = vertex_fine_tuning_apis_instance._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base="",
        )

        headers = {
            "Authorization": f"Bearer {auth_header}",
        }

    request_route = encoded_endpoint
    verbose_proxy_logger.debug("request_route %s", request_route)

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    updated_url = construct_target_url(
        base_url=base_target_url,
        requested_route=encoded_endpoint,
        default_vertex_location=vertex_location,
        default_vertex_project=vertex_project,
    )
    # base_url = httpx.URL(base_target_url)
    # updated_url = base_url.copy_with(path=encoded_endpoint)

    verbose_proxy_logger.debug("updated url %s", updated_url)

    ## check for streaming
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers=headers,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
        stream=is_streaming_request,
    )

    return received_value
