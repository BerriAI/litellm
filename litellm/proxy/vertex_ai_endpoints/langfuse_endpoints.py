"""
What is this? 

Logging Pass-Through Endpoints
"""

"""
1. Create pass-through endpoints for any LITELLM_BASE_URL/langfuse/<endpoint> map to LANGFUSE_BASE_URL/<endpoint>
"""

import ast
import asyncio
import base64
import traceback
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlencode

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
from starlette.datastructures import QueryParams

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.batches.main import FileObject
from litellm.fine_tuning.main import vertex_fine_tuning_apis_instance
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.litellm_pre_call_utils import _get_dynamic_logging_metadata
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    create_pass_through_route,
)

router = APIRouter()
default_vertex_config = None


def create_request_copy(request: Request):
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "cookies": request.cookies,
        "query_params": dict(request.query_params),
    }


@router.api_route("/langfuse/{endpoint:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def langfuse_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
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
        _get_dynamic_logging_metadata(user_api_key_dict=user_api_key_dict)
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

    base_target_url: str = (
        dynamic_langfuse_host
        or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        or "https://cloud.langfuse.com"
    )
    if not (
        base_target_url.startswith("http://") or base_target_url.startswith("https://")
    ):
        # add http:// if unset, assume communicating over private network - e.g. render
        base_target_url = "http://" + base_target_url

    encoded_endpoint = httpx.URL(endpoint).path

    # Ensure endpoint starts with '/' for proper URL construction
    if not encoded_endpoint.startswith("/"):
        encoded_endpoint = "/" + encoded_endpoint

    # Construct the full target URL using httpx
    base_url = httpx.URL(base_target_url)
    updated_url = base_url.copy_with(path=encoded_endpoint)

    # Add or update query parameters
    langfuse_public_key = dynamic_langfuse_public_key or litellm.utils.get_secret(
        secret_name="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key = dynamic_langfuse_secret_key or litellm.utils.get_secret(
        secret_name="LANGFUSE_SECRET_KEY"
    )

    langfuse_combined_key = "Basic " + b64encode(
        f"{langfuse_public_key}:{langfuse_secret_key}".encode("utf-8")
    ).decode("ascii")

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=str(updated_url),
        custom_headers={"Authorization": langfuse_combined_key},
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
    )

    return received_value
