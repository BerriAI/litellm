import traceback
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.fine_tuning.main import vertex_fine_tuning_apis_instance
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    create_pass_through_route,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.passthrough_endpoints.vertex_ai import *

router = APIRouter()

default_vertex_config: VertexPassThroughCredentials = VertexPassThroughCredentials()


def _get_vertex_env_vars() -> VertexPassThroughCredentials:
    """
    Helper to get vertex pass through config from environment variables

    The following environment variables are used:
    - DEFAULT_VERTEXAI_PROJECT (project id)
    - DEFAULT_VERTEXAI_LOCATION (location)
    - DEFAULT_GOOGLE_APPLICATION_CREDENTIALS (path to credentials file)
    """
    return VertexPassThroughCredentials(
        vertex_project=get_secret_str("DEFAULT_VERTEXAI_PROJECT"),
        vertex_location=get_secret_str("DEFAULT_VERTEXAI_LOCATION"),
        vertex_credentials=get_secret_str("DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"),
    )


def set_default_vertex_config(config: Optional[dict] = None):
    """Sets vertex configuration from provided config and/or environment variables

    Args:
        config (Optional[dict]): Configuration dictionary
        Example: {
            "vertex_project": "my-project-123",
            "vertex_location": "us-central1",
            "vertex_credentials": "os.environ/GOOGLE_CREDS"
        }
    """
    global default_vertex_config

    # Initialize config dictionary if None
    if config is None:
        default_vertex_config = _get_vertex_env_vars()
        return

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("os.environ/"):
                config[key] = litellm.get_secret(value)

    default_vertex_config = VertexPassThroughCredentials(**config)


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
    "/vertex-ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
    include_in_schema=False,
)
@router.api_route(
    "/vertex_ai/{endpoint:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    tags=["Vertex AI Pass-through", "pass-through"],
)
async def vertex_proxy_route(
    endpoint: str,
    request: Request,
    fastapi_response: Response,
):
    """
    Call LiteLLM proxy via Vertex AI SDK.

    [Docs](https://docs.litellm.ai/docs/pass_through/vertex_ai)
    """
    encoded_endpoint = httpx.URL(endpoint).path

    import re

    verbose_proxy_logger.debug("requested endpoint %s", endpoint)
    headers: dict = {}
    api_key_to_use = get_litellm_virtual_key(request=request)
    user_api_key_dict = await user_api_key_auth(
        request=request,
        api_key=api_key_to_use,
    )

    vertex_project = None
    vertex_location = None
    # Use headers from the incoming request if default_vertex_config is not set
    if default_vertex_config.vertex_project is None:
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
        vertex_project = default_vertex_config.vertex_project
        vertex_location = default_vertex_config.vertex_location
        vertex_credentials = default_vertex_config.vertex_credentials

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
    target = str(updated_url)
    is_streaming_request = False
    if "stream" in str(updated_url):
        is_streaming_request = True
        target += "?alt=sse"

    ## CREATE PASS-THROUGH
    endpoint_func = create_pass_through_route(
        endpoint=endpoint,
        target=target,
        custom_headers=headers,
    )  # dynamically construct pass-through endpoint based on incoming path
    received_value = await endpoint_func(
        request,
        fastapi_response,
        user_api_key_dict,
        stream=is_streaming_request,  # type: ignore
    )

    return received_value


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")
