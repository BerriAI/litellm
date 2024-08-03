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

router = APIRouter()
default_vertex_config = None


def set_default_vertex_config(config):
    global default_vertex_config
    if config is None:
        return

    if not isinstance(config, list):
        raise ValueError("invalid files config, expected a list is not a list")

    for element in config:
        if isinstance(element, dict):
            for key, value in element.items():
                if isinstance(value, str) and value.startswith("os.environ/"):
                    element[key] = litellm.get_secret(value)

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


async def execute_post_vertex_ai_request(
    request: Request,
    route: str,
):
    from litellm.fine_tuning.main import vertex_fine_tuning_apis_instance

    vertex_project = default_vertex_config.get("vertex_project", None)
    vertex_location = default_vertex_config.get("vertex_location", None)
    vertex_credentials = default_vertex_config.get("vertex_credentials", None)
    request_data_json = await request.json()

    response = (
        await vertex_fine_tuning_apis_instance.pass_through_vertex_ai_POST_request(
            request_data=request_data_json,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            request_route=route,
        )
    )

    return response


@router.post(
    "/vertex-ai/tuningJobs",
    dependencies=[Depends(user_api_key_auth)],
    tags=["Vertex AI endpoints"],
)
async def vertex_create_fine_tuning_job(
    request: Request,
    fastapi_response: Response,
    endpoint_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    this is a pass through endpoint for the Vertex AI API. /tuningJobs endpoint

    it uses the vertex ai credentials on the proxy and forwards to vertex ai api
    """
    try:
        response = await execute_post_vertex_ai_request(
            request=request,
            route="/tuningJobs",
        )
        return response
    except Exception as e:
        raise exception_handler(e) from e
