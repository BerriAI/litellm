import ast
import json
from typing import Dict, List, Optional

from fastapi import Request, UploadFile, status

from litellm._logging import verbose_proxy_logger
from litellm.types.router import Deployment


async def _read_request_body(request: Optional[Request]) -> Dict:
    """
    Safely read the request body and parse it as JSON.

    Parameters:
    - request: The request object to read the body from

    Returns:
    - dict: Parsed request data as a dictionary or an empty dictionary if parsing fails
    """
    try:
        if request is None:
            return {}

        # Read the request body
        body = await request.body()

        # Return empty dict if body is empty or None
        if not body:
            return {}

        # Decode the body to a string
        body_str = body.decode()

        # Attempt JSON parsing (safe for untrusted input)
        return json.loads(body_str)

    except json.JSONDecodeError:
        # Log detailed information for debugging
        verbose_proxy_logger.exception("Invalid JSON payload received.")
        return {}

    except Exception as e:
        # Catch unexpected errors to avoid crashes
        verbose_proxy_logger.exception(
            "Unexpected error reading request body - {}".format(e)
        )
        return {}


def check_file_size_under_limit(
    request_data: dict,
    file: UploadFile,
    router_model_names: List[str],
) -> bool:
    """
    Check if any files passed in request are under max_file_size_mb

    Returns True -> when file size is under max_file_size_mb limit
    Raises ProxyException -> when file size is over max_file_size_mb limit or not a premium_user
    """
    from litellm.proxy.proxy_server import (
        CommonProxyErrors,
        ProxyException,
        llm_router,
        premium_user,
    )

    file_contents_size = file.size or 0
    file_content_size_in_mb = file_contents_size / (1024 * 1024)
    if "metadata" not in request_data:
        request_data["metadata"] = {}
    request_data["metadata"]["file_size_in_mb"] = file_content_size_in_mb
    max_file_size_mb = None

    if llm_router is not None and request_data["model"] in router_model_names:
        try:
            deployment: Optional[Deployment] = (
                llm_router.get_deployment_by_model_group_name(
                    model_group_name=request_data["model"]
                )
            )
            if (
                deployment
                and deployment.litellm_params is not None
                and deployment.litellm_params.max_file_size_mb is not None
            ):
                max_file_size_mb = deployment.litellm_params.max_file_size_mb
        except Exception as e:
            verbose_proxy_logger.error(
                "Got error when checking file size: %s", (str(e))
            )

    if max_file_size_mb is not None:
        verbose_proxy_logger.debug(
            "Checking file size, file content size=%s, max_file_size_mb=%s",
            file_content_size_in_mb,
            max_file_size_mb,
        )
        if not premium_user:
            raise ProxyException(
                message=f"Tried setting max_file_size_mb for /audio/transcriptions. {CommonProxyErrors.not_premium_user.value}",
                code=status.HTTP_400_BAD_REQUEST,
                type="bad_request",
                param="file",
            )
        if file_content_size_in_mb > max_file_size_mb:
            raise ProxyException(
                message=f"File size is too large. Please check your file size. Passed file size: {file_content_size_in_mb} MB. Max file size: {max_file_size_mb} MB",
                code=status.HTTP_400_BAD_REQUEST,
                type="bad_request",
                param="file",
            )

    return True
