import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import orjson
from fastapi import Request, UploadFile, status

from litellm._logging import verbose_proxy_logger
from litellm.types.router import Deployment


@dataclass
class InternalRequestBody:
    """Wrapper to handle request body parsing and caching"""

    _parsed_body: Optional[Dict] = None

    async def get_parsed_body(self, request: Request) -> Dict:
        """Get parsed body, reading it only once"""
        if self._parsed_body is not None:
            return self._parsed_body

        try:
            if "form" in request.headers.get("content-type", ""):
                self._parsed_body = dict(await request.form())
            else:
                body = await request.body()
                self._parsed_body = orjson.loads(body) if body else {}
        except Exception:
            self._parsed_body = {}

        return self._parsed_body or {}


async def _read_request_body(request: Optional[Request]) -> Dict:
    """
    Safely read the request body and parse it as JSON.

    Parameters:
    - request: The request object to read the body from

    Returns:
    - dict: Parsed request data as a dictionary or an empty dictionary if parsing fails
    """
    if request is None:
        return {}

    internal_body = InternalRequestBody()
    return await internal_body.get_parsed_body(request)


def _safe_get_request_parsed_body(request: Optional[Request]) -> Optional[dict]:
    if request is None:
        return None
    if hasattr(request, "state") and hasattr(request.state, "parsed_body"):
        return request.state.parsed_body
    return None


def _safe_set_request_parsed_body(
    request: Optional[Request],
    parsed_body: dict,
) -> None:
    try:
        if request is None:
            return
        request.state.parsed_body = parsed_body
    except Exception as e:
        verbose_proxy_logger.debug(
            "Unexpected error setting request parsed body - {}".format(e)
        )


def _safe_get_request_headers(request: Optional[Request]) -> dict:
    """
    [Non-Blocking] Safely get the request headers
    """
    try:
        if request is None:
            return {}
        return dict(request.headers)
    except Exception as e:
        verbose_proxy_logger.debug(
            "Unexpected error reading request headers - {}".format(e)
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
