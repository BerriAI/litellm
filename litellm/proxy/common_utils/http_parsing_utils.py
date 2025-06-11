import json
from typing import Any, Dict, List, Optional

import orjson
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

        # Check if we already read and parsed the body
        _cached_request_body: Optional[dict] = _safe_get_request_parsed_body(
            request=request
        )
        if _cached_request_body is not None:
            return _cached_request_body

        _request_headers: dict = _safe_get_request_headers(request=request)
        content_type = _request_headers.get("content-type", "")

        if "form" in content_type:
            parsed_body = dict(await request.form())
        else:
            # Read the request body
            body = await request.body()

            # Return empty dict if body is empty or None
            if not body:
                parsed_body = {}
            else:
                try:
                    parsed_body = orjson.loads(body)
                except orjson.JSONDecodeError:
                    # Fall back to the standard json module which is more forgiving
                    # First decode bytes to string if needed
                    body_str = body.decode("utf-8") if isinstance(body, bytes) else body

                    # Replace invalid surrogate pairs
                    import re

                    # This regex finds incomplete surrogate pairs
                    body_str = re.sub(
                        r"[\uD800-\uDBFF](?![\uDC00-\uDFFF])", "", body_str
                    )
                    # This regex finds low surrogates without high surrogates
                    body_str = re.sub(
                        r"(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]", "", body_str
                    )

                    parsed_body = json.loads(body_str)

        # Cache the parsed result
        _safe_set_request_parsed_body(request=request, parsed_body=parsed_body)
        return parsed_body

    except (json.JSONDecodeError, orjson.JSONDecodeError):
        verbose_proxy_logger.exception("Invalid JSON payload received.")
        return {}
    except Exception as e:
        # Catch unexpected errors to avoid crashes
        verbose_proxy_logger.exception(
            "Unexpected error reading request body - {}".format(e)
        )
        return {}


def _safe_get_request_parsed_body(request: Optional[Request]) -> Optional[dict]:
    if request is None:
        return None
    if (
        hasattr(request, "scope")
        and "parsed_body" in request.scope
        and isinstance(request.scope["parsed_body"], tuple)
    ):
        accepted_keys, parsed_body = request.scope["parsed_body"]
        return {key: parsed_body[key] for key in accepted_keys}
    return None


def _safe_set_request_parsed_body(
    request: Optional[Request],
    parsed_body: dict,
) -> None:
    try:
        if request is None:
            return
        request.scope["parsed_body"] = (tuple(parsed_body.keys()), parsed_body)
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
            deployment: Optional[
                Deployment
            ] = llm_router.get_deployment_by_model_group_name(
                model_group_name=request_data["model"]
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


async def get_form_data(request: Request) -> Dict[str, Any]:
    """
    Read form data from request

    Handles when OpenAI SDKs pass form keys as `timestamp_granularities[]="word"` instead of `timestamp_granularities=["word", "sentence"]`
    """
    form = await request.form()
    form_data = dict(form)
    parsed_form_data: dict[str, Any] = {}
    for key, value in form_data.items():
        # OpenAI SDKs pass form keys as `timestamp_granularities[]="word"` instead of `timestamp_granularities=["word", "sentence"]`
        if key.endswith("[]"):
            clean_key = key[:-2]
            parsed_form_data.setdefault(clean_key, []).append(value)
        else:
            parsed_form_data[key] = value
    return parsed_form_data


async def get_request_body(request: Request) -> Dict[str, Any]:
    """
    Read the request body and parse it as JSON.
    """
    if request.headers.get("content-type") == "application/json":
        return await _read_request_body(request)
    elif (
        request.headers.get("content-type") == "multipart/form-data"
        or request.headers.get("content-type") == "application/x-www-form-urlencoded"
    ):
        return await get_form_data(request)
    else:
        raise ValueError(
            f"Unsupported content type: {request.headers.get('content-type')}"
        )
