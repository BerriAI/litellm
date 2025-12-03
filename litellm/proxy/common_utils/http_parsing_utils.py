import json
import re
from typing import Any, Collection, Dict, List, Optional

import orjson
from fastapi import Request, UploadFile, status

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.callback_utils import (
    get_metadata_variable_name_from_kwargs,
)
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
            if "metadata" in parsed_body and isinstance(parsed_body["metadata"], str):
                parsed_body["metadata"] = json.loads(parsed_body["metadata"])
        else:
            # Read the request body
            body = await request.body()

            # Return empty dict if body is empty or None
            if not body:
                parsed_body = {}
            else:
                try:
                    parsed_body = orjson.loads(body)
                except orjson.JSONDecodeError as e:
                    # First try the standard json module which is more forgiving
                    # First decode bytes to string if needed
                    body_str = body.decode("utf-8") if isinstance(body, bytes) else body

                    # Replace invalid surrogate pairs
                    # This regex finds incomplete surrogate pairs
                    body_str = re.sub(
                        r"[\uD800-\uDBFF](?![\uDC00-\uDFFF])", "", body_str
                    )
                    # This regex finds low surrogates without high surrogates
                    body_str = re.sub(
                        r"(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]", "", body_str
                    )

                    try:
                        parsed_body = json.loads(body_str)
                    except json.JSONDecodeError:
                        # If both orjson and json.loads fail, throw a proper error
                        verbose_proxy_logger.error(f"Invalid JSON payload received: {str(e)}")
                        raise ProxyException(
                            message=f"Invalid JSON payload: {str(e)}",
                            type="invalid_request_error",
                            param="request_body",
                            code=status.HTTP_400_BAD_REQUEST,
                        )

        # Cache the parsed result
        _safe_set_request_parsed_body(request=request, parsed_body=parsed_body)
        return parsed_body

    except (json.JSONDecodeError, orjson.JSONDecodeError, ProxyException) as e:
        # Re-raise ProxyException as-is
        verbose_proxy_logger.error(f"Invalid JSON payload received: {str(e)}")
        raise
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

def _safe_get_request_query_params(request: Optional[Request]) -> Dict:
    if request is None:
        return {}
    try:
        if hasattr(request, "query_params"):
            return dict(request.query_params)
        return {}
    except Exception as e:
        verbose_proxy_logger.debug(
            "Unexpected error reading request query params - {}".format(e)
        )
        return {}

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
    router_model_names: Collection[str],
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


async def convert_upload_files_to_file_data(
    form_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert FastAPI UploadFile objects to file data tuples for litellm.
    
    Converts UploadFile objects to tuples of (filename, content, content_type)
    which is the format expected by httpx and litellm's HTTP handlers.
    
    Args:
        form_data: Dictionary containing form data with potential UploadFile objects
        
    Returns:
        Dictionary with UploadFile objects converted to file data tuples
        
    Example:
        ```python
        form_data = await get_form_data(request)
        data = await convert_upload_files_to_file_data(form_data)
        # data["files"] is now [(filename, content, content_type), ...]
        ```
    """
    data = {}
    for key, value in form_data.items():
        if isinstance(value, list):
            # Check if it's a list of UploadFile objects
            if value and hasattr(value[0], "read"):
                files = []
                for f in value:
                    file_content = await f.read()
                    # Create tuple: (filename, content, content_type)
                    files.append((f.filename, file_content, f.content_type))
                data[key] = files
            else:
                data[key] = value
        elif hasattr(value, "read"):
            # Single UploadFile object - read and convert to list for consistency
            file_content = await value.read()
            data[key] = [(value.filename, file_content, value.content_type)]
        else:
            # Regular form field
            data[key] = value
    return data


async def get_request_body(request: Request) -> Dict[str, Any]:
    """
    Read the request body and parse it as JSON.
    """
    if request.method == "POST":
        if request.headers.get("content-type", "") == "application/json":
            return await _read_request_body(request)
        elif (
            "multipart/form-data" in request.headers.get("content-type", "")
            or "application/x-www-form-urlencoded" in request.headers.get("content-type", "")
        ):
            return await get_form_data(request)
        else:
            raise ValueError(
                f"Unsupported content type: {request.headers.get('content-type')}"
            )
    return {}


def get_tags_from_request_body(request_body: dict) -> List[str]:
    """
    Extract tags from request body metadata.
    
    Args:
        request_body: The request body dictionary
        
    Returns:
        List of tag names (strings), empty list if no valid tags found
    """
    metadata_variable_name = get_metadata_variable_name_from_kwargs(request_body)
    metadata = request_body.get(metadata_variable_name) or {}
    tags_in_metadata: Any = metadata.get("tags", [])
    tags_in_request_body: Any = request_body.get("tags", [])
    combined_tags: List[str] = []

    ######################################
    # Only combine tags if they are lists
    ######################################
    if isinstance(tags_in_metadata, list):
        combined_tags.extend(tags_in_metadata)
    if isinstance(tags_in_request_body, list):
        combined_tags.extend(tags_in_request_body)
    ######################################
    return [tag for tag in combined_tags if isinstance(tag, str)]


def populate_request_with_path_params(
    request_data: dict, request: Request
) -> dict:
    """
    Copy FastAPI path params into the request payload so downstream checks
    (e.g. vector store RBAC) see them the same way as body params.
    
    Since path_params may not be available during dependency injection,
    we parse the URL path directly for known patterns.
    
    Args:
        request_data: The request data dictionary to populate
        request: The FastAPI Request object
        
    Returns:
        dict: Updated request_data with path parameters added
    """    
    # Try to get path_params if available (sometimes populated by FastAPI)
    path_params = getattr(request, "path_params", None)
    if isinstance(path_params, dict) and path_params:
        for key, value in path_params.items():
            if key == "vector_store_id":
                request_data.setdefault("vector_store_id", value)
                existing_ids = request_data.get("vector_store_ids")
                if isinstance(existing_ids, list):
                    if value not in existing_ids:
                        existing_ids.append(value)
                else:
                    request_data["vector_store_ids"] = [value]
                continue
            request_data.setdefault(key, value)
        verbose_proxy_logger.debug(
            f"populate_request_with_path_params: Found path_params, vector_store_ids={request_data.get('vector_store_ids')}"
        )
        return request_data

    # Fallback: parse the URL path directly to extract vector_store_id
    _add_vector_store_id_from_path(request_data=request_data, request=request)

    return request_data


def _add_vector_store_id_from_path(request_data: dict, request: Request) -> None:
    """
    Parse the request path to find /vector_stores/{vector_store_id}/... segments.

    When found, ensure both vector_store_id and vector_store_ids are populated.
    
    Args:
        request_data: The request data dictionary to populate
        request: The FastAPI Request object
    """
    path = request.url.path
    vector_store_match = re.search(r"/vector_stores/([^/]+)/", path)
    if vector_store_match:
        vector_store_id = vector_store_match.group(1)
        verbose_proxy_logger.debug(
            f"populate_request_with_path_params: Extracted vector_store_id={vector_store_id} from path={path}"
        )
        request_data.setdefault("vector_store_id", vector_store_id)
        existing_ids = request_data.get("vector_store_ids")
        if isinstance(existing_ids, list):
            if vector_store_id not in existing_ids:
                existing_ids.append(vector_store_id)
        else:
            request_data["vector_store_ids"] = [vector_store_id]
        verbose_proxy_logger.debug(
            f"populate_request_with_path_params: Updated request_data with vector_store_ids={request_data.get('vector_store_ids')}"
        )
    else:
        verbose_proxy_logger.debug(
            f"populate_request_with_path_params: No vector_store_id present in path={path}"
        )

