"""
Generic container file handler for LiteLLM.

This module provides a single generic handler that can process any container file
endpoint defined in endpoints.json, eliminating the need for individual handler methods.
"""

import json
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import httpx
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.containers.main import (
    ContainerFileListResponse,
    ContainerFileObject,
    DeleteContainerFileResponse,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.containers.transformation import BaseContainerConfig


class EndpointConfig(TypedDict):
    """One endpoint entry of ``litellm/containers/endpoints.json``."""

    name: ReadOnly[str]
    async_name: ReadOnly[str]
    path: ReadOnly[str]
    method: ReadOnly[str]
    path_params: ReadOnly[Sequence[str]]
    query_params: ReadOnly[Sequence[str]]
    response_type: ReadOnly[str]
    is_multipart: NotRequired[ReadOnly[bool]]
    returns_binary: NotRequired[ReadOnly[bool]]


class EndpointsConfig(TypedDict):
    """The parsed ``litellm/containers/endpoints.json`` document."""

    endpoints: ReadOnly[Sequence[EndpointConfig]]


class ContainerErrorDetail(TypedDict, total=False):
    """The ``error`` object of a container API error body."""

    message: ReadOnly[str]


class ContainerResponseBody(TypedDict, total=False):
    """The fields this handler reads off a container API JSON body."""

    error: ReadOnly[ContainerErrorDetail]


_ContainerResponseModel = ContainerFileListResponse | ContainerFileObject | DeleteContainerFileResponse

# Response type mapping
RESPONSE_TYPES: Final[Mapping[str, type[_ContainerResponseModel]]] = {
    "ContainerFileListResponse": ContainerFileListResponse,
    "ContainerFileObject": ContainerFileObject,
    "DeleteContainerFileResponse": DeleteContainerFileResponse,
}

ContainerEndpointResponse = _ContainerResponseModel | bytes | ContainerResponseBody


def _load_endpoints_config() -> EndpointsConfig:
    """Load the endpoints configuration from JSON file."""
    config_path: Final = Path(__file__).parent.parent.parent / "containers" / "endpoints.json"
    with open(config_path) as f:
        return json.load(f)


def _get_endpoint_config(endpoint_name: str) -> EndpointConfig | None:
    """Get config for a specific endpoint by name."""
    config: Final = _load_endpoints_config()
    for endpoint in config["endpoints"]:
        if endpoint["name"] == endpoint_name or endpoint["async_name"] == endpoint_name:
            return endpoint
    return None


def _response_model(response_type_name: str) -> type[_ContainerResponseModel] | None:
    """The pydantic model a container endpoint's ``response_type`` names."""
    return RESPONSE_TYPES.get(response_type_name)


def _build_url(
    api_base: str,
    path_template: str,
    path_params: Mapping[str, object],
) -> str:
    """Build the full URL by substituting path parameters.

    The api_base from get_complete_url already includes /containers and may include
    query parameters. We need to parse the URL, append the path, then preserve the
    query parameters.
    """
    # api_base ends with /containers, path_template starts with /containers
    # So we need to strip /containers from the path
    path_template = path_template.removeprefix("/containers")

    # Substitute path parameters
    for param, value in path_params.items():
        encoded_value = encode_url_path_segment(value, field_name=param)
        path_template = path_template.replace(f"{{{param}}}", encoded_value)

    # Parse the api_base to extract existing query params
    parsed_base: Final = httpx.URL(api_base)

    # Append the path to the existing path (before query params)
    new_path: Final = f"{parsed_base.path.rstrip('/')}{path_template}"

    # Rebuild URL with new path, preserving query params
    final_url: Final = parsed_base.copy_with(path=new_path)

    return str(final_url)


def _build_query_params(
    query_param_names: Sequence[str],
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Build query parameters from kwargs."""
    supplied: Final = ((param_name, kwargs.get(param_name)) for param_name in query_param_names)
    return {name: value if isinstance(value, str) else str(value) for name, value in supplied if value is not None}


def error_message_from_response(response: httpx.Response) -> str:
    try:
        body: Final = response.json()
    except ValueError:
        return response.text

    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        message: Final = body["error"].get("message")
        if isinstance(message, str):
            return message

    return response.text


def raise_for_error_status(response: httpx.Response, container_provider_config: "BaseContainerConfig") -> None:
    if not httpx.codes.is_error(response.status_code):
        return
    raise container_provider_config.get_error_class(
        error_message=error_message_from_response(response),
        status_code=response.status_code,
        headers=response.headers,
    )


def _transform_response(
    response: httpx.Response,
    returns_binary: bool,
    response_type_name: str,
) -> ContainerEndpointResponse:
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    if httpx.codes.is_error(response.status_code):
        raise BaseLLMException(
            status_code=response.status_code,
            message=error_message_from_response(response),
            headers=dict(response.headers),
        )

    if returns_binary:
        return response.content

    response_json: Final[ContainerResponseBody] = response.json()
    if "error" in response_json:
        raise BaseLLMException(
            status_code=response.status_code,
            message=response_json["error"].get("message", str(response_json)),
            headers=dict(response.headers),
        )

    response_type: Final = _response_model(response_type_name)
    if response_type:
        return response_type.model_validate(response_json)
    return response_json


def _prepare_multipart_file_upload(
    file: Any,
    headers: dict[str, object],
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, object]]:
    """
    Prepare file and headers for multipart upload.

    Returns:
        Tuple of (files_dict, headers_without_content_type)
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        extract_file_data,
    )

    extracted: Final = extract_file_data(file)
    filename: Final = extracted.get("filename") or "file"
    content: Final = extracted.get("content") or b""
    content_type: Final = extracted.get("content_type") or "application/octet-stream"
    files: Final = {"file": (filename, content, content_type)}

    # Remove content-type header - httpx will set it automatically for multipart
    headers_copy: Final = headers.copy()
    headers_copy.pop("content-type", None)
    headers_copy.pop("Content-Type", None)

    return files, headers_copy


def _request_headers(
    container_provider_config: "BaseContainerConfig",
    extra_headers: dict[str, object] | None,
    litellm_params: GenericLiteLLMParams,
) -> dict[str, object]:
    """The provider auth headers for a container request."""
    return container_provider_config.validate_environment(
        headers=extra_headers or {},
        api_key=litellm_params.get("api_key", None),
    )


def _request_api_base(
    container_provider_config: "BaseContainerConfig",
    litellm_params: GenericLiteLLMParams,
) -> str:
    """The provider base URL for a container request."""
    return container_provider_config.get_complete_url(
        api_base=litellm_params.get("api_base", None),
        litellm_params=dict(litellm_params),
    )


def _sync_http_client(
    client: HTTPHandler | AsyncHTTPHandler | None,
    litellm_params: GenericLiteLLMParams,
) -> HTTPHandler:
    """The sync HTTP client for a container request, reusing the caller's when usable."""
    if client is None or not isinstance(client, HTTPHandler):
        return _get_httpx_client(params={"ssl_verify": litellm_params.get("ssl_verify", None)})
    return client


def _async_http_client(
    client: HTTPHandler | AsyncHTTPHandler | None,
    litellm_params: GenericLiteLLMParams,
) -> AsyncHTTPHandler:
    """The async HTTP client for a container request, reusing the caller's when usable."""
    if client is None or not isinstance(client, AsyncHTTPHandler):
        return get_async_httpx_client(
            llm_provider=litellm.LlmProviders.OPENAI,
            params={"ssl_verify": litellm_params.get("ssl_verify", None)},
        )
    return client


class GenericContainerHandler:
    """
    Generic handler for container file API endpoints.

    This single handler can process any endpoint defined in endpoints.json,
    eliminating the need for individual handler methods per endpoint.
    """

    def handle(
        self,
        endpoint_name: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: dict[str, object] | None = None,
        extra_query: dict[str, object] | None = None,
        timeout: float | httpx.Timeout = 600,
        _is_async: bool = False,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> Any | Coroutine[object, object, Any]:
        """
        Generic handler for any container file endpoint.

        Args:
            endpoint_name: Name of the endpoint (e.g., "list_container_files")
            container_provider_config: Provider-specific configuration
            litellm_params: LiteLLM parameters including api_key, api_base
            logging_obj: Logging object for request logging
            extra_headers: Additional HTTP headers
            extra_query: Additional query parameters
            timeout: Request timeout
            _is_async: Whether to make async request
            client: Optional HTTP client
            **kwargs: Path params and query params (e.g., container_id, file_id, after, limit)
        """
        if _is_async:
            return self._async_handle(
                endpoint_name=endpoint_name,
                container_provider_config=container_provider_config,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=timeout,
                client=client,
                **kwargs,
            )

        return self._sync_handle(
            endpoint_name=endpoint_name,
            container_provider_config=container_provider_config,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            client=client,
            **kwargs,
        )

    def _sync_handle(
        self,
        endpoint_name: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: dict[str, object] | None = None,
        extra_query: dict[str, object] | None = None,
        timeout: float | httpx.Timeout = 600,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> Any:
        """Synchronous request handler."""
        endpoint_config: Final = _get_endpoint_config(endpoint_name)
        if not endpoint_config:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        # Get HTTP client
        http_client: Final = _sync_http_client(client, litellm_params)

        # Build request
        headers = _request_headers(container_provider_config, extra_headers, litellm_params)
        if extra_headers:
            headers.update(extra_headers)

        api_base: Final = _request_api_base(container_provider_config, litellm_params)

        # Build URL with path params
        path_params: Final = {p: kwargs.get(p, "") for p in endpoint_config.get("path_params", [])}
        url: Final = _build_url(api_base, endpoint_config["path"], path_params)

        # Build query params
        query_params: Final = _build_query_params(endpoint_config.get("query_params", []), kwargs)
        if extra_query:
            query_params.update(extra_query)

        # Log request
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": query_params,
            },
        )

        # Make request
        method: Final = endpoint_config["method"].upper()
        returns_binary: Final = endpoint_config.get("returns_binary", False)
        is_multipart: Final = endpoint_config.get("is_multipart", False)

        # An empty dict passed as `params` to httpx strips any existing query
        # string from the URL (e.g. ?api-version=...).  Use None instead so
        # httpx leaves the URL's own query string intact.
        effective_params: Final = query_params or None

        try:
            if method == "GET":
                response = http_client.get(url=url, headers=headers, params=effective_params)
            elif method == "DELETE":
                response = http_client.delete(url=url, headers=headers, params=effective_params)
            elif method == "POST":
                if is_multipart and "file" in kwargs:
                    files, headers = _prepare_multipart_file_upload(kwargs["file"], headers)
                    response = http_client.post(url=url, headers=headers, params=effective_params, files=files)
                else:
                    response = http_client.post(url=url, headers=headers, params=effective_params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            return _transform_response(
                response=response,
                returns_binary=returns_binary,
                response_type_name=endpoint_config["response_type"],
            )

        except Exception as e:
            raise e

    async def _async_handle(
        self,
        endpoint_name: str,
        container_provider_config: "BaseContainerConfig",
        litellm_params: GenericLiteLLMParams,
        logging_obj: "LiteLLMLoggingObj",
        extra_headers: dict[str, object] | None = None,
        extra_query: dict[str, object] | None = None,
        timeout: float | httpx.Timeout = 600,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        **kwargs: object,
    ) -> Any:
        """Asynchronous request handler."""
        endpoint_config: Final = _get_endpoint_config(endpoint_name)
        if not endpoint_config:
            raise ValueError(f"Unknown endpoint: {endpoint_name}")

        # Get HTTP client
        http_client: Final = _async_http_client(client, litellm_params)

        # Build request
        headers = _request_headers(container_provider_config, extra_headers, litellm_params)
        if extra_headers:
            headers.update(extra_headers)

        api_base: Final = _request_api_base(container_provider_config, litellm_params)

        # Build URL with path params
        path_params: Final = {p: kwargs.get(p, "") for p in endpoint_config.get("path_params", [])}
        url: Final = _build_url(api_base, endpoint_config["path"], path_params)

        # Build query params
        query_params: Final = _build_query_params(endpoint_config.get("query_params", []), kwargs)
        if extra_query:
            query_params.update(extra_query)

        # Log request
        logging_obj.pre_call(
            input="",
            api_key="",
            additional_args={
                "api_base": url,
                "headers": headers,
                "params": query_params,
            },
        )

        # Make request
        method: Final = endpoint_config["method"].upper()
        returns_binary: Final = endpoint_config.get("returns_binary", False)
        is_multipart: Final = endpoint_config.get("is_multipart", False)

        # An empty dict passed as `params` to httpx strips any existing query
        # string from the URL (e.g. ?api-version=...).  Use None instead so
        # httpx leaves the URL's own query string intact.
        effective_params: Final = query_params or None

        try:
            if method == "GET":
                response = await http_client.get(url=url, headers=headers, params=effective_params)
            elif method == "DELETE":
                response = await http_client.delete(url=url, headers=headers, params=effective_params)
            elif method == "POST":
                if is_multipart and "file" in kwargs:
                    files, headers = _prepare_multipart_file_upload(kwargs["file"], headers)
                    response = await http_client.post(url=url, headers=headers, params=effective_params, files=files)
                else:
                    response = await http_client.post(url=url, headers=headers, params=effective_params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            return _transform_response(
                response=response,
                returns_binary=returns_binary,
                response_type_name=endpoint_config["response_type"],
            )

        except Exception as e:
            raise e


# Singleton instance
generic_container_handler: Final = GenericContainerHandler()
