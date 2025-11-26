import ast
import asyncio
import copy
import json
import traceback
from base64 import b64encode
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    status,
)
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect
from websockets.exceptions import (
    ConnectionClosedError,
    ConnectionClosedOK,
    InvalidStatus,
)

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import MAXIMUM_TRACEBACK_LINES_TO_LOG
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.passthrough import BasePassthroughUtils
from litellm.proxy._types import (
    ConfigFieldInfo,
    ConfigFieldUpdate,
    LiteLLMRoutes,
    PassThroughEndpointResponse,
    PassThroughGenericEndpoint,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardLoggingUserAPIKeyMetadata

from .streaming_handler import PassThroughStreamingHandler
from .success_handler import PassThroughEndpointLogging

router = APIRouter()

pass_through_endpoint_logging = PassThroughEndpointLogging()

# Global registry to track registered pass-through routes and prevent memory leaks
_registered_pass_through_routes: Dict[str, Dict[str, Union[str, Dict[str, Any]]]] = {}


def get_response_body(response: httpx.Response) -> Optional[dict]:
    try:
        return response.json()
    except Exception:
        return None


async def set_env_variables_in_header(custom_headers: Optional[dict]) -> Optional[dict]:
    """
    checks if any headers on config.yaml are defined as os.environ/COHERE_API_KEY etc

    only runs for headers defined on config.yaml

    example header can be

    {"Authorization": "Bearer os.environ/COHERE_API_KEY"}
    """
    if custom_headers is None:
        return None
    headers = {}
    for key, value in custom_headers.items():
        # langfuse Api requires base64 encoded headers - it's simpleer to just ask litellm users to set their langfuse public and secret keys
        # we can then get the b64 encoded keys here
        if key == "LANGFUSE_PUBLIC_KEY" or key == "LANGFUSE_SECRET_KEY":
            # langfuse requires b64 encoded headers - we construct that here
            _langfuse_public_key = custom_headers["LANGFUSE_PUBLIC_KEY"]
            _langfuse_secret_key = custom_headers["LANGFUSE_SECRET_KEY"]
            if isinstance(
                _langfuse_public_key, str
            ) and _langfuse_public_key.startswith("os.environ/"):
                _langfuse_public_key = get_secret_str(_langfuse_public_key)
            if isinstance(
                _langfuse_secret_key, str
            ) and _langfuse_secret_key.startswith("os.environ/"):
                _langfuse_secret_key = get_secret_str(_langfuse_secret_key)
            headers["Authorization"] = "Basic " + b64encode(
                f"{_langfuse_public_key}:{_langfuse_secret_key}".encode("utf-8")
            ).decode("ascii")
        else:
            # for all other headers
            headers[key] = value
            if isinstance(value, str) and "os.environ/" in value:
                verbose_proxy_logger.debug(
                    "pass through endpoint - looking up 'os.environ/' variable"
                )
                # get string section that is os.environ/
                start_index = value.find("os.environ/")
                _variable_name = value[start_index:]

                verbose_proxy_logger.debug(
                    "pass through endpoint - getting secret for variable name: %s",
                    _variable_name,
                )
                _secret_value = get_secret_str(_variable_name)
                if _secret_value is not None:
                    new_value = value.replace(_variable_name, _secret_value)
                    headers[key] = new_value
    return headers


async def chat_completion_pass_through_endpoint(  # noqa: PLR0915
    fastapi_response: Response,
    request: Request,
    adapter_id: str,
    user_api_key_dict: UserAPIKeyAuth,
):
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data = {}
    try:
        body = await request.body()
        body_str = body.decode()
        try:
            data = ast.literal_eval(body_str)
        except Exception:
            data = json.loads(body_str)

        data["adapter_id"] = adapter_id

        verbose_proxy_logger.debug(
            "Request received by LiteLLM:\n{}".format(json.dumps(data, indent=4)),
        )
        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        data = await add_litellm_data_to_request(
            data=data,  # type: ignore
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        # Check key-specific aliases
        if (
            isinstance(data["model"], str)
            and user_api_key_dict.aliases
            and isinstance(user_api_key_dict.aliases, dict)
            and data["model"] in user_api_key_dict.aliases
        ):
            data["model"] = user_api_key_dict.aliases[data["model"]]

        ### CALL HOOKS ### - modify incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=data, call_type="text_completion"
        )

        ### ROUTE THE REQUESTs ###
        router_model_names = llm_router.model_names if llm_router is not None else []
        # skip router if user passed their key
        if "api_key" in data:
            llm_response = asyncio.create_task(litellm.aadapter_completion(**data))
        elif (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            llm_response = asyncio.create_task(
                llm_router.aadapter_completion(**data, specific_deployment=True)
            )
        elif llm_router is not None and llm_router.has_model_id(
            data["model"]
        ):  # model in router model list
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and llm_router.default_deployment is not None
        ):  # model in router deployments, calling a specific deployment on the router
            llm_response = asyncio.create_task(llm_router.aadapter_completion(**data))
        elif user_model is not None:  # `litellm --model <your-model-name>`
            llm_response = asyncio.create_task(litellm.aadapter_completion(**data))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "completion: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        # Await the llm_response task
        response = await llm_response

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        verbose_proxy_logger.debug("final response: %s", response)

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
            )
        )

        verbose_proxy_logger.debug("\nResponse from Litellm:\n{}".format(response))
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.completion(): Exception occured - {}".format(
                str(e)
            )
        )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )


class HttpPassThroughEndpointHelpers(BasePassthroughUtils):
    @staticmethod
    def get_response_headers(
        headers: httpx.Headers,
        litellm_call_id: Optional[str] = None,
        custom_headers: Optional[dict] = None,
    ) -> dict:
        excluded_headers = {"transfer-encoding", "content-encoding"}

        return_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in excluded_headers
        }
        if litellm_call_id:
            return_headers["x-litellm-call-id"] = litellm_call_id
        if custom_headers:
            return_headers.update(custom_headers)

        return return_headers

    @staticmethod
    def get_endpoint_type(url: str) -> EndpointType:
        parsed_url = urlparse(url)
        if (
            ("generateContent") in url
            or ("streamGenerateContent") in url
            or ("rawPredict") in url
            or ("streamRawPredict") in url
        ):
            return EndpointType.VERTEX_AI
        elif parsed_url.hostname == "api.anthropic.com":
            return EndpointType.ANTHROPIC
        elif (
            parsed_url.hostname == "api.openai.com"
            or parsed_url.hostname == "openai.azure.com"
            or (parsed_url.hostname and "openai.com" in parsed_url.hostname)
        ):
            return EndpointType.OPENAI
        return EndpointType.GENERIC

    @staticmethod
    async def _make_non_streaming_http_request(
        request: Request,
        async_client: httpx.AsyncClient,
        url: str,
        headers: dict,
        requested_query_params: Optional[dict] = None,
        custom_body: Optional[dict] = None,
    ) -> httpx.Response:
        """
        Make a non-streaming HTTP request

        If request is GET, don't include a JSON body
        """
        if request.method == "GET":
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=requested_query_params,
            )
        else:
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=requested_query_params,
                json=custom_body,
            )
        return response

    @staticmethod
    async def non_streaming_http_request_handler(
        request: Request,
        async_client: httpx.AsyncClient,
        url: httpx.URL,
        headers: dict,
        requested_query_params: Optional[dict] = None,
        _parsed_body: Optional[dict] = None,
    ) -> httpx.Response:
        """
        Handle non-streaming HTTP requests

        Handles special cases when GET requests, multipart/form-data requests, and generic httpx requests
        """
        if request.method == "GET":
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=requested_query_params,
            )
        elif HttpPassThroughEndpointHelpers.is_multipart(request) is True:
            return await HttpPassThroughEndpointHelpers.make_multipart_http_request(
                request=request,
                async_client=async_client,
                url=url,
                headers=headers,
                requested_query_params=requested_query_params,
            )
        else:
            # Generic httpx method
            response = await async_client.request(
                method=request.method,
                url=url,
                headers=headers,
                params=requested_query_params,
                json=_parsed_body,
            )
        return response

    @staticmethod
    def is_multipart(request: Request) -> bool:
        """Check if the request is a multipart/form-data request"""
        return "multipart/form-data" in request.headers.get("content-type", "")

    @staticmethod
    async def _build_request_files_from_upload_file(
        upload_file: Union[UploadFile, StarletteUploadFile],
    ) -> Tuple[Optional[str], bytes, Optional[str]]:
        """Build a request files dict from an UploadFile object"""
        file_content = await upload_file.read()
        return (upload_file.filename, file_content, upload_file.content_type)

    @staticmethod
    async def make_multipart_http_request(
        request: Request,
        async_client: httpx.AsyncClient,
        url: httpx.URL,
        headers: dict,
        requested_query_params: Optional[dict] = None,
    ) -> httpx.Response:
        """Process multipart/form-data requests, handling both files and form fields"""
        form_data = await request.form()
        files = {}
        form_data_dict = {}

        for field_name, field_value in form_data.items():
            if isinstance(field_value, (StarletteUploadFile, UploadFile)):
                files[field_name] = (
                    await HttpPassThroughEndpointHelpers._build_request_files_from_upload_file(
                        upload_file=field_value
                    )
                )
            else:
                form_data_dict[field_name] = field_value

        # Remove content-type header - httpx will set it correctly with the new boundary
        # when it creates the multipart body from files/data parameters
        headers_copy = headers.copy()
        headers_copy.pop("content-type", None)

        response = await async_client.request(
            method=request.method,
            url=url,
            headers=headers_copy,
            params=requested_query_params,
            files=files,
            data=form_data_dict,
        )
        return response

    @staticmethod
    def _init_kwargs_for_pass_through_endpoint(
        request: Request,
        user_api_key_dict: UserAPIKeyAuth,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
        logging_obj: LiteLLMLoggingObj,
        _parsed_body: Optional[dict] = None,
        litellm_call_id: Optional[str] = None,
    ) -> dict:
        """
        Filter out litellm params from the request body
        """
        from litellm.types.utils import all_litellm_params

        _parsed_body = _parsed_body or {}

        litellm_params_in_body = {}
        for k in all_litellm_params:
            if k in _parsed_body:
                litellm_params_in_body[k] = _parsed_body.pop(k, None)

        _metadata = dict(
            StandardLoggingUserAPIKeyMetadata(
                user_api_key_hash=user_api_key_dict.api_key,
                user_api_key_alias=user_api_key_dict.key_alias,
                user_api_key_user_email=user_api_key_dict.user_email,
                user_api_key_user_id=user_api_key_dict.user_id,
                user_api_key_team_id=user_api_key_dict.team_id,
                user_api_key_org_id=user_api_key_dict.org_id,
                user_api_key_team_alias=user_api_key_dict.team_alias,
                user_api_key_end_user_id=user_api_key_dict.end_user_id,
                user_api_key_request_route=user_api_key_dict.request_route,
                user_api_key_spend=user_api_key_dict.spend,
                user_api_key_max_budget=user_api_key_dict.max_budget,
                user_api_key_budget_reset_at=(
                    user_api_key_dict.budget_reset_at.isoformat()
                    if user_api_key_dict.budget_reset_at
                    else None
                ),
                user_api_key_auth_metadata=user_api_key_dict.metadata,
            )
        )

        _metadata["user_api_key"] = user_api_key_dict.api_key

        litellm_metadata = litellm_params_in_body.pop("litellm_metadata", None)
        metadata = litellm_params_in_body.pop("metadata", None)
        if litellm_metadata:
            _metadata.update(litellm_metadata)
        if metadata:
            _metadata.update(metadata)

        _metadata = _update_metadata_with_tags_in_header(
            request=request,
            metadata=_metadata,
        )

        kwargs = {
            "litellm_params": {
                **litellm_params_in_body,  # type: ignore
                "metadata": _metadata,
                "proxy_server_request": {
                    "url": str(request.url),
                    "method": request.method,
                    "body": copy.copy(_parsed_body),  # use copy instead of deepcopy
                    "headers": request.headers,
                },
            },
            "call_type": "pass_through_endpoint",
            "litellm_call_id": litellm_call_id,
            "passthrough_logging_payload": passthrough_logging_payload,
        }

        logging_obj.model_call_details["passthrough_logging_payload"] = (
            passthrough_logging_payload
        )

        return kwargs

    @staticmethod
    def construct_target_url_with_subpath(
        base_target: str, subpath: str, include_subpath: Optional[bool]
    ) -> str:
        """
        Helper function to construct the full target URL with subpath handling.

        Args:
            base_target: The base target URL
            subpath: The captured subpath from the request
            include_subpath: Whether to include the subpath in the target URL

        Returns:
            The constructed full target URL
        """
        if not include_subpath:
            return base_target

        if not subpath:
            return base_target

        # Ensure base_target ends with / and subpath doesn't start with /
        if not base_target.endswith("/"):
            base_target = base_target + "/"
        if subpath.startswith("/"):
            subpath = subpath[1:]

        return base_target + subpath

    @staticmethod
    def _update_stream_param_based_on_request_body(
        parsed_body: dict,
        stream: Optional[bool] = None,
    ) -> Optional[bool]:
        """
        If stream is provided in the request body, use it.
        Otherwise, use the stream parameter passed to the `pass_through_request` function
        """
        if "stream" in parsed_body:
            return parsed_body.get("stream", stream)
        return stream


async def pass_through_request(  # noqa: PLR0915
    request: Request,
    target: str,
    custom_headers: dict,
    user_api_key_dict: UserAPIKeyAuth,
    custom_body: Optional[dict] = None,
    forward_headers: Optional[bool] = False,
    merge_query_params: Optional[bool] = False,
    query_params: Optional[dict] = None,
    stream: Optional[bool] = None,
    cost_per_request: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
):
    """
    Pass through endpoint handler, makes the httpx request for pass-through endpoints and ensures logging hooks are called

    Args:
        request: The incoming request
        target: The target URL
        custom_headers: The custom headers
        user_api_key_dict: The user API key dictionary
        custom_body: The custom body
        forward_headers: Whether to forward headers
        merge_query_params: Whether to merge query params
        query_params: The query params
        stream: Whether to stream the response
        cost_per_request: Optional field - cost per request to the target endpoint
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.proxy_server import proxy_logging_obj

    #########################################################
    # Initialize variables
    #########################################################
    litellm_call_id = str(uuid.uuid4())
    url: Optional[httpx.URL] = None

    # parsed request body
    _parsed_body: Optional[dict] = None
    # kwargs for pass through endpoint, contains metadata, litellm_params, call_type, litellm_call_id, passthrough_logging_payload
    kwargs: Optional[dict] = None

    #########################################################
    try:
        url = httpx.URL(target)
        headers = custom_headers
        headers = HttpPassThroughEndpointHelpers.forward_headers_from_request(
            request_headers=dict(request.headers),
            headers=headers,
            forward_headers=forward_headers,
        )

        if merge_query_params:
            # Create a new URL with the merged query params
            url = url.copy_with(
                query=urlencode(
                    HttpPassThroughEndpointHelpers.get_merged_query_parameters(
                        existing_url=url,
                        request_query_params=dict(request.query_params),
                    )
                ).encode("ascii")
            )

        endpoint_type: EndpointType = HttpPassThroughEndpointHelpers.get_endpoint_type(
            str(url)
        )

        if custom_body:
            _parsed_body = custom_body
        else:
            _parsed_body = await _read_request_body(request)
        verbose_proxy_logger.debug(
            "Pass through endpoint sending request to \nURL {}\nheaders: {}\nbody: {}\n".format(
                url, headers, _parsed_body
            )
        )

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        _parsed_body = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            data=_parsed_body,
            call_type="pass_through_endpoint",
        )
        async_client_obj = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PassThroughEndpoint,
            params={"timeout": 600},
        )
        async_client = async_client_obj.client

        # create logging object
        start_time = datetime.now()
        logging_obj = Logging(
            model="unknown",
            messages=[{"role": "user", "content": safe_dumps(_parsed_body)}],
            stream=False,
            call_type="pass_through_endpoint",
            start_time=start_time,
            litellm_call_id=litellm_call_id,
            function_id="1245",
        )
        passthrough_logging_payload = PassthroughStandardLoggingPayload(
            url=str(url),
            request_body=_parsed_body,
            request_method=getattr(request, "method", None),
            cost_per_request=cost_per_request,
        )
        kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
            user_api_key_dict=user_api_key_dict,
            _parsed_body=_parsed_body,
            passthrough_logging_payload=passthrough_logging_payload,
            litellm_call_id=litellm_call_id,
            request=request,
            logging_obj=logging_obj,
        )

        # done for supporting 'parallel_request_limiter.py' with pass-through endpoints
        logging_obj.update_environment_variables(
            model="unknown",
            user="unknown",
            optional_params={},
            litellm_params=kwargs["litellm_params"],
            call_type="pass_through_endpoint",
        )
        logging_obj.model_call_details["litellm_call_id"] = litellm_call_id

        # combine url with query params for logging
        requested_query_params: Optional[dict] = query_params or dict(
            request.query_params
        )

        requested_query_params_str = None
        if requested_query_params:
            requested_query_params_str = "&".join(
                f"{k}={v}" for k, v in requested_query_params.items()
            )

        logging_url = str(url)
        if requested_query_params_str:
            if "?" in str(url):
                logging_url = str(url) + "&" + requested_query_params_str
            else:
                logging_url = str(url) + "?" + requested_query_params_str

        logging_obj.pre_call(
            input=[{"role": "user", "content": safe_dumps(_parsed_body)}],
            api_key="",
            additional_args={
                "complete_input_dict": _parsed_body,
                "api_base": str(logging_url),
                "headers": headers,
            },
        )
        stream = (
            HttpPassThroughEndpointHelpers._update_stream_param_based_on_request_body(
                parsed_body=_parsed_body,
                stream=stream,
            )
        )

        if stream:
            req = async_client.build_request(
                "POST",
                url,
                json=_parsed_body,
                params=requested_query_params,
                headers=headers,
            )

            response = await async_client.send(req, stream=stream)

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code, detail=await e.response.aread()
                )

            return StreamingResponse(
                PassThroughStreamingHandler.chunk_processor(
                    response=response,
                    request_body=_parsed_body,
                    litellm_logging_obj=logging_obj,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    passthrough_success_handler_obj=pass_through_endpoint_logging,
                    url_route=str(url),
                ),
                headers=HttpPassThroughEndpointHelpers.get_response_headers(
                    headers=response.headers,
                    litellm_call_id=litellm_call_id,
                ),
                status_code=response.status_code,
            )

        response = (
            await HttpPassThroughEndpointHelpers.non_streaming_http_request_handler(
                request=request,
                async_client=async_client,
                url=url,
                headers=headers,
                requested_query_params=requested_query_params,
                _parsed_body=_parsed_body,
            )
        )
        verbose_proxy_logger.debug("response.headers= %s", response.headers)

        if _is_streaming_response(response) is True:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise HTTPException(
                    status_code=e.response.status_code, detail=await e.response.aread()
                )

            return StreamingResponse(
                PassThroughStreamingHandler.chunk_processor(
                    response=response,
                    request_body=_parsed_body,
                    litellm_logging_obj=logging_obj,
                    endpoint_type=endpoint_type,
                    start_time=start_time,
                    passthrough_success_handler_obj=pass_through_endpoint_logging,
                    url_route=str(url),
                ),
                headers=HttpPassThroughEndpointHelpers.get_response_headers(
                    headers=response.headers,
                    litellm_call_id=litellm_call_id,
                ),
                status_code=response.status_code,
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.text
            )

        if response.status_code >= 300:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        content = await response.aread()

        ## LOG SUCCESS
        response_body: Optional[dict] = get_response_body(response)
        passthrough_logging_payload["response_body"] = response_body
        end_time = datetime.now()
        asyncio.create_task(
            pass_through_endpoint_logging.pass_through_async_success_handler(
                httpx_response=response,
                response_body=response_body,
                url_route=str(url),
                result="",
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                cache_hit=False,
                request_body=_parsed_body,
                custom_llm_provider=custom_llm_provider,
                **kwargs,
            )
        )

        ## CUSTOM HEADERS - `x-litellm-*`
        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=litellm_call_id,
            model_id=None,
            cache_key=None,
            api_base=str(url._uri_reference),
        )

        return Response(
            content=content,
            status_code=response.status_code,
            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                headers=response.headers,
                custom_headers=custom_headers,
            ),
        )
    except Exception as e:
        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=litellm_call_id,
            model_id=None,
            cache_key=None,
            api_base=str(url._uri_reference) if url else None,
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.pass_through_endpoint(): Exception occured - {}".format(
                str(e)
            )
        )

        #########################################################
        # Monitoring: Trigger post_call_failure_hook
        # for pass through endpoint failure
        #########################################################
        request_payload: dict = _parsed_body or {}
        # add user_api_key_dict, litellm_call_id, passthrough_logging_payloa for logging
        if kwargs:
            for key, value in kwargs.items():
                request_payload[key] = value
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_payload,
            traceback_str=traceback.format_exc(
                limit=MAXIMUM_TRACEBACK_LINES_TO_LOG,
            ),
        )

        #########################################################

        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
                headers=custom_headers,
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
                headers=custom_headers,
            )


def _update_metadata_with_tags_in_header(request: Request, metadata: dict) -> dict:
    """
    If tags are in the request headers, add them to the metadata

    Used for google and vertex JS SDKs
    """
    _tags = request.headers.get("tags")
    if _tags:
        metadata["tags"] = _tags.split(",")
    return metadata


async def _parse_request_data_by_content_type(
    request: Request,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """
    Parse request data based on content type.

    Handles JSON, multipart/form-data, and URL-encoded form data.

    Returns:
        Tuple of (query_params_data, custom_body_data, file_data, stream)
    """
    content_type = request.headers.get("content-type", "")

    query_params_data = None
    custom_body_data = None
    file_data = None
    stream = None

    if "application/json" in content_type:
        # ✅ Handle JSON
        try:
            body = await request.json()
            query_params_data = body.get("query_params")
            custom_body_data = body.get("custom_body")
            stream = body.get("stream")
        except json.JSONDecodeError:
            # Handle requests with no body (e.g., DELETE requests)
            pass
    elif "multipart/form-data" in content_type:
        # ✅ Handle multipart form-data
        form = await request.form()
        if "query_params" in form:
            form_value = form["query_params"]
            if isinstance(form_value, str):
                try:
                    query_params_data = json.loads(form_value)
                except Exception:
                    query_params_data = form_value
            else:
                query_params_data = form_value

        if "custom_body" in form:
            form_value = form["custom_body"]
            if isinstance(form_value, str):
                try:
                    custom_body_data = json.loads(form_value)
                except Exception:
                    custom_body_data = form_value
            else:
                custom_body_data = form_value

        if "file" in form:
            file_data = form["file"]  # this is a Starlette UploadFile object

    elif "application/x-www-form-urlencoded" in content_type:
        # ✅ Handle URL-encoded form data
        form = await request.form()
        query_params_data = form.get("query_params")
        custom_body_data = form.get("custom_body")

    else:
        # ✅ Fallback: maybe no body, just query params
        query_params_data = dict(request.query_params) or None

    return query_params_data, custom_body_data, file_data, stream


def create_pass_through_route(
    endpoint,
    target: str,
    custom_headers: Optional[dict] = None,
    _forward_headers: Optional[bool] = False,
    _merge_query_params: Optional[bool] = False,
    dependencies: Optional[List] = None,
    include_subpath: Optional[bool] = False,
    cost_per_request: Optional[float] = None,
    custom_llm_provider: Optional[str] = None,
    is_streaming_request: Optional[bool] = False,
    query_params: Optional[dict] = None,
):
    # check if target is an adapter.py or a url
    from litellm._uuid import uuid
    from litellm.proxy.types_utils.utils import get_instance_fn

    try:
        if isinstance(target, CustomLogger):
            adapter = target
        else:
            adapter = get_instance_fn(value=target)
        adapter_id = str(uuid.uuid4())
        litellm.adapters = [{"id": adapter_id, "adapter": adapter}]

        async def endpoint_func(  # type: ignore
            request: Request,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
            subpath: str = "",  # captures sub-paths when include_subpath=True
        ):
            return await chat_completion_pass_through_endpoint(
                fastapi_response=fastapi_response,
                request=request,
                adapter_id=adapter_id,
                user_api_key_dict=user_api_key_dict,
            )

    except Exception:
        verbose_proxy_logger.debug("Defaulting to target being a url.")

        async def endpoint_func(  # type: ignore
            request: Request,
            fastapi_response: Response,
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
            subpath: str = "",  # captures sub-paths when include_subpath=True
        ):
            from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
                InitPassThroughEndpointHelpers,
            )

            path = request.url.path

            # Parse request data based on content type
            (
                query_params_data,
                custom_body_data,
                file_data,
                stream,
            ) = await _parse_request_data_by_content_type(request)

            if not InitPassThroughEndpointHelpers.is_registered_pass_through_route(
                route=path
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"Pass-through endpoint {endpoint} not found. This could have been deleted or not yet added to the proxy.",
                )

            passthrough_params = (
                InitPassThroughEndpointHelpers.get_registered_pass_through_route(
                    route=path
                )
            )
            target_params = {
                "target": target,
                "custom_headers": custom_headers,
                "forward_headers": _forward_headers,
                "merge_query_params": _merge_query_params,
                "cost_per_request": cost_per_request,
            }

            if passthrough_params is not None:
                target_params.update(passthrough_params.get("passthrough_params", {}))

            # Extract and cast parameters with proper types
            param_target = target_params.get("target") or target
            param_custom_headers = target_params.get("custom_headers", custom_headers)
            param_forward_headers = target_params.get(
                "forward_headers", _forward_headers
            )
            param_merge_query_params = target_params.get(
                "merge_query_params", _merge_query_params
            )
            param_cost_per_request = target_params.get(
                "cost_per_request", cost_per_request
            )

            # Construct the full target URL with subpath if needed
            full_target = (
                HttpPassThroughEndpointHelpers.construct_target_url_with_subpath(
                    base_target=cast(str, param_target),
                    subpath=subpath,
                    include_subpath=include_subpath,
                )
            )

            # Ensure custom_headers is a dict
            headers_dict = (
                param_custom_headers if isinstance(param_custom_headers, dict) else {}
            )

            # Ensure query_params and custom_body are dicts or None
            final_query_params = (
                query_params_data if isinstance(query_params_data, dict) else {}
            )
            if query_params:
                final_query_params.update(query_params)
            final_custom_body = (
                custom_body_data
                if isinstance(custom_body_data, dict) or custom_body_data is None
                else None
            )

            return await pass_through_request(  # type: ignore
                request=request,
                target=full_target,
                custom_headers=headers_dict,
                user_api_key_dict=user_api_key_dict,
                forward_headers=cast(Optional[bool], param_forward_headers),
                merge_query_params=cast(Optional[bool], param_merge_query_params),
                query_params=final_query_params,
                stream=is_streaming_request or stream,
                custom_body=final_custom_body,
                cost_per_request=cast(Optional[float], param_cost_per_request),
                custom_llm_provider=custom_llm_provider,
            )

    return endpoint_func


def create_websocket_passthrough_route(
    endpoint: str,
    target: str,
    custom_headers: Optional[dict] = None,
    _forward_headers: Optional[bool] = False,
    dependencies: Optional[List] = None,
    cost_per_request: Optional[float] = None,
):
    """
    Create a WebSocket passthrough route function.

    Args:
        endpoint: The endpoint path (for logging purposes)
        target: The target WebSocket URL (e.g., "wss://api.example.com/ws")
        custom_headers: Custom headers to include in the WebSocket connection
        _forward_headers: Whether to forward incoming headers
        dependencies: FastAPI dependencies to inject

    Returns:
        A WebSocket passthrough function that can be registered with app.websocket()
    """
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_websocket

    async def websocket_endpoint_func(
        websocket: WebSocket,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth_websocket),
        **kwargs,  # For additional query parameters
    ):
        """
        WebSocket passthrough endpoint function.

        This function handles the WebSocket connection by:
        1. Accepting the incoming WebSocket connection
        2. Establishing a connection to the target WebSocket
        3. Forwarding messages bidirectionally
        4. Handling connection cleanup
        """
        return await websocket_passthrough_request(
            websocket=websocket,
            target=target,
            custom_headers=custom_headers or {},
            user_api_key_dict=user_api_key_dict,
            forward_headers=_forward_headers,
            endpoint=endpoint,
            cost_per_request=cost_per_request,
            accept_websocket=True,  # Generic usage should accept the WebSocket
        )

    return websocket_endpoint_func


async def websocket_passthrough_request(  # noqa: PLR0915
    websocket: WebSocket,
    target: str,
    custom_headers: dict,
    user_api_key_dict: UserAPIKeyAuth,
    forward_headers: Optional[bool] = False,
    endpoint: Optional[str] = None,
    cost_per_request: Optional[float] = None,
    accept_websocket: bool = True,
):
    """
    WebSocket passthrough request handler.

    Args:
        websocket: The incoming WebSocket connection
        target: The target WebSocket URL
        custom_headers: Custom headers to include in the connection
        user_api_key_dict: The user API key dictionary
        forward_headers: Whether to forward incoming headers
        endpoint: The endpoint path (for logging purposes)
        cost_per_request: Optional field - cost per request to the target endpoint
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy.proxy_server import proxy_logging_obj
    from litellm.types.passthrough_endpoints.pass_through_endpoints import (
        PassthroughStandardLoggingPayload,
    )

    # Initialize tracking variables
    start_time = datetime.now()
    websocket_messages: list[dict[str, Any]] = []
    litellm_call_id = str(uuid.uuid4())

    verbose_proxy_logger.info(
        f"WebSocket passthrough ({endpoint}): Starting WebSocket connection to {target}"
    )

    # Only accept the WebSocket if requested (for generic usage)
    if accept_websocket:
        await websocket.accept()
        verbose_proxy_logger.debug(
            f"WebSocket passthrough ({endpoint}): WebSocket connection accepted"
        )

    # Prepare headers for the upstream connection
    upstream_headers = custom_headers.copy()

    if forward_headers:
        # Forward relevant headers from the incoming request
        incoming_headers = dict(websocket.headers)
        for header_name, header_value in incoming_headers.items():
            # Only forward certain headers to avoid conflicts
            if header_name.lower() in [
                "authorization",
                "x-api-key",
                "x-goog-user-project",
            ]:
                upstream_headers[header_name] = header_value

    # Initialize logging object similar to HTTP passthrough
    logging_obj = Logging(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,  # WebSockets are inherently streaming
        call_type="pass_through_endpoint",
        start_time=start_time,
        litellm_call_id=litellm_call_id,
        function_id="websocket_passthrough",
    )

    # Create passthrough logging payload
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url=target,
        request_body={},  # WebSocket doesn't have a traditional request body
        request_method="WEBSOCKET",
        cost_per_request=cost_per_request,
    )

    # Create a dummy request object for WebSocket connections to maintain compatibility
    # with the existing _init_kwargs_for_pass_through_endpoint function
    class DummyRequest:
        def __init__(
            self, url: str, method: str = "WEBSOCKET", headers: Optional[dict] = None
        ):
            self.url = url
            self.method = method
            self.headers = headers or {}

        def __str__(self):
            return f"DummyRequest(url={self.url}, method={self.method})"

    dummy_request = DummyRequest(
        url=target,
        method="WEBSOCKET",
        headers=dict(websocket.headers) if hasattr(websocket, "headers") else {},
    )

    # Initialize kwargs for logging using the same pattern as HTTP passthrough
    kwargs = HttpPassThroughEndpointHelpers._init_kwargs_for_pass_through_endpoint(
        user_api_key_dict=user_api_key_dict,
        _parsed_body={},  # WebSocket doesn't have a traditional request body
        passthrough_logging_payload=passthrough_logging_payload,
        litellm_call_id=litellm_call_id,
        request=dummy_request,  # type: ignore
        logging_obj=logging_obj,
    )

    # Update logging environment variables
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params=dict(kwargs.get("litellm_params", {})),
        call_type="pass_through_endpoint",
    )
    logging_obj.model_call_details["litellm_call_id"] = litellm_call_id

    # Pre-call logging
    logging_obj.pre_call(
        input=[{"role": "user", "content": "WebSocket connection"}],
        api_key="",
        additional_args={
            "complete_input_dict": {},
            "api_base": target,
            "headers": upstream_headers,
        },
    )

    ### CALL HOOKS ### - modify incoming data / reject request before calling the model
    websocket_data: dict[str, Any] = {}
    websocket_data = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        data=websocket_data,
        call_type="pass_through_endpoint",
    )

    try:
        verbose_proxy_logger.debug(
            f"WebSocket passthrough ({endpoint}): Establishing upstream connection to {target}"
        )
        async with connect(
            target,
            additional_headers=upstream_headers,
        ) as upstream_ws:
            verbose_proxy_logger.info(
                f"WebSocket passthrough ({endpoint}): Upstream connection established successfully"
            )

            async def forward_client_to_upstream() -> None:
                """Forward messages from client to upstream WebSocket"""
                try:
                    while True:
                        message = await websocket.receive()
                        message_type = message.get("type")
                        if message_type == "websocket.disconnect":
                            await upstream_ws.close()
                            break

                        text_data = message.get("text")
                        bytes_data = message.get("bytes")

                        if text_data is not None:
                            # Try to extract model from client setup message for Vertex AI Live
                            if endpoint and "/vertex_ai/live" in endpoint:
                                verbose_proxy_logger.debug(
                                    f"WebSocket passthrough ({endpoint}): Processing client message for model extraction"
                                )
                                try:
                                    client_message = json.loads(text_data)
                                    if (
                                        isinstance(client_message, dict)
                                        and "setup" in client_message
                                    ):
                                        setup_data = client_message["setup"]
                                        verbose_proxy_logger.debug(
                                            f"WebSocket passthrough ({endpoint}): Found setup data in client message: {setup_data}"
                                        )
                                        if (
                                            isinstance(setup_data, dict)
                                            and "model" in setup_data
                                        ):
                                            extracted_model = (
                                                _extract_model_from_vertex_ai_setup(
                                                    setup_data
                                                )
                                            )
                                            if extracted_model:
                                                kwargs["model"] = extracted_model
                                                kwargs["custom_llm_provider"] = (
                                                    "vertex_ai-language-models"
                                                )
                                                # Update logging object with correct model
                                                logging_obj.model = extracted_model
                                                logging_obj.model_call_details[
                                                    "model"
                                                ] = extracted_model
                                                logging_obj.model_call_details[
                                                    "custom_llm_provider"
                                                ] = "vertex_ai"
                                                verbose_proxy_logger.info(
                                                    f"WebSocket passthrough ({endpoint}): Successfully extracted model '{extracted_model}' and set provider to 'vertex_ai' from client setup message"
                                                )
                                            else:
                                                verbose_proxy_logger.warning(
                                                    f"WebSocket passthrough ({endpoint}): Failed to extract model from client setup data: {setup_data}"
                                                )
                                        else:
                                            verbose_proxy_logger.debug(
                                                f"WebSocket passthrough ({endpoint}): Setup data does not contain model field: {setup_data}"
                                            )
                                    else:
                                        verbose_proxy_logger.debug(
                                            f"WebSocket passthrough ({endpoint}): Client message does not contain setup data"
                                        )
                                except (json.JSONDecodeError, KeyError, TypeError) as e:
                                    verbose_proxy_logger.debug(
                                        f"WebSocket passthrough ({endpoint}): Client message is not a valid setup message: {e}"
                                    )
                                    pass  # Not a JSON message or doesn't contain setup data

                            await upstream_ws.send(text_data)
                        elif bytes_data is not None:
                            await upstream_ws.send(bytes_data)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    verbose_proxy_logger.exception(
                        f"WebSocket passthrough ({endpoint}): error forwarding client message"
                    )
                    await upstream_ws.close()

            async def forward_upstream_to_client() -> None:
                """Forward messages from upstream to client WebSocket"""
                try:
                    # Wait for the first response from upstream
                    raw_response = await upstream_ws.recv(decode=False)
                    # Ensure raw_response is bytes before decoding
                    if isinstance(raw_response, str):
                        raw_response = raw_response.encode("ascii")
                    setup_response = json.loads(raw_response.decode("ascii"))
                    verbose_proxy_logger.debug(f"Setup response: {setup_response}")

                    # Extract model and provider from setup response for Vertex AI Live
                    if endpoint and "/vertex_ai/live" in endpoint:
                        verbose_proxy_logger.debug(
                            f"WebSocket passthrough ({endpoint}): Processing server setup response for model extraction"
                        )
                        extracted_model = _extract_model_from_vertex_ai_setup(
                            setup_response
                        )
                        if extracted_model:
                            kwargs["model"] = extracted_model
                            kwargs["custom_llm_provider"] = "vertex_ai_language_models"
                            # Update logging object with correct model
                            logging_obj.model = extracted_model
                            logging_obj.model_call_details["model"] = extracted_model
                            logging_obj.model_call_details["custom_llm_provider"] = (
                                "vertex_ai_language_models"
                            )
                            verbose_proxy_logger.debug(
                                f"WebSocket passthrough ({endpoint}): Successfully extracted model '{extracted_model}' and set provider to 'vertex_ai' from server setup response"
                            )
                        else:
                            verbose_proxy_logger.warning(
                                f"WebSocket passthrough ({endpoint}): Failed to extract model from server setup response: {setup_response}"
                            )
                    else:
                        verbose_proxy_logger.debug(
                            f"WebSocket passthrough ({endpoint}): Not a Vertex AI Live endpoint, skipping model extraction"
                        )

                    # Send the setup response to the client
                    await websocket.send_text(json.dumps(setup_response))

                    # Now continuously forward messages from upstream to client
                    async for upstream_message in upstream_ws:
                        if isinstance(upstream_message, bytes):
                            await websocket.send_bytes(upstream_message)
                            # Parse and collect for cost tracking
                            try:
                                message_data = json.loads(upstream_message.decode())
                                websocket_messages.append(message_data)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass
                        else:
                            await websocket.send_text(upstream_message)
                            # Parse and collect for cost tracking
                            try:
                                message_data = json.loads(upstream_message)
                                websocket_messages.append(message_data)
                            except json.JSONDecodeError:
                                pass

                except (ConnectionClosedOK, ConnectionClosedError) as e:
                    verbose_proxy_logger.debug(
                        f"Upstream WebSocket connection closed: {e}"
                    )
                    pass
                except asyncio.CancelledError:
                    verbose_proxy_logger.debug(
                        "asyncio.CancelledError in forward_upstream_to_client"
                    )
                    raise
                except Exception as e:
                    verbose_proxy_logger.debug(
                        f"Exception in forward_upstream_to_client: {e}"
                    )
                    verbose_proxy_logger.exception(
                        f"WebSocket passthrough ({endpoint}): error forwarding upstream message"
                    )
                    raise

            # Create tasks for bidirectional message forwarding
            tasks = [
                asyncio.create_task(forward_client_to_upstream()),
                asyncio.create_task(forward_upstream_to_client()),
            ]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check for exceptions in completed tasks
            for task in done:
                exception = task.exception()
                if exception is not None:
                    raise exception

            end_time = datetime.now()

            # Update passthrough logging payload with response data
            passthrough_logging_payload["response_body"] = websocket_messages  # type: ignore
            passthrough_logging_payload["end_time"] = end_time  # type: ignore

            # Remove logging_obj from kwargs to avoid duplicate keyword argument
            success_kwargs = kwargs.copy()
            success_kwargs.pop("logging_obj", None)

            # # Add user authentication context for database logging
            # if user_api_key_dict:
            #     success_kwargs.setdefault('litellm_params', {})
            #     success_kwargs['litellm_params'].update({
            #         'proxy_server_request': {
            #             'body': {
            #                 'user': user_api_key_dict.user_id,
            #                 'team_id': user_api_key_dict.team_id,
            #                 'end_user_id': user_api_key_dict.end_user_id,
            #             }
            #         }
            #     })
            #     # Also add the user_api_key for direct access
            #     success_kwargs['user_api_key'] = user_api_key_dict.api_key

            # Create a dummy httpx.Response for WebSocket connections
            class MockWebSocketResponse:
                def __init__(self, target_url: str):
                    self.status_code = 200
                    self.text = "WebSocket connection successful"
                    self.headers: dict[str, str] = {}
                    self.request = MockWebSocketRequest(target_url)

            class MockWebSocketRequest:
                def __init__(self, target_url: str):
                    self.method = "WEBSOCKET"
                    self.url = target_url

            mock_response = MockWebSocketResponse(target)

            # Use the same success handler as HTTP passthrough endpoints
            asyncio.create_task(
                pass_through_endpoint_logging.pass_through_async_success_handler(
                    httpx_response=mock_response,  # type: ignore
                    response_body=websocket_messages,  # type: ignore
                    url_route=endpoint or "",
                    result="websocket_connection_successful",
                    start_time=start_time,
                    end_time=end_time,
                    logging_obj=logging_obj,
                    cache_hit=False,
                    request_body={},
                    **success_kwargs,
                )
            )

            # Call the proxy logging success hook
            if proxy_logging_obj:
                await proxy_logging_obj.post_call_success_hook(
                    data={},
                    user_api_key_dict=user_api_key_dict,
                    response={"status": "websocket_connection_successful"},  # type: ignore
                )

    except InvalidStatus as exc:
        verbose_proxy_logger.exception(
            f"WebSocket passthrough ({endpoint}): upstream rejected WebSocket connection"
        )

        # Prepare request payload for logging
        request_payload = {}
        if kwargs:
            for key, value in kwargs.items():
                request_payload[key] = value

        # Log the connection failure using the same pattern as HTTP
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=exc,
            request_data=request_payload,
            traceback_str=traceback.format_exc(
                limit=MAXIMUM_TRACEBACK_LINES_TO_LOG,
            ),
        )

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(
                code=getattr(exc, "status_code", 1011),
                reason="Upstream connection rejected",
            )
    except Exception as e:
        verbose_proxy_logger.exception(
            f"WebSocket passthrough ({endpoint}): unexpected error while proxying WebSocket"
        )

        # Prepare request payload for logging
        request_payload = {}
        if kwargs:
            for key, value in kwargs.items():
                request_payload[key] = value

        # Log the unexpected error using the same pattern as HTTP
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=request_payload,
            traceback_str=traceback.format_exc(
                limit=MAXIMUM_TRACEBACK_LINES_TO_LOG,
            ),
        )

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1011, reason="WebSocket passthrough error")
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()


def _is_streaming_response(response: httpx.Response) -> bool:
    _content_type = response.headers.get("content-type")
    if _content_type is not None and "text/event-stream" in _content_type:
        return True
    return False


def _extract_model_from_vertex_ai_setup(setup_response: dict) -> Optional[str]:
    """
    Extract the model name from Vertex AI Live setup response.

    The setup response can contain a model field in two formats:
    1. Direct: {"model": "projects/.../models/gemini-2.0-flash-live-preview-04-09"}
    2. Nested: {"setup": {"model": "projects/.../models/gemini-2.0-flash-live-preview-04-09"}}

    We extract just the model name: "gemini-2.0-flash-live-preview-04-09"
    """
    try:
        # Handle both direct model field and nested setup.model field
        model_path = None
        if isinstance(setup_response, dict):
            if "model" in setup_response:
                model_path = setup_response["model"]
            elif (
                "setup" in setup_response
                and isinstance(setup_response["setup"], dict)
                and "model" in setup_response["setup"]
            ):
                model_path = setup_response["setup"]["model"]

        if isinstance(model_path, str) and "/models/" in model_path:
            # Extract the model name after the last "/models/"
            model_name = model_path.split("/models/")[-1]
            return model_name
    except Exception as e:
        verbose_proxy_logger.debug(f"Error extracting model from setup response: {e}")
    return None


class SafeRouteAdder:
    """
    Wrapper class for adding routes to FastAPI app.
    Only adds routes if they don't already exist on the app.
    """

    @staticmethod
    def _is_path_registered(app: FastAPI, path: str, methods: List[str]) -> bool:
        """
        Check if a path with any of the specified methods is already registered on the app.

        Args:
            app: The FastAPI application instance
            path: The path to check (e.g., "/v1/chat/completions")
            methods: List of HTTP methods to check (e.g., ["GET", "POST"])

        Returns:
            True if the path is already registered with any of the methods, False otherwise
        """
        for route in app.routes:
            # Use getattr to safely access route attributes
            route_path = getattr(route, "path", None)
            route_methods = getattr(route, "methods", None)

            if route_path == path and route_methods is not None:
                # Check if any of the methods overlap
                if any(method in route_methods for method in methods):
                    return True
        return False

    @staticmethod
    def add_api_route_if_not_exists(
        app: FastAPI,
        path: str,
        endpoint: Any,
        methods: List[str],
        dependencies: Optional[List] = None,
    ) -> bool:
        """
        Add an API route to the app only if it doesn't already exist.

        Args:
            app: The FastAPI application instance
            path: The path for the route
            endpoint: The endpoint function/callable
            methods: List of HTTP methods
            dependencies: Optional list of dependencies

        Returns:
            True if route was added, False if it already existed
        """
        if SafeRouteAdder._is_path_registered(app=app, path=path, methods=methods):
            verbose_proxy_logger.debug(
                "Skipping route registration - path %s with methods %s already registered on app",
                path,
                methods,
            )
            return False

        app.add_api_route(
            path=path,
            endpoint=endpoint,
            methods=methods,
            dependencies=dependencies,
        )
        verbose_proxy_logger.debug(
            "Successfully added route: %s with methods %s",
            path,
            methods,
        )
        return True


class InitPassThroughEndpointHelpers:
    @staticmethod
    def add_exact_path_route(
        app: FastAPI,
        path: str,
        target: str,
        custom_headers: Optional[dict],
        forward_headers: Optional[bool],
        merge_query_params: Optional[bool],
        dependencies: Optional[List],
        cost_per_request: Optional[float],
        endpoint_id: str,
    ):
        """Add exact path route for pass-through endpoint"""
        route_key = f"{endpoint_id}:exact:{path}"

        # Check if this exact route is already registered
        if route_key in _registered_pass_through_routes:
            verbose_proxy_logger.debug(
                "Skipping duplicate exact pass through endpoint: %s (already registered)",
                path,
            )
            return

        verbose_proxy_logger.debug(
            "adding exact pass through endpoint: %s, dependencies: %s",
            path,
            dependencies,
        )

        # Use SafeRouteAdder to only add route if it doesn't exist on the app
        was_added = SafeRouteAdder.add_api_route_if_not_exists(
            app=app,
            path=path,
            endpoint=create_pass_through_route(  # type: ignore
                path,
                target,
                custom_headers,
                forward_headers,
                merge_query_params,
                dependencies,
                cost_per_request=cost_per_request,
            ),
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            dependencies=dependencies,
        )

        # Register the route to prevent duplicates only if it was added
        if was_added:
            _registered_pass_through_routes[route_key] = {
                "endpoint_id": endpoint_id,
                "path": path,
                "type": "exact",
                "passthrough_params": {
                    "target": target,
                    "custom_headers": custom_headers,
                    "forward_headers": forward_headers,
                    "merge_query_params": merge_query_params,
                    "dependencies": dependencies,
                    "cost_per_request": cost_per_request,
                },
            }

    @staticmethod
    def add_subpath_route(
        app: FastAPI,
        path: str,
        target: str,
        custom_headers: Optional[dict],
        forward_headers: Optional[bool],
        merge_query_params: Optional[bool],
        dependencies: Optional[List],
        cost_per_request: Optional[float],
        endpoint_id: str,
    ):
        """Add wildcard route for sub-paths"""
        wildcard_path = f"{path}/{{subpath:path}}"
        route_key = f"{endpoint_id}:subpath:{path}"

        # Check if this subpath route is already registered
        if route_key in _registered_pass_through_routes:
            verbose_proxy_logger.debug(
                "Skipping duplicate wildcard pass through endpoint: %s (already registered)",
                wildcard_path,
            )
            return

        verbose_proxy_logger.debug(
            "adding wildcard pass through endpoint: %s, dependencies: %s",
            wildcard_path,
            dependencies,
        )

        # Use SafeRouteAdder to only add route if it doesn't exist on the app
        was_added = SafeRouteAdder.add_api_route_if_not_exists(
            app=app,
            path=wildcard_path,
            endpoint=create_pass_through_route(  # type: ignore
                path,
                target,
                custom_headers,
                forward_headers,
                merge_query_params,
                dependencies,
                include_subpath=True,
                cost_per_request=cost_per_request,
            ),
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            dependencies=dependencies,
        )

        # Register the route to prevent duplicates only if it was added
        if was_added:
            _registered_pass_through_routes[route_key] = {
                "endpoint_id": endpoint_id,
                "path": path,
                "type": "subpath",
                "passthrough_params": {
                    "target": target,
                    "custom_headers": custom_headers,
                    "forward_headers": forward_headers,
                    "merge_query_params": merge_query_params,
                    "dependencies": dependencies,
                    "cost_per_request": cost_per_request,
                },
            }

    @staticmethod
    def remove_endpoint_routes(endpoint_id: str):
        """Remove all routes for a specific endpoint ID from the registry"""
        keys_to_remove = [
            key
            for key, value in _registered_pass_through_routes.items()
            if value["endpoint_id"] == endpoint_id
        ]
        for key in keys_to_remove:
            del _registered_pass_through_routes[key]
            verbose_proxy_logger.debug(
                "Removed pass-through route from registry: %s", key
            )

    @staticmethod
    def clear_all_pass_through_routes():
        """Clear all pass-through routes from the registry"""
        _registered_pass_through_routes.clear()

    @staticmethod
    def get_registered_pass_through_endpoints_keys() -> List[str]:
        """Get all registered pass-through endpoints from the registry"""
        return list(_registered_pass_through_routes.keys())

    @staticmethod
    def is_registered_pass_through_route(route: str) -> bool:
        """
        Check if route is a registered pass-through endpoint from DB

        Uses the in-memory registry to avoid additional DB queries
        Optimized for minimal latency

        Args:
            route: The route to check

        Returns:
            bool: True if route is a registered pass-through endpoint, False otherwise
        """
        ## CHECK IF MAPPED PASS THROUGH ENDPOINT
        for mapped_route in LiteLLMRoutes.mapped_pass_through_routes.value:
            if route.startswith(mapped_route):
                return True

        # Fast path: check if any registered route key contains this path
        # Keys are in format: "{endpoint_id}:exact:{path}" or "{endpoint_id}:subpath:{path}"
        # Extract unique paths from keys for quick checking
        for key in _registered_pass_through_routes.keys():
            parts = key.split(":", 2)  # Split into [endpoint_id, type, path]
            if len(parts) == 3:
                route_type = parts[1]
                registered_path = parts[2]
                if route_type == "exact" and route == registered_path:
                    return True
                elif route_type == "subpath":
                    if route == registered_path or route.startswith(
                        registered_path + "/"
                    ):
                        return True

        return False

    @staticmethod
    def get_registered_pass_through_route(route: str) -> Optional[Dict[str, Any]]:
        """Get passthrough params for a given route"""
        for key in _registered_pass_through_routes.keys():
            parts = key.split(":", 2)  # Split into [endpoint_id, type, path]
            if len(parts) == 3:
                route_type = parts[1]
                registered_path = parts[2]

                if route_type == "exact" and route == registered_path:
                    return _registered_pass_through_routes[key]
                elif route_type == "subpath":
                    if route == registered_path or route.startswith(
                        registered_path + "/"
                    ):
                        return _registered_pass_through_routes[key]

        return None


def _get_combined_pass_through_endpoints(
    pass_through_endpoints: Union[List[Dict], List[PassThroughGenericEndpoint]],
    config_pass_through_endpoints: List[Dict],
):
    """Get combined pass-through endpoints from db + config"""
    return pass_through_endpoints + config_pass_through_endpoints


async def initialize_pass_through_endpoints(
    pass_through_endpoints: Union[List[Dict], List[PassThroughGenericEndpoint]],
):
    """
    1. Create a global list of pass-through endpoints (db + config)
    2. Clear all existing pass-through endpoints from the FastAPI app routes
    3. Add new endpoints to the in-memory registry

    Initialize a list of pass-through endpoints by adding them to the FastAPI app routes

    Args:
        pass_through_endpoints: List of pass-through endpoints to initialize

    Returns:
        None
    """
    from litellm._uuid import uuid

    verbose_proxy_logger.debug("initializing pass through endpoints")
    from litellm.proxy._types import CommonProxyErrors, LiteLLMRoutes
    from litellm.proxy.proxy_server import (
        app,
        config_passthrough_endpoints,
        premium_user,
    )

    ## get combined pass-through endpoints from db + config
    combined_pass_through_endpoints: List[Union[Dict, PassThroughGenericEndpoint]]

    if config_passthrough_endpoints is not None:
        combined_pass_through_endpoints = _get_combined_pass_through_endpoints(  # type: ignore
            pass_through_endpoints, config_passthrough_endpoints
        )
    else:
        combined_pass_through_endpoints = pass_through_endpoints  # type: ignore

    ## clear all existing pass-through endpoints from the FastAPI app routes
    # InitPassThroughEndpointHelpers.clear_all_pass_through_routes()

    # get a list of all registered pass-through endpoints
    # mark the ones that are visited in the list
    # remove the ones that are not visited from the list
    registered_pass_through_endpoints = (
        InitPassThroughEndpointHelpers.get_registered_pass_through_endpoints_keys()
    )

    visited_endpoints = set()

    for endpoint in combined_pass_through_endpoints:
        if isinstance(endpoint, PassThroughGenericEndpoint):
            endpoint = endpoint.model_dump()

        # Auto-generate ID for backwards compatibility if not present
        if endpoint.get("id") is None:
            endpoint["id"] = str(uuid.uuid4())

        # Get the endpoint_id as a string (guaranteed to be set at this point)
        endpoint_id: str = endpoint["id"]

        _target = endpoint.get("target", None)
        _path: Optional[str] = endpoint.get("path", None)
        if _path is None:
            raise ValueError("Path is required for pass-through endpoint")
        _custom_headers = endpoint.get("headers", None)
        _custom_headers = await set_env_variables_in_header(
            custom_headers=_custom_headers
        )
        _forward_headers = endpoint.get("forward_headers", None)
        _merge_query_params = endpoint.get("merge_query_params", None)
        _auth = endpoint.get("auth", None)
        _dependencies = None
        if _auth is not None and str(_auth).lower() == "true":
            if premium_user is not True:
                raise ValueError(
                    "Error Setting Authentication on Pass Through Endpoint: {}".format(
                        CommonProxyErrors.not_premium_user.value
                    )
                )
            _dependencies = [Depends(user_api_key_auth)]
            LiteLLMRoutes.openai_routes.value.append(_path)

        if _target is None:
            continue

        # Add exact path route
        verbose_proxy_logger.debug(
            "Initializing pass through endpoint: %s (ID: %s)", _path, endpoint_id
        )
        InitPassThroughEndpointHelpers.add_exact_path_route(
            app=app,
            path=_path,
            target=_target,
            custom_headers=_custom_headers,
            forward_headers=_forward_headers,
            merge_query_params=_merge_query_params,
            dependencies=_dependencies,
            cost_per_request=endpoint.get("cost_per_request", None),
            endpoint_id=endpoint_id,
        )

        visited_endpoints.add(f"{endpoint_id}:exact:{_path}")

        # Add wildcard route for sub-paths
        if endpoint.get("include_subpath", False) is True:
            InitPassThroughEndpointHelpers.add_subpath_route(
                app=app,
                path=_path,
                target=_target,
                custom_headers=_custom_headers,
                forward_headers=_forward_headers,
                merge_query_params=_merge_query_params,
                dependencies=_dependencies,
                cost_per_request=endpoint.get("cost_per_request", None),
                endpoint_id=endpoint_id,
            )

            visited_endpoints.add(f"{endpoint_id}:subpath:{_path}")

        verbose_proxy_logger.debug(
            "Added new pass through endpoint: %s (ID: %s)", _path, endpoint_id
        )

    # remove the ones that are not visited from the list
    for endpoint_key in registered_pass_through_endpoints:
        if endpoint_key not in visited_endpoints:
            InitPassThroughEndpointHelpers.remove_endpoint_routes(endpoint_key)


async def _get_pass_through_endpoints_from_db(
    endpoint_id: Optional[str] = None,
    user_api_key_dict: Optional[UserAPIKeyAuth] = None,
) -> List[PassThroughGenericEndpoint]:
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.proxy_server import get_config_general_settings

    try:
        if user_api_key_dict is None:
            user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        return []

    pass_through_endpoint_data: Optional[List] = response.field_value
    if pass_through_endpoint_data is None:
        return []

    returned_endpoints: List[PassThroughGenericEndpoint] = []
    if endpoint_id is None:
        # Return all endpoints
        for endpoint in pass_through_endpoint_data:
            if isinstance(endpoint, dict):
                returned_endpoints.append(PassThroughGenericEndpoint(**endpoint))
            elif isinstance(endpoint, PassThroughGenericEndpoint):
                returned_endpoints.append(endpoint)
    else:
        # Find specific endpoint by ID
        found_endpoint = _find_endpoint_by_id(pass_through_endpoint_data, endpoint_id)
        if found_endpoint is not None:
            returned_endpoints.append(found_endpoint)

    return returned_endpoints


async def _filter_endpoints_by_team_allowed_routes(
    team_id: str,
    pass_through_endpoints: List[PassThroughGenericEndpoint],
    prisma_client,
) -> List[PassThroughGenericEndpoint]:
    """
    Filter pass-through endpoints based on team's allowed_passthrough_routes metadata.

    Args:
        team_id: The team ID to check permissions for
        pass_through_endpoints: List of endpoints to filter
        prisma_client: Database client

    Returns:
        Filtered list of endpoints based on team permissions

    Raises:
        HTTPException: If team is not found
    """
    # retrieve team from db
    team = await prisma_client.db.litellm_teamtable.find_unique(
        where={"team_id": team_id},
    )
    if team is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Team not found"},
        )

    # retrieve team metadata
    team_metadata = team.metadata
    if (
        team_metadata is not None
        and team_metadata.get("allowed_passthrough_routes") is not None
    ):
        ## FILTER pass_through_endpoints by allowed_passthrough_routes
        pass_through_endpoints = [
            endpoint
            for endpoint in pass_through_endpoints
            if endpoint.path in team_metadata.get("allowed_passthrough_routes")
        ]

    return pass_through_endpoints


@router.get(
    "/config/pass_through_endpoint",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PassThroughEndpointResponse,
)
@router.get(
    "/config/pass_through_endpoint/team/{team_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PassThroughEndpointResponse,
)
async def get_pass_through_endpoints(
    endpoint_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    team_id: Optional[str] = None,
):
    """
    GET configured pass through endpoint.

    If no endpoint_id given, return all configured endpoints.
    """  ## Get existing pass-through endpoint field value
    from litellm.proxy._types import CommonProxyErrors
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    pass_through_endpoints = await _get_pass_through_endpoints_from_db(
        endpoint_id=endpoint_id, user_api_key_dict=user_api_key_dict
    )

    if team_id is not None:
        pass_through_endpoints = await _filter_endpoints_by_team_allowed_routes(
            team_id=team_id,
            pass_through_endpoints=pass_through_endpoints,
            prisma_client=prisma_client,
        )

    return PassThroughEndpointResponse(endpoints=pass_through_endpoints)


@router.post(
    "/config/pass_through_endpoint/{endpoint_id}",
    dependencies=[Depends(user_api_key_auth)],
)
async def update_pass_through_endpoints(
    endpoint_id: str,
    data: PassThroughGenericEndpoint,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update a pass-through endpoint by ID.
    """
    from litellm.proxy.proxy_server import (
        get_config_general_settings,
        update_config_general_settings,
    )

    ## Get existing pass-through endpoint field value
    try:
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail={"error": "No pass-through endpoints found"},
        )

    pass_through_endpoint_data: Optional[List] = response.field_value
    if pass_through_endpoint_data is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "No pass-through endpoints found"},
        )

    # Find the endpoint to update
    found_endpoint = _find_endpoint_by_id(pass_through_endpoint_data, endpoint_id)

    if found_endpoint is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Endpoint with ID '{endpoint_id}' not found"},
        )

    # Find the index for updating the list
    endpoint_index = None
    for idx, endpoint in enumerate(pass_through_endpoint_data):
        _endpoint = (
            PassThroughGenericEndpoint(**endpoint)
            if isinstance(endpoint, dict)
            else endpoint
        )
        if _endpoint.id == endpoint_id:
            endpoint_index = idx
            break

    if endpoint_index is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Could not find index for endpoint with ID '{endpoint_id}'"
            },
        )

    # Get the update data as dict, excluding None values for partial updates
    update_data = data.model_dump(exclude_none=True)

    # Start with existing endpoint data
    endpoint_dict = found_endpoint.model_dump()

    # Update with new data (only non-None values)
    endpoint_dict.update(update_data)

    # Preserve existing ID if not provided in update and endpoint has ID
    if "id" not in update_data and found_endpoint.id is not None:
        endpoint_dict["id"] = found_endpoint.id

    # Create updated endpoint object
    updated_endpoint = PassThroughGenericEndpoint(**endpoint_dict)

    # Update the list
    pass_through_endpoint_data[endpoint_index] = endpoint_dict

    # Remove old routes from registry before they get re-registered
    InitPassThroughEndpointHelpers.remove_endpoint_routes(endpoint_id)

    ## Update db
    updated_data = ConfigFieldUpdate(
        field_name="pass_through_endpoints",
        field_value=pass_through_endpoint_data,
        config_type="general_settings",
    )

    await update_config_general_settings(
        data=updated_data, user_api_key_dict=user_api_key_dict
    )

    return PassThroughEndpointResponse(
        endpoints=[updated_endpoint] if updated_endpoint else []
    )


@router.post(
    "/config/pass_through_endpoint",
    dependencies=[Depends(user_api_key_auth)],
)
async def create_pass_through_endpoints(
    data: PassThroughGenericEndpoint,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create new pass-through endpoint
    """
    from litellm._uuid import uuid
    from litellm.proxy.proxy_server import (
        get_config_general_settings,
        update_config_general_settings,
    )

    ## Get existing pass-through endpoint field value

    try:
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        response = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=None
        )

    ## Auto-generate ID if not provided
    data_dict = data.model_dump()
    if data_dict.get("id") is None:
        data_dict["id"] = str(uuid.uuid4())

    if response.field_value is None:
        response.field_value = [data_dict]
    elif isinstance(response.field_value, List):
        response.field_value.append(data_dict)

    ## Update db
    updated_data = ConfigFieldUpdate(
        field_name="pass_through_endpoints",
        field_value=response.field_value,
        config_type="general_settings",
    )
    await update_config_general_settings(
        data=updated_data, user_api_key_dict=user_api_key_dict
    )

    # Return the created endpoint with the generated ID
    created_endpoint = PassThroughGenericEndpoint(**data_dict)
    return PassThroughEndpointResponse(endpoints=[created_endpoint])


@router.delete(
    "/config/pass_through_endpoint",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PassThroughEndpointResponse,
)
async def delete_pass_through_endpoints(
    endpoint_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a pass-through endpoint by ID.

    Returns - the deleted endpoint
    """
    from litellm.proxy.proxy_server import (
        get_config_general_settings,
        update_config_general_settings,
    )

    ## Get existing pass-through endpoint field value

    try:
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        response = ConfigFieldInfo(
            field_name="pass_through_endpoints", field_value=None
        )

    ## Update field by removing endpoint
    pass_through_endpoint_data: Optional[List] = response.field_value
    if response.field_value is None or pass_through_endpoint_data is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "There are no pass-through endpoints setup."},
        )

    # Find the endpoint to delete
    found_endpoint = _find_endpoint_by_id(pass_through_endpoint_data, endpoint_id)

    if found_endpoint is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Endpoint with ID '{}' was not found in pass-through endpoint list.".format(
                    endpoint_id
                )
            },
        )

    # Find the index for deleting from the list
    endpoint_index = None
    for idx, endpoint in enumerate(pass_through_endpoint_data):
        _endpoint = (
            PassThroughGenericEndpoint(**endpoint)
            if isinstance(endpoint, dict)
            else endpoint
        )
        if _endpoint.id == endpoint_id:
            endpoint_index = idx
            break

    if endpoint_index is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Could not find index for endpoint with ID '{endpoint_id}'"
            },
        )

    # Remove the endpoint
    pass_through_endpoint_data.pop(endpoint_index)
    response_obj = found_endpoint

    # Remove routes from registry
    InitPassThroughEndpointHelpers.remove_endpoint_routes(endpoint_id)

    ## Update db
    updated_data = ConfigFieldUpdate(
        field_name="pass_through_endpoints",
        field_value=pass_through_endpoint_data,
        config_type="general_settings",
    )
    await update_config_general_settings(
        data=updated_data, user_api_key_dict=user_api_key_dict
    )

    return PassThroughEndpointResponse(endpoints=[response_obj])


def _find_endpoint_by_id(
    endpoints_data: List,
    endpoint_id: str,
) -> Optional[PassThroughGenericEndpoint]:
    """
    Find an endpoint by ID.

    Args:
        endpoints_data: List of endpoint data (dicts or PassThroughGenericEndpoint objects)
        endpoint_id: ID to search for

    Returns:
        Found endpoint or None if not found
    """
    for endpoint in endpoints_data:
        _endpoint: Optional[PassThroughGenericEndpoint] = None
        if isinstance(endpoint, dict):
            _endpoint = PassThroughGenericEndpoint(**endpoint)
        elif isinstance(endpoint, PassThroughGenericEndpoint):
            _endpoint = endpoint

        # Only compare IDs to IDs
        if _endpoint is not None and _endpoint.id == endpoint_id:
            return _endpoint

    return None


async def initialize_pass_through_endpoints_in_db():
    """
    Gets all pass-through endpoints from db and initializes them in the proxy server.
    """
    pass_through_endpoints = await _get_pass_through_endpoints_from_db()
    await initialize_pass_through_endpoints(
        pass_through_endpoints=pass_through_endpoints
    )
