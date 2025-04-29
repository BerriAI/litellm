import ast
import asyncio
import json
import uuid
from base64 import b64encode
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import (
    ConfigFieldInfo,
    ConfigFieldUpdate,
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

    {"Authorization": "bearer os.environ/COHERE_API_KEY"}
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
        elif (
            llm_router is not None and data["model"] in llm_router.get_model_ids()
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

        verbose_proxy_logger.info("\nResponse from Litellm:\n{}".format(response))
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


class HttpPassThroughEndpointHelpers:
    @staticmethod
    def forward_headers_from_request(
        request: Request,
        headers: dict,
        forward_headers: Optional[bool] = False,
    ):
        """
        Helper to forward headers from original request
        """
        if forward_headers is True:
            request_headers = dict(request.headers)

            # Header We Should NOT forward
            request_headers.pop("content-length", None)
            request_headers.pop("host", None)

            # Combine request headers with custom headers
            headers = {**request_headers, **headers}
        return headers

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
        if ("generateContent") in url or ("streamGenerateContent") in url:
            return EndpointType.VERTEX_AI
        elif parsed_url.hostname == "api.anthropic.com":
            return EndpointType.ANTHROPIC
        return EndpointType.GENERIC

    @staticmethod
    def get_merged_query_parameters(
        existing_url: httpx.URL, request_query_params: Dict[str, Union[str, list]]
    ) -> Dict[str, Union[str, List[str]]]:
        # Get the existing query params from the target URL
        existing_query_string = existing_url.query.decode("utf-8")
        existing_query_params = parse_qs(existing_query_string)

        # parse_qs returns a dict where each value is a list, so let's flatten it
        updated_existing_query_params = {
            k: v[0] if len(v) == 1 else v for k, v in existing_query_params.items()
        }
        # Merge the query params, giving priority to the existing ones
        return {**request_query_params, **updated_existing_query_params}

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

        response = await async_client.request(
            method=request.method,
            url=url,
            headers=headers,
            params=requested_query_params,
            files=files,
            data=form_data_dict,
        )
        return response


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
):
    litellm_call_id = str(uuid.uuid4())
    url: Optional[httpx.URL] = None
    try:
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.proxy.proxy_server import proxy_logging_obj

        url = httpx.URL(target)
        headers = custom_headers
        headers = HttpPassThroughEndpointHelpers.forward_headers_from_request(
            request=request, headers=headers, forward_headers=forward_headers
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

        _parsed_body = None
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
        )
        kwargs = _init_kwargs_for_pass_through_endpoint(
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
        requested_query_params: Optional[dict] = (
            query_params or request.query_params.__dict__
        )
        if requested_query_params == request.query_params.__dict__:
            requested_query_params = None

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

        verbose_proxy_logger.debug("request method: {}".format(request.method))
        verbose_proxy_logger.debug("request url: {}".format(url))
        verbose_proxy_logger.debug("request headers: {}".format(headers))
        verbose_proxy_logger.debug(
            "requested_query_params={}".format(requested_query_params)
        )
        verbose_proxy_logger.debug("request body: {}".format(_parsed_body))

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


def _init_kwargs_for_pass_through_endpoint(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    passthrough_logging_payload: PassthroughStandardLoggingPayload,
    logging_obj: LiteLLMLoggingObj,
    _parsed_body: Optional[dict] = None,
    litellm_call_id: Optional[str] = None,
) -> dict:
    _parsed_body = _parsed_body or {}
    _litellm_metadata: Optional[dict] = _parsed_body.pop("litellm_metadata", None)
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
        )
    )
    _metadata["user_api_key"] = user_api_key_dict.api_key
    if _litellm_metadata:
        _metadata.update(_litellm_metadata)

    _metadata = _update_metadata_with_tags_in_header(
        request=request,
        metadata=_metadata,
    )

    kwargs = {
        "litellm_params": {
            "metadata": _metadata,
        },
        "call_type": "pass_through_endpoint",
        "litellm_call_id": litellm_call_id,
        "passthrough_logging_payload": passthrough_logging_payload,
    }

    logging_obj.model_call_details["passthrough_logging_payload"] = (
        passthrough_logging_payload
    )

    return kwargs


def _update_metadata_with_tags_in_header(request: Request, metadata: dict) -> dict:
    """
    If tags are in the request headers, add them to the metadata

    Used for google and vertex JS SDKs
    """
    _tags = request.headers.get("tags")
    if _tags:
        metadata["tags"] = _tags.split(",")
    return metadata


def create_pass_through_route(
    endpoint,
    target: str,
    custom_headers: Optional[dict] = None,
    _forward_headers: Optional[bool] = False,
    _merge_query_params: Optional[bool] = False,
    dependencies: Optional[List] = None,
):
    # check if target is an adapter.py or a url
    import uuid

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
            query_params: Optional[dict] = None,
            custom_body: Optional[dict] = None,
            stream: Optional[
                bool
            ] = None,  # if pass-through endpoint is a streaming request
        ):
            return await pass_through_request(  # type: ignore
                request=request,
                target=target,
                custom_headers=custom_headers or {},
                user_api_key_dict=user_api_key_dict,
                forward_headers=_forward_headers,
                merge_query_params=_merge_query_params,
                query_params=query_params,
                stream=stream,
                custom_body=custom_body,
            )

    return endpoint_func


def _is_streaming_response(response: httpx.Response) -> bool:
    _content_type = response.headers.get("content-type")
    if _content_type is not None and "text/event-stream" in _content_type:
        return True
    return False


async def initialize_pass_through_endpoints(pass_through_endpoints: list):
    verbose_proxy_logger.debug("initializing pass through endpoints")
    from litellm.proxy._types import CommonProxyErrors, LiteLLMRoutes
    from litellm.proxy.proxy_server import app, premium_user

    for endpoint in pass_through_endpoints:
        _target = endpoint.get("target", None)
        _path = endpoint.get("path", None)
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

        verbose_proxy_logger.debug(
            "adding pass through endpoint: %s, dependencies: %s", _path, _dependencies
        )
        app.add_api_route(  # type: ignore
            path=_path,
            endpoint=create_pass_through_route(  # type: ignore
                _path,
                _target,
                _custom_headers,
                _forward_headers,
                _merge_query_params,
                _dependencies,
            ),
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            dependencies=_dependencies,
        )

        verbose_proxy_logger.debug("Added new pass through endpoint: %s", _path)


@router.get(
    "/config/pass_through_endpoint",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PassThroughEndpointResponse,
)
async def get_pass_through_endpoints(
    endpoint_id: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    GET configured pass through endpoint.

    If no endpoint_id given, return all configured endpoints.
    """
    from litellm.proxy.proxy_server import get_config_general_settings

    ## Get existing pass-through endpoint field value
    try:
        response: ConfigFieldInfo = await get_config_general_settings(
            field_name="pass_through_endpoints", user_api_key_dict=user_api_key_dict
        )
    except Exception:
        return PassThroughEndpointResponse(endpoints=[])

    pass_through_endpoint_data: Optional[List] = response.field_value
    if pass_through_endpoint_data is None:
        return PassThroughEndpointResponse(endpoints=[])

    returned_endpoints = []
    if endpoint_id is None:
        for endpoint in pass_through_endpoint_data:
            if isinstance(endpoint, dict):
                returned_endpoints.append(PassThroughGenericEndpoint(**endpoint))
            elif isinstance(endpoint, PassThroughGenericEndpoint):
                returned_endpoints.append(endpoint)
    elif endpoint_id is not None:
        for endpoint in pass_through_endpoint_data:
            _endpoint: Optional[PassThroughGenericEndpoint] = None
            if isinstance(endpoint, dict):
                _endpoint = PassThroughGenericEndpoint(**endpoint)
            elif isinstance(endpoint, PassThroughGenericEndpoint):
                _endpoint = endpoint

            if _endpoint is not None and _endpoint.path == endpoint_id:
                returned_endpoints.append(_endpoint)

    return PassThroughEndpointResponse(endpoints=returned_endpoints)


@router.post(
    "/config/pass_through_endpoint/{endpoint_id}",
    dependencies=[Depends(user_api_key_auth)],
)
async def update_pass_through_endpoints(request: Request, endpoint_id: str):
    """
    Update a pass-through endpoint
    """
    pass


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

    ## Update field with new endpoint
    data_dict = data.model_dump()
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
    Delete a pass-through endpoint

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
    response_obj: Optional[PassThroughGenericEndpoint] = None
    if response.field_value is None or pass_through_endpoint_data is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "There are no pass-through endpoints setup."},
        )
    elif isinstance(response.field_value, List):
        invalid_idx: Optional[int] = None
        for idx, endpoint in enumerate(pass_through_endpoint_data):
            _endpoint: Optional[PassThroughGenericEndpoint] = None
            if isinstance(endpoint, dict):
                _endpoint = PassThroughGenericEndpoint(**endpoint)
            elif isinstance(endpoint, PassThroughGenericEndpoint):
                _endpoint = endpoint

            if _endpoint is not None and _endpoint.path == endpoint_id:
                invalid_idx = idx
                response_obj = _endpoint

        if invalid_idx is not None:
            pass_through_endpoint_data.pop(invalid_idx)

    ## Update db
    updated_data = ConfigFieldUpdate(
        field_name="pass_through_endpoints",
        field_value=pass_through_endpoint_data,
        config_type="general_settings",
    )
    await update_config_general_settings(
        data=updated_data, user_api_key_dict=user_api_key_dict
    )

    if response_obj is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Endpoint={} was not found in pass-through endpoint list.".format(
                    endpoint_id
                )
            },
        )
    return PassThroughEndpointResponse(endpoints=[response_obj])
