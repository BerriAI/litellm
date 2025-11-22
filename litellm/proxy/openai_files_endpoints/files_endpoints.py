######################################################################

#                          /v1/files Endpoints

# Equivalent of https://platform.openai.com/docs/api-reference/files
######################################################################

import asyncio
import traceback
from typing import Optional, cast, get_args

import httpx
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

import litellm
from litellm import CreateFileRequest, get_secret_str
from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
)
from litellm.proxy.utils import ProxyLogging, is_known_model
from litellm.router import Router
from litellm.types.llms.openai import (
    CREATE_FILE_REQUESTS_PURPOSE,
    OpenAIFileObject,
    OpenAIFilesPurpose,
)

from .common_utils import (
    _is_base64_encoded_unified_file_id,
    encode_file_id_with_model,
    get_credentials_for_model,
    handle_model_based_routing,
    prepare_data_with_credentials,
)

router = APIRouter()

files_config = None


def set_files_config(config):
    global files_config
    if config is None:
        return

    if not isinstance(config, list):
        raise ValueError("invalid files config, expected a list is not a list")

    for element in config:
        if isinstance(element, dict):
            for key, value in element.items():
                if isinstance(value, str) and value.startswith("os.environ/"):
                    element[key] = get_secret_str(value)

    files_config = config


def get_files_provider_config(
    custom_llm_provider: str,
):
    global files_config
    if custom_llm_provider == "vertex_ai":
        return None
    if files_config is None:
        raise ValueError("files_settings is not set, set it on your config.yaml file.")
    for setting in files_config:
        if setting.get("custom_llm_provider") == custom_llm_provider:
            return setting
    return None


def get_first_json_object(file_content_bytes: bytes) -> Optional[dict]:
    try:
        # Decode the bytes to a string and split into lines
        file_content = file_content_bytes.decode("utf-8")
        first_line = file_content.splitlines()[0].strip()

        # Parse the JSON object from the first line
        json_object = json.loads(first_line)
        return json_object
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def get_model_from_json_obj(json_object: dict) -> Optional[str]:
    body = json_object.get("body", {}) or {}
    model = body.get("model")

    return model


async def _deprecated_loadbalanced_create_file(
    llm_router: Optional[Router],
    router_model: str,
    _create_file_request: CreateFileRequest,
) -> OpenAIFileObject:
    if llm_router is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "LLM Router not initialized. Ensure models added to proxy."
            },
        )

    response = await llm_router.acreate_file(model=router_model, **_create_file_request)
    return response


async def route_create_file(
    llm_router: Optional[Router],
    _create_file_request: CreateFileRequest,
    purpose: OpenAIFilesPurpose,
    proxy_logging_obj: ProxyLogging,
    user_api_key_dict: UserAPIKeyAuth,
    target_model_names_list: List[str],
    is_router_model: bool,
    router_model: Optional[str],
    custom_llm_provider: str,
    model: Optional[str] = None,
) -> OpenAIFileObject:
    """
    Route file creation request to the appropriate provider.
    
    Priority:
    1. If model parameter provided -> use model credentials and encode ID
    2. If enable_loadbalancing_on_batch_endpoints -> deprecated loadbalancing
    3. If target_model_names_list -> managed files (requires DB)
    4. Else -> use custom_llm_provider with files_settings
    """
    
    # NEW: Handle model-based routing (no DB required)
    if model is not None:
        # Get credentials from model_list via router
        credentials = get_credentials_for_model(
            llm_router=llm_router,
            model_id=model,
            operation_context="file upload",
        )
        
        # Merge credentials into the request
        prepare_data_with_credentials(
            data=_create_file_request,  # type: ignore
            credentials=credentials,
        )
        
        # Create the file with model credentials
        response = await litellm.acreate_file(
            **_create_file_request, 
            custom_llm_provider=credentials["custom_llm_provider"]
        )  # type: ignore
        
        # Encode the file ID with model information
        if response and hasattr(response, "id") and response.id:
            original_id = response.id
            encoded_id = encode_file_id_with_model(file_id=original_id, model=model)
            response.id = encoded_id
            verbose_proxy_logger.debug(
                f"Encoded file ID: {original_id} -> {encoded_id} (model: {model})"
            )
        
        return response
    
    # EXISTING: Deprecated loadbalancing approach
    if (
        litellm.enable_loadbalancing_on_batch_endpoints is True
        and is_router_model
        and router_model is not None
    ):
        response = await _deprecated_loadbalanced_create_file(
            llm_router=llm_router,
            router_model=router_model,
            _create_file_request=_create_file_request,
        )
    elif target_model_names_list:
        managed_files_obj = proxy_logging_obj.get_proxy_hook("managed_files")
        if managed_files_obj is None:
            raise ProxyException(
                message="Managed files hook not found",
                type="None",
                param="None",
                code=500,
            )
        if llm_router is None:
            raise ProxyException(
                message="LLM Router not found",
                type="None",
                param="None",
                code=500,
            )
        if not isinstance(managed_files_obj, BaseFileEndpoints):
            raise ProxyException(
                message="Managed files hook is not a BaseFileEndpoints",
                type="None",
                param="None",
                code=500,
            )
        response = await managed_files_obj.acreate_file(
            llm_router=llm_router,
            create_file_request=_create_file_request,
            target_model_names_list=target_model_names_list,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            user_api_key_dict=user_api_key_dict,
        )
    else:
        # get configs for custom_llm_provider
        llm_provider_config = get_files_provider_config(
            custom_llm_provider=custom_llm_provider
        )
        if llm_provider_config is not None:
            # add llm_provider_config to data
            _create_file_request.update(llm_provider_config)
        _create_file_request.pop("custom_llm_provider", None)  # type: ignore
        # for now use custom_llm_provider=="openai" -> this will change as LiteLLM adds more providers for acreate_batch
        response = await litellm.acreate_file(**_create_file_request, custom_llm_provider=custom_llm_provider)  # type: ignore

    return response


@router.post(
    "/{provider}/v1/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.post(
    "/v1/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.post(
    "/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
async def create_file(
    request: Request,
    fastapi_response: Response,
    purpose: str = Form(...),
    target_model_names: str = Form(default=""),
    provider: Optional[str] = None,
    custom_llm_provider: str = Form(default="openai"),
    file: UploadFile = File(...),
    litellm_metadata: Optional[str] = Form(default=None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Upload a file that can be used across - Assistants API, Batch API 
    This is the equivalent of POST https://api.openai.com/v1/files

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/files/create

    Example Curl
    ```
    curl http://localhost:4000/v1/files \
        -H "Authorization: Bearer sk-1234" \
        -F purpose="batch" \
        -F file="@mydata.jsonl"

    ```
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        # Read the file content
        file_content = await file.read()
        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_headers(request=request)
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )

        # NEW: Extract model parameter for multi-account routing
        request_body = await _read_request_body(request=request) or {}
        model_param = (
            request_body.get("model")
            or request.query_params.get("model")
            or request.headers.get("x-litellm-model")
        )

        target_model_names_list = (
            target_model_names.split(",") if target_model_names else []
        )
        target_model_names_list = [model.strip() for model in target_model_names_list]
        # Prepare the data for forwarding

        # Replace with:
        valid_purposes = get_args(OpenAIFilesPurpose)
        if purpose not in valid_purposes:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Invalid purpose: {purpose}. Must be one of: {valid_purposes}",
                },
            )
        # Cast purpose to OpenAIFilesPurpose type
        purpose = cast(OpenAIFilesPurpose, purpose)

        data = {}
        
        # Add litellm_metadata to data if provided (from form field)
        if litellm_metadata is not None:
            data["litellm_metadata"] = litellm_metadata

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # Prepare the file data according to FileTypes
        file_data = (file.filename, file_content, file.content_type)

        ## check if model is a loadbalanced model
        router_model: Optional[str] = None
        is_router_model = False
        if litellm.enable_loadbalancing_on_batch_endpoints is True:
            json_obj = get_first_json_object(file_content_bytes=file_content)
            if json_obj:
                router_model = get_model_from_json_obj(json_object=json_obj)
                is_router_model = is_known_model(
                    model=router_model, llm_router=llm_router
                )

        _create_file_request = CreateFileRequest(
            file=file_data, purpose=cast(CREATE_FILE_REQUESTS_PURPOSE, purpose), **data
        )

        response = await route_create_file(
            llm_router=llm_router,
            _create_file_request=_create_file_request,
            purpose=purpose,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=user_api_key_dict,
            target_model_names_list=target_model_names_list,
            is_router_model=is_router_model,
            router_model=router_model,
            custom_llm_provider=custom_llm_provider,
            model=model_param,
        )

        if response is None:
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to create file. Please try again."},
            )
        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ## POST CALL HOOKS ###
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )
        if _response is not None and isinstance(_response, OpenAIFileObject):
            response = _response

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.create_file(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/files/{file_id:path}/content",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.get(
    "/v1/files/{file_id:path}/content",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.get(
    "/files/{file_id:path}/content",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
async def get_file_content(
    request: Request,
    fastapi_response: Response,
    file_id: str,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Returns information about a specific file. that can be used across - Assistants API, Batch API 
    This is the equivalent of GET https://api.openai.com/v1/files/{file_id}/content

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/files/retrieve-contents

    Example Curl
    ```
    curl http://localhost:4000/v1/files/file-abc123/content \
        -H "Authorization: Bearer sk-1234"

    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {"file_id": file_id}
    try:
        # Include original request and headers in the data
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type="afile_content",
        )

        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_headers(request=request)
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )

        ## check if file_id is a litellm managed file
        is_base64_unified_file_id = _is_base64_encoded_unified_file_id(file_id)
        if is_base64_unified_file_id:
            managed_files_obj = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_obj is None:
                raise ProxyException(
                    message="Managed files hook not found",
                    type="None",
                    param="None",
                    code=500,
                )
            if llm_router is None:
                raise ProxyException(
                    message="LLM Router not found",
                    type="None",
                    param="None",
                    code=500,
                )
            if not isinstance(managed_files_obj, BaseFileEndpoints):
                raise ProxyException(
                    message="Managed files hook is not a BaseFileEndpoints",
                    type="None",
                    param="None",
                    code=500,
                )
            model = cast(Optional[str], data.get("model"))
            if model:
                response = await llm_router.afile_content(
                    **{
                        "model": model,
                        "file_id": file_id,
                        **data,
                    }
                )  # type: ignore

            else:
                response = await managed_files_obj.afile_content(
                    **{
                        "file_id": file_id,
                        "litellm_parent_otel_span": user_api_key_dict.parent_otel_span,
                        "llm_router": llm_router,
                        **data,
                    }
                )
        else:
            # Check for model-based credential routing
            should_route, model_used, original_file_id, credentials = handle_model_based_routing(
                file_id=file_id,
                request=request,
                llm_router=llm_router,
                data=data,
                check_file_id_encoding=True,
            )
            
            if should_route:
                # Use model-based routing with credentials from config
                prepare_data_with_credentials(
                    data=data,
                    credentials=credentials,  # type: ignore
                    file_id=original_file_id,  # Use decoded file ID if from encoded ID
                )
                
                response = await litellm.afile_content(
                    custom_llm_provider=credentials["custom_llm_provider"],  # type: ignore
                    **data
                )  # type: ignore
                
                verbose_proxy_logger.debug(
                    f"Retrieved file content using model: {model_used}"
                    + (f", file_id: {file_id} -> {original_file_id}" if original_file_id else "")
                )
            else:
                # Fallback to default behavior (uses env variables or provider-based routing)
                response = await litellm.afile_content(
                    **{
                        "custom_llm_provider": custom_llm_provider,
                        "file_id": file_id,
                        **data,
                    }  # type: ignore
                )

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        httpx_response: Optional[httpx.Response] = getattr(response, "response", None)
        if httpx_response is None:
            raise ValueError(
                f"Invalid response - response.response is None - got {response}"
            )

        return Response(
            content=httpx_response.content,
            status_code=httpx_response.status_code,
            headers=httpx_response.headers,
        )

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.retrieve_file_content(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/files/{file_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.get(
    "/v1/files/{file_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.get(
    "/files/{file_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
async def get_file(
    request: Request,
    fastapi_response: Response,
    file_id: str,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Returns information about a specific file. that can be used across - Assistants API, Batch API 
    This is the equivalent of GET https://api.openai.com/v1/files/{file_id}

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/files/retrieve

    Example Curl
    ```
    curl http://localhost:4000/v1/files/file-abc123 \
        -H "Authorization: Bearer sk-1234"

    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:

        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_headers(request=request)
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )

        # Include original request and headers in the data
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type="afile_retrieve",
        )

        ## Check for model-based credential routing
        from litellm.proxy.proxy_server import llm_router
        
        should_route, model_used, original_file_id, credentials = handle_model_based_routing(
            file_id=file_id,
            request=request,
            llm_router=llm_router,
            data=data,
            check_file_id_encoding=True,
        )
        
        if should_route:
            # Use model-based routing with credentials from config
            prepare_data_with_credentials(
                data=data,
                credentials=credentials,  # type: ignore
                file_id=original_file_id,
            )

            response = await litellm.afile_retrieve(**data)  # type: ignore
            
            # Keep the encoded ID in response if it was originally encoded
            if original_file_id and response and hasattr(response, "id") and response.id:
                response.id = file_id
            
            verbose_proxy_logger.debug(
                f"Retrieved file using model: {model_used}"
                + (f", original_id: {original_file_id}" if original_file_id else "")
            )
        
        ## EXISTING: check if file_id is a litellm managed file
        elif _is_base64_encoded_unified_file_id(file_id):
            managed_files_obj = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_obj is None:
                raise ProxyException(
                    message="Managed files hook not found",
                    type="None",
                    param="None",
                    code=500,
                )
            if not isinstance(managed_files_obj, BaseFileEndpoints):
                raise ProxyException(
                    message="Managed files hook is not a BaseFileEndpoints",
                    type="None",
                    param="None",
                    code=500,
                )
            response = await managed_files_obj.afile_retrieve(
                file_id=file_id,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            )
        else:
            response = await litellm.afile_retrieve(
                custom_llm_provider=custom_llm_provider, file_id=file_id, **data  # type: ignore
            )

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.retrieve_file(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.delete(
    "/{provider}/v1/files/{file_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.delete(
    "/v1/files/{file_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.delete(
    "/files/{file_id:path}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
async def delete_file(
    request: Request,
    fastapi_response: Response,
    file_id: str,
    provider: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Deletes a specified file. that can be used across - Assistants API, Batch API 
    This is the equivalent of DELETE https://api.openai.com/v1/files/{file_id}

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/files/delete

    Example Curl
    ```
    curl http://localhost:4000/v1/files/file-abc123 \
    -X DELETE \
    -H "Authorization: Bearer $OPENAI_API_KEY"

    ```
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        custom_llm_provider = (
            provider
            or get_custom_llm_provider_from_request_headers(request=request)
            or get_custom_llm_provider_from_request_query(request=request)
            or await get_custom_llm_provider_from_request_body(request=request)
            or "openai"
        )
        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # Check for model-based credential routing
        should_route, model_used, original_file_id, credentials = handle_model_based_routing(
            file_id=file_id,
            request=request,
            llm_router=llm_router,
            data=data,
            check_file_id_encoding=True,
        )
        
        if should_route:
            # Use model-based routing with credentials from config
            prepare_data_with_credentials(
                data=data,
                credentials=credentials,  # type: ignore
                file_id=original_file_id,
            )
            
            response = await litellm.afile_delete(**data)  # type: ignore
            
            verbose_proxy_logger.debug(
                f"Deleted file using model: {model_used}"
                + (f", original_id: {original_file_id}" if original_file_id else "")
            )
        
        ## EXISTING: check if file_id is a litellm managed file
        elif _is_base64_encoded_unified_file_id(file_id):
            managed_files_obj = proxy_logging_obj.get_proxy_hook("managed_files")
            if managed_files_obj is None:
                raise ProxyException(
                    message="Managed files hook not found",
                    type="None",
                    param="None",
                    code=500,
                )
            if llm_router is None:
                raise ProxyException(
                    message="LLM Router not found",
                    type="None",
                    param="None",
                    code=500,
                )
            if not isinstance(managed_files_obj, BaseFileEndpoints):
                raise ProxyException(
                    message="Managed files hook is not a BaseFileEndpoints",
                    type="None",
                    param="None",
                    code=500,
                )

            response = await managed_files_obj.afile_delete(
                file_id=file_id,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                llm_router=llm_router,
                **data,
            )
        else:
            response = await litellm.afile_delete(
                custom_llm_provider=custom_llm_provider, file_id=file_id, **data  # type: ignore
            )

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.delete_file(): Exception occured - {}".format(
                str(e)
            )
        )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )


@router.get(
    "/{provider}/v1/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.get(
    "/v1/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
@router.get(
    "/files",
    dependencies=[Depends(user_api_key_auth)],
    tags=["files"],
)
async def list_files(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    provider: Optional[str] = None,
    target_model_names: Optional[str] = None,
    purpose: Optional[str] = None,
):
    """
    Returns information about a specific file. that can be used across - Assistants API, Batch API 
    This is the equivalent of GET https://api.openai.com/v1/files/

    Supports Identical Params as: https://platform.openai.com/docs/api-reference/files/list

    Example Curl
    ```
    curl http://localhost:4000/v1/files\
        -H "Authorization: Bearer sk-1234"

    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        version,
    )

    data: Dict = {}
    try:
        # Include original request and headers in the data
        base_llm_response_processor = ProxyBaseLLMRequestProcessing(data=data)
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type=CallTypes.alist_fine_tuning_jobs.value,
        )

        response: Optional[Any] = None
        
        # Check for model-based credential routing (no file_id encoding check for list)
        should_route, model_used, _, credentials = handle_model_based_routing(
            file_id="",  # No file_id for list endpoint
            request=request,
            llm_router=llm_router,
            data=data,
            check_file_id_encoding=False,
        )
        
        if should_route:
            # Use model-based routing with credentials from config
            data.update(credentials)  # type: ignore
            response = await litellm.afile_list(
                custom_llm_provider=credentials["custom_llm_provider"],  # type: ignore
                purpose=purpose,
                **data  # type: ignore
            )
            
            verbose_proxy_logger.debug(f"Listed files using model: {model_used}")
        
        elif target_model_names and isinstance(target_model_names, str):
            target_model_names_list = target_model_names.split(",")
            if len(target_model_names_list) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="target_model_names on list files must be a list of one model name. Example: ['gpt-4o']",
                )
            ## Use router to list fine-tuning jobs for that model
            if llm_router is None:
                raise HTTPException(
                    status_code=500,
                    detail="LLM Router not initialized. Ensure models added to proxy.",
                )
            data["model"] = target_model_names_list[0]
            response = await llm_router.afile_list(
                **data,
            )
        else:
            custom_llm_provider = (
                provider
                or get_custom_llm_provider_from_request_headers(request=request)
                or get_custom_llm_provider_from_request_query(request=request)
                or await get_custom_llm_provider_from_request_body(request=request)
                or "openai"
            )

            response = await litellm.afile_list(
                custom_llm_provider=custom_llm_provider, purpose=purpose, **data  # type: ignore
            )

        if response is None:
            raise HTTPException(
                status_code=500,
                detail="Either 'provider' or 'target_model_names' must be provided e.g. `?target_model_names=gpt-4o`",
            )

        ## POST CALL HOOKS ###
        _response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )
        if _response is not None and isinstance(_response, OpenAIFileObject):
            response = _response

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            )
        )
        return response

    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.list_files(): Exception occured - {}".format(
                str(e)
            )
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e.detail)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )
