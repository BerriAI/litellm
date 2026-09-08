import asyncio
import io
import traceback
from collections.abc import Sequence
from typing import Final, get_type_hints

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse
from starlette.datastructures import UploadFile

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    _is_form_content_type,
    coerce_numeric_form_fields,
    numeric_form_fields,
)
from litellm.proxy.route_llm_request import route_request
from litellm.types.images.main import ImageEditRequestParams
from litellm.types.llms.openai import ChatCompletionUserMessage

router: Final = APIRouter()

IMAGE_EDIT_NUMERIC_FORM_FIELDS: Final = numeric_form_fields(get_type_hints(ImageEditRequestParams))
_IMAGE_REFERENCE_PREFIXES: Final = ("http://", "https://", "data:image/")
_IMAGE_EDIT_FILE_FIELDS: Final = (
    ("image", "image[]"),
    ("mask", "mask[]"),
)


async def uploadfile_to_bytesio(upload: UploadFile) -> io.BytesIO:
    """
    Read a FastAPI UploadFile into a BytesIO and set .name so OpenAI SDK
    infers filename/content-type correctly.
    """
    data: Final = await upload.read()
    buffer: Final = io.BytesIO(data)
    buffer.name = upload.filename
    return buffer


async def batch_to_bytesio(
    uploads: Sequence[UploadFile] | None,
) -> list[io.BytesIO] | None:
    """
    Convert a sequence of UploadFiles to a list of BytesIO buffers, or None.
    """
    if not uploads:
        return None
    return [await uploadfile_to_bytesio(u) for u in uploads]


def _is_image_reference_string(value: str) -> bool:
    return value.startswith(_IMAGE_REFERENCE_PREFIXES)


def _invalid_image_field_error(field: str) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail=f"'{field}' must be a multipart file, http(s) URL, or data:image URI.",
    )


def _form_field_values(form: object, name: str) -> tuple[object, ...]:
    getlist: Final = getattr(form, "getlist", None)
    if not callable(getlist):
        return ()
    return tuple(getlist(name))


async def _coerce_image_part(value: object, field: str) -> io.BytesIO | str:
    if isinstance(value, UploadFile):
        return await uploadfile_to_bytesio(value)
    if isinstance(value, str) and _is_image_reference_string(value):
        return value
    raise _invalid_image_field_error(field)


async def _normalize_image_values(values: tuple[object, ...], field: str) -> object | None:
    if not values:
        return None
    coerced: Final = tuple([await _coerce_image_part(value, field) for value in values])
    if len(coerced) == 1 and isinstance(coerced[0], str):
        return coerced[0]
    return list(coerced)


def _json_image_values(data: dict[str, object], field: str) -> tuple[object, ...]:
    if field not in data:
        return ()
    raw: Final = data[field]
    if isinstance(raw, list):
        return tuple(raw)
    return (raw,)


async def _normalized_image_edit_fields(
    values_by_field: dict[str, tuple[object, ...]],
) -> dict[str, object]:
    image: Final = await _normalize_image_values(values_by_field["image"], "image")
    mask: Final = await _normalize_image_values(values_by_field["mask"], "mask")
    return {
        **({"image": image} if image is not None else {}),
        **({"mask": mask} if mask is not None else {}),
    }


async def _image_edit_assets_from_request(
    request: Request,
    data: dict[str, object],
) -> dict[str, object]:
    form: Final = await request.form() if _is_form_content_type(request.headers.get("content-type", "")) else None
    if form is None:
        return {
            **{key: value for key, value in data.items() if key not in {"image[]", "mask[]"}},
            **await _normalized_image_edit_fields(
                {field: _json_image_values(data, field) for field, _alias in _IMAGE_EDIT_FILE_FIELDS}
            ),
        }

    form_values: Final = {
        name: _form_field_values(form, name) for field, alias in _IMAGE_EDIT_FILE_FIELDS for name in (field, alias)
    }
    conflicts: Final = tuple(
        field for field, alias in _IMAGE_EDIT_FILE_FIELDS if form_values[field] and form_values[alias]
    )
    if conflicts:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot specify both '{conflicts[0]}' and '{conflicts[0]}[]'",
        )
    return {
        **{key: value for key, value in data.items() if key not in {"image[]", "mask[]"}},
        **await _normalized_image_edit_fields(
            {field: form_values[field] or form_values[alias] for field, alias in _IMAGE_EDIT_FILE_FIELDS}
        ),
    }


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
@router.post(
    "/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
@router.post(
    "/openai/deployments/{model:path}/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)  # azure compatible endpoint
async def image_generation(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: str | None = None,
):
    from litellm.proxy.litellm_pre_call_utils import reject_url_valued_destination
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_model,
        version,
    )

    data = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body: Final = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        if isinstance(model, str):
            reject_url_valued_destination("model", model)

        data["model"] = (
            model
            or general_settings.get("image_generation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        prompt_value: Final = data.get("prompt")
        if prompt_value is not None:
            # Reformat the image prompt as a chat message so guardrails can process it.
            user_message: Final[ChatCompletionUserMessage] = {
                "role": "user",
                "content": prompt_value,
            }
            data["messages"] = [user_message]
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="image_generation"
        )

        messages: Final = data.get("messages")
        if isinstance(messages, list) and messages:
            data["prompt"] = get_str_from_messages(messages)
        data.pop("messages", None)

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call: Final = await route_request(
            data=data,
            route_type="aimage_generation",
            llm_router=llm_router,
            user_model=user_model,
        )
        response = await llm_call

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### CALL HOOKS ### - modify outgoing data (guardrails, otel, etc.)
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )

        ### RESPONSE HEADERS ###
        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        model_id: Final = hidden_params.get("model_id", None) or ""
        cache_key: Final = hidden_params.get("cache_key", None) or ""
        api_base: Final = hidden_params.get("api_base", None) or ""
        response_cost: Final = hidden_params.get("response_cost", None) or ""
        litellm_call_id: Final = hidden_params.get("litellm_call_id", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        # Call response headers hook (matches base_process_llm_request behavior)
        callback_headers: Final = await proxy_logging_obj.post_call_response_headers_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_headers=dict(request.headers),
        )
        if callback_headers:
            fastapi_response.headers.update(callback_headers)

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error("litellm.proxy.proxy_server.image_generation(): Exception occured - %s", e)
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                openai_code=getattr(e, "code", None),
                code=getattr(e, "status_code", 500),
            )


@router.post(
    "/v1/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    tags=["images"],
)
@router.post(
    "/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    tags=["images"],
)
@router.post(
    "/openai/deployments/{model:path}/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)  # azure compatible endpoint
async def image_edit_api(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: str | None = None,
):
    """
    Follows the OpenAI Images API spec: https://platform.openai.com/docs/api-reference/images/create

    ```bash
    curl -s -D >(grep -i x-request-id >&2) \
    -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png) \
    -X POST "http://localhost:4000/v1/images/edits" \
    -H "Authorization: Bearer sk-1234" \
        -F "model=gpt-image-1" \
        -F "image[]=@soap.png" \
        -F 'prompt=Create a studio ghibli image of this'
    ```
    """
    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    parsed_body: Final = dict(
        coerce_numeric_form_fields(
            parsed_body=await _read_request_body(request=request),
            numeric_fields=IMAGE_EDIT_NUMERIC_FORM_FIELDS,
        )
    )
    with_assets: Final = await _image_edit_assets_from_request(request, parsed_body)
    data: Final = {
        **with_assets,
        **({} if "prompt" in with_assets else {"prompt": None}),
        "model": (
            model or general_settings.get("image_generation_model", None) or user_model or with_assets.get("model")
        ),
    }
    #########################################################
    # Process request
    #########################################################

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aimage_edit",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
