import asyncio
import io
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from datetime import datetime
from functools import partial
from itertools import chain
from types import MappingProxyType
from typing import Final, Protocol, get_type_hints

import anyio
import orjson
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import ORJSONResponse
from pydantic import TypeAdapter

import litellm
from litellm.constants import MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY
from litellm.images.streaming import (
    AsyncImageStream,
    completed_image_events,
    image_event_sse_frame,
    image_response_from_completed_event,
    image_stream_requested,
    is_completed_image_event,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_response,
    hierarchical_router_settings_updates,
    log_llm_api_exception,
    resolve_litellm_call_id,
)
from litellm.proxy.common_utils.http_parsing_utils import (
    coerce_numeric_form_fields,
    numeric_form_fields,
    resolve_inference_model,
)
from litellm.proxy.common_utils.openai_error_payload import (
    error_status_code,
    litellm_call_id_headers,
    openai_error_param,
    openai_error_type,
)
from litellm.proxy.route_llm_request import route_request
from litellm.types.images.main import ImageEditRequestParams
from litellm.types.llms.openai import ChatCompletionUserMessage
from litellm.types.utils import CallTypes, ImageResponse

router: Final = APIRouter()
_IMAGE_GENERATION_MODEL_ADAPTER: Final[TypeAdapter[object]] = TypeAdapter(object)

IMAGE_EDIT_NUMERIC_FORM_FIELDS: Final = numeric_form_fields(get_type_hints(ImageEditRequestParams))
IMAGE_EDIT_OPTIONAL_FIELD_DEFAULTS: Final = MappingProxyType({"prompt": None, "image": None})

IMAGE_ARRAY_FIELD: Final = "image[]"
MASK_ARRAY_FIELD: Final = "mask[]"
BRACKETED_FILE_FIELDS: Final = frozenset({IMAGE_ARRAY_FIELD, MASK_ARRAY_FIELD})


class _PostCallSuccessHook(Protocol):
    @property
    def post_call_success_hook(self) -> Callable[..., Awaitable[object]]: ...


def _post_call_success_hook_of(proxy_logging: _PostCallSuccessHook) -> Callable[..., Awaitable[object]]:
    return proxy_logging.post_call_success_hook


async def image_event_sse_stream(
    response: ImageResponse | AsyncImageStream,
    user_api_key_dict: UserAPIKeyAuth,
    request_data: dict[str, object],
    request: Request | None = None,
    *,
    event_prefix: str,
) -> AsyncGenerator[str, None]:
    """Answer a streamed image request with the provider's events as server-sent events.

    The finished image always goes through `post_call_success_hook` before it is sent. Partial images are
    held back when a post-call guardrail applies, since only the finished image can be checked."""
    from litellm.proxy.proxy_server import openai_sse_error_frame, proxy_logging_obj

    async def checked_completed_frames(image_response: ImageResponse, provider_frame: str | None) -> tuple[str, ...]:
        unchecked: Final = image_response.model_dump()
        checked: Final = await _post_call_success_hook_of(proxy_logging_obj)(
            data=request_data, user_api_key_dict=user_api_key_dict, response=image_response
        )
        served: Final = checked if isinstance(checked, ImageResponse) else image_response
        if provider_frame is not None and served.model_dump() == unchecked:
            return (provider_frame,)
        return tuple(
            image_event_sse_frame(event) for event in completed_image_events(served, event_prefix=event_prefix)
        )

    try:
        if isinstance(response, ImageResponse):
            for frame in await checked_completed_frames(response, None):
                yield frame
            return
        guarded: Final = ProxyBaseLLMRequestProcessing(data=request_data)._has_post_call_guardrails_for_passthrough()  # pyright: ignore[reportPrivateUsage]  # the gate post_call_success_hook itself applies
        async for event in response:
            if is_completed_image_event(event):
                for frame in await checked_completed_frames(
                    image_response_from_completed_event(event), image_event_sse_frame(event)
                ):
                    yield frame
            elif not guarded:
                yield image_event_sse_frame(event)
    except Exception as stream_error:  # noqa: BLE001  # stream boundary: a failure ends the SSE body as an error event
        replacement: Final = await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=stream_error,
            request_data=request_data,
        )
        yield openai_sse_error_frame(stream_error if replacement is None else replacement)
    finally:
        if isinstance(response, AsyncImageStream):
            with anyio.CancelScope(shield=True):
                await response.aclose()


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

    litellm_call_id: Final = resolve_litellm_call_id(request.headers.get("x-litellm-call-id"))
    data = {"litellm_call_id": litellm_call_id}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body: Final = await request.body()
        data = orjson.loads(body) | data

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

        streams: Final = image_stream_requested(data.get("stream"))
        data["model"] = resolve_inference_model(
            data.get("model"), general_settings, user_model, model, kind="image_generation"
        )

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        if llm_router is not None:
            requested_model: Final = _IMAGE_GENERATION_MODEL_ADAPTER.validate_python(data.get("model"))
            data.update(
                await hierarchical_router_settings_updates(
                    requested_model=requested_model,
                    user_api_key_dict=user_api_key_dict,
                    llm_router=llm_router,
                    proxy_config=proxy_config,
                    proxy_logging_obj=proxy_logging_obj,
                    apply_model_group_alias=request.scope.get(MODEL_GROUP_ALIAS_RESOLVED_SCOPE_KEY) is not True,
                )
            )

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

        if "prompt" in data:
            logging_obj, data = litellm.utils.function_setup(
                original_function="aimage_generation",
                rules_obj=litellm.utils.Rules(),
                start_time=datetime.now(),  # noqa: DTZ005  # Logging subtracts it from naive end times
                is_async_call=True,
                **data,
            )
            data["litellm_logging_obj"] = logging_obj

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call: Final = await route_request(
            data=data,
            route_type="aimage_generation",
            llm_router=llm_router,
            user_model=user_model,
            user_api_key_dict=user_api_key_dict,
        )
        generated: Final = await llm_call

        ### ALERTING ###
        asyncio.create_task(proxy_logging_obj.update_request_status(litellm_call_id=litellm_call_id, status="success"))

        async def response_headers(served: object) -> dict[str, str]:
            hidden_params: Final = getattr(served, "_hidden_params", {}) or {}
            # Call response headers hook (matches base_process_llm_request behavior)
            callback_headers: Final = await proxy_logging_obj.post_call_response_headers_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=served,
                request_headers=dict(request.headers),
            )
            return {
                **ProxyBaseLLMRequestProcessing.get_custom_headers(
                    user_api_key_dict=user_api_key_dict,
                    model_id=hidden_params.get("model_id", None) or "",
                    cache_key=hidden_params.get("cache_key", None) or "",
                    api_base=hidden_params.get("api_base", None) or "",
                    version=version,
                    response_cost=hidden_params.get("response_cost", None) or "",
                    model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                    call_id=hidden_params.get("litellm_call_id", None) or "",
                    request_data=data,
                    hidden_params=hidden_params,
                ),
                **(callback_headers or {}),
            }

        if isinstance(generated, AsyncImageStream) or (streams and isinstance(generated, ImageResponse)):
            return await create_response(
                generator=image_event_sse_stream(
                    response=generated,
                    user_api_key_dict=user_api_key_dict,
                    request_data=data,
                    event_prefix="image_generation",
                ),
                media_type="text/event-stream",
                headers=await response_headers(generated),
                request=request,
                general_settings=general_settings,
            )

        ### CALL HOOKS ### - modify outgoing data (guardrails, otel, etc.)
        response: Final = await _post_call_success_hook_of(proxy_logging_obj)(
            data=data, user_api_key_dict=user_api_key_dict, response=generated
        )

        fastapi_response.headers.update(await response_headers(response))

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=data,
            dispatched_call_type=CallTypes.aimage_generation.value,
        )
        log_llm_api_exception(e, litellm_call_id)
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=openai_error_type(e, error_status_code(e, status.HTTP_400_BAD_REQUEST)),
                param=openai_error_param(e),
                headers=litellm_call_id_headers(litellm_call_id),
                code=error_status_code(e, status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg: Final = f"{e}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=openai_error_type(e, error_status_code(e, 500)),
                param=openai_error_param(e),
                headers=litellm_call_id_headers(litellm_call_id),
                openai_code=getattr(e, "code", None),
                code=error_status_code(e, 500),
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
    image: list[UploadFile] | None = File(None),
    image_array: list[UploadFile] | None = File(None, alias=IMAGE_ARRAY_FIELD),
    mask: list[UploadFile] | None = File(None),
    mask_array: list[UploadFile] | None = File(None, alias=MASK_ARRAY_FIELD),
    model: str | None = None,
):
    """
    Follows the OpenAI Images API spec: https://platform.openai.com/docs/api-reference/images/create

    ```bash
    curl -s -D >(grep -i x-request-id >&2) \
    -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png) \
    -X POST "http://localhost:4000/v1/images/edits" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
        -F "model=gpt-image-1" \
        -F "image[]=@soap.png" \
        -F 'prompt=Create a studio ghibli image of this'
    ```
    """
    if image is not None and image_array is not None:
        raise HTTPException(status_code=422, detail="Cannot specify both 'image' and 'image[]'")
    if mask is not None and mask_array is not None:
        raise HTTPException(status_code=422, detail="Cannot specify both 'mask' and 'mask[]'")
    if image is None and image_array is not None:
        image = image_array
    if mask is None and mask_array is not None:
        mask = mask_array

    # if image is None:
    #     raise HTTPException(status_code=422, detail="Field required: image")
    # Note: Image is optional for some models (e.g., Bedrock Stability style-transfer)
    # The validation will be done at the model level if image is truly required

    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        read_request_body,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    #########################################################
    # Read request body and convert UploadFiles to BytesIO
    #########################################################
    form_fields: Final = coerce_numeric_form_fields(
        parsed_body=await read_request_body(request=request),
        numeric_fields=IMAGE_EDIT_NUMERIC_FORM_FIELDS,
    )
    data: Final = {
        key: value
        for key, value in chain(IMAGE_EDIT_OPTIONAL_FIELD_DEFAULTS.items(), form_fields.items())
        if key not in BRACKETED_FILE_FIELDS
    }
    if "stream" in data:
        data["stream"] = image_stream_requested(data["stream"])
    image_files: Final = await batch_to_bytesio(image)
    mask_files: Final = await batch_to_bytesio(mask)
    if image_files:
        data["image"] = image_files
    if mask_files:
        data["mask"] = mask_files

    for _field in ("image", "mask"):
        if _field in data and isinstance(data[_field], str):
            raise HTTPException(
                status_code=422,
                detail=f"'{_field}' must be provided as a multipart file upload, not a string.",
            )

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
            select_data_generator=partial(image_event_sse_stream, event_prefix="image_edit"),
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
