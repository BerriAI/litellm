import base64
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from fastapi import APIRouter, Depends, Request, Response, Security
from fastapi.responses import ORJSONResponse  # pyright: ignore[reportDeprecated]  # required endpoint contract
from pydantic import TypeAdapter, ValidationError
from starlette.datastructures import FormData, UploadFile

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.llms.base_llm.decisions.transformation import without_raw_prompt_cache_breakpoints
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.user_api_key_auth import (
    UserAPIKeyAuth,
    api_key_header,
    custom_litellm_key_header,
    run_pre_body_auth_checks,
    user_api_key_auth,
)
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.custom_openapi_spec import inline_request_body
from litellm.proxy.common_utils.http_parsing_utils import (
    is_form_content_type,
    read_request_body,
    safe_set_request_parsed_body,
)
from litellm.types.decisions import (
    DecisionsRequest,
    DecisionsRequestBody,
    DecisionsResponse,
    OpenAIDecisionRequest,
    OpenAIDecisionRequestBody,
    OpenAIDecisionResponse,
)
from litellm.types.extraction import ExtractionRequest, ExtractionRequestBody, ExtractionResponse

router: Final = APIRouter()
_REQUEST_DATA_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_FORM_ASSETS_ADAPTER: Final[TypeAdapter[Mapping[str, Mapping[str, object]]]] = TypeAdapter(
    Mapping[str, Mapping[str, object]]
)
_DECISIONS_REQUEST_BODY_ADAPTER: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_OPENAI_DECISION_REQUEST_BODY_ADAPTER: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(
    OpenAIDecisionRequestBody
)
_EXTRACTION_REQUEST_BODY_ADAPTER: Final[TypeAdapter[ExtractionRequestBody]] = TypeAdapter(ExtractionRequestBody)
_GENERAL_SETTINGS_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_OPTIONAL_STRING_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(str | None)
_OPTIONAL_FLOAT_ADAPTER: Final[TypeAdapter[float | None]] = TypeAdapter(float | None)
_SYSTEMONE_REQUEST_BODY: Final = inline_request_body(
    DecisionsRequest,
    {
        "model": "jev",
        "state": "Customer wrote: I was charged twice for order #4411 and want one charge refunded today.",
        "questions": {
            "is_refund_request": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "urgency": {
                "type": "choice",
                "instructions": "How urgent is this?",
                "criteria": {"low": "can wait a week", "high": "needs action today"},
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["calm", "mildly annoyed", "angry"],
            },
        },
    },
)
_DECISIONS_REQUEST_BODY: Final = inline_request_body(
    OpenAIDecisionRequest,
    {
        "model": "luna",
        "input": "Customer wrote: I was charged twice for order #4411 and want one charge refunded today.",
        "questions": [
            {"type": "predicate", "name": "is_refund_request", "instructions": "Is the customer asking for a refund?"},
            {
                "type": "choice",
                "name": "urgency",
                "instructions": "How urgent is this?",
                "choices": [
                    {"value": "low", "description": "can wait a week"},
                    {"value": "high", "description": "needs action today"},
                ],
            },
            {
                "type": "score",
                "name": "frustration",
                "instructions": "How frustrated is the customer?",
                "levels": [{"label": "calm"}, {"label": "mildly annoyed"}, {"label": "angry"}],
            },
        ],
    },
)
_EXTRACTION_REQUEST_BODY: Final = inline_request_body(
    ExtractionRequest,
    {
        "model": "clairic",
        "state": "Invoice #4471 from Acme Tools, dated 2026-03-03. 12 hammers. Total $174. Payment due in 30 days.",
        "fields": {
            "vendor": {"type": "text", "instructions": "Who sent the invoice?"},
            "invoice_date": {"type": "date"},
            "total": {"type": "number", "unit": "USD"},
            "items": {"type": "list", "items": "text", "instructions": "What was bought?"},
            "days_to_pay": {"type": "estimate", "min": 0, "max": 90},
        },
    },
)
_ApiName: TypeAlias = Literal["Decisions", "Extraction"]
_RouteType: TypeAlias = Literal["adecisions", "asystemone", "aextraction"]


async def _invalid_request(
    raw_data: Mapping[str, object],
    error: ValidationError | str,
    user_api_key_dict: UserAPIKeyAuth,
    api_name: _ApiName,
) -> Exception:
    from litellm.proxy.proxy_server import proxy_logging_obj, version

    return await ProxyBaseLLMRequestProcessing(data=dict(raw_data)).handle_llm_api_exception(
        e=BadRequestError(
            message=f"Invalid {api_name} request: {error}",
            model=str(raw_data.get("model", "")),
            llm_provider="",
        ),
        user_api_key_dict=user_api_key_dict,
        proxy_logging_obj=proxy_logging_obj,
        version=version,
    )


_FIELDS_CHECKED_AFTER_ROUTING: Final = frozenset(
    {"safety_identifier", "prompt_cache_key", "prompt_cache_options", "images", "assets", "audio", "videos"}
)


def _fields_checked_before_routing(data: Mapping[str, object], route_type: _RouteType) -> Mapping[str, object]:
    if route_type == "aextraction":
        return data
    return MappingProxyType(
        {
            key: without_raw_prompt_cache_breakpoints(value) if key == "input" else value
            for key, value in data.items()
            if key not in _FIELDS_CHECKED_AFTER_ROUTING
        }
    )


_DEPLOYMENT_ONLY_FIELDS: Final = frozenset({"model_info"})
_FORM_REQUEST_PART: Final = "request"


@dataclass(frozen=True, slots=True)
class _FormBody:
    data: Mapping[str, object]
    files: Mapping[str, UploadFile]


@dataclass(frozen=True, slots=True)
class _InvalidFormBody:
    message: str


_SystemOneFormBody: TypeAlias = _FormBody | _InvalidFormBody | None


async def _part_bytes(part: UploadFile | str) -> bytes:
    if isinstance(part, str):
        return part.encode()
    return await part.read()


def _claimed_part(asset: Mapping[str, object]) -> str | None:
    part: Final = asset.get("part")
    return part if isinstance(part, str) else None


def _form_assets(assets: object) -> Mapping[str, Mapping[str, object]]:
    try:
        return _FORM_ASSETS_ADAPTER.validate_python(assets)
    except ValidationError:
        return {}


async def _inline_asset(asset: Mapping[str, object], files: Mapping[str, UploadFile]) -> Mapping[str, object]:
    part: Final = _claimed_part(asset)
    if part is None:
        return asset
    image: Final = await files[part].read()
    detected_format: Final = get_image_type(image)
    return {
        **{key: value for key, value in asset.items() if key != "part"},
        **({} if detected_format is None else {"format": detected_format}),
        "data_base64": base64.b64encode(image).decode(),
    }


async def _inlined_form_data(form_body: _FormBody) -> Mapping[str, object]:
    if len(form_body.files) == 0:
        return form_body.data
    assets: Final = _form_assets(form_body.data.get("assets"))
    return {
        **form_body.data,
        "assets": {name: await _inline_asset(asset, form_body.files) for name, asset in assets.items()},
    }


def _paired_form_body(
    body: Mapping[str, object], parts: tuple[tuple[str, UploadFile | str], ...]
) -> _FormBody | _InvalidFormBody:
    assets: Final = _form_assets(body.get("assets"))
    claimed: Final = tuple(part for asset in assets.values() if (part := _claimed_part(asset)) is not None)
    if Counter(claimed) != Counter(name for name, _ in parts):
        return _InvalidFormBody(
            f"every part after `{_FORM_REQUEST_PART}` must be named by exactly one asset's `part`, but the parts are "
            + f"{sorted(name for name, _ in parts)} and the assets name {sorted(claimed)}"
        )
    text_parts: Final = sorted(name for name, value in parts if isinstance(value, str))
    if len(text_parts) > 0:
        return _InvalidFormBody(f"the asset parts {text_parts} must be sent as files, with a filename")
    return _FormBody(body, {name: value for name, value in parts if isinstance(value, UploadFile)})


async def _form_body(form: FormData) -> _FormBody | _InvalidFormBody:
    items: Final = form.multi_items()
    request_parts: Final = tuple(value for name, value in items if name == _FORM_REQUEST_PART)
    if len(request_parts) != 1:
        return _InvalidFormBody(f"a multipart body needs exactly one `{_FORM_REQUEST_PART}` part holding the JSON")
    try:
        body: Final = _REQUEST_DATA_ADAPTER.validate_json(await _part_bytes(request_parts[0]))
    except ValidationError as error:
        return _InvalidFormBody(f"the `{_FORM_REQUEST_PART}` part must be a JSON object: {error}")
    return _paired_form_body(body, tuple((name, value) for name, value in items if name != _FORM_REQUEST_PART))


async def _systemone_form_body(
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)],
    custom_key_header: Annotated[str | None, Security(custom_litellm_key_header)],
) -> _SystemOneFormBody:
    if not is_form_content_type(request.headers.get("content-type", "")):
        return None
    await run_pre_body_auth_checks(request=request, api_key=api_key, custom_litellm_key_header=custom_key_header)
    try:
        await read_request_body(request)
    except ProxyException as unreadable:
        return _InvalidFormBody(unreadable.message)
    form_body: Final = await _form_body(await request.form())
    if isinstance(form_body, _FormBody):
        safe_set_request_parsed_body(request, form_body.data)
    return form_body


async def _json_request_data(
    request: Request, user_api_key_dict: UserAPIKeyAuth, api_name: _ApiName
) -> Mapping[str, object]:
    if is_form_content_type(request.headers.get("content-type", "")):
        raise await _invalid_request(
            raw_data={}, error="the body must be JSON", user_api_key_dict=user_api_key_dict, api_name=api_name
        )
    try:
        return _REQUEST_DATA_ADAPTER.validate_json(await request.body())
    except ValidationError as error:
        raise await _invalid_request(raw_data={}, error=error, user_api_key_dict=user_api_key_dict, api_name=api_name)


async def _request_body(
    request: Request, form_body: _SystemOneFormBody, user_api_key_dict: UserAPIKeyAuth, api_name: _ApiName
) -> Mapping[str, object]:
    match form_body:
        case _FormBody():
            return await _inlined_form_data(form_body)
        case _InvalidFormBody(message=message):
            raise await _invalid_request(
                raw_data={}, error=message, user_api_key_dict=user_api_key_dict, api_name=api_name
            )
        case None:
            return await _json_request_data(request, user_api_key_dict, api_name)


async def _request_data(
    request: Request, form_body: _SystemOneFormBody, user_api_key_dict: UserAPIKeyAuth, api_name: _ApiName
) -> dict[str, object]:
    data: Final = await _request_body(request, form_body, user_api_key_dict, api_name)
    return {key: value for key, value in data.items() if key not in _DEPLOYMENT_ONLY_FIELDS}


async def _process_decisions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    body_adapter: (
        TypeAdapter[DecisionsRequestBody] | TypeAdapter[OpenAIDecisionRequestBody] | TypeAdapter[ExtractionRequestBody]
    ),
    route_type: _RouteType,
    form_body: _SystemOneFormBody,
) -> object:
    from litellm.proxy.proxy_server import (
        general_settings as proxy_general_settings,
    )
    from litellm.proxy.proxy_server import (
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_max_tokens,
        user_request_timeout,
        version,
    )
    from litellm.proxy.proxy_server import (
        user_api_base as proxy_user_api_base,
    )
    from litellm.proxy.proxy_server import (
        user_model as proxy_user_model,
    )
    from litellm.proxy.proxy_server import (
        user_temperature as proxy_user_temperature,
    )

    api_name: Final[_ApiName] = "Extraction" if route_type == "aextraction" else "Decisions"
    data: Final = await _request_data(request, form_body, user_api_key_dict, api_name)
    try:
        body_adapter.validate_python(_fields_checked_before_routing(data, route_type))
    except ValidationError as error:
        raise await _invalid_request(raw_data=data, error=error, user_api_key_dict=user_api_key_dict, api_name=api_name)
    general_settings: Final = _GENERAL_SETTINGS_ADAPTER.validate_python(proxy_general_settings)
    user_api_base: Final = _OPTIONAL_STRING_ADAPTER.validate_python(proxy_user_api_base)
    user_model: Final = _OPTIONAL_STRING_ADAPTER.validate_python(proxy_user_model)
    user_temperature: Final = _OPTIONAL_FLOAT_ADAPTER.validate_python(proxy_user_temperature)
    takes_generation_settings: Final = route_type != "aextraction"
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type=route_type,
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=None,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature if takes_generation_settings else None,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens if takes_generation_settings else None,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as error:
        raise await processor.handle_llm_api_exception(
            e=error,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/systemone",
    dependencies=[Depends(_systemone_form_body), Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
    summary="Ask named questions about a state (System One format)",
    responses={200: {"model": DecisionsResponse}},
    openapi_extra={"requestBody": _SYSTEMONE_REQUEST_BODY},
)
@router.post(
    "/systemone",
    dependencies=[Depends(_systemone_form_body), Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
    summary="Ask named questions about a state (System One format)",
    responses={200: {"model": DecisionsResponse}},
    openapi_extra={"requestBody": _SYSTEMONE_REQUEST_BODY},
)
async def systemone(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    form_body: Annotated[_SystemOneFormBody, Depends(_systemone_form_body)],
):
    """
    Judge a `state` against one or more named `questions` and get one calibrated answer per question name.

    Question types are `noul` (probability the answer is yes), `choice` (one label out of `criteria`, with
    `confidence` and `probabilities`), `score` (an index into the `criteria` levels, with `confidence`,
    `legend` and `probabilities`) and `box` (the image region the `instructions` describe, with `confidence`).
    TypeSafe models also take images in `assets`, each one referenced by its key in `state`. A `multipart/form-data`
    body works too, the way the TypeSafe SDK sends images: the JSON goes in a `request` part, and each asset names
    the file part holding its image in `part`. Requests that break
    these rules are rejected with a 400 before any provider is called. `model` is any decision model in the proxy model list. OpenAI decision models accept this format
    too, LiteLLM translates it and keeps the answers keyed by question name. Streaming is not supported.

    [Docs](https://docs.litellm.ai/docs/decisions)
    """
    return await _process_decisions(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        body_adapter=_DECISIONS_REQUEST_BODY_ADAPTER,
        route_type="asystemone",
        form_body=form_body,
    )


@router.post(
    "/v1/decisions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
    summary="Ask questions about an input (OpenAI Decisions format)",
    responses={200: {"model": OpenAIDecisionResponse}},
    openapi_extra={"requestBody": _DECISIONS_REQUEST_BODY},
)
@router.post(
    "/decisions",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
    summary="Ask questions about an input (OpenAI Decisions format)",
    responses={200: {"model": OpenAIDecisionResponse}},
    openapi_extra={"requestBody": _DECISIONS_REQUEST_BODY},
)
async def decisions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    """
    Judge an `input` against 1 to 128 `questions` and get the answers back in the same order.

    Question types are `predicate` (yes/no with `probability`), `choice` (one of the `choices`, with
    `probabilities`), `score` (an index into `levels`, with `probabilities`) and `box` (the image region the
    `instructions` describe, with `confidence`). Requests that break these rules are rejected with a 400 before
    any provider is called. `model` is any decision model in the proxy
    model list. System One decision models accept this format too, LiteLLM translates the request and the
    answers. Streaming is not supported.

    [Docs](https://docs.litellm.ai/docs/decisions)
    """
    return await _process_decisions(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        body_adapter=_OPENAI_DECISION_REQUEST_BODY_ADAPTER,
        route_type="adecisions",
        form_body=None,
    )


@router.post(
    "/v1/extraction",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["extraction"],
    summary="Extract typed fields from a state",
    responses={200: {"model": ExtractionResponse}},
    response_model=None,
    openapi_extra={"requestBody": _EXTRACTION_REQUEST_BODY},
)
@router.post(
    "/extraction",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["extraction"],
    summary="Extract typed fields from a state",
    responses={200: {"model": ExtractionResponse}},
    response_model=None,
    openapi_extra={"requestBody": _EXTRACTION_REQUEST_BODY},
)
async def extraction(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> object:
    """
    Read a `state` text and up to 4 images and get one typed answer per named field in `fields`.

    Field types are `text`, `number`, `date`, `time`, `datetime`, `duration`, `list` (set `items`), `estimate`
    (set `min` and `max`) and `forecast` (set `min`, `max` and `horizon`). Each answer carries its `value`, which is
    null when the input does not determine it, plus `confidence`, `p_null`, text `span`s into the state and, for
    estimates, `mean`, `median`, `p10` and `p90`. Images are data URLs or `{content_type, base64}` objects, remote
    URLs are not accepted. An unknown field type, a field name that is not 1 to 64 letters, digits, `_`, `.` or `-`
    starting with a letter or `_`, a remote image URL or more than 4 images is rejected with a 400 before any
    provider is called. Any other top-level key, `stream: true` and OpenAI generation settings such as `temperature`
    included, is rejected with a 400 naming it unless `drop_params` or the deployment's `additional_drop_params`
    drops it. Settings inside `fields` and `options` go to the provider as sent, so a field missing a setting its
    type needs or an empty `state` with no images comes back with the provider's own error. `model` is any
    extraction model in the proxy model list, such as Cloudflare Clairic. The request and the answers keep the
    provider's format.
    """
    return await _process_decisions(
        request=request,
        fastapi_response=fastapi_response,
        user_api_key_dict=user_api_key_dict,
        body_adapter=_EXTRACTION_REQUEST_BODY_ADAPTER,
        route_type="aextraction",
        form_body=None,
    )
