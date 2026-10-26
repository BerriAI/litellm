from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse  # pyright: ignore[reportDeprecated]  # required endpoint contract
from pydantic import TypeAdapter, ValidationError

from litellm.exceptions import BadRequestError
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.custom_openapi_spec import inline_request_body
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
    error: ValidationError,
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


_FIELDS_CHECKED_AFTER_ROUTING: Final = frozenset({"safety_identifier", "images", "assets", "audio", "videos"})


def _fields_checked_before_routing(data: Mapping[str, object], route_type: _RouteType) -> Mapping[str, object]:
    if route_type == "aextraction":
        return data
    return MappingProxyType({key: value for key, value in data.items() if key not in _FIELDS_CHECKED_AFTER_ROUTING})


_DEPLOYMENT_ONLY_FIELDS: Final = frozenset({"model_info"})


async def _request_data(request: Request, user_api_key_dict: UserAPIKeyAuth, api_name: _ApiName) -> dict[str, object]:
    body: Final = await request.body()
    try:
        data: Final = _REQUEST_DATA_ADAPTER.validate_json(body)
    except ValidationError as error:
        raise await _invalid_request(raw_data={}, error=error, user_api_key_dict=user_api_key_dict, api_name=api_name)
    return {key: value for key, value in data.items() if key not in _DEPLOYMENT_ONLY_FIELDS}


async def _process_decisions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    body_adapter: (
        TypeAdapter[DecisionsRequestBody] | TypeAdapter[OpenAIDecisionRequestBody] | TypeAdapter[ExtractionRequestBody]
    ),
    route_type: _RouteType,
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
    data: Final = await _request_data(request, user_api_key_dict, api_name)
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
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,  # pyright: ignore[reportDeprecated]  # required endpoint contract
    tags=["decisions"],
    summary="Ask named questions about a state (System One format)",
    responses={200: {"model": DecisionsResponse}},
    openapi_extra={"requestBody": _SYSTEMONE_REQUEST_BODY},
)
@router.post(
    "/systemone",
    dependencies=[Depends(user_api_key_auth)],
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
):
    """
    Judge a `state` against one or more named `questions` and get one calibrated answer per question name.

    Question types are `noul` (probability the answer is yes), `choice` (one label out of `criteria`, with
    `confidence` and `probabilities`), `score` (an index into the `criteria` levels, with `confidence`,
    `legend` and `probabilities`) and `box` (the image region the `instructions` describe, with `confidence`).
    TypeSafe models also take images in `assets`, each one referenced by its key in `state`. Requests that break
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
    )
