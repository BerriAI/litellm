from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, StrictBool, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import RESPONSE_COST_HEADER, normalize_drop_params
from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.image_handling import (
    async_convert_urls_to_base64,
    convert_url_to_base64,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    image_media_type,
    image_param,
    input_images,
    ir_to_systemone_response,
    remote_image_urls,
    systemone_request_to_ir,
    with_images_where,
    with_inlined_images,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.decisions.transformation import ir_to_openai_response, openai_request_to_ir
from litellm.types.decisions import (
    DecisionQuestion,
    DecisionsAsset,
    DecisionsAssetReference,
    DecisionsAssets,
    DecisionsAudio,
    DecisionsBase64Image,
    DecisionsImage,
    DecisionsIRAnswer,
    DecisionsIRBoxQuestion,
    DecisionsIRMessages,
    DecisionsIRQuestion,
    DecisionsIRRefusal,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRState,
    DecisionsJSON,
    DecisionsRequestBody,
    DecisionsResponse,
    DecisionsVideo,
    OpenAIDecisionInput,
    OpenAIDecisionQuestion,
    OpenAIDecisionRequestBody,
    OpenAIDecisionResponse,
    UnsupportedDecisionsRequest,
)
from litellm.types.utils import LlmProviders, ProviderSpecificHeader
from litellm.utils import ProviderConfigManager

SystemOneQuestions: TypeAlias = Mapping[str, DecisionQuestion | Mapping[str, object]]
OpenAIDecisionQuestions: TypeAlias = Sequence[OpenAIDecisionQuestion | Mapping[str, object]]
DecisionsImages: TypeAlias = Sequence[str | DecisionsBase64Image | Mapping[str, object]]
DecisionsAssetEntries: TypeAlias = Mapping[str, DecisionsAsset | Mapping[str, object]]
DecisionsAudioClips: TypeAlias = Sequence[DecisionsAudio | Mapping[str, object]]
DecisionsVideoClips: TypeAlias = Sequence[DecisionsVideo | Mapping[str, object]]
DecisionsRequestFormat: TypeAlias = DecisionsRequestBody | OpenAIDecisionRequestBody

_SYSTEMONE_REQUEST_ADAPTER: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_OPENAI_REQUEST_ADAPTER: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(OpenAIDecisionRequestBody)
_SAFETY_IDENTIFIER_ADAPTER: Final[TypeAdapter[str | None]] = TypeAdapter(
    str | None, config=ConfigDict(title="safety_identifier")
)
_IMAGES_ADAPTER: Final[TypeAdapter[tuple[DecisionsImage, ...] | None]] = TypeAdapter(
    tuple[DecisionsImage, ...] | None, config=ConfigDict(title="images")
)
_PROVIDER_SPECIFIC_HEADER_ADAPTER: Final[
    TypeAdapter[ProviderSpecificHeader | tuple[ProviderSpecificHeader, ...] | None]
] = TypeAdapter(ProviderSpecificHeader | tuple[ProviderSpecificHeader, ...] | None)
_HANDLER: Final = BaseLLMHTTPHandler()
_BASE64_MARKER: Final = ";base64,"


_IMAGE_ADAPTER: Final[TypeAdapter[DecisionsImage]] = TypeAdapter(DecisionsImage, config=ConfigDict(title="images"))
_IMAGE_ENTRIES_ADAPTER: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_ASSETS_ADAPTER: Final[TypeAdapter[DecisionsAssets | None]] = TypeAdapter(
    DecisionsAssets | None, config=ConfigDict(title="assets")
)
_ASSET_ENTRY_ADAPTER: Final[TypeAdapter[tuple[DecisionsAssetReference, DecisionsAsset]]] = TypeAdapter(
    tuple[DecisionsAssetReference, DecisionsAsset]
)
_ASSET_ENTRIES_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def _valid_image(image: object) -> DecisionsImage | None:
    try:
        return _IMAGE_ADAPTER.validate_python(image)
    except ValidationError:
        return None


def _image_entries(images: object) -> tuple[object, ...]:
    try:
        return _IMAGE_ENTRIES_ADAPTER.validate_python(images)
    except ValidationError:
        return ()


def _is_well_formed_image(image: DecisionsImage) -> bool:
    return _valid_image(image) is not None


def _no_image(_image: DecisionsImage) -> bool:
    return False


def _valid_asset(entry: tuple[str, object]) -> tuple[str, DecisionsAsset] | None:
    try:
        return _ASSET_ENTRY_ADAPTER.validate_python(entry)
    except ValidationError:
        return None


def _asset_entries(assets: object) -> Mapping[str, object]:
    try:
        return _ASSET_ENTRIES_ADAPTER.validate_python(assets)
    except ValidationError:
        return MappingProxyType({})


def _request_assets(assets: object, drops_params: bool) -> Mapping[str, DecisionsAsset] | None:
    if not drops_params:
        return _ASSETS_ADAPTER.validate_python(assets)
    valid: Final = MappingProxyType(
        dict(entry for entry in map(_valid_asset, _asset_entries(assets).items()) if entry is not None)
    )
    return valid or None


def _request_images(images: object, drops_params: bool) -> tuple[DecisionsImage, ...]:
    if not drops_params:
        return _IMAGES_ADAPTER.validate_python(images) or ()
    return tuple(image for image in map(_valid_image, _image_entries(images)) if image is not None)


@dataclass(frozen=True, slots=True)
class SystemOneRequestFields:
    state: DecisionsJSON | None
    questions: SystemOneQuestions | None
    images: object = None
    assets: object = None
    audio: DecisionsAudioClips | None = None
    videos: DecisionsVideoClips | None = None

    def validate(self, *, safety_identifier: str | None, drops_params: bool) -> DecisionsRequestBody:
        return _SYSTEMONE_REQUEST_ADAPTER.validate_python(
            {
                "state": self.state,
                "questions": self.questions,
                "images": _request_images(self.images, drops_params) or None,
                "assets": _request_assets(self.assets, drops_params),
                "audio": self.audio,
                "videos": self.videos,
            }
        )


@dataclass(frozen=True, slots=True)
class OpenAIDecisionRequestFields:
    input: OpenAIDecisionInput | None
    questions: OpenAIDecisionQuestions | None
    audio: DecisionsAudioClips | None = None
    videos: DecisionsVideoClips | None = None

    def validate(self, *, safety_identifier: str | None, drops_params: bool) -> OpenAIDecisionRequestBody:
        return _OPENAI_REQUEST_ADAPTER.validate_python(
            {
                "input": self.input,
                "questions": self.questions,
                "safety_identifier": safety_identifier,
                "audio": self.audio,
                "videos": self.videos,
            }
        )


RequestFields: TypeAlias = SystemOneRequestFields | OpenAIDecisionRequestFields


@dataclass(frozen=True, slots=True, repr=False)
class DecisionsCall:
    model: str
    requested_model: str
    custom_llm_provider: str
    provider_config: BaseDecisionsConfig
    request: DecisionsRequestFormat
    ir_request: DecisionsIRRequest
    api_base: str
    api_key: str | None = field(repr=False)
    logging_obj: LiteLLMLoggingObj | None
    headers: Mapping[str, str]
    timeout: float | httpx.Timeout | None
    litellm_params: Mapping[str, object] = field(repr=False)
    caller_questions: tuple[DecisionsIRQuestion, ...]


def supported_decisions_providers() -> tuple[str, ...]:
    return tuple(
        provider.value
        for provider in LlmProviders
        if ProviderConfigManager.get_provider_decisions_config(model="", provider=provider) is not None
    )


def _provider_config(model: str, custom_llm_provider: str) -> BaseDecisionsConfig:
    provider: Final = next((member for member in LlmProviders if member.value == custom_llm_provider), None)
    provider_config: Final = (
        None if provider is None else ProviderConfigManager.get_provider_decisions_config(model, provider)
    )
    if provider_config is None:
        supported: Final = ", ".join(supported_decisions_providers())
        raise litellm.BadRequestError(
            message=f"Unknown Decisions provider '{custom_llm_provider}'. Supported providers: {supported}",
            model=model,
            llm_provider=custom_llm_provider,
        )
    return provider_config


def _ir_request(request: DecisionsRequestFormat, safety_identifier: str | None) -> DecisionsIRRequest:
    match request:
        case DecisionsRequestBody():
            return replace(systemone_request_to_ir(request), safety_identifier=safety_identifier)
        case OpenAIDecisionRequestBody():
            return openai_request_to_ir(request)
        case _:
            assert_never(request)


def _drops_params(kwargs: Mapping[str, object]) -> bool:
    return litellm.drop_params is True or normalize_drop_params(kwargs.get("drop_params")) is True


def _request_safety_identifier(
    safety_identifier: object, provider_config: BaseDecisionsConfig, kwargs: Mapping[str, object]
) -> str | None:
    if not _drops_params(kwargs):
        return _SAFETY_IDENTIFIER_ADAPTER.validate_python(safety_identifier)
    if isinstance(safety_identifier, str) and provider_config.supports_safety_identifier:
        return safety_identifier
    return None


def _fields_without_unsupported_media(
    fields: RequestFields,
    provider_config: BaseDecisionsConfig,
    model: str,
    custom_llm_provider: str,
    kwargs: Mapping[str, object],
) -> RequestFields:
    if not _drops_params(kwargs):
        return fields
    return replace(
        fields,
        audio=fields.audio if provider_config.supports_audio_input(model, custom_llm_provider) else None,
        videos=fields.videos if provider_config.supports_video_input(model, custom_llm_provider) else None,
    )


@dataclass(frozen=True, slots=True)
class _UnsupportedParams:
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _NothingLeftAfterDrop:
    names: tuple[str, ...]
    missing: Literal["input to decide on", "questions to answer"]


@dataclass(frozen=True, slots=True)
class _MalformedImages:
    count: int


def _unsupported_params(
    ir_request: DecisionsIRRequest,
    provider_config: BaseDecisionsConfig,
    model: str,
    custom_llm_provider: str,
    headers: Mapping[str, str],
    supports_box_questions: bool,
) -> tuple[str, ...]:
    refuses_safety_identifier: Final = (
        ir_request.safety_identifier is not None and not provider_config.supports_safety_identifier
    )
    refused_image_param: Final = (
        None if provider_config.supports_images_in(ir_request.input, headers) else image_param(ir_request.input)
    )
    refuses_assets: Final = bool(ir_request.assets) and not provider_config.supports_assets(headers)
    refuses_box_questions: Final = _has_box_question(ir_request.questions) and not supports_box_questions
    refuses_audio: Final = bool(ir_request.audio) and not provider_config.supports_audio_input(
        model, custom_llm_provider
    )
    refuses_videos: Final = bool(ir_request.videos) and not provider_config.supports_video_input(
        model, custom_llm_provider
    )
    return (
        *(("safety_identifier",) if refuses_safety_identifier else ()),
        *(() if refused_image_param is None else (refused_image_param,)),
        *(("assets",) if refuses_assets else ()),
        *(("box questions",) if refuses_box_questions else ()),
        *(("audio",) if refuses_audio else ()),
        *(("videos",) if refuses_videos else ()),
    )


def _has_box_question(questions: Sequence[DecisionsIRQuestion]) -> bool:
    return any(isinstance(question, DecisionsIRBoxQuestion) for question in questions)


def _is_not_box_question(question: DecisionsIRQuestion) -> bool:
    return not isinstance(question, DecisionsIRBoxQuestion)


def _without_unsupported_params(
    ir_request: DecisionsIRRequest,
    provider_config: BaseDecisionsConfig,
    headers: Mapping[str, str],
    supports_box_questions: bool,
) -> DecisionsIRRequest:
    return replace(
        ir_request,
        input=(
            ir_request.input
            if provider_config.supports_images_in(ir_request.input, headers)
            else with_images_where(ir_request.input, _no_image)
        ),
        safety_identifier=ir_request.safety_identifier if provider_config.supports_safety_identifier else None,
        assets=ir_request.assets if provider_config.supports_assets(headers) else MappingProxyType({}),
        questions=(
            ir_request.questions
            if supports_box_questions
            else tuple(filter(_is_not_box_question, ir_request.questions))
        ),
    )


def _leaves_no_input(decision_input: DecisionsIRState | DecisionsIRMessages) -> bool:
    return isinstance(decision_input, DecisionsIRMessages) and not decision_input.messages


def _without_malformed_images(
    ir_request: DecisionsIRRequest, drops_params: bool
) -> DecisionsIRRequest | _MalformedImages:
    malformed: Final = sum(1 for image in input_images(ir_request.input) if not _is_well_formed_image(image))
    if not malformed:
        return ir_request
    well_formed: Final = replace(ir_request, input=with_images_where(ir_request.input, _is_well_formed_image))
    if not drops_params or _leaves_no_input(well_formed.input):
        return _MalformedImages(count=malformed)
    return well_formed


class _DeploymentBoxQuestions(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    supports_box_questions: StrictBool | None = None


def _deployment_supports_box_questions(kwargs: Mapping[str, object]) -> bool | None:
    try:
        return _DeploymentBoxQuestions.model_validate(kwargs.get("model_info") or {}).supports_box_questions
    except ValidationError:
        return None


def _provider_ir_request(
    ir_request: DecisionsIRRequest,
    *,
    provider_config: BaseDecisionsConfig,
    model: str,
    custom_llm_provider: str,
    headers: Mapping[str, str],
    kwargs: Mapping[str, object],
) -> DecisionsIRRequest | _UnsupportedParams | _NothingLeftAfterDrop:
    supports_box_questions: Final = provider_config.supports_box_questions(
        model, custom_llm_provider, headers, _deployment_supports_box_questions(kwargs)
    )
    unsupported: Final = _unsupported_params(
        ir_request, provider_config, model, custom_llm_provider, headers, supports_box_questions
    )
    if not unsupported:
        return ir_request
    if not _drops_params(kwargs):
        return _UnsupportedParams(names=unsupported)
    supported: Final = _without_unsupported_params(ir_request, provider_config, headers, supports_box_questions)
    if _leaves_no_input(supported.input):
        return _NothingLeftAfterDrop(names=unsupported, missing="input to decide on")
    if not supported.questions:
        return _NothingLeftAfterDrop(names=unsupported, missing="questions to answer")
    return supported


def _outbound_headers(
    extra_headers: Mapping[str, str] | None, custom_llm_provider: str, kwargs: Mapping[str, object]
) -> Mapping[str, str]:
    forwarded: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
        provider_specific_header=_PROVIDER_SPECIFIC_HEADER_ADAPTER.validate_python(
            kwargs.get("provider_specific_header")
        ),
        custom_llm_provider=custom_llm_provider,
    )
    return MappingProxyType({**(extra_headers or {}), **forwarded})


def prepare_call(
    *,
    model: str,
    fields: RequestFields,
    safety_identifier: str | None,
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, str] | None,
    kwargs: Mapping[str, object],
) -> DecisionsCall:
    upstream_model, provider, dynamic_api_key, dynamic_api_base = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )
    provider_config: Final = _provider_config(upstream_model, provider)
    if not upstream_model:
        raise litellm.BadRequestError(
            message="A model name is required for the Decisions API",
            model=model,
            llm_provider=provider,
        )
    try:
        canonical_model: Final = provider_config.canonical_model(upstream_model)
    except ValueError as error:
        raise litellm.BadRequestError(message=str(error), model=model, llm_provider=provider) from error
    try:
        headers: Final = _outbound_headers(extra_headers, provider, kwargs)
        request_safety_identifier: Final = _request_safety_identifier(safety_identifier, provider_config, kwargs)
        request: Final = _fields_without_unsupported_media(
            fields, provider_config, canonical_model, provider, kwargs
        ).validate(safety_identifier=request_safety_identifier, drops_params=_drops_params(kwargs))
    except ValidationError as error:
        raise litellm.BadRequestError(
            message=f"Invalid Decisions request: {error}",
            model=model,
            llm_provider=provider,
        ) from error

    try:
        resolved_api_base, static_api_key = provider_config.resolve_connection(
            dynamic_api_base or api_base, dynamic_api_key or api_key
        )
    except ValueError as error:
        raise litellm.BadRequestError(message=str(error), model=model, llm_provider=provider) from error
    if resolved_api_base is None:
        raise litellm.BadRequestError(
            message=provider_config.missing_api_base_message(provider),
            model=model,
            llm_provider=provider,
        )
    resolved_api_key: Final = provider_config.resolve_credential(
        static_api_key, resolved_api_base, {**kwargs, "api_key": api_key}
    )
    if resolved_api_key is None and provider_config.api_key_required:
        raise litellm.AuthenticationError(
            message=f"Missing API key for Decisions provider '{provider}'",
            model=model,
            llm_provider=provider,
        )

    caller_ir_request: Final = _without_malformed_images(
        _ir_request(request, request_safety_identifier), _drops_params(kwargs)
    )
    if isinstance(caller_ir_request, _MalformedImages):
        raise litellm.BadRequestError(
            message=(
                f"Invalid Decisions request: {caller_ir_request.count} input_image part(s) have an image_url that is "
                "neither a data:image/...;base64, URL nor an http(s) URL"
            ),
            model=model,
            llm_provider=provider,
        )
    ir_request: Final = _provider_ir_request(
        caller_ir_request,
        provider_config=provider_config,
        model=canonical_model,
        custom_llm_provider=provider,
        headers=headers,
        kwargs=kwargs,
    )
    if isinstance(ir_request, _UnsupportedParams):
        raise litellm.UnsupportedParamsError(
            message=(
                f"{provider} does not support parameters: {list(ir_request.names)}, for model={model}. "
                "To drop these, set `litellm.drop_params=True` or for proxy:\n\n"
                "`litellm_settings:\n drop_params: true`\n"
            ),
            model=model,
            llm_provider=provider,
        )
    if isinstance(ir_request, _NothingLeftAfterDrop):
        raise litellm.BadRequestError(
            message=(
                f"{provider} does not support parameters: {list(ir_request.names)}, for model={model}, "
                f"and dropping them leaves no {ir_request.missing}"
            ),
            model=model,
            llm_provider=provider,
        )
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if isinstance(logging_obj, LiteLLMLoggingObj):
        logging_obj.update_from_kwargs(
            kwargs=dict(kwargs),
            model=canonical_model,
            litellm_params={
                "litellm_call_id": kwargs.get("litellm_call_id"),
                "api_base": provider_config.get_complete_url(resolved_api_base, canonical_model),
            },
            custom_llm_provider=provider,
        )
    return DecisionsCall(
        model=canonical_model,
        requested_model=model,
        custom_llm_provider=provider,
        provider_config=provider_config,
        request=request,
        ir_request=ir_request,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        logging_obj=logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None,
        headers=headers,
        timeout=timeout,
        litellm_params=kwargs,
        caller_questions=caller_ir_request.questions,
    )


def _remote_image_urls(call: DecisionsCall) -> tuple[str, ...]:
    if not call.provider_config.inlines_remote_images:
        return ()
    return remote_image_urls(call.ir_request.input)


def _with_data_urls(call: DecisionsCall, data_urls: Mapping[str, str]) -> DecisionsCall:
    return replace(
        call, ir_request=replace(call.ir_request, input=with_inlined_images(call.ir_request.input, data_urls))
    )


def _with_remote_images_inlined(call: DecisionsCall) -> DecisionsCall:
    remote_urls: Final = _remote_image_urls(call)
    if not remote_urls:
        return call
    return _with_data_urls(
        call, MappingProxyType({url: convert_url_to_base64(url, timeout=call.timeout) for url in remote_urls})
    )


async def _async_with_remote_images_inlined(call: DecisionsCall) -> DecisionsCall:
    remote_urls: Final = _remote_image_urls(call)
    if not remote_urls:
        return call
    data_urls: Final = await async_convert_urls_to_base64(remote_urls, timeout=call.timeout)
    return _with_data_urls(call, MappingProxyType(dict(zip(remote_urls, data_urls, strict=True))))


def _has_media_type_in(media_types: frozenset[str], image: DecisionsImage) -> bool:
    media_type: Final = image_media_type(image)
    return media_type is None or media_type in media_types


def _checked_image_types(call: DecisionsCall) -> DecisionsCall:
    media_types: Final = call.provider_config.image_media_types
    if media_types is None:
        return call
    found: Final = (image_media_type(image) for image in input_images(call.ir_request.input))
    unsupported: Final = tuple(
        dict.fromkeys(type_ for type_ in found if type_ is not None and type_ not in media_types)
    )
    if not unsupported:
        return call
    supported: Final = with_images_where(call.ir_request.input, partial(_has_media_type_in, media_types))
    if _drops_params(call.litellm_params) and not _leaves_no_input(supported):
        return replace(call, ir_request=replace(call.ir_request, input=supported))
    raise litellm.BadRequestError(
        message=(
            f"{call.custom_llm_provider} takes only {', '.join(sorted(media_types))} images, "
            f"got {', '.join(unsupported)}"
        ),
        model=call.requested_model,
        llm_provider=call.custom_llm_provider,
    )


def _checked_image_count(call: DecisionsCall) -> DecisionsCall:
    max_images: Final = call.provider_config.max_images
    image_count: Final = len(input_images(call.ir_request.input))
    if max_images is None or image_count <= max_images:
        return call
    raise litellm.BadRequestError(
        message=f"{call.custom_llm_provider} takes at most {max_images} images per request, got {image_count}",
        model=call.requested_model,
        llm_provider=call.custom_llm_provider,
    )


def _decoded_image_bytes(image: DecisionsImage) -> int:
    if isinstance(image, DecisionsBase64Image):
        return len(image.base64.rstrip("=")) * 3 // 4
    marker_at: Final = image.lower().find(_BASE64_MARKER)
    return 0 if marker_at < 0 else len(image[marker_at + len(_BASE64_MARKER) :].rstrip("=")) * 3 // 4


def _byte_limit(limit: int) -> str:
    return f"{limit:,} bytes ({limit / (1024 * 1024):g} MiB)"


def _image_size_error(call: DecisionsCall) -> str | None:
    sizes: Final = tuple(_decoded_image_bytes(image) for image in input_images(call.ir_request.input))
    largest: Final = max(sizes, default=0)
    per_image: Final = call.provider_config.max_image_bytes
    total: Final = call.provider_config.max_total_image_bytes
    if per_image is not None and largest > per_image:
        return f"takes images of at most {_byte_limit(per_image)} each, got one of {largest:,} bytes"
    if total is not None and sum(sizes) > total:
        return f"takes at most {_byte_limit(total)} of images per request, got {sum(sizes):,} bytes"
    return None


def _checked_image_sizes(call: DecisionsCall) -> DecisionsCall:
    error: Final = _image_size_error(call)
    if error is None:
        return call
    raise litellm.BadRequestError(
        message=f"{call.custom_llm_provider} {error}",
        model=call.requested_model,
        llm_provider=call.custom_llm_provider,
    )


def _request_body(call: DecisionsCall) -> Mapping[str, object]:
    body: Final = call.provider_config.transform_decisions_request(
        model=call.model, request=call.ir_request, custom_llm_provider=call.custom_llm_provider
    )
    if isinstance(body, UnsupportedDecisionsRequest):
        raise litellm.BadRequestError(
            message=f"Decisions provider '{call.custom_llm_provider}' cannot serve this request: {body.reason}",
            model=call.requested_model,
            llm_provider=call.custom_llm_provider,
        )
    return body


def _format_response(response: DecisionsIRResponse, call: DecisionsCall) -> DecisionsResponse | OpenAIDecisionResponse:
    formatted: Final = _formatted_response(response, call)
    provider_cost: Final = call.provider_config.provider_reported_cost(response)
    formatted.set_hidden_params(
        {
            "model": f"{call.custom_llm_provider}/{call.model}",
            "custom_llm_provider": call.custom_llm_provider,
            "provider_response_model": f"{call.custom_llm_provider}/{call.model}",
            **({"additional_headers": {RESPONSE_COST_HEADER: provider_cost}} if provider_cost is not None else {}),
        }
    )
    return formatted


def _caller_answers(call: DecisionsCall, answers: Sequence[DecisionsIRAnswer]) -> tuple[DecisionsIRAnswer, ...]:
    sent_answers: Final = iter(answers)
    return tuple(
        next(sent_answers) if question in call.ir_request.questions else DecisionsIRRefusal()
        for question in call.caller_questions
    )


def _formatted_response(
    response: DecisionsIRResponse, call: DecisionsCall
) -> DecisionsResponse | OpenAIDecisionResponse:
    caller_request: Final = replace(call.ir_request, questions=call.caller_questions)
    caller_response: Final = replace(response, answers=_caller_answers(call, response.answers))
    match call.request:
        case DecisionsRequestBody():
            return ir_to_systemone_response(caller_response, caller_request)
        case OpenAIDecisionRequestBody():
            return ir_to_openai_response(caller_response, caller_request, call.requested_model)
        case _:
            assert_never(call.request)


def _map_upstream_exception(error: Exception, call: DecisionsCall) -> Exception:
    if isinstance(error, BaseLLMException) and error.status_code_is_synthesized:
        provider_label: Final = f"{call.custom_llm_provider[0].upper()}{call.custom_llm_provider[1:]}Exception"
        return litellm.APIConnectionError(
            message=f"{provider_label} - {error.message}",
            llm_provider=call.custom_llm_provider,
            model=f"{call.custom_llm_provider}/{call.model}",
        )
    return litellm.exception_type(
        model=f"{call.custom_llm_provider}/{call.model}",
        custom_llm_provider=call.custom_llm_provider,
        original_exception=error,
    )


async def asend(call: DecisionsCall) -> DecisionsResponse | OpenAIDecisionResponse:
    supported: Final = _checked_image_count(_checked_image_types(call))
    inlined: Final = _checked_image_sizes(_checked_image_types(await _async_with_remote_images_inlined(supported)))
    body: Final = _request_body(inlined)
    try:
        response: Final = await _HANDLER.adecisions(
            model=inlined.model,
            custom_llm_provider=inlined.custom_llm_provider,
            logging_obj=inlined.logging_obj,
            provider_config=inlined.provider_config,
            request=inlined.ir_request,
            body=body,
            logged_body=body if inlined is supported else _request_body(supported),
            api_base=inlined.api_base,
            api_key=inlined.api_key,
            headers=inlined.headers,
            timeout=inlined.timeout,
            litellm_params=inlined.litellm_params,
        )
    except Exception as error:
        raise _map_upstream_exception(error, inlined) from error
    return _format_response(response, inlined)


def send(call: DecisionsCall) -> DecisionsResponse | OpenAIDecisionResponse:
    supported: Final = _checked_image_count(_checked_image_types(call))
    inlined: Final = _checked_image_sizes(_checked_image_types(_with_remote_images_inlined(supported)))
    body: Final = _request_body(inlined)
    try:
        response: Final = _HANDLER.decisions(
            model=inlined.model,
            custom_llm_provider=inlined.custom_llm_provider,
            logging_obj=inlined.logging_obj,
            provider_config=inlined.provider_config,
            request=inlined.ir_request,
            body=body,
            logged_body=body if inlined is supported else _request_body(supported),
            api_base=inlined.api_base,
            api_key=inlined.api_key,
            headers=inlined.headers,
            timeout=inlined.timeout,
            litellm_params=inlined.litellm_params,
        )
    except Exception as error:
        raise _map_upstream_exception(error, inlined) from error
    return _format_response(response, inlined)


def reject_other_format(*, model: str, kwargs: Mapping[str, object], field_name: str, other_function: str) -> None:
    if field_name not in kwargs:
        return
    raise litellm.BadRequestError(
        message=f"Requests with '{field_name}' go through litellm.{other_function}",
        model=model,
        llm_provider="",
    )


def reject_images_next_to_input(*, model: str, kwargs: Mapping[str, object]) -> None:
    if not kwargs.get("images") or _drops_params(kwargs):
        return
    raise litellm.BadRequestError(
        message=(
            "images belongs to the System One format; with input (OpenAI format), "
            "send images as input_image content parts"
        ),
        model=model,
        llm_provider="",
    )
