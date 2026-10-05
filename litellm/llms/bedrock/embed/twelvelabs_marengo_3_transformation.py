"""
Request builder for Bedrock TwelveLabs Marengo Embed 3.0, whose payload nests the input under a key named after
``inputType`` instead of the flat 2.7 layout.

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.bedrock import (
    TWELVELABS_MARENGO_3_EMBEDDING_OPTIONS,
    TWELVELABS_MARENGO_3_EMBEDDING_SCOPES,
    TWELVELABS_MARENGO_3_EMBEDDING_TYPES,
    TWELVELABS_MARENGO_3_INPUT_TYPES,
    TwelveLabsMarengo3AudioRequest,
    TwelveLabsMarengo3EmbeddingRequest,
    TwelveLabsMarengo3ImageRequest,
    TwelveLabsMarengo3MultiInputRequest,
    TwelveLabsMarengo3NamedMediaSource,
    TwelveLabsMarengo3RequestBase,
    TwelveLabsMarengo3Segmentation,
    TwelveLabsMarengo3TextImageRequest,
    TwelveLabsMarengo3TextRequest,
    TwelveLabsMarengo3TimedMediaInput,
    TwelveLabsMarengo3TimedMediaOptions,
    TwelveLabsMarengo3VideoRequest,
    TwelveLabsMediaSource,
    TwelveLabsS3Location,
)
from litellm.utils import get_base64_str

MARENGO_3_MODEL_MARKER: Final = "marengo-embed-3-"
S3_URI_PREFIX: Final = "s3://"
TIMED_MEDIA_OPTION_FIELDS: Final = MappingProxyType(
    {
        "startSec": True,
        "endSec": True,
        "segmentation": True,
        "embeddingOption": True,
        "embeddingType": True,
        "embeddingScope": True,
    }
)
TIMED_MEDIA_OPTIONS: Final = TypeAdapter(TwelveLabsMarengo3TimedMediaOptions)
TIMED_INPUT_TYPES: Final = frozenset({"video", "audio"})
MARENGO_2_7_ONLY_PARAMS: Final = ("textTruncate", "lengthSec", "useFixedLengthSec", "minClipSec")
MARENGO_2_7_ONLY_FIELDS: Final = MappingProxyType({name: True for name in MARENGO_2_7_ONLY_PARAMS})


def is_marengo_3_model(model: str | None) -> bool:
    return MARENGO_3_MODEL_MARKER in (model or "")


class Marengo3Params(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    inputType: TWELVELABS_MARENGO_3_INPUT_TYPES | None = None
    input_type: TWELVELABS_MARENGO_3_INPUT_TYPES | None = None
    media_source: str | None = None
    media_sources: Mapping[str, str] | None = None
    bucketOwner: str | None = None
    startSec: float | None = None
    endSec: float | None = None
    segmentation: TwelveLabsMarengo3Segmentation | None = None
    embeddingOption: tuple[TWELVELABS_MARENGO_3_EMBEDDING_OPTIONS, ...] | None = None
    embeddingType: tuple[TWELVELABS_MARENGO_3_EMBEDDING_TYPES, ...] | None = None
    embeddingScope: tuple[TWELVELABS_MARENGO_3_EMBEDDING_SCOPES, ...] | None = None
    inferenceId: str | None = None
    textTruncate: object = None
    lengthSec: object = None
    useFixedLengthSec: object = None
    minClipSec: object = None

    @property
    def resolved_input_type(self) -> TWELVELABS_MARENGO_3_INPUT_TYPES:
        return self.inputType or self.input_type or "text"

    def timed_media_options(self) -> TwelveLabsMarengo3TimedMediaOptions:
        return TIMED_MEDIA_OPTIONS.validate_python(self.given_timed_media_options())

    def given_timed_media_options(self) -> dict[str, object]:
        return self.model_dump(include=TIMED_MEDIA_OPTION_FIELDS, exclude_none=True)

    def given_2_7_only_params(self) -> dict[str, object]:
        return self.model_dump(include=MARENGO_2_7_ONLY_FIELDS, exclude_none=True)


def _require_bucket_owner(bucket_owner: str | None) -> str:
    if bucket_owner is None:
        raise BedrockError(
            status_code=400,
            message="s3:// media requires the 'bucketOwner' parameter, the account id that owns the bucket",
        )
    return bucket_owner


def _media_source(media: str, bucket_owner: str | None) -> TwelveLabsMediaSource:
    if not media.startswith(S3_URI_PREFIX):
        inline: Final[TwelveLabsMediaSource] = {"base64String": get_base64_str(media)}
        return inline
    s3_location: Final[TwelveLabsS3Location] = {"uri": media, "bucketOwner": _require_bucket_owner(bucket_owner)}
    remote: Final[TwelveLabsMediaSource] = {"s3Location": s3_location}
    return remote


def _named_media_source(name: str, media: str, bucket_owner: str | None) -> TwelveLabsMarengo3NamedMediaSource:
    named: Final[TwelveLabsMarengo3NamedMediaSource] = {
        "name": name,
        "mediaType": "image",
        **_media_source(media, bucket_owner),
    }
    return named


def _timed_media_input(media: str, params: Marengo3Params) -> TwelveLabsMarengo3TimedMediaInput:
    timed: Final[TwelveLabsMarengo3TimedMediaInput] = {
        "mediaSource": _media_source(media, params.bucketOwner),
        **params.timed_media_options(),
    }
    return timed


def _describe(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}" for problem in error.errors()
    )


def _validated_params(inference_params: Mapping[str, object]) -> Marengo3Params:
    try:
        return Marengo3Params.model_validate(inference_params)
    except ValidationError as error:
        raise BedrockError(status_code=400, message=f"Invalid Marengo 3.0 parameters: {_describe(error)}") from error


def _reject_unless_dropped(given: Mapping[str, object], drop_params: bool, reason: str) -> None:
    if not given or drop_params:
        return
    raise BedrockError(status_code=400, message=f"{reason} {', '.join(given)}; set drop_params to drop them")


def _require(value: str | None, input_type: str, param_name: str) -> str:
    if value is None:
        raise BedrockError(status_code=400, message=f"Input type '{input_type}' requires the '{param_name}' parameter")
    return value


def _require_media_sources(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if not value:
        raise BedrockError(
            status_code=400,
            message="Input type 'multi_input' requires a non-empty 'media_sources' mapping of name to media",
        )
    return value


def _request_base(inference_id: str | None) -> TwelveLabsMarengo3RequestBase:
    if inference_id is None:
        anonymous: Final[TwelveLabsMarengo3RequestBase] = {}
        return anonymous
    identified: Final[TwelveLabsMarengo3RequestBase] = {"inferenceId": inference_id}
    return identified


def build_marengo_3_request(
    input: str, inference_params: Mapping[str, object], drop_params: bool = False
) -> TwelveLabsMarengo3EmbeddingRequest:
    params: Final = _validated_params(inference_params)
    base: Final = _request_base(params.inferenceId)
    input_type: Final = params.resolved_input_type
    _reject_unless_dropped(
        params.given_2_7_only_params(), drop_params, "Marengo 3.0 does not accept the Marengo 2.7 parameters"
    )
    if input_type not in TIMED_INPUT_TYPES:
        _reject_unless_dropped(
            params.given_timed_media_options(), drop_params, f"Input type '{input_type}' does not accept"
        )
    match input_type:
        case "text":
            text_request: Final[TwelveLabsMarengo3TextRequest] = {
                **base,
                "inputType": "text",
                "text": {"inputText": input},
            }
            return text_request
        case "image":
            image_request: Final[TwelveLabsMarengo3ImageRequest] = {
                **base,
                "inputType": "image",
                "image": {"mediaSource": _media_source(input, params.bucketOwner)},
            }
            return image_request
        case "video":
            video_request: Final[TwelveLabsMarengo3VideoRequest] = {
                **base,
                "inputType": "video",
                "video": _timed_media_input(input, params),
            }
            return video_request
        case "audio":
            audio_request: Final[TwelveLabsMarengo3AudioRequest] = {
                **base,
                "inputType": "audio",
                "audio": _timed_media_input(input, params),
            }
            return audio_request
        case "text_image":
            text_image_request: Final[TwelveLabsMarengo3TextImageRequest] = {
                **base,
                "inputType": "text_image",
                "text_image": {
                    "inputText": input,
                    "mediaSource": _media_source(
                        _require(params.media_source, input_type, "media_source"), params.bucketOwner
                    ),
                },
            }
            return text_image_request
        case "multi_input":
            media_sources: Final = tuple(
                _named_media_source(name, media, params.bucketOwner)
                for name, media in _require_media_sources(params.media_sources).items()
            )
            multi_input_request: Final[TwelveLabsMarengo3MultiInputRequest] = {
                **base,
                "inputType": "multi_input",
                "multi_input": {"inputText": input, "mediaSources": media_sources}
                if input
                else {"mediaSources": media_sources},
            }
            return multi_input_request
        case _:
            assert_never(input_type)
