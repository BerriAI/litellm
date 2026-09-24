"""
Amazon Nova Reel video generation on Bedrock (StartAsyncInvoke / GetAsyncInvoke).

Nova Reel is invoked through the Bedrock asynchronous invoke API:
- create:  POST {runtime}/async-invoke          body: {modelId, modelInput, outputDataConfig}
- status:  GET  {runtime}/async-invoke/{arn}    response: {invocationArn, status, failureMessage, ...}
- content: download output.mp4 from the S3 output location when status is Completed

The real HTTP (AWS SigV4 signing + S3 download) happens in
``litellm.llms.bedrock.videos.handler.BedrockVideoGeneration``; this config
builds/validates the request bodies and maps responses to ``VideoObject``.

Refs:
- https://docs.aws.amazon.com/nova/latest/userguide/video-req-resp-structure.html
- https://docs.aws.amazon.com/nova/latest/userguide/video-gen-access.html
- bedrock-runtime service model: StartAsyncInvoke POST /async-invoke,
  GetAsyncInvoke GET /async-invoke/{invocationArn}, AsyncInvokeStatus enum
  InProgress | Completed | Failed.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx
from httpx._types import FileContent, RequestFiles

from litellm._logging import verbose_logger
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.bedrock import (
    BedrockGetAsyncInvokeResponse,
    BedrockStartAsyncInvokeResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.types.videos.main import VideoObject

_SupportedParams: TypeAlias = list[str]
_VideoParams: TypeAlias = dict[str, object]
_VideoHeaders: TypeAlias = dict[str, str]
_VideoStringParams: TypeAlias = dict[str, str]

NOVA_REEL_DEFAULT_DURATION_SECONDS: Final = 6
NOVA_REEL_DEFAULT_FPS: Final = 24
NOVA_REEL_DEFAULT_DIMENSION: Final = "1280x720"

# AWS clientRequestToken: alphanumeric and hyphens only, at most 64 chars.
NOVA_REEL_CLIENT_REQUEST_TOKEN_MAX_LEN: Final = 64
_NOVA_REEL_TOKEN_UNSAFE: Final = re.compile(r"[^0-9A-Za-z-]")

# AWS async-invoke status enum (bedrock-runtime service model) -> OpenAI-style
# VideoObject.status values used across LiteLLM video providers.
NOVA_REEL_STATUS_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "InProgress": "processing",
        "Completed": "completed",
        "Failed": "failed",
    }
)

_UNSUPPORTED_MESSAGE: Final = (
    "video {operation} is not supported for Bedrock Nova Reel; Nova Reel exposes "
    "create (video_generation), status (video_status) and content (video_content) only"
)


def _unsupported_operation_error(operation: str) -> BedrockError:
    """400-class error for unsupported video operations.

    Verified against litellm.exception_type: a plain ValueError/NotImplementedError
    both fall through to APIConnectionError (500-class), while BedrockError carries
    status_code=400 into BadRequestError through the bedrock mapping.
    """
    return BedrockError(status_code=400, message=_UNSUPPORTED_MESSAGE.format(operation=operation))


def _user_input_error(message: str) -> BedrockError:
    """400-class error for invalid caller input (same mapping as _unsupported_operation_error).

    A plain ValueError would surface as APIConnectionError (500-class, which OpenAI
    SDKs auto-retry); BedrockError(status_code=400) maps to BadRequestError instead.
    """
    return BedrockError(status_code=400, message=message)


def _data_url_payload(image: str) -> str:
    """Validate a ``data:<mime>;base64,`` URL and return only its payload."""
    header, sep, encoded = image.partition(",")
    if not sep or not header.removeprefix("data:").endswith(";base64"):
        raise _user_input_error(
            "Nova Reel input_reference data URLs must be base64-encoded "
            "(data:image/png;base64,<payload>); got an unsupported data URL prefix."
        )
    return encoded


def _file_content_to_b64_and_format(image: FileContent) -> tuple[str, str]:
    """Read an input-reference image and return (base64, "png"|"jpeg")."""
    if isinstance(image, bytes):
        image_bytes: bytes = image
    elif isinstance(image, str):
        # Base64-encoded string, optionally wrapped in a data URL; detect the
        # format from the decoded header.
        payload: Final = _data_url_payload(image) if image.startswith("data:") else image
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as err:
            raise _user_input_error(f"Nova Reel input_reference string did not decode as base64: {err}") from err
    elif hasattr(image, "read") and callable(getattr(image, "read", None)):
        if hasattr(image, "seek"):
            image.seek(0)
        image_bytes = image.read()
        if not isinstance(image_bytes, bytes):
            raise _user_input_error(
                "Nova Reel input_reference file objects must be opened in binary mode "
                f"(read() returned {type(image_bytes).__name__}); open image files with 'rb'."
            )
    else:
        raise _user_input_error(
            f"Nova Reel input_reference must be bytes, a file-like object or a base64 string; got {type(image)!r}"
        )

    if image_bytes.startswith(b"\x89PNG"):
        image_format: str = "png"
    elif image_bytes.startswith(b"\xff\xd8"):
        image_format = "jpeg"
    else:
        raise _user_input_error(
            "Nova Reel input_reference images must be PNG or JPEG encoded; "
            f"unrecognized image header bytes {image_bytes[:8]!r}"
        )
    return base64.b64encode(image_bytes).decode("utf-8"), image_format


def _duration_seconds_from_request(request_data: Mapping[str, object] | None) -> float | None:
    """durationSeconds from the StartAsyncInvoke request envelope, for cost calculation.

    TEXT_VIDEO and MULTI_SHOT_AUTOMATED carry a single durationSeconds on
    videoGenerationConfig; MULTI_SHOT_MANUAL carries per-shot durations inside
    multiShotManualParams.shots[*].durationSeconds, so the billable duration is
    the sum of the shot durations.
    """
    if request_data is None:
        return None
    model_input: Final[object | None] = request_data.get("modelInput")
    if not isinstance(model_input, Mapping):
        return None
    manual_params: Final[object | None] = model_input.get("multiShotManualParams")
    if isinstance(manual_params, Mapping):
        shots: Final[object | None] = manual_params.get("shots")
        if isinstance(shots, list):
            total = 0.0
            saw_duration = False
            for shot in shots:
                if not isinstance(shot, Mapping):
                    continue
                # Annotated, not Final: basedpyright forbids Final assignment inside loops.
                shot_duration: object | None = shot.get("durationSeconds")
                if not isinstance(shot_duration, (int, float, str)):
                    continue
                try:
                    total += float(shot_duration)
                    saw_duration = True
                except ValueError:
                    continue
            if saw_duration:
                return total
    generation_config: Final[object | None] = model_input.get("videoGenerationConfig")
    if not isinstance(generation_config, Mapping):
        return None
    duration: Final[object | None] = generation_config.get("durationSeconds")
    if not isinstance(duration, (int, float, str)):
        return None
    try:
        return float(duration)
    except ValueError:
        return None


def _sanitize_client_request_token(token: str) -> str:
    """AWS clientRequestToken allows alphanumeric and hyphens, max 64 chars."""
    return _NOVA_REEL_TOKEN_UNSAFE.sub("-", token)[:NOVA_REEL_CLIENT_REQUEST_TOKEN_MAX_LEN]


def _request_id_from_litellm_params(litellm_params: GenericLiteLLMParams) -> str | None:
    """Best-effort litellm request id: metadata.request_id, then the litellm_call_id extra field."""
    metadata: Final = getattr(litellm_params, "metadata", None)
    if isinstance(metadata, Mapping):
        request_id: Final = metadata.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
    call_id: Final = getattr(litellm_params, "litellm_call_id", None)
    if isinstance(call_id, str) and call_id:
        return call_id
    return None


def _generation_config_from_op(op: _VideoParams, task_type: object) -> _VideoParams:
    """videoGenerationConfig from request params (pops seconds/size/dimension/fps/seed).

    durationSeconds lives on videoGenerationConfig for TEXT_VIDEO and
    MULTI_SHOT_AUTOMATED only; MULTI_SHOT_MANUAL durations live per shot
    inside multiShotManualParams.shots, so it is omitted there.
    """
    generation_config: Final[_VideoParams] = {
        "fps": NOVA_REEL_DEFAULT_FPS,
        "dimension": NOVA_REEL_DEFAULT_DIMENSION,
    }
    single_duration: Final[bool] = task_type != "MULTI_SHOT_MANUAL"
    if single_duration:
        generation_config["durationSeconds"] = NOVA_REEL_DEFAULT_DURATION_SECONDS
    seconds: Final = op.pop("seconds", None)
    if isinstance(seconds, (int, float, str)):
        try:
            parsed_seconds: Final = int(float(seconds))
        except ValueError as err:
            raise _user_input_error(f"Nova Reel seconds must be a number; got {seconds!r}") from err
        if single_duration:
            generation_config["durationSeconds"] = parsed_seconds
    size: Final = op.pop("size", None)
    if size is not None and isinstance(size, str) and "x" in size:
        generation_config["dimension"] = size.replace(" ", "")
    dimension: Final = op.pop("dimension", None)
    if dimension is not None and isinstance(dimension, str) and dimension.strip():
        generation_config["dimension"] = dimension
    fps: Final = op.pop("fps", None)
    if fps is not None:
        try:
            generation_config["fps"] = int(float(fps))
        except ValueError as err:
            raise _user_input_error(f"Nova Reel fps must be a number; got {fps!r}") from err
    seed: Final = op.pop("seed", None)
    if seed is not None:
        try:
            generation_config["seed"] = int(float(seed))
        except ValueError as err:
            raise _user_input_error(f"Nova Reel seed must be a number; got {seed!r}") from err
    return generation_config


def _task_params_from_op(task_type: object, op: _VideoParams, prompt: str, input_reference: object) -> _VideoParams:
    """Per-taskType params section for modelInput (pops multiShot params from op)."""
    if task_type == "MULTI_SHOT_AUTOMATED":
        # AWS schema: automated multi-shot takes multiShotAutomatedParams
        # (never textToVideoParams) and forbids input images.
        if input_reference is not None:
            raise _user_input_error(
                "Nova Reel MULTI_SHOT_AUTOMATED does not accept input images "
                "(input_reference/image); automated multi-shot is text-driven only."
            )
        automated_params: Final[object | None] = op.pop("multiShotAutomatedParams", None)
        automated_section: Final[_VideoParams] = {
            "multiShotAutomatedParams": (
                automated_params if isinstance(automated_params, Mapping) else {"text": prompt}
            )
        }
        return automated_section
    if task_type == "MULTI_SHOT_MANUAL":
        manual_params: Final[object | None] = op.pop("multiShotManualParams", None)
        if not isinstance(manual_params, Mapping):
            raise _user_input_error(
                "Nova Reel MULTI_SHOT_MANUAL requires multiShotManualParams in the request "
                "(shot definitions with per-shot text/images/durationSeconds); "
                f"got {manual_params!r}."
            )
        manual_section: Final[_VideoParams] = {"multiShotManualParams": manual_params}
        return manual_section
    # TEXT_VIDEO (default): textToVideoParams with the prompt and optional images.
    text_to_video_params: Final[_VideoParams] = {"text": prompt}
    if input_reference is not None:
        if isinstance(input_reference, dict):
            # Pre-built provider shape: {"format": ..., "source": {...}}
            text_to_video_params["images"] = [input_reference]  # mutable-ok: AWS images param is a list
        else:
            image_b64, image_format = _file_content_to_b64_and_format(
                input_reference  # pyright: ignore[reportArgumentType]  # untyped user input; helper validates
            )
            text_to_video_params["images"] = [  # mutable-ok: AWS images param is a list
                {"format": image_format, "source": {"bytes": image_b64}}  # mutable-ok: nested AWS image payload
            ]
    text_section: Final[_VideoParams] = {"textToVideoParams": text_to_video_params}
    return text_section


class BedrockNovaReelVideoConfig(BaseVideoConfig):
    """
    Video config for amazon.nova-reel-v1:0 (and regional variants) on Bedrock.

    Health checks: the proxy's video_generation probe calls avideo_generation()
    with only a prompt, which this config rejects with a 400 because Nova Reel
    requires a per-request output_s3_uri; the deployment is then reported
    unhealthy (the health check marks any errored probe unhealthy, 4xx included).
    Every video provider's probe creates a real video, so there is no cheaper
    repo-consistent probe. Operators should set
    ``model_info.disable_background_health_check: true`` on Nova Reel
    deployments (or put ``output_s3_uri`` in the deployment litellm_params and
    accept that each health check starts a real, billed generation).
    """

    def get_supported_openai_params(self, model: str) -> _SupportedParams:
        return [  # mutable-ok: BaseVideoConfig requires a list
            "seconds",
            "size",
            "seed",
            "fps",
            "dimension",
            "taskType",
            "output_s3_uri",
            "kmsKeyId",
            "bucketOwner",
            "input_reference",
            "image",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> _VideoParams:
        # All supported params pass through untouched; Nova Reel-specific keys
        # keep their AWS names (durationSeconds etc. are built in
        # transform_video_create_request from seconds/size).
        return dict(video_create_optional_params)  # mutable-ok: BaseVideoConfig requires a mutable param mapping

    def validate_environment(
        self,
        headers: _VideoHeaders,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> _VideoHeaders:
        if headers is None:
            headers = {}  # mutable-ok: None headers start empty before Content-Type is added
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return headers

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseVideoConfig passes headers as a dict
    ) -> BedrockError:
        """BedrockError synthesizes a response that keeps provider headers like x-amzn-RequestId."""
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: _VideoParams,
    ) -> str:
        raise NotImplementedError(
            "Nova Reel video URLs are built in BedrockVideoGeneration "
            "(AWS runtime endpoint + /async-invoke paths). Do not use get_complete_url "
            "for this config."
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: _VideoParams,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[_VideoParams, RequestFiles, str]:
        """
        Build the StartAsyncInvoke request body.

        Returns (request_body, files, "POST") where request_body is the wrapped
        async-invoke envelope {modelId, modelInput, outputDataConfig}.
        """
        op: Final[_VideoParams] = dict(  # mutable-ok: request params are popped in place while building modelInput
            video_create_optional_request_params
        )

        output_s3_uri: Final = op.pop("output_s3_uri", None)
        if not output_s3_uri or not str(output_s3_uri).strip():
            raise _user_input_error(
                "Nova Reel video generation requires an S3 output location. Pass "
                'output_s3_uri="s3://my-bucket/optional-prefix/" in the request '
                "(Bedrock writes output.mp4 there)."
            )
        if not str(output_s3_uri).startswith("s3://"):
            raise _user_input_error(
                "Nova Reel output_s3_uri must be an S3 URI starting with s3:// "
                f"(got {output_s3_uri!r}); Bedrock writes output.mp4 into that bucket."
            )
        # Pop the S3 config keys (and their snake_case aliases) before the
        # modelInput passthrough merge so they never leak into modelInput; they
        # are forwarded into outputDataConfig.s3OutputDataConfig below.
        kms_key_id: Final = op.pop("kmsKeyId", None) or op.pop("output_s3_kms_key_id", None)
        bucket_owner: Final = op.pop("bucketOwner", None) or op.pop("output_s3_bucket_owner", None)

        task_type: Final = op.pop("taskType", "TEXT_VIDEO")

        if not str(prompt or "").strip():
            raise _user_input_error("Nova Reel prompt is required and cannot be empty (or whitespace-only).")

        # Pop both reference keys unconditionally so neither leaks into modelInput;
        # input_reference wins when a caller passes both.
        popped_reference: Final = op.pop("input_reference", None)
        popped_image: Final = op.pop("image", None)
        input_reference: Final = popped_reference if popped_reference is not None else popped_image

        model_input: Final[_VideoParams] = {
            "taskType": task_type,
            "videoGenerationConfig": _generation_config_from_op(op, task_type),
        }
        model_input.update(_task_params_from_op(task_type, op, prompt, input_reference))

        # Known non-AWS video-client params have no Nova Reel mapping; drop them
        # instead of leaking junk keys into modelInput. Everything else keeps the
        # verbatim passthrough (provider-specific AWS keys like multiShotManualParams).
        for dropped_key in ("parameters", "resolution", "characters", "user", "extra_headers"):
            op.pop(dropped_key, None)
        # Pop both token keys unconditionally (caller-supplied wins over the
        # litellm request id) so neither leaks into modelInput.
        caller_token_snake: Final = op.pop("client_request_token", None)
        caller_token_camel: Final = op.pop("clientRequestToken", None)
        caller_token: Final = caller_token_snake if caller_token_snake is not None else caller_token_camel
        model_input.update(op)

        request_id: Final[str | None] = _request_id_from_litellm_params(litellm_params)
        client_request_token: Final[str | None] = (
            _sanitize_client_request_token(str(caller_token))
            if caller_token is not None
            else (_sanitize_client_request_token(request_id) if request_id is not None else None)
        )

        # Optional S3 config keys are added below before the envelope is returned.
        s3_output_config: Final[_VideoParams] = {"s3Uri": output_s3_uri}
        if kms_key_id is not None:
            s3_output_config["kmsKeyId"] = kms_key_id
        if bucket_owner is not None:
            s3_output_config["bucketOwner"] = bucket_owner
        request_body: Final[_VideoParams] = {
            "modelId": model,
            "modelInput": model_input,
            "outputDataConfig": {"s3OutputDataConfig": s3_output_config},
        }
        if client_request_token:
            # Envelope key only when populated: caller-supplied token or litellm request id.
            request_body["clientRequestToken"] = client_request_token
        return request_body, [], "POST"  # mutable-ok: HTTP files payload requires a list

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLogging | None,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        from litellm.types.videos.main import VideoObject

        try:
            response_data: Final[BedrockStartAsyncInvokeResponse] = raw_response.json()
        except ValueError as err:
            raise BedrockError(
                status_code=502,
                message=f"Nova Reel async-invoke returned a non-JSON response: {err}",
            ) from err
        invocation_arn: Final[str | None] = response_data.get("invocationArn")
        if not invocation_arn:
            raise ValueError(f"Nova Reel async-invoke response missing invocationArn: {response_data}")
        video_obj = VideoObject(
            id=encode_video_id_with_provider(invocation_arn, "bedrock", model),
            object="video",
            status="processing",
            model=model,
            created_at=_epoch_now(),
        )
        duration_seconds: Final[float | None] = _duration_seconds_from_request(request_data)
        if duration_seconds is not None:
            # Mirrors the Vertex video config: lets the video cost calculator
            # compute cost from output_cost_per_second * duration_seconds.
            video_obj.usage = {"duration_seconds": duration_seconds}  # mutable-ok: VideoObject.usage payload dict
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError(
            "Nova Reel status URLs are built and signed in BedrockVideoGeneration "
            "(GET /async-invoke/{arn}). Do not use this transform for this config."
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLogging | None,
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
        model: str | None = None,
        video_id: str | None = None,
    ) -> VideoObject:
        from litellm.types.videos.main import VideoObject

        try:
            response_data: Final[BedrockGetAsyncInvokeResponse] = raw_response.json()
        except ValueError as err:
            raise BedrockError(
                status_code=502,
                message=f"Nova Reel get-async-invoke returned a non-JSON response: {err}",
            ) from err
        invocation_arn: Final[str | None] = response_data.get("invocationArn")
        if not invocation_arn:
            raise ValueError(f"Nova Reel get-async-invoke response missing invocationArn: {response_data}")
        status_field: Final[object] = response_data.get("status")
        if not isinstance(status_field, str) or not status_field:
            raise BedrockError(
                status_code=500,
                message=(
                    "Nova Reel get-async-invoke response had an unexpected shape: "
                    f"missing or empty 'status' (observed keys: {sorted(response_data.keys())})"
                ),
            )
        raw_status: Final[str] = status_field
        if raw_status not in NOVA_REEL_STATUS_MAP:
            verbose_logger.warning("Nova Reel unmapped invocationStatus=%r; reporting processing", raw_status)
        status: Final[str] = NOVA_REEL_STATUS_MAP.get(raw_status, "processing")

        failure_message: Final[str | None] = response_data.get("failureMessage")
        video_obj = VideoObject(
            id=encode_video_id_with_provider(invocation_arn, "bedrock", model),
            object="video",
            status=status,
            model=model,
            created_at=_to_epoch(response_data.get("submitTime")),
            completed_at=(_to_epoch(response_data.get("endTime")) if status == "completed" else None),
            error=(
                {"message": failure_message}  # mutable-ok: VideoObject.error accepts a plain payload dict
                if status == "failed" and failure_message
                else None
            ),
        )
        output_config: Final = response_data.get("outputDataConfig")
        if output_config is not None:
            s3_config: Final = output_config.get("s3OutputDataConfig")
            if s3_config is not None:
                s3_uri: Final[str | None] = s3_config.get("s3Uri")
                if s3_uri:
                    # Provider detail, not usage: rides on _hidden_params (like the
                    # vertex video transforms' provider-specific fields) so cost
                    # calculators reading usage.duration_seconds never trip on it.
                    video_obj._hidden_params["output_s3_uri"] = s3_uri
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        variant: str | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError(
            "Nova Reel video content is downloaded from the S3 output location in "
            "BedrockVideoGeneration. Do not use this transform for this config."
        )

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLogging | None,
    ) -> bytes:
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise _unsupported_operation_error("remix")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLogging | None,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise _unsupported_operation_error("remix")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise _unsupported_operation_error("list")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLogging | None,
        custom_llm_provider: str | None = None,
    ) -> _VideoStringParams:
        raise _unsupported_operation_error("list")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise _unsupported_operation_error("delete")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLogging | None,
    ) -> VideoObject:
        raise _unsupported_operation_error("delete")

    @staticmethod
    def extract_invocation_arn(video_id: str) -> str:
        """Return the raw Bedrock invocationArn from a (possibly encoded) video id."""
        decoded: Final = decode_video_id_with_provider(video_id)
        arn: Final = decoded.get("video_id") or ""
        return arn or extract_original_video_id(video_id)


def _epoch_now() -> int:
    import time

    return int(time.time())


def _to_epoch(timestamp: str | float | None) -> int | None:
    """Bedrock timestamps to unix epoch seconds.

    The bedrock-runtime Smithy model declares timestampFormat: iso8601 for
    submitTime/endTime, so real GetAsyncInvoke payloads carry strings like
    "2026-01-15T10:30:00Z"; numeric epochs are accepted too.
    """
    if timestamp is None:
        return None
    try:
        return int(float(timestamp))
    except (TypeError, ValueError):
        pass
    try:
        return int(datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).timestamp())
    except ValueError:
        verbose_logger.warning("Nova Reel response carried an unparseable timestamp %r; leaving it unset", timestamp)
        return None
