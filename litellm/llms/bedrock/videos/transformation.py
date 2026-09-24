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
from typing import TYPE_CHECKING, Any, Final

import httpx
from httpx._types import FileContent, RequestFiles

from litellm._logging import verbose_logger
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.types.videos.main import VideoObject

NOVA_REEL_DEFAULT_DURATION_SECONDS: Final = 6
NOVA_REEL_DEFAULT_FPS: Final = 24
NOVA_REEL_DEFAULT_DIMENSION: Final = "1280x720"

# AWS async-invoke status enum (bedrock-runtime service model) -> OpenAI-style
# VideoObject.status values used across LiteLLM video providers.
NOVA_REEL_STATUS_MAP: Final[dict[str, str]] = {
    "InProgress": "processing",
    "Completed": "completed",
    "Failed": "failed",
}

_UNSUPPORTED_MESSAGE: Final = (
    "video {operation} is not supported for Bedrock Nova Reel; Nova Reel exposes "
    "create (video_generation), status (video_status) and content (video_content) only"
)


def _file_content_to_b64_and_format(image: FileContent) -> tuple[str, str]:
    """Read an input-reference image and return (base64, "png"|"jpeg")."""
    if isinstance(image, bytes):
        image_bytes: bytes = image
    elif isinstance(image, str):
        # Already base64-encoded string - detect format from the decoded header.
        try:
            image_bytes = base64.b64decode(image, validate=False)
        except (binascii.Error, ValueError):
            return image, "png"
    elif hasattr(image, "read") and callable(getattr(image, "read", None)):
        if hasattr(image, "seek"):
            image.seek(0)
        image_bytes = image.read()
    else:
        raise ValueError(
            f"Nova Reel input_reference must be bytes, a file-like object or a base64 string; got {type(image)!r}"
        )

    if image_bytes.startswith(b"\x89PNG"):
        image_format: str = "png"
    elif image_bytes.startswith(b"\xff\xd8"):
        image_format = "jpeg"
    else:
        image_format = "png"
    return base64.b64encode(image_bytes).decode("utf-8"), image_format


class BedrockNovaReelVideoConfig(BaseVideoConfig):
    """
    Video config for amazon.nova-reel-v1:0 (and regional variants) on Bedrock.
    """

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "seconds",
            "size",
            "seed",
            "fps",
            "dimension",
            "taskType",
            "output_s3_uri",
            "parameters",
            "input_reference",
            "image",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        # All supported params pass through untouched; Nova Reel-specific keys
        # keep their AWS names (durationSeconds etc. are built in
        # transform_video_create_request from seconds/size).
        return dict(video_create_optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if headers is None:
            headers = {}
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
        litellm_params: dict,
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
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        """
        Build the StartAsyncInvoke request body.

        Returns (request_body, None, "POST") where request_body is the wrapped
        async-invoke envelope {modelId, modelInput, outputDataConfig}.
        """
        op: Final = dict(video_create_optional_request_params)

        output_s3_uri: Final = op.pop("output_s3_uri", None)
        if not output_s3_uri or not str(output_s3_uri).strip():
            raise ValueError(
                "Nova Reel video generation requires an S3 output location. Pass "
                'output_s3_uri="s3://my-bucket/optional-prefix/" in the request '
                "(Bedrock writes output.mp4 there)."
            )

        task_type: Final = op.pop("taskType", "TEXT_VIDEO")

        text_to_video_params: Final[dict[str, object]] = {"text": prompt}
        input_reference: Final = op.pop("input_reference", None) or op.pop("image", None)
        if input_reference is not None:
            if isinstance(input_reference, dict):
                # Pre-built provider shape: {"format": ..., "source": {...}}
                text_to_video_params["images"] = [input_reference]
            else:
                image_b64, image_format = _file_content_to_b64_and_format(input_reference)
                text_to_video_params["images"] = [{"format": image_format, "source": {"bytes": image_b64}}]

        generation_config: Final[dict[str, object]] = {
            "durationSeconds": NOVA_REEL_DEFAULT_DURATION_SECONDS,
            "fps": NOVA_REEL_DEFAULT_FPS,
            "dimension": NOVA_REEL_DEFAULT_DIMENSION,
        }
        seconds: Final = op.pop("seconds", None)
        if seconds is not None:
            try:
                generation_config["durationSeconds"] = int(float(seconds))
            except (TypeError, ValueError):
                verbose_logger.debug("Nova Reel ignoring non-numeric seconds=%r", seconds)
        size: Final = op.pop("size", None)
        if size is not None and isinstance(size, str) and "x" in size:
            generation_config["dimension"] = size.replace(" ", "")
        dimension: Final = op.pop("dimension", None)
        if dimension is not None and isinstance(dimension, str) and dimension.strip():
            generation_config["dimension"] = dimension
        fps: Final = op.pop("fps", None)
        if fps is not None:
            generation_config["fps"] = fps
        seed: Final = op.pop("seed", None)
        if seed is not None:
            generation_config["seed"] = seed

        model_input: Final[dict[str, object]] = {
            "taskType": task_type,
            "textToVideoParams": text_to_video_params,
            "videoGenerationConfig": generation_config,
        }

        # Remaining provider-specific keys (e.g. multiShotManualParams) are passed
        # through verbatim on the modelInput body.
        model_input.update(op)

        request_body: Final[dict[str, object]] = {
            "modelId": model,
            "modelInput": model_input,
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
        }
        return request_body, None, "POST"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        from litellm.types.videos.main import VideoObject

        response_data: Final = raw_response.json()
        invocation_arn: Final = response_data.get("invocationArn")
        if not invocation_arn:
            raise ValueError(f"Nova Reel async-invoke response missing invocationArn: {response_data}")
        video_obj = VideoObject(
            id=encode_video_id_with_provider(invocation_arn, "bedrock", model),
            object="video",
            status="processing",
            model=model,
            created_at=_epoch_now(),
        )
        usage: Final[dict[str, object]] = {}
        duration: Final = (
            ((request_data or {}).get("modelInput") or {}).get("videoGenerationConfig", {}).get("durationSeconds")
        )
        if duration is not None:
            # Mirrors the Vertex video config: lets the video cost calculator
            # compute cost from output_cost_per_second * duration_seconds.
            usage["duration_seconds"] = float(duration)
        if usage:
            video_obj.usage = usage
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError(
            "Nova Reel status URLs are built and signed in BedrockVideoGeneration "
            "(GET /async-invoke/{arn}). Do not use this transform for this config."
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: str | None = None,
        client: Any = None,
        model: str | None = None,
        video_id: str | None = None,
    ) -> VideoObject:
        from litellm.types.videos.main import VideoObject

        response_data: Final[dict] = raw_response.json()
        invocation_arn: Final = response_data.get("invocationArn")
        if not invocation_arn:
            raise ValueError(f"Nova Reel get-async-invoke response missing invocationArn: {response_data}")
        raw_status: Final = str(response_data.get("status") or response_data.get("invocationStatus") or "InProgress")
        status: Final = NOVA_REEL_STATUS_MAP.get(raw_status, "processing")

        video_obj = VideoObject(
            id=encode_video_id_with_provider(invocation_arn, "bedrock", model),
            object="video",
            status=status,
            model=model,
            created_at=_to_epoch(response_data.get("submitTime")),
            completed_at=(_to_epoch(response_data.get("endTime")) if status == "completed" else None),
            error=(
                {"message": response_data["failureMessage"]}
                if status == "failed" and response_data.get("failureMessage")
                else None
            ),
        )
        output_config: Final = (response_data.get("outputDataConfig") or {}).get("s3OutputDataConfig") or {}
        s3_uri: Final = output_config.get("s3Uri")
        if s3_uri:
            video_obj.usage = {"output_s3_uri": s3_uri}
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError(
            "Nova Reel video content is downloaded from the S3 output location in "
            "BedrockVideoGeneration. Do not use this transform for this config."
        )

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> bytes:
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE.format(operation="remix"))

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE.format(operation="remix"))

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE.format(operation="list"))

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE.format(operation="list"))

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE.format(operation="delete"))

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> VideoObject:
        raise NotImplementedError(_UNSUPPORTED_MESSAGE.format(operation="delete"))

    @staticmethod
    def extract_invocation_arn(video_id: str) -> str:
        """Return the raw Bedrock invocationArn from a (possibly encoded) video id."""
        decoded: Final = decode_video_id_with_provider(video_id)
        arn: Final = decoded.get("video_id") or ""
        return arn or extract_original_video_id(video_id)


def _epoch_now() -> int:
    import time

    return int(time.time())


def _to_epoch(timestamp: Any) -> int | None:
    if timestamp is None:
        return None
    try:
        return int(float(timestamp))
    except (TypeError, ValueError):
        return None
