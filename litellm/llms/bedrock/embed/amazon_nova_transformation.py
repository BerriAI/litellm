"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock Amazon Nova /invoke and /async-invoke format.

Why separate file? Make it easy to see how transformation works

Supports:
- Synchronous embeddings (SINGLE_EMBEDDING)
- Asynchronous embeddings with segmentation (SEGMENTED_EMBEDDING)
- Multimodal inputs: text, image, video, audio
- Multiple embedding purposes and dimensions

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/nova-embed.html
"""

import base64
from collections.abc import Mapping, Sequence
from itertools import zip_longest
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import ModelNotMappedError
from litellm.litellm_core_utils.audio_utils.utils import calculate_request_duration
from litellm.types.utils import (
    Embedding,
    EmbeddingResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


def _estimated_item_tokens(item: Mapping[str, object]) -> int:
    truncated: Final = item.get("truncatedCharLength")
    embedding: Final = item.get("embedding")
    if isinstance(truncated, int):
        return truncated // 4
    return len(embedding) // 4 if isinstance(embedding, list) else 0


def _estimated_input_tokens(response: Mapping[str, object]) -> int:
    items: Final = response.get("embeddings")
    if isinstance(items, list):
        return sum(_estimated_item_tokens(item) for item in items if isinstance(item, Mapping) and "embedding" in item)
    return _estimated_item_tokens({"embedding": response.get("embedding")})


_EMBEDDING_PARAMS: Final = TypeAdapter(dict[str, object])


class _InlineSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    encoded: str = Field(alias="bytes")


class _InlineMedia(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: _InlineSource


class _ClipPrices(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_cost_per_token: float = 0.0
    input_cost_per_audio_per_second: float = 0.0
    input_cost_per_video_per_second: float = 0.0


def _clip_prices(model: str) -> _ClipPrices:
    try:
        return _ClipPrices.model_validate(litellm.get_model_info(model=model, custom_llm_provider="bedrock"))
    except (ModelNotMappedError, ValidationError):
        return _ClipPrices()


def _seconds_costing_at_least(seconds: float | None, per_second: float, floor_cost: float) -> float | None:
    return seconds if seconds is not None and seconds * per_second >= floor_cost else None


def _embedding_params(request_data: Mapping[str, object]) -> Mapping[str, object]:
    params: Final = request_data.get("singleEmbeddingParams") or request_data.get("segmentedEmbeddingParams")
    try:
        return _EMBEDDING_PARAMS.validate_python(params)
    except ValidationError:
        return {}


def _is_image_request(request_data: Mapping[str, object]) -> bool:
    return "image" in _embedding_params(request_data)


def _is_audio_or_video_request(request_data: Mapping[str, object]) -> bool:
    params: Final = _embedding_params(request_data)
    return "audio" in params or "video" in params


def _inline_media_bytes(media: object) -> bytes | None:
    try:
        return base64.b64decode(_InlineMedia.model_validate(media).source.encoded)
    except ValueError:
        return None


def _mp4_box(content: bytes, box_type: bytes, start: int, end: int) -> tuple[int, int] | None:
    offset = start  # rebind-ok: walks boxes whose sizes are only known as each one is read
    while offset + 8 <= end:
        size = int.from_bytes(content[offset : offset + 4], "big")
        header = 16 if size == 1 else 8
        length = (
            end - offset
            if size == 0
            else int.from_bytes(content[offset + 8 : offset + 16], "big")
            if size == 1
            else size
        )
        if length < header or offset + length > end:
            return None
        if content[offset + 4 : offset + 8] == box_type:
            return offset + header, offset + length
        offset += length
    return None


def _mp4_duration_seconds(content: bytes) -> float | None:
    """Reads the movie duration from the `moov/mvhd` box of an MP4, MOV or 3GP file."""
    moov: Final = _mp4_box(content, b"moov", 0, len(content))
    mvhd: Final = _mp4_box(content, b"mvhd", *moov) if moov else None
    if not mvhd or mvhd[0] == mvhd[1]:
        return None
    timescale_at: Final = mvhd[0] + (20 if content[mvhd[0]] == 1 else 12)
    duration_end: Final = timescale_at + (12 if content[mvhd[0]] == 1 else 8)
    if duration_end > mvhd[1]:
        return None
    timescale: Final = int.from_bytes(content[timescale_at : timescale_at + 4], "big")
    duration_field: Final = content[timescale_at + 4 : duration_end]
    duration: Final = int.from_bytes(duration_field, "big")
    is_known: Final = duration > 0 and duration_field.count(0xFF) < len(duration_field)
    return duration / timescale if timescale > 0 and is_known else None


def _audio_seconds(request_data: Mapping[str, object]) -> float | None:
    content: Final = _inline_media_bytes(_embedding_params(request_data).get("audio"))
    seconds: Final = calculate_request_duration(content) if content else None
    return seconds if seconds and seconds > 0 else None


def _video_seconds(request_data: Mapping[str, object]) -> float | None:
    content: Final = _inline_media_bytes(_embedding_params(request_data).get("video"))
    return _mp4_duration_seconds(content) if content else None


class AmazonNovaEmbeddingConfig:
    """
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/nova-embed.html

    Amazon Nova Multimodal Embeddings supports:
    - Text, image, video, and audio inputs
    - Synchronous (InvokeModel) and asynchronous (StartAsyncInvoke) APIs
    - Multiple embedding purposes and dimensions
    """

    servable_encoding_formats: Final = ("float", "base64")

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> list[str]:
        return [
            "dimensions",
            "encoding_format",
        ]

    def map_openai_params(self, non_default_params: Mapping[str, object], optional_params: dict) -> dict:
        """Map OpenAI-style parameters to Nova parameters."""
        for k, v in non_default_params.items():
            if k == "dimensions":
                # Map OpenAI dimensions to Nova embedding_dimension
                optional_params["embedding_dimension"] = v
            elif k == "encoding_format" and v == "base64":
                optional_params["encoding_format"] = v
        return optional_params

    def _parse_data_url(self, data_url: str) -> tuple[str, str]:
        """
        Parse a data URL to extract the media type and base64 data.

        Args:
            data_url: Data URL in format: data:image/jpeg;base64,/9j/4AAQ...

        Returns:
            tuple: (media_type, base64_data)
                media_type: e.g., "image/jpeg", "video/mp4", "audio/mpeg"
                base64_data: The base64-encoded data without the prefix
        """
        if not data_url.startswith("data:"):
            raise ValueError(f"Invalid data URL format: {data_url[:50]}...")

        # Split by comma to separate metadata from data
        # Format: data:image/jpeg;base64,<base64_data>
        if "," not in data_url:
            raise ValueError(f"Invalid data URL format (missing comma): {data_url[:50]}...")

        metadata, base64_data = data_url.split(",", 1)

        # Extract media type from metadata
        # Remove 'data:' prefix and ';base64' suffix
        metadata = metadata[5:]  # Remove 'data:'

        if ";" in metadata:
            media_type = metadata.split(";")[0]
        else:
            media_type = metadata

        return media_type, base64_data

    def _transform_request(
        self,
        input: str,
        inference_params: dict[str, object],
        async_invoke_route: bool = False,
        model_id: str | None = None,
        output_s3_uri: str | None = None,
    ) -> dict[str, object]:
        """
        Transform OpenAI-style input to Nova format.

        Only handles OpenAI params (dimensions). All other Nova-specific params
        should be passed via inference_params and will be passed through as-is.

        Args:
            input: The input text or media reference
            inference_params: Additional parameters (will be passed through)
            async_invoke_route: Whether this is for async invoke
            model_id: Model ID (for async invoke)
            output_s3_uri: S3 URI for output (for async invoke)

        Returns:
            dict: Nova embedding request
        """
        # Determine task type
        task_type: Final = "SEGMENTED_EMBEDDING" if async_invoke_route else "SINGLE_EMBEDDING"

        # Build the base request structure
        request: Final[dict[str, object]] = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": task_type,
        }

        # Start with inference_params (user-provided params)
        embedding_params: Final = inference_params.copy()

        embedding_params.pop("output_s3_uri", None)

        # Map OpenAI dimensions to embeddingDimension if provided
        if "dimensions" in embedding_params:
            embedding_params["embeddingDimension"] = embedding_params.pop("dimensions")
        elif "embedding_dimension" in embedding_params:
            embedding_params["embeddingDimension"] = embedding_params.pop("embedding_dimension")

        # Add required embeddingPurpose if not provided (required by Nova API)
        if "embeddingPurpose" not in embedding_params:
            embedding_params["embeddingPurpose"] = "GENERIC_INDEX"

        # Add required embeddingDimension if not provided (required by Nova API)
        if "embeddingDimension" not in embedding_params:
            embedding_params["embeddingDimension"] = 3072

        # For text/media input, add basic structure if user hasn't provided text/image/video/audio
        if (
            "text" not in embedding_params
            and "image" not in embedding_params
            and "video" not in embedding_params
            and "audio" not in embedding_params
        ):
            # Check if input is a data URL (e.g., data:image/jpeg;base64,...)
            if input.startswith("data:"):
                # Parse the data URL to extract media type and base64 data
                media_type, base64_data = self._parse_data_url(input)

                if media_type.startswith("image/"):
                    # Extract image format from MIME type (e.g., image/jpeg -> jpeg)
                    image_format = media_type.split("/")[1].lower()
                    # Nova API expects specific formats
                    if image_format == "jpg":
                        image_format = "jpeg"

                    embedding_params["image"] = {
                        "format": image_format,
                        "source": {"bytes": base64_data},
                    }
                elif media_type.startswith("video/"):
                    # Handle video data URLs
                    video_format: Final = media_type.split("/")[1].lower()
                    embedding_params["video"] = {
                        "format": video_format,
                        "source": {"bytes": base64_data},
                    }
                elif media_type.startswith("audio/"):
                    # Handle audio data URLs
                    audio_format: Final = media_type.split("/")[1].lower()
                    embedding_params["audio"] = {
                        "format": audio_format,
                        "source": {"bytes": base64_data},
                    }
                else:
                    # Fallback to text for unknown types
                    embedding_params["text"] = {"value": input, "truncationMode": "END"}
            elif input.startswith("s3://"):
                # S3 URL - default to text for now, user should specify modality
                embedding_params["text"] = {
                    "source": {"s3Location": {"uri": input}},
                    "truncationMode": "END",  # Required by Nova API
                }
            else:
                # Plain text input
                embedding_params["text"] = {
                    "value": input,
                    "truncationMode": "END",  # Required by Nova API
                }

        # Set the embedding params in the request
        if task_type == "SINGLE_EMBEDDING":
            request["singleEmbeddingParams"] = embedding_params
        else:
            request["segmentedEmbeddingParams"] = embedding_params

        # For async invoke, wrap in the async invoke format
        if async_invoke_route and model_id:
            return self._wrap_async_invoke_request(
                model_input=request,
                model_id=model_id,
                output_s3_uri=output_s3_uri,
            )

        return request

    def transform_request(
        self,
        input: str,
        inference_params: dict[str, object],  # mutable-ok: mirrors override contract
        async_invoke_route: bool = False,
        model_id: str | None = None,
        output_s3_uri: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        return self._transform_request(input, inference_params, async_invoke_route, model_id, output_s3_uri)

    def _wrap_async_invoke_request(
        self,
        model_input: dict[str, object],
        model_id: str,
        output_s3_uri: str | None = None,
    ) -> dict[str, object]:
        """
        Wrap the transformed request in the AWS Bedrock async invoke format.

        Args:
            model_input: The transformed Nova embedding request
            model_id: The model identifier (without async_invoke prefix)
            output_s3_uri: S3 URI for output data config

        Returns:
            dict: The wrapped async invoke request
        """
        import urllib.parse

        # Clean the model ID
        unquoted_model_id = urllib.parse.unquote(model_id)
        if unquoted_model_id.startswith("async_invoke/"):
            unquoted_model_id = unquoted_model_id.replace("async_invoke/", "")

        # Validate that the S3 URI is not empty
        if not output_s3_uri or output_s3_uri.strip() == "":
            raise ValueError("output_s3_uri is required for async invoke requests")

        return {
            "modelId": unquoted_model_id,
            "modelInput": model_input,
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
        }

    def _transform_response(
        self,
        response_list: list[dict],
        model: str,
        batch_data: Sequence[Mapping[str, object]] | None = None,
        metered_input_tokens: Sequence[int | None] = (),
    ) -> EmbeddingResponse:
        """
        Transform Nova response to OpenAI format.

        Nova response format:
        {
            "embeddings": [
                {
                    "embeddingType": "TEXT" | "IMAGE" | "VIDEO" | "AUDIO" | "AUDIO_VIDEO_COMBINED",
                    "embedding": [0.1, 0.2, ...],
                    "truncatedCharLength": 100  # Optional, only for text
                }
            ]
        }
        """
        embeddings: Final[list[Embedding]] = []

        for response in response_list:
            # Nova response has an "embeddings" array
            if "embeddings" in response and isinstance(response["embeddings"], list):
                for item in response["embeddings"]:
                    if "embedding" in item:
                        embedding = Embedding(
                            embedding=item["embedding"],
                            index=len(embeddings),
                            object="embedding",
                        )
                        embeddings.append(embedding)
            elif "embedding" in response:
                # Direct embedding response (fallback)
                embedding = Embedding(
                    embedding=response["embedding"],
                    index=len(embeddings),
                    object="embedding",
                )
                embeddings.append(embedding)

        requests: Final[Sequence[Mapping[str, object]]] = batch_data or ()
        prices: Final = _clip_prices(model)
        estimated_tokens: Final = tuple(_estimated_input_tokens(response) for response in response_list)
        estimate_costs: Final = tuple(tokens * prices.input_cost_per_token for tokens in estimated_tokens)
        audio_seconds: Final = tuple(
            _seconds_costing_at_least(_audio_seconds(request_data), prices.input_cost_per_audio_per_second, estimate)
            for request_data, estimate in zip(requests, estimate_costs)
        )
        video_seconds: Final = tuple(
            _seconds_costing_at_least(_video_seconds(request_data), prices.input_cost_per_video_per_second, estimate)
            for request_data, estimate in zip(requests, estimate_costs)
        )
        billed_tokens: Final = tuple(
            0
            if audio is not None or video is not None
            else None
            if _is_audio_or_video_request(request_data)
            else metered
            for request_data, metered, audio, video in zip(requests, metered_input_tokens, audio_seconds, video_seconds)
        )
        input_tokens: Final = tuple(
            (estimate or 0) if tokens is None else tokens
            for estimate, tokens in zip_longest(estimated_tokens, billed_tokens)
        )
        is_image: Final = tuple(_is_image_request(request_data) for request_data in requests)
        image_count: Final = sum(is_image)
        audio_length_seconds: Final = sum(seconds for seconds in audio_seconds if seconds is not None)
        video_length_seconds: Final = sum(seconds for seconds in video_seconds if seconds is not None)
        prompt_tokens: Final = sum(input_tokens)
        prompt_tokens_details: Final = (
            PromptTokensDetailsWrapper(
                image_count=image_count or None,
                audio_length_seconds=audio_length_seconds or None,
                video_length_seconds=video_length_seconds or None,
                text_tokens=sum(
                    tokens for tokens, image in zip_longest(input_tokens, is_image, fillvalue=False) if not image
                ),
            )
            if image_count > 0 or audio_length_seconds > 0 or video_length_seconds > 0
            else None
        )
        usage: Final = Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
            prompt_tokens_details=prompt_tokens_details,
        )

        return EmbeddingResponse(data=embeddings, model=model, usage=usage)

    def transform_response(
        self,
        response_list: list[dict[str, object]],  # mutable-ok: mirrors override contract
        model: str,
        batch_data: list[dict[str, object]] | None = None,  # mutable-ok: mirrors override contract
        metered_input_tokens: Sequence[int | None] = (),
    ) -> EmbeddingResponse:
        return self._transform_response(response_list, model, batch_data, metered_input_tokens)

    def _transform_async_invoke_response(self, response: Mapping[str, object], model: str) -> EmbeddingResponse:
        """
        Transform async invoke response (invocation ARN) to OpenAI format.

        AWS async invoke returns:
        {
            "invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"
        }

        We transform this to a job-like embedding response with the ARN in hidden params.
        """
        invocation_arn: Final = response.get("invocationArn", "")

        # Create a placeholder embedding object for the job
        embedding: Final = Embedding(
            embedding=[],  # Empty embedding for async jobs
            index=0,
            object="embedding",
        )

        # Create usage object (empty for async jobs)
        usage: Final = Usage(prompt_tokens=0, total_tokens=0)

        # Create hidden params with job ID
        from litellm.types.llms.base import HiddenParams

        hidden_params: Final = HiddenParams()
        setattr(hidden_params, "_invocation_arn", invocation_arn)

        return EmbeddingResponse(
            data=[embedding],
            model=model,
            usage=usage,
            hidden_params=hidden_params,
        )

    def transform_async_invoke_response(
        self,
        response: dict[str, object],  # mutable-ok: mirrors override contract
        model: str,
    ) -> EmbeddingResponse:
        return self._transform_async_invoke_response(response, model)
