"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock TwelveLabs Marengo /invoke and /async-invoke format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
Marengo 3.0 docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
"""

from collections.abc import Mapping
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import assert_never

from litellm.llms.bedrock.embed.twelvelabs_marengo_3_transformation import (
    build_marengo_3_request,
    is_marengo_3_model,
)
from litellm.types.llms.bedrock import (
    TWELVELABS_EMBEDDING_INPUT_TYPES,
    TWELVELABS_MARENGO_3_INPUT_TYPES,
    TwelveLabsAsyncInvokeRequest,
    TwelveLabsMarengo3EmbeddingRequest,
    TwelveLabsMarengoEmbeddingRequest,
    TwelveLabsOutputDataConfig,
    TwelveLabsS3Location,
    TwelveLabsS3OutputDataConfig,
)
from litellm.types.utils import Embedding, EmbeddingResponse, PromptTokensDetailsWrapper, Usage


class MarengoEmbeddingItem(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    embedding: tuple[float, ...]


class MarengoInvokeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data: tuple[MarengoEmbeddingItem, ...] = ()
    embedding: tuple[float, ...] | None = None
    embeddings: tuple[MarengoEmbeddingItem, ...] = ()

    def vectors(self) -> tuple[tuple[float, ...], ...]:
        if self.data:
            return tuple(item.embedding for item in self.data)
        if self.embedding is not None:
            return (self.embedding,)
        return tuple(item.embedding for item in self.embeddings)


class MarengoBilledMultiInput(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    inputText: str | None = None
    mediaSources: tuple[Mapping[str, object], ...] = ()


class MarengoBilledRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    inputType: TWELVELABS_MARENGO_3_INPUT_TYPES | None = None
    multi_input: MarengoBilledMultiInput | None = None


INVOKE_RESPONSES: Final = TypeAdapter(tuple[MarengoInvokeResponse, ...])
BILLED_REQUESTS: Final = TypeAdapter(tuple[MarengoBilledRequest, ...])


def _billed_units(request: MarengoBilledRequest) -> tuple[int, int]:
    input_type: Final = request.inputType
    match input_type:
        case "text":
            return (1, 0)
        case "image":
            return (0, 1)
        case "text_image":
            return (1, 1)
        case "multi_input":
            multi_input: Final = request.multi_input or MarengoBilledMultiInput()
            return (1 if multi_input.inputText else 0, len(multi_input.mediaSources))
        case "video" | "audio" | None:
            return (0, 0)
        case _:
            assert_never(input_type)


def _billed_usage(batch_data: list[dict] | None) -> Usage:
    units: Final = tuple(_billed_units(request) for request in BILLED_REQUESTS.validate_python(batch_data or ()))
    query_count: Final = sum(text_requests for text_requests, _ in units)
    image_count: Final = sum(images for _, images in units)
    details: Final = (
        PromptTokensDetailsWrapper(query_count=query_count or None, image_count=image_count or None)
        if query_count or image_count
        else None
    )
    return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=details)


class TwelveLabsMarengoEmbeddingConfig:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html

    Supports text, image, video, and audio inputs.
    - InvokeModel: text and image inputs
    - StartAsyncInvoke: video, audio, image, and text inputs

    Marengo 3.0 (model ids containing "marengo-embed-3") nests the input under a key named after inputType and
    adds the text_image and multi_input input types; that payload is built by build_marengo_3_request.
    """

    def __init__(self, model: str | None = None) -> None:
        self.is_marengo_3: Final = is_marengo_3_model(model)

    def get_supported_openai_params(self) -> list[str]:
        return [
            "encoding_format",
            "textTruncate",
            "embeddingOption",
            "startSec",
            "lengthSec",
            "useFixedLengthSec",
            "minClipSec",
            "input_type",
            "endSec",
            "segmentation",
            "embeddingType",
            "embeddingScope",
            "inferenceId",
            "media_source",
            "media_sources",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                # TwelveLabs doesn't have encoding_format, but we can map it to embeddingOption
                if v == "float" and not self.is_marengo_3:
                    optional_params["embeddingOption"] = ["visual-text", "visual-image"]
            elif k == "textTruncate":
                optional_params["textTruncate"] = v
            elif k == "embeddingOption":
                optional_params["embeddingOption"] = v
            elif k == "input_type":
                # Map input_type to inputType for Bedrock
                optional_params["inputType"] = v
            elif k in (
                "startSec",
                "lengthSec",
                "useFixedLengthSec",
                "minClipSec",
                "endSec",
                "segmentation",
                "embeddingType",
                "embeddingScope",
                "inferenceId",
                "media_source",
                "media_sources",
            ):
                optional_params[k] = v
        return optional_params

    def _extract_bucket_owner_from_params(self, inference_params: dict) -> str:
        """
        Extract bucket owner from inference parameters.
        """
        return inference_params.get("bucketOwner", "")

    def _is_s3_url(self, input: str) -> bool:
        """Check if input is an S3 URL."""
        return input.startswith("s3://")

    def _transform_request(
        self,
        input: str,
        inference_params: dict,
        async_invoke_route: bool = False,
        model_id: str | None = None,
        output_s3_uri: str | None = None,
    ) -> TwelveLabsMarengoEmbeddingRequest | TwelveLabsMarengo3EmbeddingRequest | TwelveLabsAsyncInvokeRequest:
        """
        Transform OpenAI-style input to TwelveLabs Marengo format/async-invoke format.

        Supports:
        - Text inputs (for both invoke and async-invoke)
        - Image inputs (for both invoke and async-invoke)
        - Video inputs (async-invoke only)
        - Audio inputs (async-invoke only)
        - S3 URLs for all media types (async-invoke only)
        - Marengo 3.0 only: text_image and multi_input inputs (nested payload)
        """
        input_type: Final = cast(
            TWELVELABS_EMBEDDING_INPUT_TYPES,
            inference_params.get("inputType") or inference_params.get("input_type") or "text",
        )

        if input_type in ["video", "audio"] and not async_invoke_route:
            raise ValueError(
                f"Input type '{input_type}' requires async_invoke route. "
                f"Use model format: 'bedrock/async_invoke/model_id'"
            )

        if self.is_marengo_3:
            marengo_3_request: Final = build_marengo_3_request(input=input, inference_params=inference_params)
            if async_invoke_route and model_id:
                return self._wrap_async_invoke_request(
                    model_input=marengo_3_request, model_id=model_id, output_s3_uri=output_s3_uri
                )
            return marengo_3_request

        transformed_request: Final[TwelveLabsMarengoEmbeddingRequest] = {"inputType": input_type}

        if input_type == "text":
            transformed_request["inputText"] = input
            # Set default textTruncate if not specified
            if "textTruncate" not in inference_params:
                transformed_request["textTruncate"] = "end"

        elif input_type in ["image", "video", "audio"]:
            if self._is_s3_url(input):
                # S3 URL input
                s3_location: Final[TwelveLabsS3Location] = {"uri": input}
                bucket_owner: Final = self._extract_bucket_owner_from_params(inference_params)
                if bucket_owner:
                    s3_location["bucketOwner"] = bucket_owner

                transformed_request["mediaSource"] = {"s3Location": s3_location}
            else:
                # Base64 encoded input
                if input.startswith("data:"):
                    # Extract base64 data from data URL
                    b64_str = input.split(",", 1)[1] if "," in input else input
                else:
                    # Direct base64 string
                    from litellm.utils import get_base64_str

                    b64_str = get_base64_str(input)

                transformed_request["mediaSource"] = {"base64String": b64_str}

        # Apply any additional inference parameters
        for k, v in inference_params.items():
            if k not in [
                "inputType",
                "input_type",  # Exclude both camelCase and snake_case
                "inputText",
                "mediaSource",
                "bucketOwner",  # Don't include bucketOwner in the request
            ]:  # Don't override core fields
                transformed_request[k] = v

        # If async invoke route, wrap in the async invoke format
        if async_invoke_route and model_id:
            return self._wrap_async_invoke_request(
                model_input=transformed_request,
                model_id=model_id,
                output_s3_uri=output_s3_uri,
            )

        return transformed_request

    def _wrap_async_invoke_request(
        self,
        model_input: TwelveLabsMarengoEmbeddingRequest | TwelveLabsMarengo3EmbeddingRequest,
        model_id: str,
        output_s3_uri: str | None = None,
    ) -> TwelveLabsAsyncInvokeRequest:
        """
        Wrap the transformed request in the correct AWS Bedrock async invoke format.

        Args:
            model_input: The transformed TwelveLabs Marengo embedding request
            model_id: The model identifier (without async_invoke prefix)
            output_s3_uri: Optional S3 URI for output data config

        Returns:
            TwelveLabsAsyncInvokeRequest: The wrapped async invoke request
        """
        import urllib.parse

        # Clean the model ID
        unquoted_model_id = urllib.parse.unquote(model_id)
        if unquoted_model_id.startswith("async_invoke/"):
            unquoted_model_id = unquoted_model_id.replace("async_invoke/", "")

        # Validate that the S3 URI is not empty
        if not output_s3_uri or output_s3_uri.strip() == "":
            raise ValueError("output_s3_uri cannot be empty for async invoke requests")

        return TwelveLabsAsyncInvokeRequest(
            modelId=unquoted_model_id,
            modelInput=model_input,
            outputDataConfig=TwelveLabsOutputDataConfig(
                s3OutputDataConfig=TwelveLabsS3OutputDataConfig(s3Uri=output_s3_uri)
            ),
        )

    def _transform_response(
        self, response_list: list[dict], model: str, batch_data: list[dict] | None = None
    ) -> EmbeddingResponse:
        vectors: Final = tuple(
            vector for response in INVOKE_RESPONSES.validate_python(response_list) for vector in response.vectors()
        )
        embeddings: Final = [
            Embedding(embedding=list(vector), index=index, object="embedding") for index, vector in enumerate(vectors)
        ]
        return EmbeddingResponse(data=embeddings, model=model, usage=_billed_usage(batch_data))

    def _transform_async_invoke_response(self, response: dict, model: str) -> EmbeddingResponse:
        """
        Transform async invoke response (invocation ARN) to OpenAI format.

        AWS async invoke returns:
        {
            "invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"
        }

        We transform this to a job-like embedding response:
        {
            "object": "list",
            "data": [
                {
                    "object": "embedding_job_id:1234567890",
                    "embedding": [],
                    "index": 0
                }
            ],
            "model": "model",
            "usage": {}
        }
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
