"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock TwelveLabs Marengo /invoke and /async-invoke format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
"""

from typing import List, Optional, Union, cast

from litellm.types.llms.bedrock import (
    TWELVELABS_EMBEDDING_INPUT_TYPES,
    TwelveLabsAsyncInvokeRequest,
    TwelveLabsMarengoEmbeddingRequest,
    TwelveLabsOutputDataConfig,
    TwelveLabsS3Location,
    TwelveLabsS3OutputDataConfig,
)
from litellm.types.utils import Embedding, EmbeddingResponse, Usage


class TwelveLabsMarengoEmbeddingConfig:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html

    Supports text, image, video, and audio inputs.
    - InvokeModel: text and image inputs
    - StartAsyncInvoke: video, audio, image, and text inputs
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> List[str]:
        return [
            "encoding_format",
            "textTruncate",
            "embeddingOption",
            "startSec",
            "lengthSec",
            "useFixedLengthSec",
            "minClipSec",
            "input_type",
        ]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "encoding_format":
                # TwelveLabs doesn't have encoding_format, but we can map it to embeddingOption
                if v == "float":
                    optional_params["embeddingOption"] = ["visual-text", "visual-image"]
            elif k == "textTruncate":
                optional_params["textTruncate"] = v
            elif k == "embeddingOption":
                optional_params["embeddingOption"] = v
            elif k == "input_type":
                # Map input_type to inputType for Bedrock
                optional_params["inputType"] = v
            elif k in ["startSec", "lengthSec", "useFixedLengthSec", "minClipSec"]:
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
        model_id: Optional[str] = None,
        output_s3_uri: Optional[str] = None,
    ) -> Union[TwelveLabsMarengoEmbeddingRequest, TwelveLabsAsyncInvokeRequest]:
        """
        Transform OpenAI-style input to TwelveLabs Marengo format/async-invoke format.

        Supports:
        - Text inputs (for both invoke and async-invoke)
        - Image inputs (for both invoke and async-invoke)
        - Video inputs (async-invoke only)
        - Audio inputs (async-invoke only)
        - S3 URLs for all media types (async-invoke only)
        """
        # Get input_type or default to "text"
        input_type = cast(
            TWELVELABS_EMBEDDING_INPUT_TYPES,
            inference_params.get("inputType") or inference_params.get("input_type") or "text"
        )

        # Validate that async-invoke is used for video/audio
        if input_type in ["video", "audio"] and not async_invoke_route:
            raise ValueError(
                f"Input type '{input_type}' requires async_invoke route. "
                f"Use model format: 'bedrock/async_invoke/model_id'"
            )

        transformed_request: TwelveLabsMarengoEmbeddingRequest = {
            "inputType": input_type
        }

        if input_type == "text":
            transformed_request["inputText"] = input
            # Set default textTruncate if not specified
            if "textTruncate" not in inference_params:
                transformed_request["textTruncate"] = "end"

        elif input_type in ["image", "video", "audio"]:
            if self._is_s3_url(input):
                # S3 URL input
                s3_location: TwelveLabsS3Location = {"uri": input}
                bucket_owner = self._extract_bucket_owner_from_params(inference_params)
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
                transformed_request[k] = v  # type: ignore

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
        model_input: TwelveLabsMarengoEmbeddingRequest,
        model_id: str,
        output_s3_uri: Optional[str] = None,
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
        self, response_list: List[dict], model: str
    ) -> EmbeddingResponse:
        """
        Transform TwelveLabs response to OpenAI format.
        Handles the actual TwelveLabs response format: {"data": [{"embedding": [...]}]}
        """
        embeddings: List[Embedding] = []
        total_tokens = 0

        for response in response_list:
            # TwelveLabs response format has a "data" field containing the embeddings
            if "data" in response and isinstance(response["data"], list):
                for item in response["data"]:
                    if "embedding" in item:
                        # Single embedding response
                        embedding = Embedding(
                            embedding=item["embedding"],
                            index=len(embeddings),
                            object="embedding",
                        )
                        embeddings.append(embedding)

                        # Estimate token count (rough approximation)
                        if "inputTextTokenCount" in item:
                            total_tokens += item["inputTextTokenCount"]
                        else:
                            # Rough estimate: 1 token per 4 characters for text, or use embedding size
                            total_tokens += len(item["embedding"]) // 4
            elif "embedding" in response:
                # Direct embedding response (fallback for other formats)
                embedding = Embedding(
                    embedding=response["embedding"],
                    index=len(embeddings),
                    object="embedding",
                )
                embeddings.append(embedding)

                # Estimate token count (rough approximation)
                if "inputTextTokenCount" in response:
                    total_tokens += response["inputTextTokenCount"]
                else:
                    # Rough estimate: 1 token per 4 characters for text
                    total_tokens += len(response.get("inputText", "")) // 4
            elif "embeddings" in response:
                # Multiple embeddings response (from video/audio)
                for i, emb in enumerate(response["embeddings"]):
                    embedding = Embedding(
                        embedding=emb["embedding"],
                        index=len(embeddings),
                        object="embedding",
                    )
                    embeddings.append(embedding)
                    total_tokens += len(emb["embedding"]) // 4  # Rough estimate

        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)

        return EmbeddingResponse(data=embeddings, model=model, usage=usage)

    def _transform_async_invoke_response(
        self, response: dict, model: str
    ) -> EmbeddingResponse:
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
        invocation_arn = response.get("invocationArn", "")

        # Create a placeholder embedding object for the job
        embedding = Embedding(
            embedding=[],  # Empty embedding for async jobs
            index=0,
            object="embedding",
        )

        # Create usage object (empty for async jobs)
        usage = Usage(prompt_tokens=0, total_tokens=0)

        # Create hidden params with job ID
        from litellm.types.llms.base import HiddenParams

        hidden_params = HiddenParams()
        setattr(hidden_params, "_invocation_arn", invocation_arn)

        return EmbeddingResponse(
            data=[embedding],
            model=model,
            usage=usage,
            hidden_params=hidden_params,
        )
