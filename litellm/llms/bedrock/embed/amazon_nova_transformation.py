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

from typing import List, Optional

from litellm.types.utils import Embedding, EmbeddingResponse, Usage


class AmazonNovaEmbeddingConfig:
    """
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/nova-embed.html
    
    Amazon Nova Multimodal Embeddings supports:
    - Text, image, video, and audio inputs
    - Synchronous (InvokeModel) and asynchronous (StartAsyncInvoke) APIs
    - Multiple embedding purposes and dimensions
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> List[str]:
        return [
            "dimensions",
        ]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        """Map OpenAI-style parameters to Nova parameters."""
        for k, v in non_default_params.items():
            if k == "dimensions":
                # Map OpenAI dimensions to Nova embedding_dimension
                optional_params["embedding_dimension"] = v
            elif k in self.get_supported_openai_params():
                optional_params[k] = v
        return optional_params

    def _transform_request(
        self,
        input: str,
        inference_params: dict,
        async_invoke_route: bool = False,
        model_id: Optional[str] = None,
        output_s3_uri: Optional[str] = None,
    ) -> dict:
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
        task_type = "SEGMENTED_EMBEDDING" if async_invoke_route else "SINGLE_EMBEDDING"
        
        # Build the base request structure
        request: dict = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": task_type,
        }
        
        # Start with inference_params (user-provided params)
        embedding_params = inference_params.copy()
        
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
        
        # For text input, add basic text structure if user hasn't provided text/image/video/audio
        if "text" not in embedding_params and "image" not in embedding_params and "video" not in embedding_params and "audio" not in embedding_params:
            # Default to text if no modality specified
            if input.startswith("s3://"):
                embedding_params["text"] = {
                    "source": {"s3Location": {"uri": input}},
                    "truncationMode": "END"  # Required by Nova API
                }
            else:
                embedding_params["text"] = {
                    "value": input,
                    "truncationMode": "END"  # Required by Nova API
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

    def _wrap_async_invoke_request(
        self,
        model_input: dict,
        model_id: str,
        output_s3_uri: Optional[str] = None,
    ) -> dict:
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
            "outputDataConfig": {
                "s3OutputDataConfig": {
                    "s3Uri": output_s3_uri
                }
            },
        }

    def _transform_response(
        self, response_list: List[dict], model: str
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
        embeddings: List[Embedding] = []
        total_tokens = 0
        
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
                        
                        # Estimate token count
                        # For text, use truncatedCharLength if available
                        if "truncatedCharLength" in item:
                            total_tokens += item["truncatedCharLength"] // 4
                        else:
                            # Rough estimate based on embedding dimension
                            total_tokens += len(item["embedding"]) // 4
            elif "embedding" in response:
                # Direct embedding response (fallback)
                embedding = Embedding(
                    embedding=response["embedding"],
                    index=len(embeddings),
                    object="embedding",
                )
                embeddings.append(embedding)
                total_tokens += len(response["embedding"]) // 4
        
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
        
        We transform this to a job-like embedding response with the ARN in hidden params.
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

