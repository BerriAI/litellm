"""
Transformation logic from OpenAI /v1/embeddings format to Bedrock TwelveLabs Marengo /invoke format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
"""

from typing import List

from litellm.types.llms.bedrock import (
    TwelveLabsMarengoEmbeddingRequest,
)
from litellm.types.utils import Embedding, EmbeddingResponse, Usage
from litellm.utils import get_base64_str, is_base64_encoded


class TwelveLabsMarengoEmbeddingConfig:
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html

    Supports text and image inputs for Phase 1.
    Video and audio support will be added in Phase 2.
    """

    def __init__(self) -> None:
        pass

    def get_supported_openai_params(self) -> List[str]:
        return ["encoding_format", "textTruncate", "embeddingOption"]

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
        return optional_params

    def _transform_request(
        self, input: str, inference_params: dict
    ) -> TwelveLabsMarengoEmbeddingRequest:
        """
        Transform OpenAI-style input to TwelveLabs Marengo format.
        Phase 1: Supports text and image inputs only.
        """
        # Check if input is base64 encoded image
        is_encoded = is_base64_encoded(input)

        if is_encoded:
            # Image input
            b64_str = get_base64_str(input)
            transformed_request = TwelveLabsMarengoEmbeddingRequest(
                inputType="image", mediaSource={"base64String": b64_str}
            )
        else:
            # Text input
            transformed_request = TwelveLabsMarengoEmbeddingRequest(
                inputType="text", inputText=input
            )

            # Set default textTruncate if not specified
            if "textTruncate" not in inference_params:
                transformed_request["textTruncate"] = "end"

        # Apply any additional inference parameters
        for k, v in inference_params.items():
            if k not in [
                "inputType",
                "inputText",
                "mediaSource",
            ]:  # Don't override core fields
                transformed_request[k] = v  # type: ignore

        return transformed_request

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
