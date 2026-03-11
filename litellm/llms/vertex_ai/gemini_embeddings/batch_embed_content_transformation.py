"""
Transformation logic from OpenAI /v1/embeddings format to Google AI Studio /batchEmbedContents format. 

Why separate file? Make it easy to see how transformation works
"""

from typing import Dict, List, Optional, Tuple

from litellm.types.llms.openai import EmbeddingInput
from litellm.types.llms.vertex_ai import (
    BlobType,
    ContentType,
    EmbedContentRequest,
    FileDataType,
    PartType,
    VertexAIBatchEmbeddingsRequestBody,
    VertexAIBatchEmbeddingsResponseObject,
)
from litellm.types.utils import Embedding, EmbeddingResponse, Usage
from litellm.utils import get_formatted_prompt, token_counter

SUPPORTED_EMBEDDING_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "audio/mpeg",
    "audio/wav",
    "video/mp4",
    "video/quicktime",
    "application/pdf",
}


def _is_file_reference(s: str) -> bool:
    """Check if string is a Gemini file reference (files/...)."""
    return isinstance(s, str) and s.startswith("files/")


def _is_gcs_url(s: str) -> bool:
    """Check if string is a GCS URL (gs://...)."""
    return isinstance(s, str) and s.startswith("gs://")


def _infer_mime_type_from_gcs_url(gcs_url: str) -> str:
    """
    Infer MIME type from GCS URL file extension.
    
    Args:
        gcs_url: GCS URL like gs://bucket/path/to/file.png
    
    Returns:
        str: Inferred MIME type
    
    Raises:
        ValueError: If file extension is not supported
    """
    extension_to_mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".pdf": "application/pdf",
    }
    
    gcs_url_lower = gcs_url.lower()
    for ext, mime_type in extension_to_mime.items():
        if gcs_url_lower.endswith(ext):
            return mime_type
    
    raise ValueError(
        f"Unable to infer MIME type from GCS URL: {gcs_url}. "
        f"Supported extensions: {', '.join(extension_to_mime.keys())}"
    )


def _parse_data_url(data_url: str) -> Tuple[str, str]:
    """
    Parse a data URL to extract the media type and base64 data.
    
    Args:
        data_url: Data URL in format: data:image/jpeg;base64,/9j/4AAQ...
    
    Returns:
        tuple: (media_type, base64_data)
            media_type: e.g., "image/jpeg", "video/mp4", "audio/mpeg"
            base64_data: The base64-encoded data without the prefix
    
    Raises:
        ValueError: If data URL format is invalid or MIME type is unsupported
    """
    if not data_url.startswith("data:"):
        raise ValueError(f"Invalid data URL format: {data_url[:50]}...")
    
    if "," not in data_url:
        raise ValueError(f"Invalid data URL format (missing comma): {data_url[:50]}...")
    
    metadata, base64_data = data_url.split(",", 1)
    
    metadata = metadata[5:]
    
    if ";" in metadata:
        media_type = metadata.split(";")[0]
    else:
        media_type = metadata
    
    if media_type not in SUPPORTED_EMBEDDING_MIME_TYPES:
        raise ValueError(
            f"Unsupported MIME type for embedding: {media_type}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EMBEDDING_MIME_TYPES))}"
        )
    
    return media_type, base64_data


def _is_multimodal_input(input: EmbeddingInput) -> bool:
    """
    Check if the input contains multimodal data (data URIs, file references, or GCS URLs).
    
    Args:
        input: EmbeddingInput (str or List[str])
    
    Returns:
        bool: True if any element is a data URI, file reference, or GCS URL
    """
    if isinstance(input, str):
        input_list = [input]
    else:
        input_list = input
    
    for element in input_list:
        if isinstance(element, str):
            if element.startswith("data:") and ";base64," in element:
                return True
            if _is_file_reference(element):
                return True
            if _is_gcs_url(element):
                return True
    
    return False


def transform_openai_input_gemini_content(
    input: EmbeddingInput, model: str, optional_params: dict
) -> VertexAIBatchEmbeddingsRequestBody:
    """
    The content to embed. Only the parts.text fields will be counted.
    """
    gemini_model_name = "models/{}".format(model)
    
    gemini_params = optional_params.copy()
    if "dimensions" in gemini_params:
        gemini_params["outputDimensionality"] = gemini_params.pop("dimensions")
    
    requests: List[EmbedContentRequest] = []
    if isinstance(input, str):
        request = EmbedContentRequest(
            model=gemini_model_name,
            content=ContentType(parts=[PartType(text=input)]),
            **gemini_params
        )
        requests.append(request)
    else:
        for i in input:
            request = EmbedContentRequest(
                model=gemini_model_name,
                content=ContentType(parts=[PartType(text=i)]),
                **gemini_params
            )
            requests.append(request)

    return VertexAIBatchEmbeddingsRequestBody(requests=requests)


def transform_openai_input_gemini_embed_content(
    input: EmbeddingInput,
    model: str,
    optional_params: dict,
    resolved_files: Optional[Dict[str, Dict[str, str]]] = None,
) -> dict:
    """
    Transform OpenAI embedding input to Gemini embedContent format (multimodal).
    
    Args:
        input: EmbeddingInput (str or List[str]) with text, data URIs, or file references
        model: Model name
        optional_params: Additional parameters (taskType, outputDimensionality, etc.)
        resolved_files: Dict mapping file names (files/abc) to {mime_type, uri}
    
    Returns:
        dict: Gemini embedContent request body with content.parts
    """
    resolved_files = resolved_files or {}
    
    gemini_params = optional_params.copy()
    if "dimensions" in gemini_params:
        gemini_params["outputDimensionality"] = gemini_params.pop("dimensions")
    
    input_list = [input] if isinstance(input, str) else input
    parts: List[PartType] = []
    
    for element in input_list:
        if not isinstance(element, str):
            raise ValueError(f"Unsupported input type: {type(element)}")
        
        if element.startswith("data:") and ";base64," in element:
            mime_type, base64_data = _parse_data_url(element)
            blob: BlobType = {"mime_type": mime_type, "data": base64_data}
            parts.append(PartType(inline_data=blob))
        elif _is_gcs_url(element):
            mime_type = _infer_mime_type_from_gcs_url(element)
            file_data: FileDataType = {
                "mime_type": mime_type,
                "file_uri": element,
            }
            parts.append(PartType(file_data=file_data))
        elif _is_file_reference(element):
            if element not in resolved_files:
                raise ValueError(f"File reference {element} not resolved")
            file_info = resolved_files[element]
            file_data_ref: FileDataType = {
                "mime_type": file_info["mime_type"],
                "file_uri": file_info["uri"],
            }
            parts.append(PartType(file_data=file_data_ref))
        else:
            parts.append(PartType(text=element))
    
    request_body: dict = {
        "content": ContentType(parts=parts),
        **gemini_params,
    }
    
    return request_body


def process_embed_content_response(
    input: EmbeddingInput,
    model_response: EmbeddingResponse,
    model: str,
    response_json: dict,
) -> EmbeddingResponse:
    """
    Process Gemini embedContent response (single embedding for multimodal input).
    
    Args:
        input: Original input
        model_response: EmbeddingResponse to populate
        model: Model name
        response_json: Raw JSON response from embedContent endpoint
    
    Returns:
        EmbeddingResponse with single embedding
    """
    if "embedding" not in response_json:
        raise ValueError(f"embedContent response missing 'embedding' field: {response_json}")
    
    embedding_data = response_json["embedding"]
    
    openai_embedding = Embedding(
        embedding=embedding_data["values"],
        index=0,
        object="embedding",
    )
    
    model_response.data = [openai_embedding]
    model_response.model = model
    
    if _is_multimodal_input(input):
        prompt_tokens = 0
    else:
        input_text = get_formatted_prompt(data={"input": input}, call_type="embedding")
        prompt_tokens = token_counter(model=model, text=input_text)
    model_response.usage = Usage(
        prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
    )
    
    return model_response


def process_response(
    input: EmbeddingInput,
    model_response: EmbeddingResponse,
    model: str,
    _predictions: VertexAIBatchEmbeddingsResponseObject,
) -> EmbeddingResponse:
    openai_embeddings: List[Embedding] = []
    for embedding in _predictions["embeddings"]:
        openai_embedding = Embedding(
            embedding=embedding["values"],
            index=0,
            object="embedding",
        )
        openai_embeddings.append(openai_embedding)

    model_response.data = openai_embeddings
    model_response.model = model

    input_text = get_formatted_prompt(data={"input": input}, call_type="embedding")
    prompt_tokens = token_counter(model=model, text=input_text)
    model_response.usage = Usage(
        prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
    )

    return model_response
