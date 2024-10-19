from datetime import datetime
from typing import Dict, Optional

from litellm.types.utils import EmbeddingResponse


def convert_dict_to_embedding_response(
    model_response_object: Optional[EmbeddingResponse],
    response_object: Optional[Dict],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    hidden_params: Optional[Dict],
    _response_headers: Optional[Dict],
) -> EmbeddingResponse:
    if response_object is None:
        raise Exception("Error in response object format")

    if model_response_object is None:
        model_response_object = EmbeddingResponse()

    if "model" in response_object:
        model_response_object.model = response_object["model"]

    if "object" in response_object:
        model_response_object.object = response_object["object"]

    model_response_object.data = response_object["data"]

    if "usage" in response_object and response_object["usage"] is not None:
        model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
        model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
        model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

    if start_time is not None and end_time is not None:
        model_response_object._response_ms = (  # type: ignore
            end_time - start_time
        ).total_seconds() * 1000  # return response latency in ms like openai

    if hidden_params is not None:
        model_response_object._hidden_params = hidden_params

    if _response_headers is not None:
        model_response_object._response_headers = _response_headers

    return model_response_object
