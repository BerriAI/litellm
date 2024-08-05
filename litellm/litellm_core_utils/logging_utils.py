from typing import Any

import litellm

"""
Helper utils used for logging callbacks
"""


ALL_LITELLM_RESPONSE_TYPES = [
    litellm.ModelResponse,
    litellm.EmbeddingResponse,
    litellm.ImageResponse,
    litellm.TranscriptionResponse,
    litellm.TextCompletionResponse,
]


def convert_litellm_response_object_to_dict(response_obj: Any) -> dict:
    """
    Convert a LiteLLM response object to a dictionary

    """
    if isinstance(response_obj, dict):
        return response_obj
    for _type in ALL_LITELLM_RESPONSE_TYPES:
        if isinstance(response_obj, _type):
            return response_obj.model_dump()

    # If it's not a LiteLLM type, return the object as is
    return dict(response_obj)
