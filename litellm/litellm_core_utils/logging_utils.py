from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from litellm import ModelResponse as _ModelResponse

    LiteLLMModelResponse = _ModelResponse
else:
    LiteLLMModelResponse = Any


import litellm

"""
Helper utils used for logging callbacks
"""


def convert_litellm_response_object_to_dict(response_obj: Any) -> dict:
    """
    Convert a LiteLLM response object to a dictionary

    """
    if isinstance(response_obj, dict):
        return response_obj
    for _type in litellm.ALL_LITELLM_RESPONSE_TYPES:
        if isinstance(response_obj, _type):
            return response_obj.model_dump()

    # If it's not a LiteLLM type, return the object as is
    return dict(response_obj)


def convert_litellm_response_object_to_str(
    response_obj: Union[Any, LiteLLMModelResponse]
) -> Optional[str]:
    """
    Get the string of the response object from LiteLLM

    """
    if isinstance(response_obj, litellm.ModelResponse):
        response_str = ""
        for choice in response_obj.choices:
            if isinstance(choice, litellm.Choices):
                if choice.message.content and isinstance(choice.message.content, str):
                    response_str += choice.message.content
        return response_str

    return None
