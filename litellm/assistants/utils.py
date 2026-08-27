from typing import Final

import litellm

from ..exceptions import UnsupportedParamsError
from ..types.llms.openai import *


def get_optional_params_add_message(
    role: str | None,
    content: str | list[MessageContentTextObject | MessageContentImageFileObject | MessageContentImageURLObject] | None,
    attachments: list[Attachment] | None,
    metadata: dict | None,
    custom_llm_provider: str,
    **kwargs,
):
    """
    Azure doesn't support 'attachments' for creating a message

    Reference - https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-reference-messages?tabs=python#create-message
    """
    passed_params: Final = locals()
    custom_llm_provider = passed_params.pop("custom_llm_provider")
    special_params: Final = passed_params.pop("kwargs")
    for k, v in special_params.items():
        passed_params[k] = v

    default_params: Final = {
        "role": None,
        "content": None,
        "attachments": None,
        "metadata": None,
    }

    non_default_params = {k: v for k, v in passed_params.items() if (k in default_params and v != default_params[k])}
    optional_params = {}

    ## raise exception if non-default value passed for non-openai/azure embedding calls
    def _check_valid_arg(supported_params):
        if len(non_default_params.keys()) > 0:
            keys: Final = list(non_default_params.keys())
            for k in keys:
                if litellm.drop_params is True and k not in supported_params:  # drop the unsupported non-default values
                    non_default_params.pop(k, None)
                elif k not in supported_params:
                    raise litellm.utils.UnsupportedParamsError(
                        status_code=500,
                        message=f"k={k}, not supported by {custom_llm_provider}. Supported params={supported_params}. To drop it from the call, set `litellm.drop_params = True`.",
                    )
            return non_default_params

    if custom_llm_provider == "openai":
        optional_params = non_default_params
    elif custom_llm_provider == "azure":
        supported_params: Final = litellm.AzureOpenAIAssistantsAPIConfig().get_supported_openai_create_message_params()
        _check_valid_arg(supported_params=supported_params)
        optional_params = litellm.AzureOpenAIAssistantsAPIConfig().map_openai_params_create_message_params(
            non_default_params=non_default_params, optional_params=optional_params
        )
    for k in passed_params:
        if k not in default_params:
            optional_params[k] = passed_params[k]
    return optional_params


def get_optional_params_image_gen(
    n: int | None = None,
    quality: str | None = None,
    response_format: str | None = None,
    size: str | None = None,
    style: str | None = None,
    user: str | None = None,
    custom_llm_provider: str | None = None,
    **kwargs,
):
    # retrieve all parameters passed to the function
    passed_params: Final = locals()
    custom_llm_provider = passed_params.pop("custom_llm_provider")
    special_params: Final = passed_params.pop("kwargs")
    for k, v in special_params.items():
        passed_params[k] = v

    default_params: Final = {
        "n": None,
        "quality": None,
        "response_format": None,
        "size": None,
        "style": None,
        "user": None,
    }

    non_default_params = {k: v for k, v in passed_params.items() if (k in default_params and v != default_params[k])}
    optional_params = {}

    ## raise exception if non-default value passed for non-openai/azure embedding calls
    def _check_valid_arg(supported_params):
        if len(non_default_params.keys()) > 0:
            keys: Final = list(non_default_params.keys())
            for k in keys:
                if litellm.drop_params is True and k not in supported_params:  # drop the unsupported non-default values
                    non_default_params.pop(k, None)
                elif k not in supported_params:
                    raise UnsupportedParamsError(
                        status_code=500,
                        message=f"Setting user/encoding format is not supported by {custom_llm_provider}. To drop it from the call, set `litellm.drop_params = True`.",
                    )
            return non_default_params

    if (
        custom_llm_provider == "openai"
        or custom_llm_provider == "azure"
        or custom_llm_provider in litellm.openai_compatible_providers
    ):
        optional_params = non_default_params
    elif custom_llm_provider == "bedrock":
        supported_params = ["size"]
        _check_valid_arg(supported_params=supported_params)
        if size is not None:
            width, height = size.split("x")
            optional_params["width"] = int(width)
            optional_params["height"] = int(height)
    elif custom_llm_provider == "vertex_ai":
        supported_params = ["n"]
        """
        All params here: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/imagegeneration?project=adroit-crow-413218
        """
        _check_valid_arg(supported_params=supported_params)
        if n is not None:
            optional_params["sampleCount"] = int(n)

    for k in passed_params:
        if k not in default_params:
            optional_params[k] = passed_params[k]
    return optional_params
