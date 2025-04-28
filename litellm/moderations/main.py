from typing import Dict, Optional

import openai

import litellm

############ Instantiated classes ############
from litellm.main import openai_chat_completions as openai_api_client
from litellm.moderations.utils import ModerationAPIUtils
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIModerationResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client

############# Moderations API #######################


@client
def moderation(
    input: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> OpenAIModerationResponse:
    # only supports open ai for now
    api_key = (
        api_key
        or litellm.api_key
        or litellm.openai_key
        or get_secret_str("OPENAI_API_KEY")
    )

    optional_params = GenericLiteLLMParams(**kwargs)
    try:
        (
            model,
            custom_llm_provider,
            _dynamic_api_key,
            _dynamic_api_base,
        ) = litellm.get_llm_provider(
            model=model or "",
            custom_llm_provider=custom_llm_provider,
            api_base=optional_params.api_base,
            api_key=optional_params.api_key,
        )
    except litellm.BadRequestError:
        # `model` is optional field for moderation - get_llm_provider will throw BadRequestError if model is not set / not recognized
        pass

    openai_client = kwargs.get("client", None)
    if openai_client is None:
        openai_client = openai.OpenAI(
            api_key=api_key,
        )

    # update litellm_logging_obj with request params (used for logging the correct values in the logging callbacks)
    ModerationAPIUtils.init_litellm_logging_obj_for_moderations_call(
        custom_llm_provider=custom_llm_provider,
        model=model,
        user=kwargs.get("user", None),
        **kwargs,
    )

    if model is not None:
        response = openai_client.moderations.create(input=input, model=model)
    else:
        response = openai_client.moderations.create(input=input)

    response_dict: Dict = response.model_dump()
    return litellm.utils.LiteLLMResponseObjectHandler.convert_to_moderation_response(
        response_object=response_dict,
    )


@client
async def amoderation(
    input: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> OpenAIModerationResponse:
    from openai import AsyncOpenAI

    ############################################################
    ######### Pre-Request Setup #################################
    ############################################################
    # only supports open ai for now
    api_key = (
        api_key
        or litellm.api_key
        or litellm.openai_key
        or get_secret_str("OPENAI_API_KEY")
    )
    openai_client = kwargs.get("client", None)
    if openai_client is None or not isinstance(openai_client, AsyncOpenAI):
        # call helper to get OpenAI client
        # _get_openai_client maintains in-memory caching logic for OpenAI clients
        _openai_client: AsyncOpenAI = openai_api_client._get_openai_client(  # type: ignore
            is_async=True,
            api_key=api_key,
        )
    else:
        _openai_client = openai_client

    optional_params = GenericLiteLLMParams(**kwargs)
    try:
        (
            model,
            custom_llm_provider,
            _dynamic_api_key,
            _dynamic_api_base,
        ) = litellm.get_llm_provider(
            model=model or "",
            custom_llm_provider=custom_llm_provider,
            api_base=optional_params.api_base,
            api_key=optional_params.api_key,
        )
    except litellm.BadRequestError:
        # `model` is optional field for moderation - get_llm_provider will throw BadRequestError if model is not set / not recognized
        pass

    # update litellm_logging_obj with request params (used for logging the correct values in the logging callbacks)
    ModerationAPIUtils.init_litellm_logging_obj_for_moderations_call(
        custom_llm_provider=custom_llm_provider,
        model=model,
        user=kwargs.get("user", None),
        **kwargs,
    )

    ############################################################
    ######### Make API Call #################################
    ############################################################

    if model is not None:
        response = await _openai_client.moderations.create(input=input, model=model)
    else:
        response = await _openai_client.moderations.create(input=input)

    ############################################################
    ######### Post-Request Processing #################################
    ############################################################
    response_dict: Dict = response.model_dump()
    return litellm.utils.LiteLLMResponseObjectHandler.convert_to_moderation_response(
        response_object=response_dict,
    )
