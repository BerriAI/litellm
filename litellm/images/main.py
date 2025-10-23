import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, List, Literal, Optional, Union, cast, overload

import httpx

import litellm
from litellm import Logging, client, exception_type, get_litellm_params
from litellm.constants import DEFAULT_IMAGE_ENDPOINT_MODEL
from litellm.constants import request_timeout as DEFAULT_REQUEST_TIMEOUT
from litellm.exceptions import LiteLLMUnknownProvider
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.mock_functions import mock_image_generation
from litellm.llms.base_llm import BaseImageEditConfig, BaseImageGenerationConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.custom_llm import CustomLLM

#################### Initialize provider clients ####################
llm_http_handler: BaseLLMHTTPHandler = BaseLLMHTTPHandler()
from litellm.main import (
    azure_chat_completions,
    base_llm_aiohttp_handler,
    base_llm_http_handler,
    bedrock_image_generation,
    openai_chat_completions,
    openai_image_variations,
    vertex_image_generation,
)

###########################################
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import ImageGenerationRequestQuality
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    LITELLM_IMAGE_VARIATION_PROVIDERS,
    FileTypes,
    LlmProviders,
    all_litellm_params,
)
from litellm.utils import (
    ImageResponse,
    ProviderConfigManager,
    get_llm_provider,
    get_optional_params_image_gen,
)

from .utils import ImageEditRequestUtils


##### Image Generation #######################
@client
async def aimage_generation(*args, **kwargs) -> ImageResponse:
    """
    Asynchronously calls the `image_generation` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `image_generation` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `image_generation` function.

    Returns:
    - `response` (Any): The response returned by the `image_generation` function.
    """
    loop = asyncio.get_event_loop()
    model = args[0] if len(args) > 0 else kwargs["model"]
    ### PASS ARGS TO Image Generation ###
    kwargs["aimg_generation"] = True
    custom_llm_provider = None
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(image_generation, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=kwargs.get("api_base", None)
        )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)

        response: Optional[ImageResponse] = None
        if isinstance(init_response, dict):
            response = ImageResponse(**init_response)
        elif isinstance(init_response, ImageResponse):  ## CACHING SCENARIO
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response  # type: ignore

        if response is None:
            raise ValueError(
                "Unable to get Image Response. Please pass a valid llm_provider."
            )

        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


# fmt: off

# Overload for when aimg_generation=True (returns Coroutine)
@overload
def image_generation(
    prompt: str,
    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[Union[str, ImageGenerationRequestQuality]] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    aimg_generation: Literal[True],
    **kwargs,
) -> Coroutine[Any, Any, ImageResponse]: 
    ...



# Overload for when aimg_generation=False or not specified (returns ImageResponse)
@overload
def image_generation(
    prompt: str,
    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[Union[str, ImageGenerationRequestQuality]] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    *,
    aimg_generation: Literal[False] = False,
    **kwargs,
) -> ImageResponse: 
    ...

# fmt: on


@client
def image_generation(  # noqa: PLR0915
    prompt: str,
    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[Union[str, ImageGenerationRequestQuality]] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    custom_llm_provider=None,
    **kwargs,
) -> Union[
    ImageResponse,
    Coroutine[Any, Any, ImageResponse],
]:
    """
    Maps the https://api.openai.com/v1/images/generations endpoint.

    Currently supports just Azure + OpenAI.
    """
    try:
        args = locals()
        aimg_generation = kwargs.get("aimg_generation", False)
        litellm_call_id = kwargs.get("litellm_call_id", None)
        logger_fn = kwargs.get("logger_fn", None)
        mock_response: Optional[str] = kwargs.get("mock_response", None)  # type: ignore
        proxy_server_request = kwargs.get("proxy_server_request", None)
        azure_ad_token_provider = kwargs.get("azure_ad_token_provider", None)
        model_info = kwargs.get("model_info", None)
        metadata = kwargs.get("metadata", {})
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        client = kwargs.get("client", None)
        extra_headers = kwargs.get("extra_headers", None)
        headers: dict = kwargs.get("headers", None) or {}
        base_model = kwargs.get("base_model", None)
        if extra_headers is not None:
            headers.update(extra_headers)
        model_response: ImageResponse = litellm.utils.ImageResponse()
        dynamic_api_key: Optional[str] = None
        if model is not None or custom_llm_provider is not None:
            model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
                model=model,  # type: ignore
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
            )
        else:
            model = "dall-e-2"
            custom_llm_provider = "openai"  # default to dall-e-2 on openai
        model_response._hidden_params["model"] = model
        openai_params = [
            "user",
            "request_timeout",
            "api_base",
            "api_version",
            "api_key",
            "deployment_id",
            "organization",
            "base_url",
            "default_headers",
            "timeout",
            "max_retries",
            "n",
            "quality",
            "size",
            "style",
        ]
        litellm_params = all_litellm_params
        default_params = openai_params + litellm_params
        non_default_params = {
            k: v for k, v in kwargs.items() if k not in default_params
        }  # model-specific params - pass them straight to the model/provider

        image_generation_config: Optional[BaseImageGenerationConfig] = None
        if (
            custom_llm_provider is not None
            and custom_llm_provider in LlmProviders._member_map_.values()
        ):
            image_generation_config = (
                ProviderConfigManager.get_provider_image_generation_config(
                    model=base_model or model,
                    provider=LlmProviders(custom_llm_provider),
                )
            )

        optional_params = get_optional_params_image_gen(
            model=base_model or model,
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            style=style,
            user=user,
            custom_llm_provider=custom_llm_provider,
            provider_config=image_generation_config,
            **non_default_params,
        )

        litellm_params_dict = get_litellm_params(**kwargs)

        logging: Logging = litellm_logging_obj
        logging.update_environment_variables(
            model=model,
            user=user,
            optional_params=optional_params,
            litellm_params={
                "timeout": timeout,
                "azure": False,
                "litellm_call_id": litellm_call_id,
                "logger_fn": logger_fn,
                "proxy_server_request": proxy_server_request,
                "model_info": model_info,
                "metadata": metadata,
                "preset_cache_key": None,
                "stream_response": {},
            },
            custom_llm_provider=custom_llm_provider,
        )
        if "custom_llm_provider" not in logging.model_call_details:
            logging.model_call_details["custom_llm_provider"] = custom_llm_provider
        if mock_response is not None:
            return mock_image_generation(model=model, mock_response=mock_response)

        if custom_llm_provider == "azure":
            # azure configs
            api_type = get_secret_str("AZURE_API_TYPE") or "azure"

            api_base = api_base or litellm.api_base or get_secret_str("AZURE_API_BASE")

            api_version = (
                api_version
                or litellm.api_version
                or get_secret_str("AZURE_API_VERSION")
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret_str("AZURE_OPENAI_API_KEY")
                or get_secret_str("AZURE_API_KEY")
            )

            azure_ad_token = optional_params.pop(
                "azure_ad_token", None
            ) or get_secret_str("AZURE_AD_TOKEN")

            default_headers = {
                "Content-Type": "application/json",
                "api-key": api_key,
            }
            for k, v in default_headers.items():
                if k not in headers:
                    headers[k] = v

            model_response = azure_chat_completions.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                azure_ad_token=azure_ad_token,
                azure_ad_token_provider=azure_ad_token_provider,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                api_version=api_version,
                aimg_generation=aimg_generation,
                client=client,
                headers=headers,
                litellm_params=litellm_params_dict,
            )
        #########################################################
        # Providers using llm_http_handler
        #########################################################
        elif custom_llm_provider in (
            litellm.LlmProviders.RECRAFT,
            litellm.LlmProviders.AIML,
            litellm.LlmProviders.GEMINI,
        ):
            if image_generation_config is None:
                raise ValueError(
                    f"image generation config is not supported for {custom_llm_provider}"
                )

            return llm_http_handler.image_generation_handler(
                api_key=api_key,
                model=model,
                prompt=prompt,
                image_generation_provider_config=image_generation_config,
                image_generation_optional_request_params=optional_params,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params_dict,
                logging_obj=litellm_logging_obj,
                timeout=timeout,
                client=client,
            )
        elif custom_llm_provider == "azure_ai":
            from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

            api_base = AzureFoundryModelInfo.get_api_base(api_base)
            api_key = AzureFoundryModelInfo.get_api_key(api_key)
            if extra_headers is not None:
                optional_params["extra_headers"] = extra_headers

            default_headers = {
                "Content-Type": "application/json",
                "api-key": api_key,
            }
            for k, v in default_headers.items():
                if k not in headers:
                    headers[k] = v

            model_response = azure_chat_completions.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                azure_ad_token=None,
                azure_ad_token_provider=azure_ad_token_provider,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                api_version=api_version,
                aimg_generation=aimg_generation,
                client=client,
                headers=headers,
                litellm_params=litellm_params_dict,
            )
        elif (
            custom_llm_provider == "openai"
            or custom_llm_provider == LlmProviders.LITELLM_PROXY.value
            or custom_llm_provider in litellm.openai_compatible_providers
        ):
            model_response = openai_chat_completions.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                api_key=api_key or dynamic_api_key,
                api_base=api_base,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                aimg_generation=aimg_generation,
                client=client,
            )
        elif custom_llm_provider == "bedrock":
            if model is None:
                raise Exception("Model needs to be set for bedrock")
            model_response = bedrock_image_generation.image_generation(  # type: ignore
                model=model,
                prompt=prompt,
                timeout=timeout,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                aimg_generation=aimg_generation,
                client=client,
                api_base=api_base,
                api_key=api_key,
            )
        elif custom_llm_provider == "vertex_ai":
            vertex_ai_project = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret_str("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret_str("VERTEXAI_LOCATION")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret_str("VERTEXAI_CREDENTIALS")
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret_str("VERTEXAI_API_BASE")
                or get_secret_str("VERTEX_API_BASE")
            )

            model_response = vertex_image_generation.image_generation(
                model=model,
                prompt=prompt,
                timeout=timeout,
                logging_obj=litellm_logging_obj,
                optional_params=optional_params,
                model_response=model_response,
                vertex_project=vertex_ai_project,
                vertex_location=vertex_ai_location,
                vertex_credentials=vertex_credentials,
                aimg_generation=aimg_generation,
                api_base=api_base,
                client=client,
            )
        elif (
            custom_llm_provider in litellm._custom_providers
        ):  # Assume custom LLM provider
            # Get the Custom Handler
            custom_handler: Optional[CustomLLM] = None
            for item in litellm.custom_provider_map:
                if item["provider"] == custom_llm_provider:
                    custom_handler = item["custom_handler"]

            if custom_handler is None:
                raise LiteLLMUnknownProvider(
                    model=model, custom_llm_provider=custom_llm_provider
                )

            ## ROUTE LLM CALL ##
            if aimg_generation is True:
                async_custom_client: Optional[AsyncHTTPHandler] = None
                if client is not None and isinstance(client, AsyncHTTPHandler):
                    async_custom_client = client

                ## CALL FUNCTION
                model_response = custom_handler.aimage_generation(  # type: ignore
                    model=model,
                    prompt=prompt,
                    api_key=api_key,
                    api_base=api_base,
                    model_response=model_response,
                    optional_params=optional_params,
                    logging_obj=litellm_logging_obj,
                    timeout=timeout,
                    client=async_custom_client,
                )
            else:
                custom_client: Optional[HTTPHandler] = None
                if client is not None and isinstance(client, HTTPHandler):
                    custom_client = client

                ## CALL FUNCTION
                model_response = custom_handler.image_generation(
                    model=model,
                    prompt=prompt,
                    api_key=api_key,
                    api_base=api_base,
                    model_response=model_response,
                    optional_params=optional_params,
                    logging_obj=litellm_logging_obj,
                    timeout=timeout,
                    client=custom_client,
                )

        return model_response
    except Exception as e:
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=locals(),
            extra_kwargs=kwargs,
        )


@client
async def aimage_variation(*args, **kwargs) -> ImageResponse:
    """
    Asynchronously calls the `image_variation` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `image_variation` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `image_variation` function.

    Returns:
    - `response` (Any): The response returned by the `image_variation` function.
    """
    loop = asyncio.get_event_loop()
    model = kwargs.get("model", None)
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    ### PASS ARGS TO Image Generation ###
    kwargs["async_call"] = True
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(image_variation, *args, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        if custom_llm_provider is None and model is not None:
            _, custom_llm_provider, _, _ = get_llm_provider(
                model=model, api_base=kwargs.get("api_base", None)
            )

        # Await normally
        init_response = await loop.run_in_executor(None, func_with_context)
        if isinstance(init_response, dict) or isinstance(
            init_response, ImageResponse
        ):  ## CACHING SCENARIO
            if isinstance(init_response, dict):
                init_response = ImageResponse(**init_response)
            response = init_response
        elif asyncio.iscoroutine(init_response):
            response = await init_response  # type: ignore
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )


@client
def image_variation(
    image: FileTypes,
    model: str = "dall-e-2",  # set to dall-e-2 by default - like OpenAI.
    n: int = 1,
    response_format: Literal["url", "b64_json"] = "url",
    size: Optional[str] = None,
    user: Optional[str] = None,
    **kwargs,
) -> ImageResponse:
    # get non-default params
    client = kwargs.get("client", None)
    # get logging object
    litellm_logging_obj = cast(LiteLLMLoggingObj, kwargs.get("litellm_logging_obj"))

    # get the litellm params
    litellm_params = get_litellm_params(**kwargs)
    # get the custom llm provider
    model, custom_llm_provider, dynamic_api_key, api_base = get_llm_provider(
        model=model,
        custom_llm_provider=litellm_params.get("custom_llm_provider", None),
        api_base=litellm_params.get("api_base", None),
        api_key=litellm_params.get("api_key", None),
    )

    # route to the correct provider w/ the params
    try:
        llm_provider = LlmProviders(custom_llm_provider)
        image_variation_provider = LITELLM_IMAGE_VARIATION_PROVIDERS(llm_provider)
    except ValueError:
        raise ValueError(
            f"Invalid image variation provider: {custom_llm_provider}. Supported providers are: {LITELLM_IMAGE_VARIATION_PROVIDERS}"
        )
    model_response = ImageResponse()

    response: Optional[ImageResponse] = None

    provider_config = ProviderConfigManager.get_provider_model_info(
        model=model or "",  # openai defaults to dall-e-2
        provider=llm_provider,
    )

    if provider_config is None:
        raise ValueError(
            f"image variation provider has no known model info config - required for getting api keys, etc.: {custom_llm_provider}. Supported providers are: {LITELLM_IMAGE_VARIATION_PROVIDERS}"
        )

    api_key = provider_config.get_api_key(litellm_params.get("api_key", None))
    api_base = provider_config.get_api_base(litellm_params.get("api_base", None))

    if image_variation_provider == LITELLM_IMAGE_VARIATION_PROVIDERS.OPENAI:
        if api_key is None:
            raise ValueError("API key is required for OpenAI image variations")
        if api_base is None:
            raise ValueError("API base is required for OpenAI image variations")

        response = openai_image_variations.image_variations(
            model_response=model_response,
            api_key=api_key,
            api_base=api_base,
            model=model,
            image=image,
            timeout=litellm_params.get("timeout", None),
            custom_llm_provider=custom_llm_provider,
            logging_obj=litellm_logging_obj,
            optional_params={},
            litellm_params=litellm_params,
        )
    elif image_variation_provider == LITELLM_IMAGE_VARIATION_PROVIDERS.TOPAZ:
        if api_key is None:
            raise ValueError("API key is required for Topaz image variations")
        if api_base is None:
            raise ValueError("API base is required for Topaz image variations")

        response = base_llm_aiohttp_handler.image_variations(
            model_response=model_response,
            api_key=api_key,
            api_base=api_base,
            model=model,
            image=image,
            timeout=litellm_params.get("timeout", None) or DEFAULT_REQUEST_TIMEOUT,
            custom_llm_provider=custom_llm_provider,
            logging_obj=litellm_logging_obj,
            optional_params={},
            litellm_params=litellm_params,
            client=client,
        )

    # return the response
    if response is None:
        raise ValueError(
            f"Invalid image variation provider: {custom_llm_provider}. Supported providers are: {LITELLM_IMAGE_VARIATION_PROVIDERS}"
        )
    return response


@client
def image_edit(
    image: Union[FileTypes, List[FileTypes]],
    prompt: str,
    model: Optional[str] = None,
    mask: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[Union[str, ImageGenerationRequestQuality]] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    user: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[ImageResponse, Coroutine[Any, Any, ImageResponse]]:
    """
    Maps the image edit functionality, similar to OpenAI's images/edits endpoint.
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("async_call", False) is True

        # add images / or return a single image
        images = image if isinstance(image, list) else [image]

        headers_from_kwargs = kwargs.get("headers")
        merged_extra_headers: Dict[str, Any] = {}
        if isinstance(headers_from_kwargs, dict):
            merged_extra_headers.update(headers_from_kwargs)
        if isinstance(extra_headers, dict):
            merged_extra_headers.update(extra_headers)

        if merged_extra_headers:
            extra_headers = dict(merged_extra_headers)

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)
        model, custom_llm_provider, _, _ = get_llm_provider(
            model=model or DEFAULT_IMAGE_ENDPOINT_MODEL,
            custom_llm_provider=custom_llm_provider,
        )

        # get provider config
        image_edit_provider_config: Optional[BaseImageEditConfig] = (
            ProviderConfigManager.get_provider_image_edit_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if image_edit_provider_config is None:
            raise ValueError(f"image edit is not supported for {custom_llm_provider}")

        local_vars.update(kwargs)
        # Get ImageEditOptionalRequestParams with only valid parameters
        image_edit_optional_params: ImageEditOptionalRequestParams = (
            ImageEditRequestUtils.get_requested_image_edit_optional_param(local_vars)
        )

        # Get optional parameters for the responses API
        image_edit_request_params: Dict = (
            ImageEditRequestUtils.get_optional_params_image_edit(
                model=model,
                image_edit_provider_config=image_edit_provider_config,
                image_edit_optional_params=image_edit_optional_params,
            )
        )

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=model,
            user=user,
            optional_params=dict(image_edit_request_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
                **image_edit_request_params,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler with _is_async flag instead of directly calling the async handler
        return base_llm_http_handler.image_edit_handler(
            model=model,
            image=images,
            prompt=prompt,
            image_edit_provider_config=image_edit_provider_config,
            image_edit_optional_request_params=image_edit_request_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or DEFAULT_REQUEST_TIMEOUT,
            _is_async=_is_async,
            client=kwargs.get("client"),
        )

    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aimage_edit(
    image: Union[FileTypes, List[FileTypes]],
    model: str,
    prompt: str,
    mask: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[Union[str, ImageGenerationRequestQuality]] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    user: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> ImageResponse:
    """
    Asynchronously calls the `image_edit` function with the given arguments and keyword arguments.

    Parameters:
    - `args` (tuple): Positional arguments to be passed to the `image_edit` function.
    - `kwargs` (dict): Keyword arguments to be passed to the `image_edit` function.

    Returns:
    - `response` (Any): The response returned by the `image_edit` function.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["async_call"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=local_vars.get("base_url", None)
            )

        images = image if isinstance(image, list) else [image]

        func = partial(
            image_edit,
            image=images,
            prompt=prompt,
            mask=mask,
            model=model,
            n=n,
            quality=quality,
            response_format=response_format,
            size=size,
            user=user,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
