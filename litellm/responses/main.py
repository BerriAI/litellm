from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.responses.utils import (
    ResponsesAPIRequestParams,
    get_optional_params_responses_api,
)
from litellm.types.llms.openai import (
    Reasoning,
    ResponseIncludable,
    ResponseInputParam,
    ResponseTextConfigParam,
    ToolChoice,
    ToolParam,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


async def aresponses(
    input: Union[str, ResponseInputParam],
    model: str,
    include: Optional[List[ResponseIncludable]] = None,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[Reasoning] = None,
    store: Optional[bool] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    text: Optional[ResponseTextConfigParam] = None,
    tool_choice: Optional[ToolChoice] = None,
    tools: Optional[Iterable[ToolParam]] = None,
    top_p: Optional[float] = None,
    truncation: Optional[Literal["auto", "disabled"]] = None,
    user: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    **kwargs,
):

    # get llm provider logic
    litellm_params = GenericLiteLLMParams(**kwargs)
    model, custom_llm_provider, dynamic_api_key, dynamic_api_base = (
        litellm.get_llm_provider(
            model=model,
            custom_llm_provider=kwargs.get("custom_llm_provider", None),
            api_base=litellm_params.api_base,
            api_key=litellm_params.api_key,
        )
    )

    # get provider config
    responses_api_provider_config: Optional[BaseResponsesAPIConfig] = (
        ProviderConfigManager.get_provider_responses_api_config(
            model=model,
            provider=litellm.LlmProviders(custom_llm_provider),
        )
    )

    if responses_api_provider_config is None:
        raise litellm.BadRequestError(
            model=model,
            llm_provider=custom_llm_provider,
            message=f"Responses API not available for custom_llm_provider={custom_llm_provider}, model: {model}",
        )

    # Get all parameters using locals() and combine with kwargs
    all_params = {**locals(), **kwargs}

    # Get optional parameters for the responses API
    responses_api_request_params: ResponsesAPIRequestParams = (
        get_optional_params_responses_api(
            model=model,
            responses_api_provider_config=responses_api_provider_config,
            optional_params={**locals(), **kwargs},
        )
    )

    response = await base_llm_http_handler.async_response_api_handler(
        model=model,
        input=input,
        responses_api_provider_config=responses_api_provider_config,
        responses_api_request_params=responses_api_request_params,
    )
    return response

    pass


def responses(
    input: Union[str, ResponseInputParam],
    model: str,
    include: Optional[List[ResponseIncludable]] = None,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[Reasoning] = None,
    store: Optional[bool] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    text: Optional[ResponseTextConfigParam] = None,
    tool_choice: Optional[ToolChoice] = None,
    tools: Optional[Iterable[ToolParam]] = None,
    top_p: Optional[float] = None,
    truncation: Optional[Literal["auto", "disabled"]] = None,
    user: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
):
    pass
