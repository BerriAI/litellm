from typing import Any, Dict, Iterable, List, Literal, Optional, Union, get_type_hints

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
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
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponseTextConfigParam,
    ToolChoice,
    ToolParam,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

from .streaming_iterator import ResponsesAPIStreamingIterator

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


def get_requested_response_api_optional_param(
    params: Dict[str, Any]
) -> ResponsesAPIOptionalRequestParams:
    """
    Filter parameters to only include those defined in ResponsesAPIOptionalRequestParams.

    Args:
        params: Dictionary of parameters to filter

    Returns:
        ResponsesAPIOptionalRequestParams instance with only the valid parameters
    """
    valid_keys = get_type_hints(ResponsesAPIOptionalRequestParams).keys()
    filtered_params = {k: v for k, v in params.items() if k in valid_keys}
    return ResponsesAPIOptionalRequestParams(**filtered_params)


@client
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
) -> Union[ResponsesAPIResponse, ResponsesAPIStreamingIterator]:
    litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
    litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)

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
    local_vars = locals()
    local_vars.update(kwargs)
    # Get ResponsesAPIOptionalRequestParams with only valid parameters
    response_api_optional_params: ResponsesAPIOptionalRequestParams = (
        get_requested_response_api_optional_param(local_vars)
    )

    # Get optional parameters for the responses API
    responses_api_request_params: Dict = get_optional_params_responses_api(
        model=model,
        responses_api_provider_config=responses_api_provider_config,
        response_api_optional_params=response_api_optional_params,
    )

    # Pre Call logging
    litellm_logging_obj.update_environment_variables(
        model=model,
        user=user,
        optional_params=dict(responses_api_request_params),
        litellm_params={
            "litellm_call_id": litellm_call_id,
            **responses_api_request_params,
        },
        custom_llm_provider=custom_llm_provider,
    )

    response = await base_llm_http_handler.async_response_api_handler(
        model=model,
        input=input,
        responses_api_provider_config=responses_api_provider_config,
        response_api_optional_request_params=responses_api_request_params,
        custom_llm_provider=custom_llm_provider,
        litellm_params=litellm_params,
        logging_obj=litellm_logging_obj,
        extra_headers=extra_headers,
        extra_body=extra_body,
        timeout=timeout,
    )
    return response


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
