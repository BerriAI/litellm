import asyncio
import contextvars
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Coroutine,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    Union,
    cast,
)

import httpx
from pydantic import BaseModel

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    update_responses_input_with_model_file_ids,
)
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    PromptObject,
    Reasoning,
    ResponseIncludable,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ToolChoice,
    ToolParam,
)

# Handle ResponseText import with fallback
if TYPE_CHECKING:
    from litellm.types.llms.openai import ResponseText  # type: ignore
else:
    ResponseText = str  # Fallback for ResponseText import
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
else:
    MCPTool = Any

from .streaming_iterator import BaseResponsesAPIStreamingIterator

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
litellm_completion_transformation_handler = LiteLLMCompletionTransformationHandler()
#################################################


def mock_responses_api_response(
    mock_response: str = "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",
):
    return ResponsesAPIResponse(
        **{  # type: ignore
            "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
            "object": "response",
            "created_at": 1741476542,
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": "gpt-4.1-2025-04-14",
            "output": [
                {
                    "type": "message",
                    "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": mock_response,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 36,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 87,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 123,
            },
            "user": None,
            "metadata": {},
        }
    )


async def aresponses_api_with_mcp(
    input: Union[str, ResponseInputParam],
    model: str,
    include: Optional[List[ResponseIncludable]] = None,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    prompt: Optional[PromptObject] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[Reasoning] = None,
    store: Optional[bool] = None,
    background: Optional[bool] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    text: Optional["ResponseText"] = None,
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
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
    """
    Async version of responses API with MCP integration.

    When MCP tools with server_url="litellm_proxy" are provided, this function will:
    1. Get available tools from the MCP server manager
    2. Insert the tools into the messages/input
    3. Call the standard responses API
    4. If require_approval="never" and tool calls are returned, automatically execute them
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    # Parse MCP tools and separate from other tools
    (
        mcp_tools_with_litellm_proxy,
        other_tools,
    ) = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(tools)

    # Process MCP tools through the complete pipeline (fetch + filter + deduplicate + transform)
    # Extract user_api_key_auth from litellm_metadata (where it's added by add_user_api_key_auth_to_request_metadata)
    user_api_key_auth = kwargs.get("user_api_key_auth") or kwargs.get("litellm_metadata", {}).get("user_api_key_auth")

    # Get original MCP tools (for events) and OpenAI tools (for LLM) by reusing existing methods
    (
        original_mcp_tools,
        tool_server_map,
    ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
        user_api_key_auth=user_api_key_auth,
        mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
    )
    openai_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
        original_mcp_tools
    )

    # Combine with other tools
    all_tools = openai_tools + other_tools if (openai_tools or other_tools) else None

    # Prepare call parameters for reuse
    call_params = {
        "include": include,
        "instructions": instructions,
        "max_output_tokens": max_output_tokens,
        "prompt": prompt,
        "metadata": metadata,
        "parallel_tool_calls": parallel_tool_calls,
        "reasoning": reasoning,
        "store": store,
        "background": background,
        "stream": stream,
        "temperature": temperature,
        "text": text,
        "tool_choice": tool_choice,
        "top_p": top_p,
        "truncation": truncation,
        "user": user,
        "extra_headers": extra_headers,
        "extra_query": extra_query,
        "extra_body": extra_body,
        "timeout": timeout,
        "custom_llm_provider": custom_llm_provider,
        **kwargs,
    }

    # Handle MCP streaming if requested
    if stream and mcp_tools_with_litellm_proxy:
        # Generate MCP discovery events using the already processed tools
        from litellm._uuid import uuid
        from litellm.responses.mcp.mcp_streaming_iterator import (
            create_mcp_list_tools_events,
        )

        base_item_id = f"mcp_{uuid.uuid4().hex[:8]}"
        mcp_discovery_events = await create_mcp_list_tools_events(
            mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
            user_api_key_auth=user_api_key_auth,
            base_item_id=base_item_id,
            pre_processed_mcp_tools=original_mcp_tools,
        )

        return LiteLLM_Proxy_MCP_Handler._create_mcp_streaming_response(
            input=input,
            model=model,
            all_tools=all_tools,
            mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
            mcp_discovery_events=mcp_discovery_events,
            call_params=call_params,
            previous_response_id=previous_response_id,
            tool_server_map=tool_server_map,
            **kwargs,
        )

    # Determine if we should auto-execute tools
    should_auto_execute = bool(
        mcp_tools_with_litellm_proxy
    ) and LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(
        mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy
    )

    # Prepare parameters for the initial call
    initial_call_params = LiteLLM_Proxy_MCP_Handler._prepare_initial_call_params(
        call_params=call_params, should_auto_execute=should_auto_execute
    )

    #########################################################
    # Make initial response API call
    #########################################################
    response = await aresponses(
        input=input,
        model=model,
        tools=all_tools,
        previous_response_id=previous_response_id,
        **initial_call_params,
    )

    verbose_logger.debug("Initial response %s", response)

    #########################################################
    # Auto-Execute Tools Handling
    # If auto-execute tools is True, then we need to execute the tool calls
    #########################################################
    if should_auto_execute and isinstance(
        response, ResponsesAPIResponse
    ):  # type: ignore
        tool_calls = LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_response(
            response=response
        )

        if tool_calls:
            user_api_key_auth = kwargs.get("litellm_metadata", {}).get(
                "user_api_key_auth"
            )
            
            # Extract MCP auth headers from the request to pass to MCP server
            secret_fields: Optional[Dict[str, Any]] = kwargs.get("secret_fields")
            (
                mcp_auth_header,
                mcp_server_auth_headers,
                oauth2_headers,
                raw_headers_from_request,
            ) = ResponsesAPIRequestUtils.extract_mcp_headers_from_request(
                secret_fields=secret_fields,
                tools=tools,
            )
            
            tool_results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
                tool_server_map=tool_server_map,
                tool_calls=tool_calls,
                user_api_key_auth=user_api_key_auth,
                mcp_auth_header=mcp_auth_header,
                mcp_server_auth_headers=mcp_server_auth_headers,
                oauth2_headers=oauth2_headers,
                raw_headers=raw_headers_from_request,
            )

            if tool_results:
                follow_up_input = LiteLLM_Proxy_MCP_Handler._create_follow_up_input(
                    response=response, tool_results=tool_results, original_input=input
                )

                # Prepare parameters for follow-up call (restores original stream setting)
                follow_up_call_params = (
                    LiteLLM_Proxy_MCP_Handler._prepare_follow_up_call_params(
                        call_params=call_params, original_stream_setting=stream or False
                    )
                )

                # Create tool execution events for streaming if needed
                tool_execution_events = []
                if stream:
                    tool_execution_events = (
                        LiteLLM_Proxy_MCP_Handler._create_tool_execution_events(
                            tool_calls=tool_calls, tool_results=tool_results
                        )
                    )

                final_response = await LiteLLM_Proxy_MCP_Handler._make_follow_up_call(
                    follow_up_input=follow_up_input,
                    model=model,
                    all_tools=all_tools,
                    response_id=response.id,
                    **follow_up_call_params,
                )

                # If streaming and we have tool execution events, wrap the response
                if (
                    stream
                    and tool_execution_events
                    and (
                        hasattr(final_response, "__aiter__")
                        or hasattr(final_response, "__iter__")
                    )
                ):
                    from litellm.responses.mcp.mcp_streaming_iterator import (
                        MCPEnhancedStreamingIterator,
                    )

                    final_response = MCPEnhancedStreamingIterator(
                        tool_server_map=tool_server_map,
                        base_iterator=final_response,
                        mcp_events=tool_execution_events,
                    )

                # Add custom output elements to the final response (for non-streaming)
                elif isinstance(final_response, ResponsesAPIResponse):
                    # Fetch MCP tools again for output elements (without OpenAI transformation)
                    (
                        mcp_tools_for_output,
                        _,
                    ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
                        user_api_key_auth=user_api_key_auth,
                        mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
                    )
                    final_response = (
                        LiteLLM_Proxy_MCP_Handler._add_mcp_output_elements_to_response(
                            response=final_response,
                            mcp_tools_fetched=mcp_tools_for_output,
                            tool_results=tool_results,
                        )
                    )
                return final_response

    return response


@client
async def aresponses(
    input: Union[str, ResponseInputParam],
    model: str,
    include: Optional[List[ResponseIncludable]] = None,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    prompt: Optional[PromptObject] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[Reasoning] = None,
    store: Optional[bool] = None,
    background: Optional[bool] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    text: Optional["ResponseText"] = None,
    text_format: Optional[Union[Type["BaseModel"], dict]] = None,
    tool_choice: Optional[ToolChoice] = None,
    tools: Optional[Iterable[ToolParam]] = None,
    top_p: Optional[float] = None,
    truncation: Optional[Literal["auto", "disabled"]] = None,
    user: Optional[str] = None,
    service_tier: Optional[str] = None,
    safety_identifier: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
    """
    Async: Handles responses API requests by reusing the synchronous function
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aresponses"] = True

        # Convert text_format to text parameter if provided
        text = ResponsesAPIRequestUtils.convert_text_format_to_text_param(
            text_format=text_format, text=text
        )
        if text is not None:
            # Update local_vars to include the converted text parameter
            local_vars["text"] = text

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=local_vars.get("base_url", None)
            )

        func = partial(
            responses,
            input=input,
            model=model,
            include=include,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            prompt=prompt,
            metadata=metadata,
            parallel_tool_calls=parallel_tool_calls,
            previous_response_id=previous_response_id,
            reasoning=reasoning,
            store=store,
            background=background,
            stream=stream,
            temperature=temperature,
            text=text,
            tool_choice=tool_choice,
            tools=tools,
            top_p=top_p,
            truncation=truncation,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            service_tier=service_tier,
            safety_identifier=safety_identifier,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        # Update the responses_api_response_id with the model_id
        if isinstance(response, ResponsesAPIResponse):
            response = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                responses_api_response=response,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                custom_llm_provider=custom_llm_provider,
            )

        if response is None:
            raise ValueError(
                f"Got an unexpected None response from the Responses API: {response}"
            )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def responses(
    input: Union[str, ResponseInputParam],
    model: str,
    include: Optional[List[ResponseIncludable]] = None,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    prompt: Optional[PromptObject] = None,
    metadata: Optional[Dict[str, Any]] = None,
    parallel_tool_calls: Optional[bool] = None,
    previous_response_id: Optional[str] = None,
    reasoning: Optional[Reasoning] = None,
    store: Optional[bool] = None,
    background: Optional[bool] = None,
    stream: Optional[bool] = None,
    temperature: Optional[float] = None,
    text: Optional["ResponseText"] = None,
    text_format: Optional[Union[Type["BaseModel"], dict]] = None,
    tool_choice: Optional[ToolChoice] = None,
    tools: Optional[Iterable[ToolParam]] = None,
    top_p: Optional[float] = None,
    truncation: Optional[Literal["auto", "disabled"]] = None,
    user: Optional[str] = None,
    service_tier: Optional[str] = None,
    safety_identifier: Optional[str] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    allowed_openai_params: Optional[List[str]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
):
    """
    Synchronous version of the Responses API.
    Uses the synchronous HTTP handler to make requests.
    """
    local_vars = locals()
    from litellm.responses.mcp.litellm_proxy_mcp_handler import (
        LiteLLM_Proxy_MCP_Handler,
    )

    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aresponses", False) is True

        # Convert text_format to text parameter if provided
        text = ResponsesAPIRequestUtils.convert_text_format_to_text_param(
            text_format=text_format, text=text
        )
        if text is not None:
            # Update local_vars to include the converted text parameter
            local_vars["text"] = text

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        #########################################################
        # MOCK RESPONSE LOGIC
        #########################################################
        if litellm_params.mock_response and isinstance(
            litellm_params.mock_response, str
        ):
            return mock_responses_api_response(
                mock_response=litellm_params.mock_response
            )

        (
            model,
            custom_llm_provider,
            dynamic_api_key,
            dynamic_api_base,
        ) = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=litellm_params.api_base,
            api_key=litellm_params.api_key,
        )
        
        #########################################################
        # Update input with provider-specific file IDs if managed files are used
        #########################################################
        input = cast(Union[str, ResponseInputParam], update_responses_input_with_model_file_ids(input=input))
        local_vars["input"] = input
        
        #########################################################
        # Native MCP Responses API
        #########################################################
        if LiteLLM_Proxy_MCP_Handler._should_use_litellm_mcp_gateway(tools=tools):
            return aresponses_api_with_mcp(
                input=input,
                model=model,
                include=include,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                prompt=prompt,
                metadata=metadata,
                parallel_tool_calls=parallel_tool_calls,
                previous_response_id=previous_response_id,
                reasoning=reasoning,
                store=store,
                background=background,
                stream=stream,
                temperature=temperature,
                text=text,
                tool_choice=tool_choice,
                tools=tools,
                top_p=top_p,
                truncation=truncation,
                user=user,
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                custom_llm_provider=custom_llm_provider,
                **kwargs,
            )

        # get provider config
        responses_api_provider_config: Optional[BaseResponsesAPIConfig] = (
            ProviderConfigManager.get_provider_responses_api_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        local_vars.update(kwargs)
        # Get ResponsesAPIOptionalRequestParams with only valid parameters
        response_api_optional_params: ResponsesAPIOptionalRequestParams = (
            ResponsesAPIRequestUtils.get_requested_response_api_optional_param(
                local_vars
            )
        )

        if responses_api_provider_config is None:
            return litellm_completion_transformation_handler.response_api_handler(
                model=model,
                input=input,
                responses_api_request=response_api_optional_params,
                custom_llm_provider=custom_llm_provider,
                _is_async=_is_async,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )

        # Get optional parameters for the responses API
        responses_api_request_params: Dict = (
            ResponsesAPIRequestUtils.get_optional_params_responses_api(
                model=model,
                responses_api_provider_config=responses_api_provider_config,
                response_api_optional_params=response_api_optional_params,
                allowed_openai_params=allowed_openai_params,
            )
        )

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=model,
            user=user,
            optional_params=dict(responses_api_request_params),
            litellm_params={
                **responses_api_request_params,
                "litellm_call_id": litellm_call_id,
                "metadata": metadata,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler with _is_async flag instead of directly calling the async handler
        response = base_llm_http_handler.response_api_handler(
            model=model,
            input=input,
            responses_api_provider_config=responses_api_provider_config,
            response_api_optional_request_params=responses_api_request_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            fake_stream=responses_api_provider_config.should_fake_stream(
                model=model, stream=stream, custom_llm_provider=custom_llm_provider
            ),
            litellm_metadata=kwargs.get("litellm_metadata", {}),
            shared_session=kwargs.get("shared_session"),
        )

        # Update the responses_api_response_id with the model_id
        if isinstance(response, ResponsesAPIResponse):
            response = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                responses_api_response=response,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                custom_llm_provider=custom_llm_provider,
            )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def adelete_responses(
    response_id: str,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> DeleteResponseResult:
    """
    Async version of the DELETE Responses API

    DELETE /v1/responses/{response_id} endpoint in the responses API

    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["adelete_responses"] = True

        # get custom llm provider from response_id
        decoded_response_id: DecodedResponseId = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id,
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        func = partial(
            delete_responses,
            response_id=response_id,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
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
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete_responses(
    response_id: str,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[DeleteResponseResult, Coroutine[Any, Any, DeleteResponseResult]]:
    """
    Synchronous version of the DELETE Responses API

    DELETE /v1/responses/{response_id} endpoint in the responses API

    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("adelete_responses", False) is True

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        # get custom llm provider from response_id
        decoded_response_id: DecodedResponseId = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id,
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        if custom_llm_provider is None:
            raise ValueError("custom_llm_provider is required but passed as None")

        # get provider config
        responses_api_provider_config: Optional[BaseResponsesAPIConfig] = (
            ProviderConfigManager.get_provider_responses_api_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if responses_api_provider_config is None:
            raise ValueError(
                f"DELETE responses is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "response_id": response_id,
            },
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler with _is_async flag instead of directly calling the async handler
        response = base_llm_http_handler.delete_response_api_handler(
            response_id=response_id,
            custom_llm_provider=custom_llm_provider,
            responses_api_provider_config=responses_api_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def aget_responses(
    response_id: str,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> ResponsesAPIResponse:
    """
    Async: Fetch a response by its ID.

    GET /v1/responses/{response_id} endpoint in the responses API

    Args:
        response_id: The ID of the response to fetch.
        custom_llm_provider: Optional provider name. If not specified, will be decoded from response_id.

    Returns:
        The response object with complete information about the stored response.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aget_responses"] = True

        # get custom llm provider from response_id
        decoded_response_id: DecodedResponseId = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id,
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        func = partial(
            get_responses,
            response_id=response_id,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        # Update the responses_api_response_id with the model_id
        if isinstance(response, ResponsesAPIResponse):
            response = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                responses_api_response=response,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                custom_llm_provider=custom_llm_provider,
            )
        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def get_responses(
    response_id: str,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[ResponsesAPIResponse, Coroutine[Any, Any, ResponsesAPIResponse]]:
    """
    Fetch a response by its ID.

    GET /v1/responses/{response_id} endpoint in the responses API

    Args:
        response_id: The ID of the response to fetch.
        custom_llm_provider: Optional provider name. If not specified, will be decoded from response_id.

    Returns:
        The response object with complete information about the stored response.
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aget_responses", False) is True

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        # get custom llm provider from response_id
        decoded_response_id: DecodedResponseId = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id,
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        if custom_llm_provider is None:
            raise ValueError("custom_llm_provider is required but passed as None")

        # get provider config
        responses_api_provider_config: Optional[BaseResponsesAPIConfig] = (
            ProviderConfigManager.get_provider_responses_api_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if responses_api_provider_config is None:
            raise ValueError(
                f"GET responses is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "response_id": response_id,
            },
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler with _is_async flag instead of directly calling the async handler
        response = base_llm_http_handler.get_responses(
            response_id=response_id,
            custom_llm_provider=custom_llm_provider,
            responses_api_provider_config=responses_api_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        # Update the responses_api_response_id with the model_id
        if isinstance(response, ResponsesAPIResponse):
            response = ResponsesAPIRequestUtils._update_responses_api_response_id_with_model_id(
                responses_api_response=response,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
                custom_llm_provider=custom_llm_provider,
            )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def alist_input_items(
    response_id: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
    include: Optional[List[str]] = None,
    limit: int = 20,
    order: Literal["asc", "desc"] = "desc",
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Async: List input items for a response"""
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["alist_input_items"] = True

        decoded_response_id = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        func = partial(
            list_input_items,
            response_id=response_id,
            after=after,
            before=before,
            include=include,
            limit=limit,
            order=order,
            extra_headers=extra_headers,
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
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def list_input_items(
    response_id: str,
    after: Optional[str] = None,
    before: Optional[str] = None,
    include: Optional[List[str]] = None,
    limit: int = 20,
    order: Literal["asc", "desc"] = "desc",
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[Dict, Coroutine[Any, Any, Dict]]:
    """List input items for a response"""
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("alist_input_items", False) is True

        litellm_params = GenericLiteLLMParams(**kwargs)

        decoded_response_id = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        if custom_llm_provider is None:
            raise ValueError("custom_llm_provider is required but passed as None")

        responses_api_provider_config: Optional[BaseResponsesAPIConfig] = (
            ProviderConfigManager.get_provider_responses_api_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if responses_api_provider_config is None:
            raise ValueError(
                f"list_input_items is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)

        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={"response_id": response_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )

        response = base_llm_http_handler.list_responses_input_items(
            response_id=response_id,
            custom_llm_provider=custom_llm_provider,
            responses_api_provider_config=responses_api_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            after=after,
            before=before,
            include=include,
            limit=limit,
            order=order,
            extra_headers=extra_headers,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def acancel_responses(
    response_id: str,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> ResponsesAPIResponse:
    """
    Async version of the POST Cancel Responses API

    POST /v1/responses/{response_id}/cancel endpoint in the responses API

    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acancel_responses"] = True

        # get custom llm provider from response_id
        decoded_response_id: DecodedResponseId = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id,
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        func = partial(
            cancel_responses,
            response_id=response_id,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
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
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def cancel_responses(
    response_id: str,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[ResponsesAPIResponse, Coroutine[Any, Any, ResponsesAPIResponse]]:
    """
    Synchronous version of the POST Responses API

    POST /v1/responses/{response_id}/cancel endpoint in the responses API

    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("acancel_responses", False) is True

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        # get custom llm provider from response_id
        decoded_response_id: DecodedResponseId = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                response_id=response_id,
            )
        )
        response_id = decoded_response_id.get("response_id") or response_id
        custom_llm_provider = (
            decoded_response_id.get("custom_llm_provider") or custom_llm_provider
        )

        if custom_llm_provider is None:
            raise ValueError("custom_llm_provider is required but passed as None")

        # get provider config
        responses_api_provider_config: Optional[BaseResponsesAPIConfig] = (
            ProviderConfigManager.get_provider_responses_api_config(
                model=None,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if responses_api_provider_config is None:
            raise ValueError(
                f"CANCEL responses is not supported for {custom_llm_provider}"
            )

        local_vars.update(kwargs)

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={
                "response_id": response_id,
            },
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler with _is_async flag instead of directly calling the async handler
        response = base_llm_http_handler.cancel_response_api_handler(
            response_id=response_id,
            custom_llm_provider=custom_llm_provider,
            responses_api_provider_config=responses_api_provider_config,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            shared_session=kwargs.get("shared_session"),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
