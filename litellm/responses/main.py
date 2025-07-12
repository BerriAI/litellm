import asyncio
import contextvars
from functools import partial
from typing import Any, Coroutine, Dict, Iterable, List, Literal, Optional, Tuple, Union

import httpx

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
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
    ResponseTextConfigParam,
    ToolChoice,
    ToolParam,
)
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client
from litellm._logging import verbose_logger

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
        id="resp_mock_123",
        object="response",
        status="completed",
        status_details=None,
        output=[
            ResponseContent(
                type="text",
                text=mock_response,
                content=mock_response,
            )
        ],
        usage=ResponseAPIUsage(
            completion_tokens=50,
            prompt_tokens=25,
            total_tokens=75,
        ),
        created=1234567890,
        model="mock-model",
    )


class MCPResponsesAPIHelper:
    """Helper class with static methods for MCP integration with Responses API."""
    
    @staticmethod
    def _parse_mcp_tools(tools: Optional[Iterable[ToolParam]]) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Parse tools and separate MCP tools with litellm_proxy from other tools.
        
        Returns:
            Tuple of (mcp_tools_with_litellm_proxy, other_tools)
        """
        mcp_tools_with_litellm_proxy: List[Dict[str, Any]] = []
        other_tools: List[Any] = []
        
        if tools:
            for tool in tools:
                if (isinstance(tool, dict) and 
                    tool.get("type") == "mcp" and 
                    tool.get("server_url") == "litellm_proxy"):
                    mcp_tools_with_litellm_proxy.append(tool)
                else:
                    other_tools.append(tool)
        
        return mcp_tools_with_litellm_proxy, other_tools
    
    @staticmethod
    async def _get_mcp_tools_from_manager(user_api_key_auth: Any) -> List[Any]:
        """Get available tools from the MCP server manager."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
        
        return await global_mcp_server_manager.list_tools(user_api_key_auth=user_api_key_auth)
    
    @staticmethod
    def _transform_mcp_tools_to_openai(mcp_tools: List[Any]) -> List[Any]:
        """Transform MCP tools to OpenAI-compatible format."""
        from litellm.experimental_mcp_client.tools import transform_mcp_tool_to_openai_tool
        
        openai_tools = []
        for mcp_tool in mcp_tools:
            openai_tool = transform_mcp_tool_to_openai_tool(mcp_tool)
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    @staticmethod
    def _should_auto_execute_tools(
        mcp_tools_with_litellm_proxy: List[Dict[str, Any]], 
        response: ResponsesAPIResponse
    ) -> bool:
        """Check if we should auto-execute tool calls."""
        return (
            mcp_tools_with_litellm_proxy and 
            isinstance(response, ResponsesAPIResponse) and
            any(tool.get("require_approval") == "never" for tool in mcp_tools_with_litellm_proxy)
        )
    
    @staticmethod
    def _extract_tool_calls_from_response(response: ResponsesAPIResponse) -> List[Any]:
        """Extract tool calls from the response output."""
        tool_calls = []
        for output_item in response.output:
            # Check if this is a function call output item
            if (isinstance(output_item, dict) and 
                output_item.get("type") == "function_call"):
                tool_calls.append(output_item)
            elif hasattr(output_item, 'type') and output_item.type == "function_call":
                # Handle pydantic model case
                tool_calls.append(output_item)
        
        return tool_calls
    
    @staticmethod
    def _extract_tool_call_details(tool_call) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract tool name, arguments, and call_id from a tool call."""
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name")
            tool_arguments = tool_call.get("arguments")
            tool_call_id = tool_call.get("call_id") or tool_call.get("id")
        else:
            tool_name = getattr(tool_call, "name", None)
            tool_arguments = getattr(tool_call, "arguments", None)
            tool_call_id = getattr(tool_call, "call_id", None) or getattr(tool_call, "id", None)
        
        return tool_name, tool_arguments, tool_call_id
    
    @staticmethod
    def _parse_tool_arguments(tool_arguments: Any) -> Dict[str, Any]:
        """Parse tool arguments, handling both string and dict formats."""
        import json
        
        if isinstance(tool_arguments, str):
            try:
                return json.loads(tool_arguments)
            except json.JSONDecodeError:
                return {}
        else:
            return tool_arguments or {}
    
    @staticmethod
    async def _execute_tool_calls(
        tool_calls: List[Any], 
        user_api_key_auth: Any
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
        
        tool_results = []
        for tool_call in tool_calls:
            try:
                tool_name, tool_arguments, tool_call_id = MCPResponsesAPIHelper._extract_tool_call_details(tool_call)
                
                if not tool_name:
                    verbose_logger.warning(f"Tool call missing name: {tool_call}")
                    continue
                
                parsed_arguments = MCPResponsesAPIHelper._parse_tool_arguments(tool_arguments)
                
                result = await global_mcp_server_manager.call_tool(
                    name=tool_name,
                    arguments=parsed_arguments,
                    user_api_key_auth=user_api_key_auth,
                )
                
                # Format result for inclusion in response
                result_text = str(result.content[0].text) if result.content else "Tool executed successfully"
                tool_results.append({
                    "tool_call_id": tool_call_id,
                    "result": result_text
                })
                
            except Exception as e:
                verbose_logger.exception(f"Error executing MCP tool call: {e}")
                tool_results.append({
                    "tool_call_id": tool_call_id if 'tool_call_id' in locals() else "unknown",
                    "result": f"Error executing tool: {str(e)}"
                })
        
        return tool_results
    
    @staticmethod
    def _create_follow_up_input(response: ResponsesAPIResponse, tool_results: List[Dict[str, Any]]) -> List[Any]:
        """Create follow-up input with tool results in proper format."""
        follow_up_input = []
        
        # Add the original response (assistant message with tool calls)
        # Transform response output back to input format for follow-up
        for output_item in response.output:
            if isinstance(output_item, dict):
                if output_item.get("type") == "function_call":
                    # Add function call to input
                    follow_up_input.append({
                        "type": "function_call",
                        "call_id": output_item.get("call_id") or output_item.get("id"),
                        "name": output_item.get("name"),
                        "arguments": output_item.get("arguments")
                    })
                elif output_item.get("type") == "message":
                    # Add message content
                    follow_up_input.append({
                        "type": "message",
                        "role": output_item.get("role", "assistant"),
                        "content": output_item.get("content", "")
                    })
        
        # Add tool results
        for tool_result in tool_results:
            follow_up_input.append({
                "type": "function_call_output",
                "call_id": tool_result["tool_call_id"],
                "output": tool_result["result"]
            })
        
        return follow_up_input
    
    @staticmethod
    async def _make_follow_up_call(
        follow_up_input: List[Any],
        model: str,
        all_tools: Optional[List[Any]],
        response_id: str,
        **call_params: Any
    ) -> ResponsesAPIResponse:
        """Make follow-up response API call with tool results."""
        return await aresponses(
            input=follow_up_input,
            model=model,
            tools=all_tools,  # Keep tools for potential future calls
            previous_response_id=response_id,  # Link to previous response
            **call_params
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
    from litellm.proxy._types import UserAPIKeyAuth
    
    # Parse MCP tools and separate from other tools
    mcp_tools_with_litellm_proxy, other_tools = MCPResponsesAPIHelper._parse_mcp_tools(tools)
    
    # Get available tools from MCP manager if we have MCP tools
    openai_tools = []
    if mcp_tools_with_litellm_proxy:
        user_api_key_auth = kwargs.get("user_api_key_auth")
        mcp_tools = await MCPResponsesAPIHelper._get_mcp_tools_from_manager(user_api_key_auth)
        openai_tools = MCPResponsesAPIHelper._transform_mcp_tools_to_openai(mcp_tools)
    
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
    
    # Make initial response API call
    response = await aresponses(
        input=input,
        model=model,
        tools=all_tools,
        previous_response_id=previous_response_id,
        **call_params
    )
    
    # Check if we need to auto-execute tool calls
    if MCPResponsesAPIHelper._should_auto_execute_tools(mcp_tools_with_litellm_proxy, response):
        tool_calls = MCPResponsesAPIHelper._extract_tool_calls_from_response(response)
        
        if tool_calls:
            user_api_key_auth = kwargs.get("user_api_key_auth")
            tool_results = await MCPResponsesAPIHelper._execute_tool_calls(tool_calls, user_api_key_auth)
            
            if tool_results:
                follow_up_input = MCPResponsesAPIHelper._create_follow_up_input(response, tool_results)
                
                final_response = await MCPResponsesAPIHelper._make_follow_up_call(
                    follow_up_input=follow_up_input,
                    model=model,
                    all_tools=all_tools,
                    response_id=response.id,
                    **call_params
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
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
):
    """
    Synchronous version of the Responses API.
    Uses the synchronous HTTP handler to make requests.
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aresponses", False) is True

        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        ## MOCK RESPONSE LOGIC
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
                **kwargs,
            )

        # Get optional parameters for the responses API
        responses_api_request_params: Dict = (
            ResponsesAPIRequestUtils.get_optional_params_responses_api(
                model=model,
                responses_api_provider_config=responses_api_provider_config,
                response_api_optional_params=response_api_optional_params,
            )
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
