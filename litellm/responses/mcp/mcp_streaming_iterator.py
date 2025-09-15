import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from litellm._logging import verbose_logger
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import (
    MCPCallArgumentsDeltaEvent,
    MCPCallArgumentsDoneEvent,
    MCPCallCompletedEvent,
    MCPCallFailedEvent,
    MCPCallInProgressEvent,
    MCPListToolsCompletedEvent,
    MCPListToolsFailedEvent,
    MCPListToolsInProgressEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
    ResponsesAPIStreamingResponse,
    ToolParam,
)

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
else:
    MCPTool = Any


async def create_mcp_list_tools_events(
    mcp_tools_with_litellm_proxy: List[ToolParam],
    user_api_key_auth: Any,
    base_item_id: str,
    pre_processed_mcp_tools: List[Any]
) -> List[ResponsesAPIStreamingResponse]:
    """Create MCP discovery events using pre-processed tools from the parent"""
    
    events: List[ResponsesAPIStreamingResponse] = []
    
    try:
        # Extract MCP server names
        mcp_servers = []
        for tool in mcp_tools_with_litellm_proxy:
            if isinstance(tool, dict) and "server_url" in tool:
                server_url = tool.get("server_url")
                if isinstance(server_url, str) and server_url.startswith("litellm_proxy/mcp/"):
                    server_name = server_url.split("/")[-1]
                    mcp_servers.append(server_name)
        
        # Emit list tools in progress event
        in_progress_event = MCPListToolsInProgressEvent(
            type=ResponsesAPIStreamEvents.MCP_LIST_TOOLS_IN_PROGRESS,
            sequence_number=1,
            output_index=0,
            item_id=base_item_id,
        )
        events.append(in_progress_event)
        
        # Use the pre-processed MCP tools that were already fetched, filtered, and deduplicated by the parent
        filtered_mcp_tools = pre_processed_mcp_tools
        
        # Convert tools to dict format for the event
        mcp_tools_dict = []
        for tool in filtered_mcp_tools:
            if hasattr(tool, 'model_dump') and callable(getattr(tool, 'model_dump')):
                # Type cast to help mypy understand this is safe after hasattr check
                mcp_tools_dict.append(cast(Any, tool).model_dump())
            elif hasattr(tool, '__dict__'):
                mcp_tools_dict.append(tool.__dict__)
            else:
                mcp_tools_dict.append({"name": getattr(tool, 'name', str(tool))})
        
        # Emit list tools completed event
        completed_event = MCPListToolsCompletedEvent(
            type=ResponsesAPIStreamEvents.MCP_LIST_TOOLS_COMPLETED,
            sequence_number=2,
            output_index=0,
            item_id=base_item_id,
        )
        events.append(completed_event)
        
        # Add output_item.done event with the actual tools list (matching OpenAI format)
        from litellm.types.llms.openai import OutputItemDoneEvent

        # Extract server label from the first MCP tool config
        server_label = ""
        if mcp_tools_with_litellm_proxy:
            first_tool = mcp_tools_with_litellm_proxy[0]
            if isinstance(first_tool, dict):
                server_label_value = first_tool.get("server_label", "")
                server_label = str(server_label_value) if server_label_value is not None else ""
        
        # Format tools for OpenAI output_item.done format
        formatted_tools = []
        for tool in filtered_mcp_tools:
            tool_dict = {
                "name": getattr(tool, 'name', 'unknown'),
                "description": getattr(tool, 'description', ''),
                "annotations": {"read_only": False},
            }
            
            # Add input_schema if available
            if hasattr(tool, 'inputSchema'):
                tool_dict["input_schema"] = getattr(tool, 'inputSchema')
            elif hasattr(tool, 'input_schema'):
                tool_dict["input_schema"] = getattr(tool, 'input_schema')
            
            formatted_tools.append(tool_dict)
        
        # Create the output_item.done event with MCP tools list
        output_item_done_event = OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            output_index=0,
            item={
                "id": base_item_id,
                "type": "mcp_list_tools",
                "server_label": server_label,
                "tools": formatted_tools
            }
        )
        events.append(output_item_done_event)
        
        verbose_logger.debug(f"Created {len(events)} MCP discovery events")
        
    except Exception as e:
        verbose_logger.error(f"Error creating MCP list tools events: {e}")
        import traceback
        traceback.print_exc()
        
        # Emit failed event on error
        failed_event = MCPListToolsFailedEvent(
            type=ResponsesAPIStreamEvents.MCP_LIST_TOOLS_FAILED,
            sequence_number=2,
            output_index=0,
            item_id=base_item_id,
        )
        events.append(failed_event)
        
        # Still emit output_item.done event even on failure (with empty tools list)
        from litellm.types.llms.openai import OutputItemDoneEvent
        
        output_item_done_event = OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            output_index=0,
            item={
                "id": base_item_id,
                "type": "mcp_list_tools",
                "server_label": "",
                "tools": []
            }
        )
        events.append(output_item_done_event)
    
    return events


def create_mcp_call_events(
    tool_name: str, 
    tool_call_id: str, 
    arguments: str,
    result: Optional[str] = None,
    base_item_id: Optional[str] = None,
    sequence_start: int = 1
) -> List[ResponsesAPIStreamingResponse]:
    """Create MCP call events following OpenAI's specification"""
    events: List[ResponsesAPIStreamingResponse] = []
    item_id = base_item_id or f"mcp_{uuid.uuid4().hex[:8]}"
    
    # MCP call in progress event
    in_progress_event = MCPCallInProgressEvent(
        type=ResponsesAPIStreamEvents.MCP_CALL_IN_PROGRESS,
        sequence_number=sequence_start,
        output_index=0,
        item_id=item_id,
    )
    events.append(in_progress_event)
    
    # MCP call arguments delta event (streaming the arguments)
    arguments_delta_event = MCPCallArgumentsDeltaEvent(
        type=ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DELTA,
        output_index=0,
        item_id=item_id,
        delta=arguments,  # JSON string with arguments
        sequence_number=sequence_start + 1,
    )
    events.append(arguments_delta_event)
    
    # MCP call arguments done event
    arguments_done_event = MCPCallArgumentsDoneEvent(
        type=ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DONE,
        output_index=0,
        item_id=item_id,
        arguments=arguments,  # Complete JSON string with finalized arguments
        sequence_number=sequence_start + 2,
    )
    events.append(arguments_done_event)
    
    # MCP call completed event (or failed if result indicates failure)
    if result is not None:
        completed_event = MCPCallCompletedEvent(
            type=ResponsesAPIStreamEvents.MCP_CALL_COMPLETED,
            sequence_number=sequence_start + 3,
            item_id=item_id,
            output_index=0,
        )
        events.append(completed_event)
        
        # Add output_item.done event with the tool call result
        from litellm.types.llms.openai import OutputItemDoneEvent
        
        output_item_done_event = OutputItemDoneEvent(
            type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
            output_index=0,
            item={
                "id": item_id,
                "type": "mcp_call",
                "approval_request_id": f"mcpr_{uuid.uuid4().hex[:8]}",
                "arguments": arguments,
                "error": None,
                "name": tool_name,
                "output": result,
                "server_label": "litellm"
            },
        )
        events.append(output_item_done_event)
    else:
        failed_event = MCPCallFailedEvent(
            type=ResponsesAPIStreamEvents.MCP_CALL_FAILED,
            sequence_number=sequence_start + 3,
            item_id=item_id,
            output_index=0,
        )
        events.append(failed_event)
    
    return events


class MCPEnhancedStreamingIterator(BaseResponsesAPIStreamingIterator):
    """
    A complete MCP streaming iterator that handles the entire flow:
    1. Immediately emits MCP discovery events
    2. Makes the first LLM call and streams its response
    3. Handles tool execution and follow-up calls for auto-execute tools
    4. Emits tool execution events in the stream
    """
    
    def __init__(
        self,
        base_iterator: Any,  # Can be None - will be created internally
        mcp_events: List[ResponsesAPIStreamingResponse],
        mcp_tools_with_litellm_proxy: Optional[List[Any]] = None,
        user_api_key_auth: Any = None,
        original_request_params: Optional[Dict[str, Any]] = None
    ):
        # MCP setup
        self.mcp_tools_with_litellm_proxy = mcp_tools_with_litellm_proxy or []
        self.user_api_key_auth = user_api_key_auth
        self.original_request_params = original_request_params or {}
        self.should_auto_execute = self._should_auto_execute_tools()
        
        # Streaming state management
        self.phase = "mcp_discovery"  # mcp_discovery -> initial_response -> tool_execution -> follow_up_response -> finished
        self.finished = False
        
        # Event queues and generation flags
        self.mcp_discovery_events: List[ResponsesAPIStreamingResponse] = mcp_events  # Pre-generated MCP discovery events
        self.tool_execution_events: List[ResponsesAPIStreamingResponse] = []
        self.mcp_discovery_generated = True  # Events are already generated
        self.mcp_events = mcp_events  # Store the initial MCP events for backward compatibility
        
        # Iterator references
        self.base_iterator: Optional[Union[Any, ResponsesAPIResponse]] = base_iterator  # Will be created when needed
        self.follow_up_iterator: Optional[Any] = None
        
        # Response collection for tool execution
        self.collected_response: Optional[ResponsesAPIResponse] = None
        
        # Set up model metadata (will be updated when we get the real iterator)
        self.model = self.original_request_params.get('model', 'unknown')
        self.litellm_metadata = {}
        self.custom_llm_provider = self.original_request_params.get('custom_llm_provider', None)
        
        # Mark as async iterator
        self.is_async = True
        
    def _should_auto_execute_tools(self) -> bool:
        """Check if tools should be auto-executed"""
        from litellm.responses.mcp.litellm_proxy_mcp_handler import (
            LiteLLM_Proxy_MCP_Handler,
        )
        return LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(
            self.mcp_tools_with_litellm_proxy
        )

    def __aiter__(self):
        return self

    async def __anext__(self) -> ResponsesAPIStreamingResponse:
        """
        Phase-based streaming:
        1. mcp_discovery - Emit MCP discovery events
        2. initial_response - Stream the first LLM response  
        3. tool_execution - Emit tool execution events
        4. follow_up_response - Stream the follow-up response
        5. finished - End iteration
        """
        
        # Phase 1: MCP Discovery Events
        if self.phase == "mcp_discovery":
            # Generate MCP discovery events if not already done
            # MCP discovery events are already generated and available
            
            # Emit MCP discovery events
            if self.mcp_discovery_events:
                return self.mcp_discovery_events.pop(0)
            
            # All MCP discovery events emitted, move to next phase
            verbose_logger.debug("MCP discovery phase complete, transitioning to initial_response")
            self.phase = "initial_response"
            await self._create_initial_response_iterator()
            # Fall through to process the initial response immediately
        
        # Phase 2: Initial Response Stream
        if self.phase == "initial_response":
            if self.base_iterator:
                # Check if base_iterator is actually iterable
                if hasattr(self.base_iterator, '__anext__'):
                    try:
                        chunk = await cast(Any, self.base_iterator).__anext__()  # type: ignore[attr-defined]
                        
                        # If auto-execution is enabled, check for completed responses
                        if self.should_auto_execute and self._is_response_completed(chunk):
                            # Collect the response for tool execution
                            response_obj = getattr(chunk, 'response', None)
                            if isinstance(response_obj, ResponsesAPIResponse):
                                self.collected_response = response_obj
                            # Move to tool execution phase after emitting this chunk
                            self.phase = "tool_execution"
                            await self._generate_tool_execution_events()
                        
                        return chunk
                    except StopAsyncIteration:
                        # Initial response ended, move to next phase
                        if self.should_auto_execute and self.collected_response:
                            self.phase = "tool_execution"
                            await self._generate_tool_execution_events()
                        else:
                            self.phase = "finished"
                            raise
                else:
                    # base_iterator is not async iterable (likely a ResponsesAPIResponse)
                    # Collect it for tool execution if needed
                    if self.should_auto_execute and isinstance(self.base_iterator, ResponsesAPIResponse):
                        self.collected_response = self.base_iterator
                        self.phase = "tool_execution"
                        await self._generate_tool_execution_events()
                    else:
                        self.phase = "finished"
                        raise StopAsyncIteration
        
        # Phase 3: Tool Execution Events
        if self.phase == "tool_execution":
            # Emit any queued tool execution events
            if self.tool_execution_events:
                return self.tool_execution_events.pop(0)
            
            # Move to follow-up response phase
            self.phase = "follow_up_response"
            await self._create_follow_up_iterator()
        
        # Phase 4: Follow-up Response Stream
        if self.phase == "follow_up_response":
            if self.follow_up_iterator:
                try:
                    return await cast(Any, self.follow_up_iterator).__anext__()  # type: ignore[attr-defined]
                except StopAsyncIteration:
                    self.phase = "finished"
                    raise
            else:
                self.phase = "finished"
                raise StopAsyncIteration
        
        # Phase 5: Finished
        if self.phase == "finished":
            raise StopAsyncIteration
            
        # Should not reach here
        raise StopAsyncIteration
    
    def _is_response_completed(self, chunk: ResponsesAPIStreamingResponse) -> bool:
        """Check if this chunk indicates the response is completed"""
        from litellm.types.llms.openai import ResponsesAPIStreamEvents
        return getattr(chunk, 'type', None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    
    
    async def _create_initial_response_iterator(self) -> None:
        """Create the initial response iterator by making the first LLM call"""
        try:
            # Import the core aresponses function that doesn't have MCP logic
            from litellm.responses.main import aresponses

            # Make the initial response API call - but avoid the MCP wrapper
            params = self.original_request_params.copy()
            params['stream'] = True  # Ensure streaming
            
            # Use the pre-fetched all_tools from original_request_params (no re-processing needed)
            params_for_llm = {}
            for key, value in params.items():
                params_for_llm[key] = value  # Copy all params as-is since tools are already processed
            
            tools_count = len(params_for_llm.get('tools', []))
            verbose_logger.debug(f"Making LLM call with {tools_count} tools")
            response = await aresponses(**params_for_llm)
            
            # Set the base iterator
            if hasattr(response, '__aiter__') or hasattr(response, '__iter__'):
                self.base_iterator = response
                # Copy metadata from the real iterator
                self.model = getattr(response, 'model', self.model)
                self.litellm_metadata = getattr(response, 'litellm_metadata', {})
                self.custom_llm_provider = getattr(response, 'custom_llm_provider', self.custom_llm_provider)
                verbose_logger.debug(f"Created base iterator: {type(self.base_iterator)}")
            else:
                # Non-streaming response - this shouldn't happen but handle it
                verbose_logger.warning(f"Got non-streaming response: {type(response)}")
                self.base_iterator = None
                self.phase = "finished"
                
        except Exception as e:
            verbose_logger.error(f"Error creating initial response iterator: {e}")
            import traceback
            traceback.print_exc()
            self.base_iterator = None
            self.phase = "finished"
    
    async def _generate_tool_execution_events(self) -> None:
        """Generate tool execution events and execute tools"""
        if not self.collected_response:
            return
            
        import uuid

        from litellm.responses.mcp.litellm_proxy_mcp_handler import (
            LiteLLM_Proxy_MCP_Handler,
        )
        
        try:
            # Extract tool calls from the response
            if self.collected_response is not None:
                tool_calls = LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_response(self.collected_response)  # type: ignore[arg-type]
            else:
                tool_calls = []
            if not tool_calls:
                return
            
            for tool_call in tool_calls:
                tool_name, tool_arguments, tool_call_id = LiteLLM_Proxy_MCP_Handler._extract_tool_call_details(tool_call)
                if tool_name and tool_call_id:
                    # Create MCP call events for this tool execution
                    call_events = create_mcp_call_events(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        arguments=tool_arguments or "{}",  # JSON string with arguments
                        result=None,  # Will be set after execution
                        base_item_id=f"mcp_{uuid.uuid4().hex[:8]}",
                        sequence_start=len(self.tool_execution_events) + 1
                    )
                    # Add the in_progress and arguments events (not the completed event yet)
                    self.tool_execution_events.extend(call_events[:-1])
            
            # Execute the tools
            tool_results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
                tool_calls=tool_calls,
                user_api_key_auth=self.user_api_key_auth
            )
            
            # Create completion events and output_item.done events for tool execution
            for tool_result in tool_results:
                tool_call_id = tool_result.get("tool_call_id", "unknown")
                result_text = tool_result.get("result", "")
                
                # Find matching tool name and arguments
                tool_name = "unknown"
                tool_arguments = "{}"
                for tool_call in tool_calls:
                    name, args, call_id = LiteLLM_Proxy_MCP_Handler._extract_tool_call_details(tool_call)
                    if call_id == tool_call_id:
                        tool_name = name or "unknown"
                        tool_arguments = args or "{}"
                        break
                
                item_id = f"mcp_{uuid.uuid4().hex[:8]}"
                
                # Create the completion event
                completed_event = MCPCallCompletedEvent(
                    type=ResponsesAPIStreamEvents.MCP_CALL_COMPLETED,
                    sequence_number=len(self.tool_execution_events) + 1,
                    item_id=item_id,
                    output_index=0,
                )
                self.tool_execution_events.append(completed_event)
                
                # Create output_item.done event with the tool call result
                from litellm.types.llms.openai import OutputItemDoneEvent
                
                output_item_done_event = OutputItemDoneEvent(
                    type=ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                    output_index=0,
                    item={
                        "id": item_id,
                        "type": "mcp_call",
                        "approval_request_id": f"mcpr_{uuid.uuid4().hex[:8]}",
                        "arguments": tool_arguments,
                        "error": None,
                        "name": tool_name,
                        "output": result_text,
                        "server_label": "litellm"  # or extract from tool config
                    },
                )
                self.tool_execution_events.append(output_item_done_event)
            
            # Store tool results for follow-up call
            self.tool_results = tool_results
            
        except Exception as e:
            verbose_logger.error(f"Error in tool execution: {e}")
            import traceback
            traceback.print_exc()
            self.tool_results = []
    
    async def _create_follow_up_iterator(self) -> None:
        """Create the follow-up response iterator with tool results"""
        if not self.collected_response or not hasattr(self, 'tool_results'):
            return
            
        from litellm.responses.main import aresponses
        from litellm.responses.mcp.litellm_proxy_mcp_handler import (
            LiteLLM_Proxy_MCP_Handler,
        )
        
        try:
            # Create follow-up input
            if self.collected_response is not None:
                follow_up_input = LiteLLM_Proxy_MCP_Handler._create_follow_up_input(
                    response=self.collected_response,  # type: ignore[arg-type]
                    tool_results=self.tool_results,
                    original_input=self.original_request_params.get('input')
                )
                
                # Make follow-up call with streaming
                follow_up_params = self.original_request_params.copy()
                follow_up_params.update({
                    'input': follow_up_input,
                    'previous_response_id': self.collected_response.id,  # type: ignore[attr-defined]
                    'stream': True
                })
            else:
                return
            # Remove tool_choice to avoid forcing more tool calls
            follow_up_params.pop('tool_choice', None)
            
            follow_up_response = await aresponses(**follow_up_params)
            
            # Set up the follow-up iterator
            if hasattr(follow_up_response, '__aiter__'):
                self.follow_up_iterator = follow_up_response
            
        except Exception as e:
            verbose_logger.error(f"Error creating follow-up iterator: {e}")
            import traceback
            traceback.print_exc()
            self.follow_up_iterator = None


    def __iter__(self):
        return self

    def __next__(self) -> ResponsesAPIStreamingResponse:
        # First, emit any queued MCP events
        if self.mcp_events:  # type: ignore[attr-defined]
            return self.mcp_events.pop(0)  # type: ignore[attr-defined]
        
        # Then delegate to the base iterator
        if not self.is_async:
            try:
                if self.base_iterator and hasattr(self.base_iterator, '__next__'):
                    return next(cast(Any, self.base_iterator))  # type: ignore[arg-type]
                else:
                    raise StopIteration
            except StopIteration:
                self.finished = True
                raise
        else:
            raise RuntimeError("Cannot use sync iteration on async iterator")
