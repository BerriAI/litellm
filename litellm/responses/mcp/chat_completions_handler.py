"""Helpers for handling MCP-aware `/chat/completions` requests."""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    cast,
)

from litellm._logging import verbose_logger
from litellm.responses.mcp.litellm_proxy_mcp_handler import (
    LiteLLM_Proxy_MCP_Handler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.utils import ModelResponse, ModelResponseStream, StreamingChoices
from litellm.utils import CustomStreamWrapper


def _add_mcp_metadata_to_response(
    response: Union[ModelResponse, CustomStreamWrapper],
    openai_tools: Optional[List],
    tool_calls: Optional[List] = None,
    tool_results: Optional[List] = None,
) -> None:
    """
    Add MCP metadata to response's provider_specific_fields.

    This function adds MCP-related information to the response so that
    clients can access which tools were available, which were called, and
    what results were returned.

    For ModelResponse: adds to choices[].message.provider_specific_fields
    For CustomStreamWrapper: stores in _hidden_params and automatically adds to
    final chunk's delta.provider_specific_fields via CustomStreamWrapper._add_mcp_metadata_to_final_chunk()
    """
    if isinstance(response, CustomStreamWrapper):
        # For streaming, store MCP metadata in _hidden_params
        # CustomStreamWrapper._add_mcp_metadata_to_final_chunk() will automatically
        # add it to the final chunk's delta.provider_specific_fields
        if not hasattr(response, "_hidden_params"):
            response._hidden_params = {}

        mcp_metadata = {}
        if openai_tools:
            mcp_metadata["mcp_list_tools"] = openai_tools
        if tool_calls:
            mcp_metadata["mcp_tool_calls"] = tool_calls
        if tool_results:
            mcp_metadata["mcp_call_results"] = tool_results

        if mcp_metadata:
            response._hidden_params["mcp_metadata"] = mcp_metadata
        return

    if not isinstance(response, ModelResponse):
        return

    if not hasattr(response, "choices") or not response.choices:
        return

    # Add MCP metadata to all choices' messages
    for choice in response.choices:
        message = getattr(choice, "message", None)
        if message is not None:
            # Get existing provider_specific_fields or create new dict
            provider_fields = getattr(message, "provider_specific_fields", None) or {}

            # Add MCP metadata
            if openai_tools:
                provider_fields["mcp_list_tools"] = openai_tools
            if tool_calls:
                provider_fields["mcp_tool_calls"] = tool_calls
            if tool_results:
                provider_fields["mcp_call_results"] = tool_results

            # Set the provider_specific_fields
            setattr(message, "provider_specific_fields", provider_fields)


class _SyncIteratorWrapper:
    """Wraps an async iterator for synchronous iteration."""

    def __init__(self, async_iterator: Any, loop: Any) -> None:
        self._async_iterator = async_iterator
        self._loop = loop
        self._iterator: Any = None

    def __iter__(self) -> "_SyncIteratorWrapper":
        return self

    def __next__(self) -> Any:
        import asyncio

        if self._iterator is None:
            aiter_result = self._async_iterator.__aiter__()
            if hasattr(aiter_result, "__await__"):
                self._iterator = self._loop.run_until_complete(aiter_result)
            else:
                self._iterator = aiter_result
        try:
            return self._loop.run_until_complete(self._iterator.__anext__())
        except StopAsyncIteration:
            raise StopIteration


class MCPStreamingIterator:
    """
    Async iterator that drives the MCP tool-execution loop for streaming responses.

    Phases:
    1. Yield chunks from the initial LLM stream.
    2. When the stream ends, execute any tool calls (MCP or A2A).
    3. Yield chunks from the follow-up LLM stream.
    """

    def __init__(
        self,
        stream_wrapper: Any,
        messages: List,
        tool_server_map: Any,
        user_api_key_auth: Any,
        mcp_auth_header: Optional[str],
        mcp_server_auth_headers: Any,
        oauth2_headers: Any,
        raw_headers: Any,
        litellm_call_id: Optional[str],
        litellm_trace_id: Optional[str],
        openai_tools: List,
        base_call_args: Dict[str, Any],
        agent_tool_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.stream_wrapper = stream_wrapper
        self.messages = messages
        self.tool_server_map = tool_server_map
        self.user_api_key_auth = user_api_key_auth
        self.mcp_auth_header = mcp_auth_header
        self.mcp_server_auth_headers = mcp_server_auth_headers
        self.oauth2_headers = oauth2_headers
        self.raw_headers = raw_headers
        self.litellm_call_id = litellm_call_id
        self.litellm_trace_id = litellm_trace_id
        self.openai_tools = openai_tools
        self.base_call_args = base_call_args
        self.agent_tool_map = agent_tool_map or {}
        self.collected_chunks: List[ModelResponseStream] = []
        self.tool_calls: Optional[List] = None
        self.tool_results: Optional[List] = None
        self.complete_response: Optional[ModelResponse] = None
        self.stream_exhausted = False
        self.tool_execution_done = False
        self.follow_up_stream: Optional[CustomStreamWrapper] = None
        self.follow_up_non_stream: Optional[ModelResponse] = None
        self.follow_up_iterator: Any = None
        self.follow_up_exhausted = False

    def __aiter__(self) -> "MCPStreamingIterator":
        return self

    def _add_mcp_list_tools_to_chunk(
        self, chunk: ModelResponseStream
    ) -> ModelResponseStream:
        """Add mcp_list_tools to the first chunk."""
        from litellm.types.utils import add_provider_specific_fields

        if not self.openai_tools:
            return chunk

        if hasattr(chunk, "choices") and chunk.choices:
            for choice in chunk.choices:
                if (
                    isinstance(choice, StreamingChoices)
                    and hasattr(choice, "delta")
                    and choice.delta
                ):
                    provider_fields = dict(
                        getattr(choice.delta, "provider_specific_fields", None) or {}
                    )
                    provider_fields["mcp_list_tools"] = self.openai_tools
                    add_provider_specific_fields(choice.delta, provider_fields)

        return chunk

    def _add_mcp_tool_metadata_to_final_chunk(
        self, chunk: ModelResponseStream
    ) -> ModelResponseStream:
        """Add mcp_tool_calls and mcp_call_results to the final chunk."""
        from litellm.types.utils import add_provider_specific_fields

        if hasattr(chunk, "choices") and chunk.choices:
            for choice in chunk.choices:
                if (
                    isinstance(choice, StreamingChoices)
                    and hasattr(choice, "delta")
                    and choice.delta
                ):
                    attr_value = getattr(choice.delta, "provider_specific_fields", None)
                    provider_fields = (
                        dict(attr_value) if isinstance(attr_value, dict) else {}
                    )

                    if self.tool_calls:
                        provider_fields["mcp_tool_calls"] = self.tool_calls
                    if self.tool_results:
                        provider_fields["mcp_call_results"] = self.tool_results

                    add_provider_specific_fields(choice.delta, provider_fields)

        return chunk

    async def __anext__(self) -> Any:
        # Phase 1: Collect and yield initial stream chunks
        if not self.stream_exhausted:
            if not hasattr(self, "_stream_iterator"):
                self._stream_iterator = self.stream_wrapper.__aiter__()
                _add_mcp_metadata_to_response(
                    response=self.stream_wrapper,
                    openai_tools=self.openai_tools,
                )

            try:
                chunk = await self._stream_iterator.__anext__()
                self.collected_chunks.append(chunk)

                if len(self.collected_chunks) == 1:
                    chunk = self._add_mcp_list_tools_to_chunk(chunk)

                is_final = (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and hasattr(chunk.choices[0], "finish_reason")
                    and chunk.choices[0].finish_reason is not None
                )

                if is_final:
                    self.stream_exhausted = True
                    await self._process_tool_calls()
                    chunk = self._add_mcp_tool_metadata_to_final_chunk(chunk)
                    if self.tool_results and self.complete_response:
                        await self._prepare_follow_up_call()

                return chunk
            except StopAsyncIteration:
                self.stream_exhausted = True
                await self._process_tool_calls()
                if self.collected_chunks:
                    final_chunk = self.collected_chunks[-1]
                    final_chunk = self._add_mcp_tool_metadata_to_final_chunk(
                        final_chunk
                    )
                    if self.tool_results and self.complete_response:
                        await self._prepare_follow_up_call()
                    return final_chunk

        # Phase 2: Yield follow-up stream chunks if available
        if self.follow_up_stream and not self.follow_up_exhausted:
            if not self.follow_up_iterator:
                self.follow_up_iterator = self.follow_up_stream.__aiter__()
                verbose_logger.debug("Follow-up stream iterator created")

            try:
                chunk = await self.follow_up_iterator.__anext__()
                verbose_logger.debug("Follow-up chunk yielded: %s", chunk)
                return chunk
            except StopAsyncIteration:
                self.follow_up_exhausted = True
                verbose_logger.debug("Follow-up stream exhausted")
                raise StopAsyncIteration

        # Phase 3: emit non-streaming follow-up answer as a synthetic final chunk
        if self.follow_up_non_stream is not None:
            from litellm.types.utils import ModelResponseStream, StreamingChoices

            non_stream = self.follow_up_non_stream
            self.follow_up_non_stream = None
            # Build a minimal streaming chunk from the ModelResponse
            content = ""
            if non_stream.choices:
                content = getattr(non_stream.choices[0].message, "content", "") or ""
            synthetic = ModelResponseStream(
                id=non_stream.id,
                model=non_stream.model or "",
                choices=[
                    StreamingChoices(
                        finish_reason="stop",
                        index=0,
                        delta={"role": "assistant", "content": content},  # type: ignore[arg-type]
                    )
                ],
            )
            return synthetic

        raise StopAsyncIteration

    async def _process_tool_calls(self) -> None:
        """Build complete response from collected chunks and execute any tool calls."""
        from litellm.main import stream_chunk_builder

        if self.tool_execution_done:
            return

        self.tool_execution_done = True

        if not self.collected_chunks:
            return

        complete_response = stream_chunk_builder(
            chunks=self.collected_chunks,
            messages=self.messages,
        )

        if isinstance(complete_response, ModelResponse):
            self.complete_response = complete_response
            self.tool_calls = (
                LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_chat_response(
                    response=complete_response
                )
            )

            if self.tool_calls:
                self.tool_results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
                    tool_server_map=self.tool_server_map,
                    tool_calls=self.tool_calls,
                    user_api_key_auth=self.user_api_key_auth,
                    mcp_auth_header=self.mcp_auth_header,
                    mcp_server_auth_headers=self.mcp_server_auth_headers,
                    oauth2_headers=self.oauth2_headers,
                    raw_headers=self.raw_headers,
                    litellm_call_id=self.litellm_call_id,
                    litellm_trace_id=self.litellm_trace_id,
                    agent_tool_map=self.agent_tool_map,
                )

    async def _prepare_follow_up_call(self) -> None:
        """Initiate the follow-up streaming call with tool results."""
        if self.follow_up_stream is not None:
            return

        if not self.tool_results or not self.complete_response:
            return

        follow_up_messages = (
            LiteLLM_Proxy_MCP_Handler._create_follow_up_messages_for_chat(
                original_messages=self.messages,
                response=self.complete_response,
                tool_results=self.tool_results,
            )
        )

        follow_up_call_args = {
            **self.base_call_args,
            "messages": follow_up_messages,
            "stream": True,
            "_skip_mcp_handler": True,
        }

        import litellm

        follow_up_response = await litellm.acompletion(**follow_up_call_args)

        if isinstance(follow_up_response, CustomStreamWrapper):
            self.follow_up_stream = follow_up_response
            verbose_logger.debug("Follow-up stream created successfully")
        elif isinstance(follow_up_response, ModelResponse):
            # Provider returned a non-streaming response despite stream=True.
            # Store it so __anext__ can yield it as a synthetic final chunk rather
            # than silently dropping the follow-up answer.
            self.follow_up_non_stream = follow_up_response
            verbose_logger.debug(
                "Follow-up response is non-streaming ModelResponse; will emit as final chunk"
            )
        else:
            verbose_logger.warning(
                "Follow-up response is unexpected type %s, answer may be dropped",
                type(follow_up_response),
            )


class MCPStreamWrapper(CustomStreamWrapper):
    """
    Thin ``CustomStreamWrapper`` subclass that delegates async iteration to
    an ``MCPStreamingIterator`` so that the MCP tool-execution loop is
    transparent to callers that consume the stream normally.
    """

    def __init__(
        self,
        original_wrapper: CustomStreamWrapper,
        custom_iterator: MCPStreamingIterator,
    ) -> None:
        super().__init__(
            completion_stream=None,
            model=getattr(original_wrapper, "model", "unknown"),
            logging_obj=getattr(original_wrapper, "logging_obj", None),
            custom_llm_provider=getattr(original_wrapper, "custom_llm_provider", None),
            stream_options=getattr(original_wrapper, "stream_options", None),
            make_call=getattr(original_wrapper, "make_call", None),
            _response_headers=getattr(original_wrapper, "_response_headers", None),
        )
        self._original_wrapper = original_wrapper
        self._custom_iterator = custom_iterator
        if hasattr(original_wrapper, "_hidden_params"):
            self._hidden_params = original_wrapper._hidden_params
        self._sync_iterator: Optional[_SyncIteratorWrapper] = None
        self._sync_loop: Any = None

    def __aiter__(self) -> MCPStreamingIterator:
        return self._custom_iterator

    def __iter__(self) -> _SyncIteratorWrapper:
        import asyncio

        if self._sync_iterator is None:
            self._sync_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._sync_loop)
            self._sync_iterator = _SyncIteratorWrapper(
                self._custom_iterator, self._sync_loop
            )
        return self._sync_iterator

    def __next__(self) -> Any:
        if self._sync_iterator is None:
            self.__iter__()
        assert self._sync_iterator is not None
        return next(self._sync_iterator)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original_wrapper, name)


async def acompletion_with_mcp(  # noqa: PLR0915
    model: str,
    messages: List,
    tools: Optional[List] = None,
    **kwargs: Any,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Async completion with MCP integration.

    This function handles MCP tool integration following the same pattern as aresponses_api_with_mcp.
    It's designed to be called from the synchronous completion() function and return a coroutine.

    When MCP tools with server_url="litellm_proxy" are provided, this function will:
    1. Get available tools from the MCP server manager
    2. Transform them to OpenAI format
    3. Call acompletion with the transformed tools
    4. If require_approval="never" and tool calls are returned, automatically execute them
    5. Make a follow-up call with the tool results
    """
    from litellm import acompletion as litellm_acompletion
    from litellm.proxy.agent_endpoints.registry_orchestrator import RegistryOrchestrator

    # Parse MCP tools and separate from other tools
    (
        mcp_tools_with_litellm_proxy,
        other_tools,
    ) = LiteLLM_Proxy_MCP_Handler._parse_mcp_tools(tools)

    # Parse A2A agent tools from what remains
    (
        agent_tool_configs,
        other_tools,
    ) = RegistryOrchestrator.parse_agent_tool_configs(other_tools)

    if not mcp_tools_with_litellm_proxy and not agent_tool_configs:
        # No MCP or agent tools, proceed with regular completion
        return await litellm_acompletion(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )

    # Extract user_api_key_auth from metadata or kwargs
    user_api_key_auth = kwargs.get("user_api_key_auth") or (
        (kwargs.get("metadata", {}) or {}).get("user_api_key_auth")
    )

    # Extract MCP auth headers before fetching tools (needed for dynamic auth)
    (
        mcp_auth_header,
        mcp_server_auth_headers,
        oauth2_headers,
        raw_headers,
    ) = ResponsesAPIRequestUtils.extract_mcp_headers_from_request(
        secret_fields=kwargs.get("secret_fields"),
        tools=tools,
    )

    # Process MCP tools (pass auth headers for dynamic auth)
    (
        deduplicated_mcp_tools,
        tool_server_map,
    ) = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
        user_api_key_auth=user_api_key_auth,
        mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy,
        litellm_trace_id=kwargs.get("litellm_trace_id"),
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
    )

    # Apply per-tool semantic filter if any MCP or agent tool config has semantic_filter=true
    if any(
        isinstance(t, dict) and t.get("semantic_filter")
        for t in list(mcp_tools_with_litellm_proxy) + list(agent_tool_configs)
    ):
        deduplicated_mcp_tools = await RegistryOrchestrator.apply_semantic_filter(
            tools=deduplicated_mcp_tools,
            messages=messages,
        )

    openai_tools = LiteLLM_Proxy_MCP_Handler._transform_mcp_tools_to_openai(
        deduplicated_mcp_tools,
        target_format="chat",
    )

    # Wrap registered A2A agents as function tools
    agent_function_tools: List = []
    agent_tool_map: dict = {}
    if agent_tool_configs:
        agent_function_tools, agent_tool_map = (
            await RegistryOrchestrator.resolve_agent_tools(
                user_api_key_auth=user_api_key_auth,
            )
        )

    # Combine all tool types
    combined = openai_tools + agent_function_tools + other_tools
    all_tools: Optional[List] = combined if combined else None

    # Determine if we should auto-execute tools (MCP or agent tools with require_approval="never")
    should_auto_execute = LiteLLM_Proxy_MCP_Handler._should_auto_execute_tools(
        mcp_tools_with_litellm_proxy=mcp_tools_with_litellm_proxy
    ) or any(
        isinstance(t, dict) and t.get("require_approval") == "never"
        for t in agent_tool_configs
    )

    # Prepare call parameters
    # Remove keys that shouldn't be passed to acompletion
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in ["acompletion"]}

    base_call_args = {
        "model": model,
        "messages": messages,
        "tools": all_tools,
        "_skip_mcp_handler": True,  # Prevent recursion
        **clean_kwargs,
    }

    # If not auto-executing, just make the call with transformed tools
    if not should_auto_execute:
        response = await litellm_acompletion(**base_call_args)
        if isinstance(response, (ModelResponse, CustomStreamWrapper)):
            _add_mcp_metadata_to_response(
                response=response,
                openai_tools=openai_tools,
            )
        return response

    # For auto-execute: handle streaming vs non-streaming differently
    stream = kwargs.get("stream", False)
    mock_tool_calls = base_call_args.pop("mock_tool_calls", None)

    if stream:
        # Streaming mode: make initial call with streaming, collect chunks, detect tool calls
        initial_call_args = dict(base_call_args)
        initial_call_args["stream"] = True
        if mock_tool_calls is not None:
            initial_call_args["mock_tool_calls"] = mock_tool_calls
        _agent_tool_map = agent_tool_map  # capture for closure

        # Make initial streaming call
        initial_stream = await litellm_acompletion(**initial_call_args)

        if not isinstance(initial_stream, CustomStreamWrapper):
            # Not a stream, return as-is
            if isinstance(initial_stream, ModelResponse):
                _add_mcp_metadata_to_response(
                    response=initial_stream,
                    openai_tools=openai_tools,
                )
            return initial_stream

        # Create the MCP streaming iterator (module-level class)
        iterator = MCPStreamingIterator(
            stream_wrapper=initial_stream,
            messages=messages,
            tool_server_map=tool_server_map,
            user_api_key_auth=user_api_key_auth,
            mcp_auth_header=mcp_auth_header,
            mcp_server_auth_headers=mcp_server_auth_headers,
            oauth2_headers=oauth2_headers,
            raw_headers=raw_headers,
            litellm_call_id=kwargs.get("litellm_call_id"),
            litellm_trace_id=kwargs.get("litellm_trace_id"),
            openai_tools=openai_tools,
            base_call_args=base_call_args,
            agent_tool_map=_agent_tool_map,
        )

        return cast(CustomStreamWrapper, MCPStreamWrapper(initial_stream, iterator))

    # Non-streaming mode: use existing logic
    initial_call_args = dict(base_call_args)
    initial_call_args["stream"] = False
    if mock_tool_calls is not None:
        initial_call_args["mock_tool_calls"] = mock_tool_calls

    # Make initial call
    initial_response = await litellm_acompletion(**initial_call_args)

    if not isinstance(initial_response, ModelResponse):
        return initial_response

    # Extract tool calls from response
    tool_calls = LiteLLM_Proxy_MCP_Handler._extract_tool_calls_from_chat_response(
        response=initial_response
    )

    if not tool_calls:
        _add_mcp_metadata_to_response(
            response=initial_response,
            openai_tools=openai_tools,
        )
        return initial_response

    # Execute tool calls (MCP + A2A agents)
    tool_results = await LiteLLM_Proxy_MCP_Handler._execute_tool_calls(
        tool_server_map=tool_server_map,
        tool_calls=tool_calls,
        user_api_key_auth=user_api_key_auth,
        mcp_auth_header=mcp_auth_header,
        mcp_server_auth_headers=mcp_server_auth_headers,
        oauth2_headers=oauth2_headers,
        raw_headers=raw_headers,
        litellm_call_id=kwargs.get("litellm_call_id"),
        litellm_trace_id=kwargs.get("litellm_trace_id"),
        agent_tool_map=agent_tool_map,
    )

    if not tool_results:
        _add_mcp_metadata_to_response(
            response=initial_response,
            openai_tools=openai_tools,
            tool_calls=tool_calls,
        )
        return initial_response

    # Create follow-up messages with tool results
    follow_up_messages = LiteLLM_Proxy_MCP_Handler._create_follow_up_messages_for_chat(
        original_messages=messages,
        response=initial_response,
        tool_results=tool_results,
    )

    # Make follow-up call with original stream setting
    follow_up_call_args = dict(base_call_args)
    follow_up_call_args["messages"] = follow_up_messages
    follow_up_call_args["stream"] = stream

    response = await litellm_acompletion(**follow_up_call_args)
    if isinstance(response, (ModelResponse, CustomStreamWrapper)):
        _add_mcp_metadata_to_response(
            response=response,
            openai_tools=openai_tools,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
    return response
