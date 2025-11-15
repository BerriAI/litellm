"""
Transformation and completion handling for Apple Foundation Models provider.

This provider integrates with Apple's on-device Foundation Models and does not use HTTP.
Refactored to follow LiteLLM patterns with class-based architecture and DRY principles.
"""

import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    import httpx
else:
    httpx = None

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base import BaseLLM
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

from ..common_utils import get_apple_async_session_class, get_apple_session_class


class AppleFoundationModelsConfig(BaseConfig):
    """
    Configuration and transformation logic for Apple Foundation Models provider.
    """

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_guardrails: Optional[bool] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "apple_foundation_models"

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "temperature",
            "max_tokens",
            "tools",
            "tool_choice",
            "response_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        """Map OpenAI params to Apple Foundation Models params."""
        for param, value in non_default_params.items():
            if param in self.get_supported_openai_params(model):
                optional_params[param] = value
        return optional_params

    def validate_environment(self, *args: Any, **kwargs: Any) -> dict:
        """
        Validate environment for Apple Foundation Models.

        No API key needed - availability is checked via client.is_ready().
        """
        headers = kwargs.get("headers", {})
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request for Apple Foundation Models.

        Not used since this provider doesn't make HTTP requests.
        """
        return {}

    def transform_response(
        self,
        model: str,
        raw_response: Any,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform response from Apple Foundation Models.

        Not used since responses are handled directly in completion methods.
        """
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Any]
    ) -> BaseLLMException:
        """Get appropriate error class for Apple Foundation Models errors."""
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers if isinstance(headers, dict) else {},
        )

    def _extract_prompt_and_instructions(
        self, messages: List[AllMessageValues]
    ) -> tuple[str, Optional[str]]:
        """
        Extract system instructions and convert messages to prompt string.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Tuple of (prompt, system_instructions)
        """
        system_messages = []
        prompt_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if content is None:
                continue

            content_str = str(content) if not isinstance(content, str) else content

            if role == "system":
                system_messages.append(content_str)
            elif role in ("user", "assistant"):
                prompt_messages.append(f"{role}: {content_str}")

        system_instructions = "\n\n".join(system_messages) if system_messages else None
        prompt = "\n".join(prompt_messages)

        return prompt, system_instructions

    def _parse_tool_calls_from_transcript(
        self, transcript: List[Dict]
    ) -> Optional[List[ChatCompletionMessageToolCall]]:
        """
        Parse Apple's transcript to extract tool calls in OpenAI format.
        """
        tool_calls = []

        for entry in transcript:
            if entry.get("type") == "tool_call":
                tool_call = ChatCompletionMessageToolCall(
                    id=entry.get("tool_id", ""),
                    type="function",
                    function=Function(
                        name=entry.get("tool_name", ""),
                        arguments=entry.get("arguments", "{}"),
                    ),
                )
                tool_calls.append(tool_call)

        return tool_calls if tool_calls else None

    def _extract_json_schema(self, optional_params: dict) -> Optional[Dict]:
        """
        Extract JSON schema from response_format parameter.

        Supports:
        1. Pydantic BaseModel classes
        2. Dict with json_schema
        3. Dict with response_schema (legacy)
        """
        response_format = optional_params.get("response_format")
        if not response_format:
            return None

        if not isinstance(response_format, dict):
            try:
                response_format = type_to_response_format_param(response_format)
            except Exception as e:
                verbose_logger.warning(
                    f"Failed to convert response_format to schema: {e}"
                )
                return None

        if not response_format or not isinstance(response_format, dict):
            return None

        if "json_schema" in response_format:
            return response_format["json_schema"].get("schema")
        elif "response_schema" in response_format:
            return response_format["response_schema"]

        return None

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Apple Foundation Models doesn't provide token counts,
        so we use a simple estimation (roughly 4 chars per token).
        """
        return len(text) // 4 if text else 0

    def _build_usage(self, prompt: str, response_text: str) -> Usage:
        """Build usage information with token estimates."""
        prompt_tokens = self._estimate_tokens(prompt)
        completion_tokens = self._estimate_tokens(response_text)

        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class _SyncStreamingAdapter:
    """
    Adapter turning Apple SDK's sync StreamChunk iterator into LiteLLM chunks.
    """

    def __init__(
        self,
        sdk_iterator: Iterator[Any],
        prompt: str,
        config: AppleFoundationModelsConfig,
    ):
        self.sdk_iterator = sdk_iterator
        self.prompt = prompt
        self.config = config
        self.full_response = ""
        self._finished = False

    def __iter__(self) -> "_SyncStreamingAdapter":
        return self

    def __next__(self) -> GenericStreamingChunk:
        if self._finished:
            raise StopIteration

        try:
            chunk = next(self.sdk_iterator)
        except StopIteration:
            return self._complete()

        chunk_text = getattr(chunk, "content", None)
        if chunk_text is None:
            chunk_text = str(chunk)
        if not chunk_text:
            return self.__next__()

        self.full_response += chunk_text
        return GenericStreamingChunk(
            text=chunk_text,
            tool_use=None,
            is_finished=False,
            finish_reason="",
            usage=None,
            index=0,
        )

    def _complete(self) -> GenericStreamingChunk:
        self._finished = True
        usage_obj = self.config._build_usage(self.prompt, self.full_response)
        return GenericStreamingChunk(
            text="",
            tool_use=None,
            is_finished=True,
            finish_reason="stop",
            usage={
                "prompt_tokens": usage_obj.prompt_tokens,
                "completion_tokens": usage_obj.completion_tokens,
                "total_tokens": usage_obj.total_tokens,
            },
            index=0,
        )


class _AsyncStreamingAdapter:
    """
    Adapter turning Apple SDK's async StreamChunk iterator into LiteLLM chunks.
    """

    def __init__(
        self,
        sdk_iterator: AsyncIterator[Any],
        prompt: str,
        config: AppleFoundationModelsConfig,
    ):
        self.sdk_iterator = sdk_iterator
        self.prompt = prompt
        self.config = config
        self.full_response = ""
        self._finished = False

    def __aiter__(self) -> "_AsyncStreamingAdapter":
        return self

    async def __anext__(self) -> GenericStreamingChunk:
        if self._finished:
            raise StopAsyncIteration

        chunk = await self._get_next_chunk()
        if chunk is None:
            return self._complete()

        chunk_text = getattr(chunk, "content", None)
        if chunk_text is None:
            chunk_text = str(chunk)
        if not chunk_text:
            return await self.__anext__()

        self.full_response += chunk_text
        return GenericStreamingChunk(
            text=chunk_text,
            tool_use=None,
            is_finished=False,
            finish_reason="",
            usage=None,
            index=0,
        )

    async def _get_next_chunk(self) -> Optional[Any]:
        try:
            return await self.sdk_iterator.__anext__()
        except StopAsyncIteration:
            return None

    def _complete(self) -> GenericStreamingChunk:
        self._finished = True
        usage_obj = self.config._build_usage(self.prompt, self.full_response)
        return GenericStreamingChunk(
            text="",
            tool_use=None,
            is_finished=True,
            finish_reason="stop",
            usage={
                "prompt_tokens": usage_obj.prompt_tokens,
                "completion_tokens": usage_obj.completion_tokens,
                "total_tokens": usage_obj.total_tokens,
            },
            index=0,
        )


class AppleFoundationModelsLLM(BaseLLM):
    """
    Handler class for Apple Foundation Models provider.

    Follows LiteLLM's BaseLLM pattern for consistency with other providers.
    """

    def __init__(self):
        super().__init__()
        self.config = AppleFoundationModelsConfig()

    def _build_tool_functions(
        self, optional_params: dict
    ) -> Optional[List[Callable[..., Any]]]:
        """
        Extract and prepare tool functions from optional parameters.

        Support two patterns:
        1. Just tool_functions (list or dict of callables) - Apple SDK can introspect
        2. tools (OpenAI schemas) + optional tool_functions (implementations)

        Args:
            optional_params: Dictionary containing tool_functions and/or tools parameters

        Returns:
            List of callable functions, or None if no tools provided
        """
        tool_functions_param = optional_params.get("tool_functions")
        tools_param = optional_params.get("tools")

        if tool_functions_param:
            if isinstance(tool_functions_param, dict):
                return list(tool_functions_param.values())
            return tool_functions_param
        elif tools_param:
            raise ValueError(
                "Apple Foundation Models requires callable implementations for each tool. "
                "Pass them via `tool_functions` (list or dict) so the SDK can invoke the actual Python functions."
            )

        return None

    def _make_session(
        self,
        session_cls: Any,
        system_instructions: Optional[str],
        tool_functions: Optional[List[Callable[..., Any]]],
    ) -> Any:
        """
        Create a session instance with the given class and parameters.

        Args:
            session_cls: Session or AsyncSession class
            system_instructions: System instructions for the session
            tool_functions: List of callable tool functions

        Returns:
            Session or AsyncSession instance
        """
        return session_cls(
            instructions=system_instructions or "You are a helpful assistant.",
            tools=tool_functions,
        )

    def _create_session(
        self,
        system_instructions: Optional[str],
        optional_params: dict,
        async_mode: bool = False,
    ) -> Any:
        """
        Create Apple Foundation Models session (sync or async based on flag).
        """
        session_cls = (
            get_apple_async_session_class() if async_mode else get_apple_session_class()
        )
        tool_functions = self._build_tool_functions(optional_params)
        return self._make_session(session_cls, system_instructions, tool_functions)

    def _extract_generation_params(self, optional_params: dict) -> dict:
        """Extract and prepare generation parameters."""
        params = {}
        if "temperature" in optional_params:
            params["temperature"] = optional_params["temperature"]
        if "max_tokens" in optional_params:
            params["max_tokens"] = optional_params["max_tokens"]
        # Note: SDK doesn't support seed parameter
        return params

    def _convert_tools_to_callables(
        self,
        tools: List[Dict],
        tool_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> List[Callable[..., Any]]:
        """
        Convert OpenAI tool format to callable functions for new SDK.

        The new SDK expects tools as a list of callables passed at session creation.
        """
        callables = []

        for tool in tools:
            if tool.get("type") != "function":
                continue

            func_def = tool.get("function", {})
            func_name = func_def.get("name")
            func_description = func_def.get("description", "")

            if tool_functions and func_name in tool_functions:
                tool_func = tool_functions[func_name]
                if not tool_func.__doc__:
                    tool_func.__doc__ = func_description
                callables.append(tool_func)

        return callables

    def _execute_generation(
        self,
        session: Any,
        prompt: str,
        generation_params: dict,
        schema: Optional[Dict] = None,
    ) -> tuple[str, Optional[List[ChatCompletionMessageToolCall]]]:
        """
        Sync generation execution path.

        Uses the new SDK's unified generate() method which handles both
        text and structured generation based on parameters.

        Note: Tools are registered at session creation in the new SDK,
        not during generation.
        """
        kwargs: dict = {}
        try:
            kwargs = self._build_generation_kwargs(generation_params, schema)
            verbose_logger.debug(
                f"Calling session.generate with prompt length={len(prompt)}, kwargs={kwargs}"
            )
            response = session.generate(prompt, **kwargs)
            return self._process_generation_response(
                response, schema_supplied=bool(schema)
            )
        except Exception as e:
            verbose_logger.error(
                f"Generation failed with schema={bool(schema)}, "
                f"kwargs={kwargs}, "
                f"error type={type(e).__name__}"
            )
            self._raise_generation_error(e)
            raise  # Ensure we never fall through without returning

    async def _execute_async_generation(
        self,
        session: Any,
        prompt: str,
        generation_params: dict,
        schema: Optional[Dict] = None,
    ) -> tuple[str, Optional[List[ChatCompletionMessageToolCall]]]:
        """
        Async generation execution path.

        Uses the SDK's unified generate() method (async version from AsyncSession).
        """
        kwargs: dict = {}
        try:
            kwargs = self._build_generation_kwargs(generation_params, schema)
            verbose_logger.debug(
                f"Calling async session.generate with prompt length={len(prompt)}, kwargs={kwargs}"
            )
            response = await session.generate(prompt, **kwargs)
            return self._process_generation_response(
                response, schema_supplied=bool(schema)
            )
        except Exception as e:
            verbose_logger.error(
                f"Async generation failed with schema={bool(schema)}, "
                f"kwargs={kwargs}, "
                f"error type={type(e).__name__}"
            )
            self._raise_generation_error(e)
            raise  # Ensure we never fall through without returning

    def _convert_tool_calls(
        self, sdk_tool_calls: List
    ) -> List[ChatCompletionMessageToolCall]:
        """
        Convert SDK's ToolCall objects to OpenAI format.

        New SDK returns ToolCall dataclass with id, type, and function attributes.
        """
        litellm_tool_calls = []

        for tool_call in sdk_tool_calls:
            litellm_tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=Function(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    ),
                )
            )

        return litellm_tool_calls

    def _build_generation_kwargs(
        self, generation_params: dict, schema: Optional[Dict]
    ) -> dict:
        """
        Prepare kwargs for Session.generate/AsyncSession.generate.

        Note: The Apple SDK expects JSON schemas in a specific format.
        If the model can't produce output matching the schema, it will
        raise a DecodingFailureError.
        """
        kwargs = generation_params.copy()
        if schema:
            if not isinstance(schema, dict):
                verbose_logger.warning(
                    f"Schema must be a dict, got {type(schema)}. Ignoring schema."
                )
            elif "type" not in schema:
                verbose_logger.warning(
                    "Schema missing 'type' field. This may cause deserialization errors."
                )
                kwargs["schema"] = schema
            else:
                kwargs["schema"] = schema
                verbose_logger.debug(f"Using schema with type: {schema.get('type')}")
        return kwargs

    def _process_generation_response(
        self, response: Any, schema_supplied: bool
    ) -> tuple[str, Optional[List[ChatCompletionMessageToolCall]]]:
        """Normalize SDK responses regardless of sync/async path."""
        if schema_supplied and hasattr(response, "parsed"):
            response_text = json.dumps(response.parsed)
        else:
            response_text = getattr(response, "text", "")

        tool_calls = (
            self._convert_tool_calls(response.tool_calls)
            if getattr(response, "tool_calls", None)
            else None
        )
        return response_text, tool_calls

    def _raise_generation_error(self, error: Exception) -> None:
        error_type = type(error).__name__
        error_msg = str(error)

        verbose_logger.error(f"Generation failed: {error_msg}")

        # Provide more helpful error messages for common SDK errors
        if "DecodingFailureError" in error_type or "deserialize" in error_msg.lower():
            helpful_msg = (
                f"Apple Foundation Models SDK failed to parse structured output. "
                f"This usually means the model's response didn't match the expected schema. "
                f"Original error: {error_msg}. "
                f"Try: 1) Simplifying your schema, 2) Adding clearer instructions in the prompt, "
                f"or 3) Using response_format without strict schema enforcement."
            )
            raise RuntimeError(helpful_msg) from error
        elif "tool" in error_msg.lower() and "not found" in error_msg.lower():
            helpful_msg = (
                f"Tool function error: {error_msg}. "
                f"Make sure all tools defined in 'tools' parameter have corresponding "
                f"implementations in 'tool_functions' parameter."
            )
            raise RuntimeError(helpful_msg) from error
        else:
            raise RuntimeError(f"Generation failed: {error_msg}") from error

    def _prepare_completion_inputs(
        self, messages: List[AllMessageValues], optional_params: dict
    ) -> tuple[str, Optional[str], dict, Optional[Dict]]:
        """
        Shared helper used by sync + async completion paths to gather
        prompt/system context, generation params, and structured schema.
        """
        prompt, system_instructions = self.config._extract_prompt_and_instructions(
            messages
        )
        generation_params = self._extract_generation_params(optional_params)
        schema = self.config._extract_json_schema(optional_params)
        return prompt, system_instructions, generation_params, schema

    def _build_completion_context(
        self,
        messages: List[AllMessageValues],
        optional_params: dict,
        async_mode: bool,
    ) -> tuple[str, dict, Optional[Dict], Any]:
        """
        Helper returning prompt, generation params, schema, and session for a completion run.
        """
        (
            prompt,
            system_instructions,
            generation_params,
            schema,
        ) = self._prepare_completion_inputs(messages, optional_params)
        session = self._create_session(
            system_instructions, optional_params, async_mode=async_mode
        )
        return prompt, generation_params, schema, session

    def _wrap_stream(
        self,
        iterator: Union[
            Iterator[GenericStreamingChunk], AsyncIterator[GenericStreamingChunk]
        ],
        model: str,
        logging_obj: LiteLLMLoggingObj,
    ) -> CustomStreamWrapper:
        """Consistent CustomStreamWrapper factory used by sync + async streaming paths."""
        return CustomStreamWrapper(
            completion_stream=iterator,
            model=model,
            custom_llm_provider="apple_foundation_models",
            logging_obj=logging_obj,
        )

    def _build_stream_iterator(
        self,
        session: Any,
        prompt: str,
        generation_params: dict,
        async_mode: bool,
    ) -> Union[_SyncStreamingAdapter, _AsyncStreamingAdapter]:
        """
        Create either the sync or async streaming iterator.
        """
        if async_mode:
            async_stream_gen = self._async_stream_generator(
                session, prompt, generation_params
            )
            return _AsyncStreamingAdapter(
                sdk_iterator=async_stream_gen,
                prompt=prompt,
                config=self.config,
            )

        stream_gen = self._stream_generator(session, prompt, generation_params)
        return _SyncStreamingAdapter(
            sdk_iterator=stream_gen,
            prompt=prompt,
            config=self.config,
        )

    def _build_model_response(
        self,
        model_response: ModelResponse,
        response_text: str,
        prompt: str,
        tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None,
    ) -> ModelResponse:
        """Build complete ModelResponse."""
        model_response.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content=response_text,
                    role="assistant",
                    tool_calls=tool_calls,
                ),
            )
        ]

        # Set usage tracking (type: ignore needed as usage is dynamically added)
        model_response.usage = self.config._build_usage(prompt, response_text)  # type: ignore

        return model_response

    def _stream_generator(
        self,
        session: Any,
        prompt: str,
        generation_params: dict,
    ) -> Iterator[Any]:
        """
        Sync generator for streaming responses.

        The new SDK's Session.generate(stream=True) returns a sync iterator
        of StreamChunk objects.
        """
        for chunk in session.generate(prompt, stream=True, **generation_params):
            if getattr(chunk, "content", None):
                yield chunk

    async def _async_stream_generator(
        self,
        session: Any,
        prompt: str,
        generation_params: dict,
    ) -> AsyncIterator[Any]:
        # When stream=True, session.generate() returns an async generator directly
        # (not a coroutine), so we don't await it
        stream_iterator = session.generate(prompt, stream=True, **generation_params)
        async for chunk in stream_iterator:
            yield chunk

    def _stream_completion(
        self,
        model: str,
        messages: List[AllMessageValues],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        async_mode: bool,
    ) -> CustomStreamWrapper:
        """Shared streaming handler for sync + async paths."""
        prompt, generation_params, _, session = self._build_completion_context(
            messages, optional_params, async_mode=async_mode
        )
        iterator = self._build_stream_iterator(
            session=session,
            prompt=prompt,
            generation_params=generation_params,
            async_mode=async_mode,
        )

        return self._wrap_stream(iterator, model=model, logging_obj=logging_obj)

    def completion_non_streaming(
        self,
        model: str,
        messages: List[AllMessageValues],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
    ) -> ModelResponse:
        """Handle synchronous non-streaming completion."""
        prompt, generation_params, schema, session = self._build_completion_context(
            messages, optional_params, async_mode=False
        )

        response_text, tool_calls = self._execute_generation(
            session=session,
            prompt=prompt,
            generation_params=generation_params,
            schema=schema,
        )

        return self._build_model_response(
            model_response, response_text, prompt, tool_calls
        )

    async def acompletion_non_streaming(
        self,
        model: str,
        messages: List[AllMessageValues],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
    ) -> ModelResponse:
        """
        Handle async non-streaming completion.

        Uses AsyncClient and AsyncSession for proper async support.
        """
        prompt, generation_params, schema, session = self._build_completion_context(
            messages, optional_params, async_mode=True
        )
        response_text, tool_calls = await self._execute_async_generation(
            session=session,
            prompt=prompt,
            generation_params=generation_params,
            schema=schema,
        )

        return self._build_model_response(
            model_response, response_text, prompt, tool_calls
        )

    def dispatch_completion(
        self,
        *,
        model: str,
        messages: List[AllMessageValues],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        stream: bool,
        async_mode: bool,
    ) -> Union[ModelResponse, CustomStreamWrapper, Coroutine[Any, Any, ModelResponse]]:
        """
        Central router that picks the appropriate handler based on stream + async flags.
        """
        if stream:
            return self._stream_completion(
                model=model,
                messages=messages,
                model_response=model_response,
                logging_obj=logging_obj,
                optional_params=optional_params,
                async_mode=async_mode,
            )

        if async_mode:
            return self.acompletion_non_streaming(
                model=model,
                messages=messages,
                model_response=model_response,
                logging_obj=logging_obj,
                optional_params=optional_params,
            )
        return self.completion_non_streaming(
            model=model,
            messages=messages,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
        )


def completion(
    model: str,
    messages: List[AllMessageValues],
    model_response: ModelResponse,
    logging_obj: LiteLLMLoggingObj,
    optional_params: dict,
    litellm_params: dict,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    client=None,
    acompletion: bool = False,
    stream: Optional[bool] = False,
    timeout: Optional[Union[float, int, "httpx.Timeout"]] = None,
) -> Union[ModelResponse, CustomStreamWrapper, Coroutine[Any, Any, ModelResponse]]:
    """
    Handle completion requests for Apple Foundation Models.

    Main entry point that routes to appropriate handler based on
    streaming and async flags.
    """
    verbose_logger.debug(
        f"Apple Foundation Models completion called with model={model}"
    )

    llm = AppleFoundationModelsLLM()

    # Convert None to False for stream parameter
    stream_value = stream if stream is not None else False

    verbose_logger.debug(
        "Apple Foundation Models dispatching with stream=%s async=%s",
        stream_value,
        acompletion,
    )
    return llm.dispatch_completion(
        model=model,
        messages=messages,
        model_response=model_response,
        logging_obj=logging_obj,
        optional_params=optional_params,
        stream=stream_value,
        async_mode=acompletion,
    )
