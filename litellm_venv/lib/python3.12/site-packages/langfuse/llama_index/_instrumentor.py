import uuid
from contextlib import contextmanager
from logging import getLogger
from typing import Any, Dict, List, Optional

import httpx

from langfuse import Langfuse
from langfuse.client import StatefulTraceClient, StateType
from langfuse.types import MaskFunction
from langfuse.utils.langfuse_singleton import LangfuseSingleton

from ._context import InstrumentorContext
from ._event_handler import LlamaIndexEventHandler
from ._span_handler import LlamaIndexSpanHandler

try:
    from llama_index.core.instrumentation import get_dispatcher
except ImportError:
    raise ModuleNotFoundError(
        "Please install llama-index to use the Langfuse llama-index integration: 'pip install llama-index'"
    )

logger = getLogger(__name__)


class LlamaIndexInstrumentor:
    """Instrumentor for exporting LlamaIndex instrumentation module spans to Langfuse.

    This beta integration is currently under active development and subject to change.
    Please provide feedback to the Langfuse team: https://github.com/langfuse/langfuse/issues/1931

    For production setups, please use the existing callback-based integration (LlamaIndexCallbackHandler).

    Usage:
        instrumentor = LlamaIndexInstrumentor()
        instrumentor.start()

        # After calling start(), all LlamaIndex executions will be automatically traced

        # To trace a specific execution or set custom trace ID/params, use the context manager:
        with instrumentor.observe(trace_id="unique_trace_id", user_id="user123"):
            # Your LlamaIndex code here
            index = get_llama_index_index()
            response = index.as_query_engine().query("Your query here")

        instrumentor.flush()

    The instrumentor will automatically capture and log events and spans from LlamaIndex
    to Langfuse, providing detailed observability into your LLM application.

    Args:
        public_key (Optional[str]): Langfuse public key
        secret_key (Optional[str]): Langfuse secret key
        host (Optional[str]): Langfuse API host
        debug (Optional[bool]): Enable debug logging
        threads (Optional[int]): Number of threads for async operations
        flush_at (Optional[int]): Number of items to flush at
        flush_interval (Optional[int]): Flush interval in seconds
        max_retries (Optional[int]): Maximum number of retries for failed requests
        timeout (Optional[int]): Timeout for requests in seconds
        httpx_client (Optional[httpx.Client]): Custom HTTPX client
        enabled (Optional[bool]): Enable/disable the instrumentor
        sample_rate (Optional[float]): Sample rate for logging (0.0 to 1.0)
        mask (langfuse.types.MaskFunction): Masking function for 'input' and 'output' fields in events. Function must take a single keyword argument `data` and return a serializable, masked version of the data.
        environment (optional): The tracing environment. Can be any lowercase alphanumeric string with hyphens and underscores that does not start with 'langfuse'. Can bet set via `LANGFUSE_TRACING_ENVIRONMENT` environment variable.
    """

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        debug: Optional[bool] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
        enabled: Optional[bool] = None,
        sample_rate: Optional[float] = None,
        mask: Optional[MaskFunction] = None,
        environment: Optional[str] = None,
    ):
        self._langfuse = LangfuseSingleton().get(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            debug=debug,
            threads=threads,
            flush_at=flush_at,
            flush_interval=flush_interval,
            max_retries=max_retries,
            timeout=timeout,
            httpx_client=httpx_client,
            enabled=enabled,
            sample_rate=sample_rate,
            mask=mask,
            sdk_integration="llama-index_instrumentation",
            environment=environment,
        )
        self._span_handler = LlamaIndexSpanHandler(langfuse_client=self._langfuse)
        self._event_handler = LlamaIndexEventHandler(langfuse_client=self._langfuse)
        self._context = InstrumentorContext()

    def start(self):
        """Start the automatic tracing of LlamaIndex operations.

        Once called, all subsequent LlamaIndex executions will be automatically traced
        and logged to Langfuse without any additional code changes required.

        Example:
            ```python
            instrumentor = LlamaIndexInstrumentor()
            instrumentor.start()

            # From this point, all LlamaIndex operations are automatically traced
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            response = query_engine.query("What is the capital of France?")

            # The above operations will be automatically logged to Langfuse
            instrumentor.flush()
            ```
        """
        self._context.reset()
        dispatcher = get_dispatcher()

        # Span Handler
        if not any(
            isinstance(handler, type(self._span_handler))
            for handler in dispatcher.span_handlers
        ):
            dispatcher.add_span_handler(self._span_handler)

        # Event Handler
        if not any(
            isinstance(handler, type(self._event_handler))
            for handler in dispatcher.event_handlers
        ):
            dispatcher.add_event_handler(self._event_handler)

    def stop(self):
        """Stop the automatic tracing of LlamaIndex operations.

        This method removes the span and event handlers from the LlamaIndex dispatcher,
        effectively stopping the automatic tracing and logging to Langfuse.

        After calling this method, LlamaIndex operations will no longer be automatically
        traced unless `start()` is called again.

        Example:
            ```python
            instrumentor = LlamaIndexInstrumentor()
            instrumentor.start()

            # LlamaIndex operations are automatically traced here

            instrumentor.stop()

            # LlamaIndex operations are no longer automatically traced
            ```
        """
        self._context.reset()
        dispatcher = get_dispatcher()

        # Span Handler, in-place filter
        dispatcher.span_handlers[:] = filter(
            lambda h: not isinstance(h, type(self._span_handler)),
            dispatcher.span_handlers,
        )

        # Event Handler, in-place filter
        dispatcher.event_handlers[:] = filter(
            lambda h: not isinstance(h, type(self._event_handler)),
            dispatcher.event_handlers,
        )

    @contextmanager
    def observe(
        self,
        *,
        trace_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        update_parent: Optional[bool] = None,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        version: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        public: Optional[bool] = None,
    ):
        """Access context manager for observing and tracing LlamaIndex operations.

        This method allows you to wrap LlamaIndex operations in a context that
        automatically traces and logs them to Langfuse. It provides fine-grained
        control over the trace properties and ensures proper instrumentation.

        Args:
            trace_id (Optional[str]): Unique identifier for the trace. If not provided, a UUID will be generated.
            parent_observation_id (Optional[str]): ID of the parent observation, if any.
            update_parent (Optional[bool]): Whether to update the parent trace.
            trace_name (Optional[str]): Name of the trace.
            user_id (Optional[str]): ID of the user associated with this trace.
            session_id (Optional[str]): ID of the session associated with this trace.
            version (Optional[str]): Version information for this trace.
            release (Optional[str]): Release information for this trace.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the trace.
            tags (Optional[List[str]]): Tags associated with this trace.
            public (Optional[bool]): Whether this trace should be public.

        Yields:
            StatefulTraceClient: A client for interacting with the current trace.

        Example:
            ```python
            instrumentor = LlamaIndexInstrumentor()

            with instrumentor.observe(trace_id="unique_id", user_id="user123"):
                # LlamaIndex operations here will be traced
                index.as_query_engine().query("What is the capital of France?")

            # Tracing stops after the context manager exits

            instrumentor.flush()
            ```

        Note:
            If the instrumentor is not already started, this method will start it
            for the duration of the context and stop it afterwards.
        """
        was_instrumented = self._is_instrumented

        if not was_instrumented:
            self.start()

        if parent_observation_id is not None and trace_id is None:
            logger.warning(
                "trace_id must be provided if parent_observation_id is provided. Ignoring parent_observation_id."
            )
            parent_observation_id = None

        final_trace_id = trace_id or str(uuid.uuid4())

        self._context.update(
            is_user_managed_trace=True,
            trace_id=final_trace_id,
            parent_observation_id=parent_observation_id,
            update_parent=update_parent,
            trace_name=trace_name,
            user_id=user_id,
            session_id=session_id,
            version=version,
            release=release,
            metadata=metadata,
            tags=tags,
            public=public,
        )

        yield self._get_trace_client(final_trace_id)

        self._context.reset()

        if not was_instrumented:
            self.stop()

    @property
    def _is_instrumented(self) -> bool:
        """Check if the dispatcher is instrumented."""
        dispatcher = get_dispatcher()

        return any(
            isinstance(handler, type(self._span_handler))
            for handler in dispatcher.span_handlers
        ) and any(
            isinstance(handler, type(self._event_handler))
            for handler in dispatcher.event_handlers
        )

    def _get_trace_client(self, trace_id: str) -> StatefulTraceClient:
        return StatefulTraceClient(
            client=self._langfuse.client,
            id=trace_id,
            trace_id=trace_id,
            task_manager=self._langfuse.task_manager,
            state_type=StateType.TRACE,
            environment=self._langfuse.environment,
        )

    @property
    def client_instance(self) -> Langfuse:
        """Return the Langfuse client instance associated with this instrumentor.

        This property provides access to the underlying Langfuse client, allowing
        direct interaction with Langfuse functionality if needed.

        Returns:
            Langfuse: The Langfuse client instance.
        """
        return self._langfuse

    def flush(self) -> None:
        """Flush any pending tasks in the task manager.

        This method ensures that all queued tasks are sent to Langfuse immediately.
        It's useful for scenarios where you want to guarantee that all instrumentation
        data has been transmitted before your application terminates or moves on to
        a different phase.

        Note:
            This method is a wrapper around the `flush()` method of the underlying
            Langfuse client instance. It's provided here for convenience and to maintain
            a consistent interface within the instrumentor.

        Example:
            ```python
            instrumentor = LlamaIndexInstrumentor(langfuse_client)
            # ... perform some operations ...
            instrumentor.flush()  # Ensure all data is sent to Langfuse
            ```
        """
        self.client_instance.flush()
