from collections import OrderedDict
from functools import wraps

import sentry_sdk
from sentry_sdk.ai.monitoring import set_ai_pipeline_name, record_token_usage
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.ai.utils import set_data_normalized
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import Span
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import logger, capture_internal_exceptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, List, Callable, Dict, Union, Optional
    from uuid import UUID

try:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from langchain_core.callbacks import (
        manager,
        BaseCallbackHandler,
    )
    from langchain_core.agents import AgentAction, AgentFinish
except ImportError:
    raise DidNotEnable("langchain not installed")


DATA_FIELDS = {
    "temperature": SPANDATA.AI_TEMPERATURE,
    "top_p": SPANDATA.AI_TOP_P,
    "top_k": SPANDATA.AI_TOP_K,
    "function_call": SPANDATA.AI_FUNCTION_CALL,
    "tool_calls": SPANDATA.AI_TOOL_CALLS,
    "tools": SPANDATA.AI_TOOLS,
    "response_format": SPANDATA.AI_RESPONSE_FORMAT,
    "logit_bias": SPANDATA.AI_LOGIT_BIAS,
    "tags": SPANDATA.AI_TAGS,
}

# To avoid double collecting tokens, we do *not* measure
# token counts for models for which we have an explicit integration
NO_COLLECT_TOKEN_MODELS = [
    "openai-chat",
    "anthropic-chat",
    "cohere-chat",
    "huggingface_endpoint",
]


class LangchainIntegration(Integration):
    identifier = "langchain"
    origin = f"auto.ai.{identifier}"

    # The most number of spans (e.g., LLM calls) that can be processed at the same time.
    max_spans = 1024

    def __init__(
        self, include_prompts=True, max_spans=1024, tiktoken_encoding_name=None
    ):
        # type: (LangchainIntegration, bool, int, Optional[str]) -> None
        self.include_prompts = include_prompts
        self.max_spans = max_spans
        self.tiktoken_encoding_name = tiktoken_encoding_name

    @staticmethod
    def setup_once():
        # type: () -> None
        manager._configure = _wrap_configure(manager._configure)


class WatchedSpan:
    span = None  # type: Span
    num_completion_tokens = 0  # type: int
    num_prompt_tokens = 0  # type: int
    no_collect_tokens = False  # type: bool
    children = []  # type: List[WatchedSpan]
    is_pipeline = False  # type: bool

    def __init__(self, span):
        # type: (Span) -> None
        self.span = span


class SentryLangchainCallback(BaseCallbackHandler):  # type: ignore[misc]
    """Base callback handler that can be used to handle callbacks from langchain."""

    span_map = OrderedDict()  # type: OrderedDict[UUID, WatchedSpan]

    max_span_map_size = 0

    def __init__(self, max_span_map_size, include_prompts, tiktoken_encoding_name=None):
        # type: (int, bool, Optional[str]) -> None
        self.max_span_map_size = max_span_map_size
        self.include_prompts = include_prompts

        self.tiktoken_encoding = None
        if tiktoken_encoding_name is not None:
            import tiktoken  # type: ignore

            self.tiktoken_encoding = tiktoken.get_encoding(tiktoken_encoding_name)

    def count_tokens(self, s):
        # type: (str) -> int
        if self.tiktoken_encoding is not None:
            return len(self.tiktoken_encoding.encode_ordinary(s))
        return 0

    def gc_span_map(self):
        # type: () -> None

        while len(self.span_map) > self.max_span_map_size:
            run_id, watched_span = self.span_map.popitem(last=False)
            self._exit_span(watched_span, run_id)

    def _handle_error(self, run_id, error):
        # type: (UUID, Any) -> None
        if not run_id or run_id not in self.span_map:
            return

        span_data = self.span_map[run_id]
        if not span_data:
            return
        sentry_sdk.capture_exception(error, span_data.span.scope)
        span_data.span.__exit__(None, None, None)
        del self.span_map[run_id]

    def _normalize_langchain_message(self, message):
        # type: (BaseMessage) -> Any
        parsed = {"content": message.content, "role": message.type}
        parsed.update(message.additional_kwargs)
        return parsed

    def _create_span(self, run_id, parent_id, **kwargs):
        # type: (SentryLangchainCallback, UUID, Optional[Any], Any) -> WatchedSpan

        watched_span = None  # type: Optional[WatchedSpan]
        if parent_id:
            parent_span = self.span_map.get(parent_id)  # type: Optional[WatchedSpan]
            if parent_span:
                watched_span = WatchedSpan(parent_span.span.start_child(**kwargs))
                parent_span.children.append(watched_span)
        if watched_span is None:
            watched_span = WatchedSpan(sentry_sdk.start_span(**kwargs))

        if kwargs.get("op", "").startswith("ai.pipeline."):
            if kwargs.get("name"):
                set_ai_pipeline_name(kwargs.get("name"))
            watched_span.is_pipeline = True

        watched_span.span.__enter__()
        self.span_map[run_id] = watched_span
        self.gc_span_map()
        return watched_span

    def _exit_span(self, span_data, run_id):
        # type: (SentryLangchainCallback, WatchedSpan, UUID) -> None

        if span_data.is_pipeline:
            set_ai_pipeline_name(None)

        span_data.span.__exit__(None, None, None)
        del self.span_map[run_id]

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        tags=None,
        parent_run_id=None,
        metadata=None,
        **kwargs,
    ):
        # type: (SentryLangchainCallback, Dict[str, Any], List[str], UUID, Optional[List[str]], Optional[UUID], Optional[Dict[str, Any]], Any) -> Any
        """Run when LLM starts running."""
        with capture_internal_exceptions():
            if not run_id:
                return
            all_params = kwargs.get("invocation_params", {})
            all_params.update(serialized.get("kwargs", {}))
            watched_span = self._create_span(
                run_id,
                kwargs.get("parent_run_id"),
                op=OP.LANGCHAIN_RUN,
                name=kwargs.get("name") or "Langchain LLM call",
                origin=LangchainIntegration.origin,
            )
            span = watched_span.span
            if should_send_default_pii() and self.include_prompts:
                set_data_normalized(span, SPANDATA.AI_INPUT_MESSAGES, prompts)
            for k, v in DATA_FIELDS.items():
                if k in all_params:
                    set_data_normalized(span, v, all_params[k])

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, Dict[str, Any], List[List[BaseMessage]], UUID, Any) -> Any
        """Run when Chat Model starts running."""
        with capture_internal_exceptions():
            if not run_id:
                return
            all_params = kwargs.get("invocation_params", {})
            all_params.update(serialized.get("kwargs", {}))
            watched_span = self._create_span(
                run_id,
                kwargs.get("parent_run_id"),
                op=OP.LANGCHAIN_CHAT_COMPLETIONS_CREATE,
                name=kwargs.get("name") or "Langchain Chat Model",
                origin=LangchainIntegration.origin,
            )
            span = watched_span.span
            model = all_params.get(
                "model", all_params.get("model_name", all_params.get("model_id"))
            )
            watched_span.no_collect_tokens = any(
                x in all_params.get("_type", "") for x in NO_COLLECT_TOKEN_MODELS
            )

            if not model and "anthropic" in all_params.get("_type"):
                model = "claude-2"
            if model:
                span.set_data(SPANDATA.AI_MODEL_ID, model)
            if should_send_default_pii() and self.include_prompts:
                set_data_normalized(
                    span,
                    SPANDATA.AI_INPUT_MESSAGES,
                    [
                        [self._normalize_langchain_message(x) for x in list_]
                        for list_ in messages
                    ],
                )
            for k, v in DATA_FIELDS.items():
                if k in all_params:
                    set_data_normalized(span, v, all_params[k])
            if not watched_span.no_collect_tokens:
                for list_ in messages:
                    for message in list_:
                        self.span_map[run_id].num_prompt_tokens += self.count_tokens(
                            message.content
                        ) + self.count_tokens(message.type)

    def on_llm_new_token(self, token, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, str, UUID, Any) -> Any
        """Run on new LLM token. Only available when streaming is enabled."""
        with capture_internal_exceptions():
            if not run_id or run_id not in self.span_map:
                return
            span_data = self.span_map[run_id]
            if not span_data or span_data.no_collect_tokens:
                return
            span_data.num_completion_tokens += self.count_tokens(token)

    def on_llm_end(self, response, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, LLMResult, UUID, Any) -> Any
        """Run when LLM ends running."""
        with capture_internal_exceptions():
            if not run_id:
                return

            token_usage = (
                response.llm_output.get("token_usage") if response.llm_output else None
            )

            span_data = self.span_map[run_id]
            if not span_data:
                return

            if should_send_default_pii() and self.include_prompts:
                set_data_normalized(
                    span_data.span,
                    SPANDATA.AI_RESPONSES,
                    [[x.text for x in list_] for list_ in response.generations],
                )

            if not span_data.no_collect_tokens:
                if token_usage:
                    record_token_usage(
                        span_data.span,
                        token_usage.get("prompt_tokens"),
                        token_usage.get("completion_tokens"),
                        token_usage.get("total_tokens"),
                    )
                else:
                    record_token_usage(
                        span_data.span,
                        span_data.num_prompt_tokens,
                        span_data.num_completion_tokens,
                    )

            self._exit_span(span_data, run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, Union[Exception, KeyboardInterrupt], UUID, Any) -> Any
        """Run when LLM errors."""
        with capture_internal_exceptions():
            self._handle_error(run_id, error)

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, Dict[str, Any], Dict[str, Any], UUID, Any) -> Any
        """Run when chain starts running."""
        with capture_internal_exceptions():
            if not run_id:
                return
            watched_span = self._create_span(
                run_id,
                kwargs.get("parent_run_id"),
                op=(
                    OP.LANGCHAIN_RUN
                    if kwargs.get("parent_run_id") is not None
                    else OP.LANGCHAIN_PIPELINE
                ),
                name=kwargs.get("name") or "Chain execution",
                origin=LangchainIntegration.origin,
            )
            metadata = kwargs.get("metadata")
            if metadata:
                set_data_normalized(watched_span.span, SPANDATA.AI_METADATA, metadata)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, Dict[str, Any], UUID, Any) -> Any
        """Run when chain ends running."""
        with capture_internal_exceptions():
            if not run_id or run_id not in self.span_map:
                return

            span_data = self.span_map[run_id]
            if not span_data:
                return
            self._exit_span(span_data, run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, Union[Exception, KeyboardInterrupt], UUID, Any) -> Any
        """Run when chain errors."""
        self._handle_error(run_id, error)

    def on_agent_action(self, action, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, AgentAction, UUID, Any) -> Any
        with capture_internal_exceptions():
            if not run_id:
                return
            watched_span = self._create_span(
                run_id,
                kwargs.get("parent_run_id"),
                op=OP.LANGCHAIN_AGENT,
                name=action.tool or "AI tool usage",
                origin=LangchainIntegration.origin,
            )
            if action.tool_input and should_send_default_pii() and self.include_prompts:
                set_data_normalized(
                    watched_span.span, SPANDATA.AI_INPUT_MESSAGES, action.tool_input
                )

    def on_agent_finish(self, finish, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, AgentFinish, UUID, Any) -> Any
        with capture_internal_exceptions():
            if not run_id:
                return

            span_data = self.span_map[run_id]
            if not span_data:
                return
            if should_send_default_pii() and self.include_prompts:
                set_data_normalized(
                    span_data.span, SPANDATA.AI_RESPONSES, finish.return_values.items()
                )
            self._exit_span(span_data, run_id)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, Dict[str, Any], str, UUID, Any) -> Any
        """Run when tool starts running."""
        with capture_internal_exceptions():
            if not run_id:
                return
            watched_span = self._create_span(
                run_id,
                kwargs.get("parent_run_id"),
                op=OP.LANGCHAIN_TOOL,
                name=serialized.get("name") or kwargs.get("name") or "AI tool usage",
                origin=LangchainIntegration.origin,
            )
            if should_send_default_pii() and self.include_prompts:
                set_data_normalized(
                    watched_span.span,
                    SPANDATA.AI_INPUT_MESSAGES,
                    kwargs.get("inputs", [input_str]),
                )
                if kwargs.get("metadata"):
                    set_data_normalized(
                        watched_span.span, SPANDATA.AI_METADATA, kwargs.get("metadata")
                    )

    def on_tool_end(self, output, *, run_id, **kwargs):
        # type: (SentryLangchainCallback, str, UUID, Any) -> Any
        """Run when tool ends running."""
        with capture_internal_exceptions():
            if not run_id or run_id not in self.span_map:
                return

            span_data = self.span_map[run_id]
            if not span_data:
                return
            if should_send_default_pii() and self.include_prompts:
                set_data_normalized(span_data.span, SPANDATA.AI_RESPONSES, output)
            self._exit_span(span_data, run_id)

    def on_tool_error(self, error, *args, run_id, **kwargs):
        # type: (SentryLangchainCallback, Union[Exception, KeyboardInterrupt], UUID, Any) -> Any
        """Run when tool errors."""
        self._handle_error(run_id, error)


def _wrap_configure(f):
    # type: (Callable[..., Any]) -> Callable[..., Any]

    @wraps(f)
    def new_configure(*args, **kwargs):
        # type: (Any, Any) -> Any

        integration = sentry_sdk.get_client().get_integration(LangchainIntegration)
        if integration is None:
            return f(*args, **kwargs)

        with capture_internal_exceptions():
            new_callbacks = []  # type: List[BaseCallbackHandler]
            if "local_callbacks" in kwargs:
                existing_callbacks = kwargs["local_callbacks"]
                kwargs["local_callbacks"] = new_callbacks
            elif len(args) > 2:
                existing_callbacks = args[2]
                args = (
                    args[0],
                    args[1],
                    new_callbacks,
                ) + args[3:]
            else:
                existing_callbacks = []

            if existing_callbacks:
                if isinstance(existing_callbacks, list):
                    for cb in existing_callbacks:
                        new_callbacks.append(cb)
                elif isinstance(existing_callbacks, BaseCallbackHandler):
                    new_callbacks.append(existing_callbacks)
                else:
                    logger.debug("Unknown callback type: %s", existing_callbacks)

            already_added = False
            for callback in new_callbacks:
                if isinstance(callback, SentryLangchainCallback):
                    already_added = True

            if not already_added:
                new_callbacks.append(
                    SentryLangchainCallback(
                        integration.max_spans,
                        integration.include_prompts,
                        integration.tiktoken_encoding_name,
                    )
                )
        return f(*args, **kwargs)

    return new_configure
