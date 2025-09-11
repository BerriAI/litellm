import inspect
from functools import wraps

import sentry_sdk.utils
from sentry_sdk import start_span
from sentry_sdk.tracing import Span
from sentry_sdk.utils import ContextVar

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Callable, Any

_ai_pipeline_name = ContextVar("ai_pipeline_name", default=None)


def set_ai_pipeline_name(name):
    # type: (Optional[str]) -> None
    _ai_pipeline_name.set(name)


def get_ai_pipeline_name():
    # type: () -> Optional[str]
    return _ai_pipeline_name.get()


def ai_track(description, **span_kwargs):
    # type: (str, Any) -> Callable[..., Any]
    def decorator(f):
        # type: (Callable[..., Any]) -> Callable[..., Any]
        def sync_wrapped(*args, **kwargs):
            # type: (Any, Any) -> Any
            curr_pipeline = _ai_pipeline_name.get()
            op = span_kwargs.get("op", "ai.run" if curr_pipeline else "ai.pipeline")

            with start_span(name=description, op=op, **span_kwargs) as span:
                for k, v in kwargs.pop("sentry_tags", {}).items():
                    span.set_tag(k, v)
                for k, v in kwargs.pop("sentry_data", {}).items():
                    span.set_data(k, v)
                if curr_pipeline:
                    span.set_data("ai.pipeline.name", curr_pipeline)
                    return f(*args, **kwargs)
                else:
                    _ai_pipeline_name.set(description)
                    try:
                        res = f(*args, **kwargs)
                    except Exception as e:
                        event, hint = sentry_sdk.utils.event_from_exception(
                            e,
                            client_options=sentry_sdk.get_client().options,
                            mechanism={"type": "ai_monitoring", "handled": False},
                        )
                        sentry_sdk.capture_event(event, hint=hint)
                        raise e from None
                    finally:
                        _ai_pipeline_name.set(None)
                    return res

        async def async_wrapped(*args, **kwargs):
            # type: (Any, Any) -> Any
            curr_pipeline = _ai_pipeline_name.get()
            op = span_kwargs.get("op", "ai.run" if curr_pipeline else "ai.pipeline")

            with start_span(name=description, op=op, **span_kwargs) as span:
                for k, v in kwargs.pop("sentry_tags", {}).items():
                    span.set_tag(k, v)
                for k, v in kwargs.pop("sentry_data", {}).items():
                    span.set_data(k, v)
                if curr_pipeline:
                    span.set_data("ai.pipeline.name", curr_pipeline)
                    return await f(*args, **kwargs)
                else:
                    _ai_pipeline_name.set(description)
                    try:
                        res = await f(*args, **kwargs)
                    except Exception as e:
                        event, hint = sentry_sdk.utils.event_from_exception(
                            e,
                            client_options=sentry_sdk.get_client().options,
                            mechanism={"type": "ai_monitoring", "handled": False},
                        )
                        sentry_sdk.capture_event(event, hint=hint)
                        raise e from None
                    finally:
                        _ai_pipeline_name.set(None)
                    return res

        if inspect.iscoroutinefunction(f):
            return wraps(f)(async_wrapped)
        else:
            return wraps(f)(sync_wrapped)

    return decorator


def record_token_usage(
    span, prompt_tokens=None, completion_tokens=None, total_tokens=None
):
    # type: (Span, Optional[int], Optional[int], Optional[int]) -> None
    ai_pipeline_name = get_ai_pipeline_name()
    if ai_pipeline_name:
        span.set_data("ai.pipeline.name", ai_pipeline_name)
    if prompt_tokens is not None:
        span.set_measurement("ai_prompt_tokens_used", value=prompt_tokens)
    if completion_tokens is not None:
        span.set_measurement("ai_completion_tokens_used", value=completion_tokens)
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens
    if total_tokens is not None:
        span.set_measurement("ai_total_tokens_used", total_tokens)
