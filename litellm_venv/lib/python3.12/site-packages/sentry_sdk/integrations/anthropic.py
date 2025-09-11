from functools import wraps
from typing import TYPE_CHECKING

import sentry_sdk
from sentry_sdk.ai.monitoring import record_token_usage
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import _check_minimum_version, DidNotEnable, Integration
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.utils import (
    capture_internal_exceptions,
    event_from_exception,
    package_version,
)

try:
    from anthropic.resources import AsyncMessages, Messages

    if TYPE_CHECKING:
        from anthropic.types import MessageStreamEvent
except ImportError:
    raise DidNotEnable("Anthropic not installed")

if TYPE_CHECKING:
    from typing import Any, AsyncIterator, Iterator
    from sentry_sdk.tracing import Span


class AnthropicIntegration(Integration):
    identifier = "anthropic"
    origin = f"auto.ai.{identifier}"

    def __init__(self, include_prompts=True):
        # type: (AnthropicIntegration, bool) -> None
        self.include_prompts = include_prompts

    @staticmethod
    def setup_once():
        # type: () -> None
        version = package_version("anthropic")
        _check_minimum_version(AnthropicIntegration, version)

        Messages.create = _wrap_message_create(Messages.create)
        AsyncMessages.create = _wrap_message_create_async(AsyncMessages.create)


def _capture_exception(exc):
    # type: (Any) -> None
    event, hint = event_from_exception(
        exc,
        client_options=sentry_sdk.get_client().options,
        mechanism={"type": "anthropic", "handled": False},
    )
    sentry_sdk.capture_event(event, hint=hint)


def _calculate_token_usage(result, span):
    # type: (Messages, Span) -> None
    input_tokens = 0
    output_tokens = 0
    if hasattr(result, "usage"):
        usage = result.usage
        if hasattr(usage, "input_tokens") and isinstance(usage.input_tokens, int):
            input_tokens = usage.input_tokens
        if hasattr(usage, "output_tokens") and isinstance(usage.output_tokens, int):
            output_tokens = usage.output_tokens

    total_tokens = input_tokens + output_tokens
    record_token_usage(span, input_tokens, output_tokens, total_tokens)


def _get_responses(content):
    # type: (list[Any]) -> list[dict[str, Any]]
    """
    Get JSON of a Anthropic responses.
    """
    responses = []
    for item in content:
        if hasattr(item, "text"):
            responses.append(
                {
                    "type": item.type,
                    "text": item.text,
                }
            )
    return responses


def _collect_ai_data(event, input_tokens, output_tokens, content_blocks):
    # type: (MessageStreamEvent, int, int, list[str]) -> tuple[int, int, list[str]]
    """
    Count token usage and collect content blocks from the AI streaming response.
    """
    with capture_internal_exceptions():
        if hasattr(event, "type"):
            if event.type == "message_start":
                usage = event.message.usage
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
            elif event.type == "content_block_start":
                pass
            elif event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    content_blocks.append(event.delta.text)
            elif event.type == "content_block_stop":
                pass
            elif event.type == "message_delta":
                output_tokens += event.usage.output_tokens

    return input_tokens, output_tokens, content_blocks


def _add_ai_data_to_span(
    span, integration, input_tokens, output_tokens, content_blocks
):
    # type: (Span, AnthropicIntegration, int, int, list[str]) -> None
    """
    Add token usage and content blocks from the AI streaming response to the span.
    """
    with capture_internal_exceptions():
        if should_send_default_pii() and integration.include_prompts:
            complete_message = "".join(content_blocks)
            span.set_data(
                SPANDATA.AI_RESPONSES,
                [{"type": "text", "text": complete_message}],
            )
        total_tokens = input_tokens + output_tokens
        record_token_usage(span, input_tokens, output_tokens, total_tokens)
        span.set_data(SPANDATA.AI_STREAMING, True)


def _sentry_patched_create_common(f, *args, **kwargs):
    # type: (Any, *Any, **Any) -> Any
    integration = kwargs.pop("integration")
    if integration is None:
        return f(*args, **kwargs)

    if "messages" not in kwargs:
        return f(*args, **kwargs)

    try:
        iter(kwargs["messages"])
    except TypeError:
        return f(*args, **kwargs)

    span = sentry_sdk.start_span(
        op=OP.ANTHROPIC_MESSAGES_CREATE,
        description="Anthropic messages create",
        origin=AnthropicIntegration.origin,
    )
    span.__enter__()

    result = yield f, args, kwargs

    # add data to span and finish it
    messages = list(kwargs["messages"])
    model = kwargs.get("model")

    with capture_internal_exceptions():
        span.set_data(SPANDATA.AI_MODEL_ID, model)
        span.set_data(SPANDATA.AI_STREAMING, False)

        if should_send_default_pii() and integration.include_prompts:
            span.set_data(SPANDATA.AI_INPUT_MESSAGES, messages)

        if hasattr(result, "content"):
            if should_send_default_pii() and integration.include_prompts:
                span.set_data(SPANDATA.AI_RESPONSES, _get_responses(result.content))
            _calculate_token_usage(result, span)
            span.__exit__(None, None, None)

        # Streaming response
        elif hasattr(result, "_iterator"):
            old_iterator = result._iterator

            def new_iterator():
                # type: () -> Iterator[MessageStreamEvent]
                input_tokens = 0
                output_tokens = 0
                content_blocks = []  # type: list[str]

                for event in old_iterator:
                    input_tokens, output_tokens, content_blocks = _collect_ai_data(
                        event, input_tokens, output_tokens, content_blocks
                    )
                    if event.type != "message_stop":
                        yield event

                _add_ai_data_to_span(
                    span, integration, input_tokens, output_tokens, content_blocks
                )
                span.__exit__(None, None, None)

            async def new_iterator_async():
                # type: () -> AsyncIterator[MessageStreamEvent]
                input_tokens = 0
                output_tokens = 0
                content_blocks = []  # type: list[str]

                async for event in old_iterator:
                    input_tokens, output_tokens, content_blocks = _collect_ai_data(
                        event, input_tokens, output_tokens, content_blocks
                    )
                    if event.type != "message_stop":
                        yield event

                _add_ai_data_to_span(
                    span, integration, input_tokens, output_tokens, content_blocks
                )
                span.__exit__(None, None, None)

            if str(type(result._iterator)) == "<class 'async_generator'>":
                result._iterator = new_iterator_async()
            else:
                result._iterator = new_iterator()

        else:
            span.set_data("unknown_response", True)
            span.__exit__(None, None, None)

    return result


def _wrap_message_create(f):
    # type: (Any) -> Any
    def _execute_sync(f, *args, **kwargs):
        # type: (Any, *Any, **Any) -> Any
        gen = _sentry_patched_create_common(f, *args, **kwargs)

        try:
            f, args, kwargs = next(gen)
        except StopIteration as e:
            return e.value

        try:
            try:
                result = f(*args, **kwargs)
            except Exception as exc:
                _capture_exception(exc)
                raise exc from None

            return gen.send(result)
        except StopIteration as e:
            return e.value

    @wraps(f)
    def _sentry_patched_create_sync(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        integration = sentry_sdk.get_client().get_integration(AnthropicIntegration)
        kwargs["integration"] = integration

        return _execute_sync(f, *args, **kwargs)

    return _sentry_patched_create_sync


def _wrap_message_create_async(f):
    # type: (Any) -> Any
    async def _execute_async(f, *args, **kwargs):
        # type: (Any, *Any, **Any) -> Any
        gen = _sentry_patched_create_common(f, *args, **kwargs)

        try:
            f, args, kwargs = next(gen)
        except StopIteration as e:
            return await e.value

        try:
            try:
                result = await f(*args, **kwargs)
            except Exception as exc:
                _capture_exception(exc)
                raise exc from None

            return gen.send(result)
        except StopIteration as e:
            return e.value

    @wraps(f)
    async def _sentry_patched_create_async(*args, **kwargs):
        # type: (*Any, **Any) -> Any
        integration = sentry_sdk.get_client().get_integration(AnthropicIntegration)
        kwargs["integration"] = integration

        return await _execute_async(f, *args, **kwargs)

    return _sentry_patched_create_async
