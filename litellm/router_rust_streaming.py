"""
The stream wrappers of the Rust router core: host-side objects around a streamed response.

Every attempt the Rust core runs that returns a stream is wrapped here before the core sees the
outcome, the way the Python router wraps each attempt's stream:

- a chat stream (``CustomStreamWrapper``) has its first chunk read inside the attempt, so a stream
  that fails before any token fails the attempt and the core retries or falls back as for any error
- a chat, ``/v1/messages`` or Responses stream that fails later, before the caller received content,
  hands the failure back to the core: the core decides whether a same-group retry or a fallback takes
  over (``RouterCoreHandle.stream_policy`` / ``retry_stream`` / ``fallback_stream``), and the wrapper
  splices what that call returns into the stream the caller already holds

The wrappers decide nothing about routing: what may take over, the retry budget and every
deployment pick come from the core. They hold live Python iterators, frames and hidden params,
which is why they live in the host. Ported from the Python router's ``_acompletion_streaming_iterator``,
``_completion_streaming_iterator``, ``_aanthropic_messages_streaming_iterator`` and
``_aresponses_streaming_iterator``, without importing it.
"""

import contextlib
import dataclasses
import weakref
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Final,
    Generic,
    NoReturn,
    TypeAlias,
    TypeVar,
    cast,  # noqa: TID251  # untyped stream iterators and provider chunk lists
)

import anyio

import litellm.exceptions as litellm_exceptions
from litellm._logging import verbose_router_logger
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_retry_headers_to_response,
    get_hidden_params_dict,
    replace_complexity_router_headers,
)
from litellm.litellm_core_utils.hidden_params import HIDDEN_PARAMS_ATTR, get_hidden_params, set_hidden_params
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponseStream, Usage

if TYPE_CHECKING:
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
    from litellm.router_python_host import PythonHost
    from litellm.router_rust_core import RustCore
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import ContinuationOutput, ContinuedPartial
    from litellm.types.llms.openai import ResponseAPIUsage, ResponseIncompleteEvent

# A stream that never emits content or an error could grow a held-frame buffer without bound, so
# hitting either cap commits the stream instead.
MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS: Final = 200
MAX_HELD_PRE_OUTPUT_RESPONSES_EVENTS: Final = 200

# The hidden params and headers a fallback's response carries, as dicts the proxy writes into.
_Prepared: TypeAlias = "tuple[dict[str, object], dict[str, object]]"
_RestT: Final = TypeVar("_RestT", Iterator[object], AsyncIterator[object])


# --- the attempt a stream came from -------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class StreamAttempt:
    """The attempt that returned a stream, and the core that can still recover it.

    ``request`` is the call's prepared request, the one every attempt of the call is built from;
    ``kwargs`` are this attempt's own kwargs. The origin fields are what the core reads to decide
    what may take over a failure of this stream.
    """

    core: "RustCore"
    op: str
    request: Mapping[str, object]
    kwargs: Mapping[str, object]
    group: str
    order: int | None
    depth: int
    level: int
    deployment_id: str | None
    attempted_retries: int
    committed_budget: int | None = None
    # The deployment's max_parallel_requests slot the attempt holds; an open chat stream takes it.
    slot: contextlib.ExitStack | None = None

    @property
    def model_group(self) -> str:
        return self.group

    def origin(self) -> Mapping[str, object]:
        return {
            "group": self.group,
            "order": self.order,
            "depth": self.depth,
            "level": self.level,
            "deployment_id": self.deployment_id,
            "attempted_retries": self.attempted_retries,
            "committed_budget": self.committed_budget,
        }

    def policy(self, trigger: BaseException | None = None) -> Mapping[str, object]:
        return self.core.stream_policy(self.request, self.origin(), trigger)

    def can_retry(self) -> bool:
        return self.policy().get("can_retry") is True

    def can_fall_back(self) -> bool:
        return self.policy().get("can_fall_back") is True

    def refusal_fallback_available(self) -> bool:
        return self.policy().get("refusal_fallback") is True

    def retry_budget(self, trigger: BaseException) -> tuple[int, bool]:
        policy: Final = self.policy(trigger)
        budget: Final = policy.get("retry_budget")
        return (budget if isinstance(budget, int) else 0), policy.get("policy_applies") is True

    def should_retry_status(self, status_code: int) -> bool:
        return bool(self.core._host.invoke("should_retry")(status_code))  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # the wrapper serves its own core's host

    async def fallback(self, trigger: BaseException, request: Mapping[str, object] | None = None) -> object:
        """The fallback walk for this stream's failure; raises when no fallback answers."""
        return await self.core.recover_stream("fallback", self.op, request or self.request, self.origin(), trigger)

    def fallback_sync(self, trigger: BaseException) -> object:
        return self.core.recover_stream_sync("fallback", self.op, self.request, self.origin(), trigger)

    async def retry(self, trigger: BaseException) -> "tuple[object, int, int] | BaseException":
        """A same-group retry: the retried response with the retry count and budget, or the newest failure."""
        return await self.core.retry_stream(self.op, self.request, self.origin(), trigger)


# --- the first chunk ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _FirstChunk(Generic[_RestT]):
    """The chunk an attempt already read off a stream (None: the provider closed it empty) and the
    iterator it was read from, which carries on after it."""

    chunk: object
    rest: _RestT


def _mid_stream_retry_trigger(error: MidStreamFallbackError) -> Exception:
    """The provider's own error, which is what the retry policy and the retry rules classify."""
    return error.original_exception if error.original_exception is not None else error


async def _aread_first_chunk(stream: CustomStreamWrapper) -> _FirstChunk[AsyncIterator[object]]:
    """Read the first chunk while the attempt still holds the request, so a stream that fails before
    any token fails the attempt. Iterates the way ``async for`` does: one ``__aiter__`` call, then
    ``__anext__`` on what it returned."""
    rest: Final = stream.__aiter__()
    try:
        return _FirstChunk(chunk=await rest.__anext__(), rest=rest)
    except StopAsyncIteration:
        return _FirstChunk(chunk=None, rest=rest)
    except MidStreamFallbackError as e:
        raise _mid_stream_retry_trigger(e) from e


def _read_first_chunk(stream: CustomStreamWrapper) -> _FirstChunk[Iterator[object]]:
    rest: Final = iter(stream)
    try:
        return _FirstChunk(chunk=next(rest), rest=rest)
    except StopIteration:
        return _FirstChunk(chunk=None, rest=rest)
    except MidStreamFallbackError as e:
        raise _mid_stream_retry_trigger(e) from e


def _chunks_with_first(model_response: Iterable[object], taken: _FirstChunk[Iterator[object]]) -> Iterator[object]:
    if taken.chunk is None:
        return
    yield taken.chunk
    yield from taken.rest


async def _achunks_with_first(
    model_response: AsyncIterable[object], taken: _FirstChunk[AsyncIterator[object]]
) -> AsyncIterator[object]:
    if taken.chunk is None:
        return
    yield taken.chunk
    async for item in taken.rest:
        yield item


# --- shared helpers -----------------------------------------------------------------------------


_CONTENT_DELTA_KEYS: Final = (
    "content",
    "tool_calls",
    "function_call",
    "reasoning_content",
    "thinking_blocks",
    "reasoning_items",
    "audio",
    "images",
    "annotations",
)


def _delta_has_content(delta: object) -> bool:
    get: Final[object] = getattr(delta, "get", None)
    if not callable(get):
        return False
    for key in _CONTENT_DELTA_KEYS:
        value: object = get(key)
        if value:
            return True
    return False


def _stream_chunks_have_generated_content(chunks: Sequence[object]) -> bool:
    for chunk in chunks:
        choices: object = getattr(chunk, "choices", None)
        if not isinstance(choices, Sequence) or not choices:
            continue
        delta: object = getattr(choices[0], "delta", None)  # pyright: ignore[reportUnknownArgumentType]  # a chunk choice
        if _delta_has_content(delta):
            return True
    return False


def _raise_original_mid_stream_error(e: MidStreamFallbackError) -> NoReturn:
    """Surface the provider error the stream wrapper carried instead of the internal MidStreamFallbackError."""
    if e.original_exception is not None:
        raise e.original_exception from e
    raise e


def _stream_pre_content_error(error: Exception, model: str, llm_provider: str = "anthropic") -> MidStreamFallbackError:
    """The envelope the fallback chain judges a failure by when the client has received no content yet."""
    return MidStreamFallbackError(
        message=str(error),
        model=model,
        llm_provider=llm_provider,
        original_exception=error,
        is_pre_first_chunk=True,
    )


def _context_window_stream_fallback_error(
    error: "MidStreamFallbackError | litellm_exceptions.ContextWindowExceededError",
    model: object,
    llm_provider: str,
    has_output: bool,
) -> MidStreamFallbackError:
    if isinstance(error, litellm_exceptions.ContextWindowExceededError):
        if has_output:
            raise error
        return _stream_pre_content_error(error, model if isinstance(model, str) else "", llm_provider)
    if isinstance(error.original_exception, litellm_exceptions.ContextWindowExceededError) and (
        has_output or error.generated_content or error.emitted_disqualifying_content
    ):
        raise error.original_exception
    return error


def _is_context_window(error: "MidStreamFallbackError | litellm_exceptions.ContextWindowExceededError") -> bool:
    return isinstance(error, litellm_exceptions.ContextWindowExceededError) or isinstance(
        error.original_exception, litellm_exceptions.ContextWindowExceededError
    )


def _fallback_trigger(e: MidStreamFallbackError, unwrapped: tuple[type[Exception], ...]) -> Exception:
    """The error the fallback walk dispatches on: the provider's own for the kinds with a list of their own."""
    return e.original_exception if isinstance(e.original_exception, unwrapped) else e


def _prepare_fallback_hidden_params(
    fallback_response: object,
) -> _Prepared:
    fallback_hidden_params: Final = get_hidden_params_dict(fallback_response)
    fallback_headers: Final = fallback_hidden_params.get("additional_headers")
    if not isinstance(fallback_headers, dict):
        return fallback_hidden_params, {}
    return fallback_hidden_params, cast("dict[str, object]", fallback_headers)  # cast-ok: a dict of headers


def _apply_fallback_hidden_params_to_item(
    fallback_item: object,
    prepared_fallback_hidden_params: _Prepared,
) -> None:
    if fallback_item is None or not hasattr(fallback_item, HIDDEN_PARAMS_ATTR):
        return
    fallback_hidden_params, fallback_headers = prepared_fallback_hidden_params
    item_hidden_params: Final = get_hidden_params_dict(fallback_item)
    item_headers: object = item_hidden_params.get("additional_headers")  # rebind-ok: replaced when not a dict
    if not isinstance(item_headers, dict):
        item_headers = {}  # rebind-ok: rebound as the stream advances
    set_hidden_params(
        fallback_item,
        {
            **item_hidden_params,
            **fallback_hidden_params,
            "additional_headers": {**cast("dict[str, object]", item_headers), **fallback_headers},  # cast-ok: checked
        },
    )


def _combine_fallback_usage(fallback_item: ModelResponseStream, complete_response_object_usage: Usage | None) -> None:
    """Merge partial-stream usage with fallback-stream usage on the chunk. A usage given as a plain
    dict comes out all-zero: ``combine_usage_objects`` reads token counts as attributes."""
    from litellm.cost_calculator import BaseTokenUsageProcessor

    usage: Final = cast("Usage | None", getattr(fallback_item, "usage", None))  # cast-ok: a chunk's usage
    usage_objects: Final = [usage] if usage is not None else []
    if complete_response_object_usage is not None:
        usage_objects.append(complete_response_object_usage)
    fallback_item.usage = BaseTokenUsageProcessor.combine_usage_objects(usage_objects=usage_objects)  # pyright: ignore[reportAttributeAccessIssue]  # usage is not a declared chunk field  # rebind-ok: the Python router rewrites the chunk's usage in place


# --- mid-stream continuation --------------------------------------------------------------------

CONTINUATION_FLAG: Final = "enable_mid_stream_fallback_continuation"
NO_FLAG: Final = object()
# The routing-body key naming the deployments a continuation may not land on (``continuation_deployments``).
CONTINUATION_EXCLUDED_IDS: Final = "continuation_excluded_ids"


def continuation_flag(value: object) -> bool:
    """``_is_flag_enabled``: True, or a string ``str_to_bool`` reads as True."""
    from litellm.secret_managers.main import str_to_bool

    return value is True or (isinstance(value, str) and str_to_bool(value) is True)


def register_continuation_prefill_check(host: "PythonHost") -> None:
    """``_register_continuation_prefill_check``: process-global on purpose, as in the Python router."""
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import ContinuationPrefillDeploymentCheck

    host.callbacks.add_litellm_callback(ContinuationPrefillDeploymentCheck())


def continuation_routing(core: "RustCore", request: Mapping[str, object]) -> Mapping[str, object]:
    """The routing fields of a continuation request: the deployments that cannot continue it, which the
    core leaves out of selection the way ``continuation_deployments`` filters the healthy list."""
    from litellm.litellm_core_utils.routing_common_utils import MID_STREAM_CONTINUATION_KWARG
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import (
        MID_STREAM_CONTINUATION_MARKER,
        deployment_can_continue,
    )

    if request.get(MID_STREAM_CONTINUATION_KWARG) is not MID_STREAM_CONTINUATION_MARKER:
        return {}
    registry: Final = core.registry()
    excluded: Final = [
        deployment.model_info.id
        for record in registry.all()
        if (deployment := registry.deployment(str(_record_id(record)))) is not None
        and not deployment_can_continue(deployment.model_dump(exclude_none=True))
    ]
    return {"body": {CONTINUATION_EXCLUDED_IDS: excluded}}


def _record_id(record: Mapping[str, object]) -> object:
    model_info: Final = record.get("model_info")
    return model_info.get("id") if isinstance(model_info, Mapping) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # a plain record


def _names_deployment_that_cannot_continue(core: "RustCore", target: object) -> bool:
    from litellm.router_utils.fallback_event_handlers import fallback_target_name
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import deployment_can_continue

    name: Final = fallback_target_name(target)
    registry: Final = core.registry()
    model_names: Final = {record.get("model_name") for record in registry.all()}
    if name is None or name in model_names or not registry.has_deployment(name):
        return False
    deployment: Final = registry.deployment(name)
    return deployment is not None and not deployment_can_continue(deployment.model_dump(exclude_none=True))


def _mid_stream_continuation_eligible(
    attempt: StreamAttempt, e: MidStreamFallbackError, cut_deployment_id: object
) -> bool:
    """``_mid_stream_continuation_eligible``: whether a stream that broke after plain assistant text may
    continue via prefill."""
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.router_utils.fallback_event_handlers import (
        continuation_fallback_targets,
        fallback_lookup_groups,
        fallback_target_overrides_request_params,
    )
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import constrains_continuation
    from litellm.types.guardrails import GuardrailEventHooks

    core: Final = attempt.core
    if not core.enable_mid_stream_fallback_continuation:
        return False
    if not e.generated_content or e.emitted_disqualifying_content:
        return False
    if isinstance(e.original_exception, litellm_exceptions.ContentPolicyViolationError):
        return False
    if isinstance(cut_deployment_id, str) and not core.registry().has_deployment(cut_deployment_id):
        return False
    if constrains_continuation(_params_sent_to_deployment(attempt, cut_deployment_id)):
        return False
    request: Final = attempt.request
    requested_group: Final = request.get("model")
    lookup_groups: Final = fallback_lookup_groups(
        request, requested_group if isinstance(requested_group, str) else None
    )
    targets: Final = continuation_fallback_targets(core.settings().fallbacks, lookup_groups)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # fallbacks as Python spells them
    if any(fallback_target_overrides_request_params(target) for target in targets):
        return False
    if any(_names_deployment_that_cannot_continue(core, target) for target in targets):
        return False
    return not any(
        isinstance(callback, CustomGuardrail)
        and callback._event_hook_is_event_type(  # pyright: ignore[reportPrivateUsage]  # the only check of a guardrail's hooks
            GuardrailEventHooks.post_call
        )
        for callback in core._host.callback_list("callbacks")  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # the core's own host; read as getattr(litellm, name)
    )


def _extend_trailing_text_block(blocks: Sequence[Mapping[str, object]], text: str) -> Sequence[Mapping[str, object]]:
    """Continue the last text block in place so the prefill reads as the text the caller already received."""
    last: Final = blocks[-1] if blocks else None
    if last is not None and last.get("type") == "text" and isinstance(last_text := last.get("text"), str):
        return [*blocks[:-1], {**last, "text": f"{last_text}{text}"}]
    return [*blocks, {"type": "text", "text": text}]


def _build_completion_continuation_input(
    messages: Sequence[Mapping[str, object]], generated_content: str
) -> Sequence[Mapping[str, object]] | None:
    """The partial output as an assistant prefill, or a trailing ``prefix: True`` assistant turn extended in
    place. None when the request ends in a plain assistant turn, in a prefill turn whose content this cannot
    extend, or when the cut came mid-echo of the prefill."""
    from pydantic import TypeAdapter, ValidationError

    from litellm.router_utils.pre_call_checks.continuation_prefill_check import continued_text

    last: Final = messages[-1] if messages else None
    if last is None or last.get("role") != "assistant":
        return [*messages, {"role": "assistant", "content": generated_content, "prefix": True}]
    if last.get("prefix") is not True:
        return None
    new_text: Final = continued_text(last, generated_content)
    if new_text is None:
        return None
    content: Final = last.get("content")
    if content is None or isinstance(content, str):
        return [*messages[:-1], {**last, "content": f"{content or ''}{new_text}", "prefix": True}]
    try:
        blocks: Final = TypeAdapter(list[dict[str, object]]).validate_python(content)
    except ValidationError:
        return None
    return [*messages[:-1], {**last, "content": _extend_trailing_text_block(blocks, new_text), "prefix": True}]


def _price_cut_partial(cut_stream: CustomStreamWrapper, partial_response: object) -> "ContinuedPartial | None":
    """Price the cut stream's partial, plus any partial it inherited, before a fallback can reuse its logging."""
    from litellm.cost_calculator import BaseTokenUsageProcessor
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import ContinuedPartial

    cut_logging: Final = cut_stream.logging_obj
    partial_usage: Final = getattr(partial_response, "usage", None)
    if partial_response is None or not isinstance(partial_usage, Usage):
        return None
    details: Final = cut_logging.model_call_details
    inherited_usage: Final = details.get("continued_partial_usage")
    inherited_cost: Final = details.get("continued_partial_cost")
    inherited_id: Final = details.get("continued_response_id")
    priced_partial: Final = partial_response.model_copy(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # a ModelResponse
        update={"model": cut_stream.model} if cut_stream.model else {}  # pyright: ignore[reportUnknownMemberType]  # untyped attribute
    )
    priced_partial._hidden_params = {  # pyright: ignore[reportUnknownMemberType]  # no public accessor
        key: value for key, value in get_hidden_params_dict(partial_response).items() if key != "response_cost"
    }
    if LiteLLMLogging.usage_capture_callback is not None:
        from litellm._internal_context import internal_sub_call

        with internal_sub_call(cut_logging.litellm_call_id):
            cut_logging.capture_usage_before_logging(priced_partial)  # pyright: ignore[reportUnknownArgumentType]  # a ModelResponse
    return ContinuedPartial(
        usage=BaseTokenUsageProcessor.combine_usage_objects(
            usage_objects=[partial_usage, inherited_usage] if isinstance(inherited_usage, Usage) else [partial_usage]
        ),
        response_cost=(cut_logging.response_cost_calculator(result=priced_partial) or 0.0)  # pyright: ignore[reportUnknownArgumentType]  # a ModelResponse
        + (inherited_cost if isinstance(inherited_cost, float) else 0.0),
        response_id=inherited_id if isinstance(inherited_id, str) else str(getattr(partial_response, "id", "")),
    )


def _carry_partial_spend_to_fallback(fallback_response: object, cut_partial: "ContinuedPartial | None") -> None:
    """Bill what the cut legs consumed on the continuation's success log."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    fallback_logging: Final = getattr(fallback_response, "logging_obj", None)
    if cut_partial is None or not isinstance(fallback_logging, LiteLLMLogging):
        return
    fallback_logging.record_continued_partial(
        usage=cut_partial.usage,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # a Usage
        response_cost=cut_partial.response_cost,
        response_id=cut_partial.response_id,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _Continuation:
    """What a chat continuation sends the fallback and splices back."""

    request: Mapping[str, object]
    output: "ContinuationOutput"
    partial: "ContinuedPartial | None"
    usage: Usage | None


async def _chat_continuation(
    attempt: StreamAttempt, e: MidStreamFallbackError, model_response: CustomStreamWrapper
) -> _Continuation:
    """The request a fallback continues a chat stream from, once it broke after content. Raises the
    provider's error when the continuation is declined or the output ceiling is spent."""
    from litellm.constants import MID_STREAM_FAILURE_LOG_WAIT_SECONDS
    from litellm.litellm_core_utils.routing_common_utils import MID_STREAM_CONTINUATION_KWARG
    from litellm.main import (
        stream_chunk_builder,  # pyright: ignore[reportUnknownVariableType]  # untyped chunk list argument
    )
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import (
        MID_STREAM_CONTINUATION_MARKER,
        ContinuationOutput,
        continuation_output_ceilings,
    )

    request: Final = attempt.request
    raw_messages: Final = request.get("messages")
    messages: Final = cast("list[Mapping[str, object]]", raw_messages or [])  # cast-ok: chat messages
    complete: Final = stream_chunk_builder(chunks=model_response.chunks, messages=messages)  # pyright: ignore[reportUnknownArgumentType]  # provider chunk list
    usage: Final = cast("Usage | None", getattr(complete, "usage", None))  # cast-ok: a built response's usage
    continuation_messages: Final = _build_completion_continuation_input(messages, e.generated_content)
    output: Final = (
        ContinuationOutput(
            e.generated_content,
            continuation_messages[-1],
            str(request.get("model") or ""),
            complete,
            checks_post_call_rules=request.get(MID_STREAM_CONTINUATION_KWARG) is not MID_STREAM_CONTINUATION_MARKER,
        )
        if continuation_messages is not None
        else None
    )
    partial: Final = _price_cut_partial(model_response, complete) if output is not None else None
    emitted_tokens: Final = int(getattr(usage, "completion_tokens", 0) or 0)
    reduced: Final = continuation_output_ceilings(request, emitted_tokens)
    if reduced is None or continuation_messages is None or output is None:
        _raise_original_mid_stream_error(e)
    await model_response.wait_for_failure_logging(timeout=MID_STREAM_FAILURE_LOG_WAIT_SECONDS)
    return _Continuation(
        request={
            **request,
            **reduced,
            "messages": continuation_messages,
            MID_STREAM_CONTINUATION_KWARG: MID_STREAM_CONTINUATION_MARKER,
        },
        output=output,
        partial=partial,
        usage=usage,
    )


# --- chat completions ---------------------------------------------------------------------------


class FallbackAwareStreamWrapper(CustomStreamWrapper):
    """A chat stream built around the attempt the core picked first, which repoints itself when a
    fallback takes over."""

    fallback_headers_adopted: bool = False

    def adopt_fallback_response_headers(
        self,
        fallback_response: object,
        prepared_fallback_hidden_params: _Prepared,
    ) -> None:
        """Repoint this wrapper at the deployment that served the stream. Replaces rather than merges,
        so the failed attempt's request id, counters, ``model_id`` and ``api_base`` cannot leak."""
        self._response_headers = getattr(fallback_response, "_response_headers", None)
        fallback_hidden_params, fallback_headers = prepared_fallback_hidden_params
        if fallback_hidden_params:
            self._hidden_params = {**fallback_hidden_params, "additional_headers": dict(fallback_headers)}
            self._base_hidden_params = {**self._hidden_params, "response_cost": None}
        self.fallback_headers_adopted = True


def _adopt_fallback_response_headers(
    wrapper_ref: "weakref.ref[FallbackAwareStreamWrapper]", fallback_response: object
) -> _Prepared:
    prepared: Final = _prepare_fallback_hidden_params(fallback_response)
    adopting_wrapper: Final = wrapper_ref()
    if adopting_wrapper is not None:
        adopting_wrapper.adopt_fallback_response_headers(fallback_response, prepared)
    return prepared


def _chat_wrapper_init(
    wrapper: FallbackAwareStreamWrapper, stream: object, model_response: CustomStreamWrapper
) -> None:
    CustomStreamWrapper.__init__(  # pyright: ignore[reportUnknownMemberType]  # the base constructor is untyped
        wrapper,
        completion_stream=stream,
        model=model_response.model,  # pyright: ignore[reportUnknownMemberType]  # an untyped attribute
        custom_llm_provider=model_response.custom_llm_provider,
        logging_obj=model_response.logging_obj,
        _response_headers=getattr(model_response, "_response_headers", None),
    )
    inner_chunks: Final[object] = getattr(model_response, "chunks", None)
    if isinstance(inner_chunks, list):
        chunk_list: Final = cast("list[ModelResponseStream]", inner_chunks)  # cast-ok: the inner stream's chunks
        wrapper.chunks = chunk_list  # rebind-ok: initializes the wrapper being built
    hidden_params: Final = get_hidden_params(model_response)
    if hidden_params is not None:
        wrapper._hidden_params = dict(hidden_params)  # pyright: ignore[reportPrivateUsage]  # the wrapper's own field  # rebind-ok: initializes the wrapper being built


class _AsyncChatStream(FallbackAwareStreamWrapper):
    def __init__(self, generator: AsyncGenerator[object, None], model_response: CustomStreamWrapper) -> None:  # pyright: ignore[reportMissingSuperCall]  # _chat_wrapper_init calls it
        _chat_wrapper_init(self, generator, model_response)
        self._async_generator: Final = generator

    def __aiter__(self) -> "_AsyncChatStream":  # pyright: ignore[reportIncompatibleMethodOverride]  # iterates the generator
        return self

    async def __anext__(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # yields what the stream yields
        return await self._async_generator.__anext__()


class _SyncChatStream(FallbackAwareStreamWrapper):
    def __init__(self, generator: Generator[object, None, None], model_response: CustomStreamWrapper) -> None:  # pyright: ignore[reportMissingSuperCall]  # _chat_wrapper_init calls it
        _chat_wrapper_init(self, generator, model_response)
        self.chunks = []  # the sync twin keeps no inner chunk list
        hidden_params: Final = get_hidden_params(model_response)
        if hidden_params is not None:
            self._hidden_params = dict(hidden_params)
        self._sync_generator: Final = generator

    def __iter__(self) -> "_SyncChatStream":  # pyright: ignore[reportIncompatibleMethodOverride]  # iterates the generator
        return self

    def __next__(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # yields what the stream yields
        return next(self._sync_generator)


def _chat_has_output(model_response: CustomStreamWrapper) -> bool:
    return bool(getattr(model_response, "response_uptil_now", "")) or _stream_chunks_have_generated_content(
        model_response.chunks
    )


def wrap_chat_stream(  # noqa: C901  # one function per Python wrapper it ports
    attempt: StreamAttempt,
    model_response: CustomStreamWrapper,
    first_chunk: _FirstChunk[AsyncIterator[object]],
    held_slot: contextlib.ExitStack | None = None,
) -> FallbackAwareStreamWrapper:
    """``_acompletion_streaming_iterator``: a mid-stream error before content falls back through the core;
    after content, the provider's error is raised. ``held_slot`` (the deployment's max_parallel_requests
    slot) is released when the stream is exhausted, closed, or falls back."""
    slot: Final = held_slot if held_slot is not None else contextlib.ExitStack()

    async def close_model_response() -> None:
        if not hasattr(model_response, "aclose"):
            return
        try:
            await model_response.aclose()
        except BaseException as e:  # noqa: BLE001  # closing is best-effort
            verbose_router_logger.debug("stream_with_fallbacks: error closing model_response: %s", e)

    async def stream_with_fallbacks() -> AsyncGenerator[object, None]:
        fallback_response: object = None  # rebind-ok: tracked for cleanup
        try:
            async for item in _achunks_with_first(model_response, first_chunk):
                yield item
        except (MidStreamFallbackError, litellm_exceptions.ContextWindowExceededError) as stream_error:
            e: Final = (
                _context_window_stream_fallback_error(
                    stream_error,
                    getattr(model_response, "model", ""),
                    model_response.custom_llm_provider or "",
                    _chat_has_output(model_response),
                )
                if _is_context_window(stream_error)
                else cast("MidStreamFallbackError", stream_error)  # cast-ok: not a context window error
            )
            with anyio.CancelScope(shield=True):
                await close_model_response()
                slot.close()
            committed: Final = bool(
                not e.is_pre_first_chunk
                and (e.generated_content or _stream_chunks_have_generated_content(model_response.chunks))
            )
            cut_deployment_id: Final = (get_hidden_params(model_response) or {}).get("model_id")
            if committed and not _mid_stream_continuation_eligible(attempt, e, cut_deployment_id):
                _raise_original_mid_stream_error(e)
            trigger: Final = _fallback_trigger(e, (litellm_exceptions.ContextWindowExceededError,))
            try:
                continuation: Final = await _chat_continuation(attempt, e, model_response) if committed else None
                fallback_response = await attempt.fallback(  # rebind-ok: rebound as the stream advances
                    trigger, continuation.request if continuation is not None else None
                )
                if hasattr(fallback_response, "__aiter__"):
                    if continuation is not None:
                        _carry_partial_spend_to_fallback(fallback_response, continuation.partial)
                    prepared = _adopt_fallback_response_headers(wrapper_ref, fallback_response)  # rebind-ok: re-adopted
                    settled = False  # rebind-ok: rebound as the stream advances
                    fallback_stream: Final = cast("AsyncIterable[object]", fallback_response)  # cast-ok: checked
                    fallback_items: Final = (
                        continuation.output.splice(fallback_stream) if continuation is not None else fallback_stream
                    )
                    async for fallback_item in fallback_items:
                        if not settled:
                            settled = True
                            # a fallback that failed over again only repoints itself once it yields
                            prepared = _adopt_fallback_response_headers(wrapper_ref, fallback_response)
                        _apply_fallback_hidden_params_to_item(fallback_item, prepared)
                        if (
                            fallback_item
                            and isinstance(fallback_item, ModelResponseStream)
                            and hasattr(fallback_item, "usage")
                        ):
                            _combine_fallback_usage(
                                fallback_item,
                                continuation.usage if continuation is not None else None,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # a Usage
                            )
                        yield fallback_item
                else:
                    # A fallback that answers without a stream yields None (an oddity kept as is).
                    yield None
            except Exception as fallback_error:
                verbose_router_logger.error("Fallback also failed: %s", fallback_error)
                if isinstance(trigger, MidStreamFallbackError) and trigger.original_exception is not None:
                    raise trigger.original_exception from fallback_error
                raise trigger from fallback_error
        finally:
            with anyio.CancelScope(shield=True):
                await close_model_response()
                slot.close()
                if fallback_response is not None and hasattr(fallback_response, "aclose"):
                    try:
                        await fallback_response.aclose()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
                    except BaseException as close_error:  # noqa: BLE001  # closing is best-effort
                        verbose_router_logger.debug(
                            "stream_with_fallbacks: error closing fallback_response: %s", close_error
                        )

    wrapped_response: Final = _AsyncChatStream(stream_with_fallbacks(), model_response)
    # weak, so the generator closing over it does not keep the wrapper alive
    wrapper_ref: Final[weakref.ref[FallbackAwareStreamWrapper]] = weakref.ref(wrapped_response)
    return wrapped_response


def wrap_chat_stream_sync(
    attempt: StreamAttempt, model_response: CustomStreamWrapper, first_chunk: _FirstChunk[Iterator[object]]
) -> FallbackAwareStreamWrapper:
    """``_completion_streaming_iterator``: the sync twin, which never continues after content."""

    def stream_with_fallbacks() -> Generator[object, None, None]:
        fallback_response: object = None  # rebind-ok: tracked for cleanup
        try:
            yield from _chunks_with_first(model_response, first_chunk)
        except (MidStreamFallbackError, litellm_exceptions.ContextWindowExceededError) as stream_error:
            e: Final = (
                _context_window_stream_fallback_error(
                    stream_error,
                    getattr(model_response, "model", ""),
                    model_response.custom_llm_provider or "",
                    _chat_has_output(model_response),
                )
                if _is_context_window(stream_error)
                else cast("MidStreamFallbackError", stream_error)  # cast-ok: not a context window error
            )
            if attempt.request.get("disable_fallbacks") is True or (
                not e.is_pre_first_chunk
                and (e.generated_content or _stream_chunks_have_generated_content(model_response.chunks))
            ):
                _raise_original_mid_stream_error(e)
            trigger: Final = _fallback_trigger(e, (litellm_exceptions.ContextWindowExceededError,))
            try:
                fallback_response = attempt.fallback_sync(trigger)  # rebind-ok: rebound as the stream advances
                if hasattr(fallback_response, "__iter__"):
                    prepared = _adopt_fallback_response_headers(wrapper_ref, fallback_response)  # rebind-ok: re-adopted
                    settled = False  # rebind-ok: rebound as the stream advances
                    for fallback_item in cast("Iterable[object]", fallback_response):  # cast-ok: checked
                        if not settled:
                            settled = True
                            prepared = _adopt_fallback_response_headers(wrapper_ref, fallback_response)
                        _apply_fallback_hidden_params_to_item(fallback_item, prepared)
                        if (
                            fallback_item
                            and isinstance(fallback_item, ModelResponseStream)
                            and hasattr(fallback_item, "usage")
                        ):
                            _combine_fallback_usage(fallback_item, None)
                        yield fallback_item
                else:
                    yield None
            except Exception as fallback_error:
                verbose_router_logger.error("Fallback also failed: %s", fallback_error)
                if isinstance(trigger, MidStreamFallbackError) and trigger.original_exception is not None:
                    raise trigger.original_exception from fallback_error
                raise trigger from fallback_error
        finally:
            for closable in (model_response, fallback_response):
                close = getattr(closable, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as close_error:  # noqa: BLE001  # closing is best-effort
                        verbose_router_logger.debug("stream_with_fallbacks: error closing: %s", close_error)

    wrapped_response: Final = _SyncChatStream(stream_with_fallbacks(), model_response)
    wrapper_ref: Final[weakref.ref[FallbackAwareStreamWrapper]] = weakref.ref(wrapped_response)
    return wrapped_response


# --- /v1/messages -------------------------------------------------------------------------------


class FallbackAwareAnthropicMessagesStream:
    """A ``/v1/messages`` byte stream that carries the ``_hidden_params`` the proxy reads its response
    headers off, through retries and fallbacks of the stream it wraps."""

    def __init__(
        self, async_generator: AsyncGenerator[bytes, None], source_iterator: object, attempt: StreamAttempt
    ) -> None:
        self._async_generator: Final = async_generator
        self._source_iterator: object = source_iterator
        self.attempt: Final = attempt
        self.fallback_headers_adopted = False
        self._hidden_params: dict[str, object] = dict(  # mutable-ok: the proxy writes headers into it
            getattr(source_iterator, "_hidden_params", None) or {}
        )
        self._followed_source_params: object = None

    @property
    def has_buffered_provider_output(self) -> bool:
        return getattr(self._source_iterator, "has_buffered_provider_output", False) is True

    @property
    def chunks(self) -> object:
        return getattr(self._source_iterator, "chunks", None)

    @property
    def messages(self) -> object:
        return getattr(self._source_iterator, "messages", None)

    @property
    def model(self) -> object:
        return getattr(self._source_iterator, "model", None)

    def adopt_fallback_source(self, fallback_response: object) -> None:
        self._source_iterator = fallback_response
        self.fallback_headers_adopted = True

    def partial_response_on_disconnect(self) -> "ResponseIncompleteEvent | None":
        from litellm.responses.streaming_iterator import partial_responses_incomplete_event_on_disconnect

        return partial_responses_incomplete_event_on_disconnect(self._source_iterator)

    def follow_source_attribution(self) -> None:
        """Mirror the source's current attribution onto this wrapper whenever the source has adopted a
        new one: a retry's or a fallback's stream carries a wrapper of its own."""
        source: Final = self._source_iterator
        if not getattr(source, "fallback_headers_adopted", False):
            return
        source_params: Final = getattr(source, "_hidden_params", None)
        if source_params is None or source_params is self._followed_source_params:
            return
        self._followed_source_params = source_params
        hidden_params, headers = _prepare_fallback_hidden_params(source)
        self.merge_fallback_hidden_params(hidden_params, headers)

    def __aiter__(self) -> "FallbackAwareAnthropicMessagesStream":
        return self

    async def __anext__(self) -> bytes:
        return await self._async_generator.__anext__()

    async def aclose(self) -> None:
        await self._async_generator.aclose()

    def merge_fallback_hidden_params(
        self, fallback_hidden_params: Mapping[str, object], fallback_headers: Mapping[str, object]
    ) -> None:
        existing_headers: Final = cast(  # cast-ok: additional_headers is a dict when present
            "dict[str, object]", self._hidden_params.get("additional_headers") or {}
        )
        self._hidden_params = {
            **self._hidden_params,
            **fallback_hidden_params,
            "additional_headers": dict(replace_complexity_router_headers(existing_headers, fallback_headers)),
        }


def _anthropic_stream_forwards_ping_live(chunk: object, has_generated_content: bool) -> bool:
    from litellm.llms.anthropic.messages.native.streaming_iterator import is_anthropic_ping_chunk

    return not has_generated_content and is_anthropic_ping_chunk(chunk)


def _is_retriable_anthropic_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _anthropic_stream_error_is_gateway_verdict(chunk: object) -> bool:
    from litellm.llms.anthropic.messages.native.agentic_streaming_iterator import is_server_fulfilled_tool_leak_error

    return is_server_fulfilled_tool_leak_error(chunk)


def _anthropic_stream_should_decline_fallback(has_generated_content: bool, error: MidStreamFallbackError) -> bool:
    return has_generated_content or not error.is_pre_first_chunk


def _stream_raised_error_status(error: Exception) -> int | None:
    raw_status: Final = getattr(error, "status_code", None)
    if isinstance(raw_status, int):
        return raw_status
    if isinstance(raw_status, str) and raw_status.isdigit():
        return int(raw_status)
    response_status: Final = getattr(getattr(error, "response", None), "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _stream_fallback_error_for_raised(
    attempt: StreamAttempt, error: Exception, model: str, has_generated_content: bool, llm_provider: str = "anthropic"
) -> MidStreamFallbackError | None:
    """The pre-stream retry rule (408, 409, 429, 5xx); None means the raise propagates unchanged."""
    if has_generated_content:
        return None
    status_code: Final = _stream_raised_error_status(error)
    if status_code is None or attempt.should_retry_status(status_code):
        return _stream_pre_content_error(error, model, llm_provider)
    return None


def _stream_commits_now(chunk: object, has_generated_content: bool, buffered_chunk_count: int) -> bool:
    from litellm.llms.anthropic.messages.native.streaming_iterator import is_anthropic_content_delta_chunk

    if has_generated_content:
        return False
    return is_anthropic_content_delta_chunk(chunk) or buffered_chunk_count >= MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS


def _anthropic_recoverable_frame_error(
    attempt: StreamAttempt,
    error_event: tuple[str, str, int] | None,
    chunk: object,
    has_generated_content: bool,
    model_group: str,
) -> Exception | None:
    """The exception a provider ``event: error`` frame before content recovers through when a retry of
    its class or a fallback can still take over; None forwards the frame as the provider sent it."""
    from litellm.llms.anthropic.common_utils import anthropic_error_frame_exception

    if has_generated_content or error_event is None:
        return None
    error_type, message, status_code = error_event
    if not _is_retriable_anthropic_status(status_code) or _anthropic_stream_error_is_gateway_verdict(chunk):
        return None
    frame_error: Final = anthropic_error_frame_exception(error_type, message, status_code, model_group)
    budget, _ = attempt.retry_budget(frame_error)
    if budget > attempt.attempted_retries:
        return frame_error
    if attempt.can_fall_back():
        return frame_error
    return None


def wrap_anthropic_stream(
    attempt: StreamAttempt, response: AsyncIterator[bytes]
) -> FallbackAwareAnthropicMessagesStream:
    """``_aanthropic_messages_streaming_iterator``: lifecycle frames are held while a retry or a fallback
    can still take over; an error frame, a raised error or a safeguard refusal before content is
    retried in the group, then falls back, as the core decides."""
    from litellm.llms.anthropic.messages.native.fake_stream_iterator import FakeAnthropicMessagesStreamIterator
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        aclose_if_supported,
        aclose_superseded_attempt,
        parse_anthropic_error_event,
        parse_anthropic_refusal_stop_details,
    )
    from litellm.llms.anthropic.messages.native.utils import safeguard_refusal_error

    source_iterator: Final = response
    replays_a_finished_agentic_turn: Final = isinstance(source_iterator, FakeAnthropicMessagesStreamIterator)
    model: Final = attempt.group

    async def close_source() -> None:
        with anyio.CancelScope(shield=True), contextlib.suppress(BaseException):
            await aclose_if_supported(source_iterator)

    async def close_superseded() -> None:
        with anyio.CancelScope(shield=True), contextlib.suppress(BaseException):
            await aclose_superseded_attempt(source_iterator)

    async def stream_with_fallbacks() -> AsyncGenerator[bytes, None]:
        policy: Final = attempt.policy()
        has_generated_content = not (  # rebind-ok: set once content is seen, the buffer cap is hit, or nothing can take over
            policy.get("can_retry") is True or policy.get("can_fall_back") is True
        )
        buffered: tuple[bytes, ...] = ()  # rebind-ok: flushed once committed or on decline
        try:
            async for chunk in source_iterator:
                if _anthropic_stream_forwards_ping_live(chunk, has_generated_content):
                    yield chunk
                    continue
                if _stream_commits_now(chunk, has_generated_content, len(buffered)):
                    has_generated_content = True
                # A transport can split one SSE data line across byte chunks, so pre-content detection
                # parses the held frames plus the current chunk.
                parse_window = (
                    b"".join(c for c in (*buffered, chunk) if isinstance(c, (bytes, bytearray)))  # pyright: ignore[reportUnnecessaryIsInstance]  # bridge-path chunks are not always bytes
                    if not has_generated_content and isinstance(chunk, (bytes, bytearray))  # pyright: ignore[reportUnnecessaryIsInstance]  # see above
                    else chunk
                )
                error_event = parse_anthropic_error_event(parse_window)
                recoverable_frame_error = _anthropic_recoverable_frame_error(
                    attempt, error_event, chunk, has_generated_content, model
                )
                refusal_stop_details = (
                    parse_anthropic_refusal_stop_details(parse_window)
                    if not has_generated_content and error_event is None
                    else None
                )
                if (
                    refusal_stop_details is not None
                    and not replays_a_finished_agentic_turn
                    and attempt.refusal_fallback_available()
                ):
                    refusal_error = safeguard_refusal_error(model=model, stop_details=refusal_stop_details)
                    raise MidStreamFallbackError(
                        message=refusal_error.message,
                        model=model,
                        llm_provider="anthropic",
                        original_exception=refusal_error,
                        is_pre_first_chunk=True,
                    )
                if not has_generated_content and error_event is None:
                    buffered = (*buffered, chunk)
                    continue
                if recoverable_frame_error is not None and error_event is not None:
                    raise MidStreamFallbackError(
                        message=error_event[1],
                        model=model,
                        llm_provider="anthropic",
                        original_exception=recoverable_frame_error,
                        is_pre_first_chunk=True,
                    )
                for buffered_chunk in buffered:
                    yield buffered_chunk
                buffered = ()
                yield chunk
            for buffered_chunk in buffered:
                yield buffered_chunk
        except Exception as stream_error:  # noqa: BLE001  # any raised provider error must reach the fallback gate
            async for item in _anthropic_recover_stream_error(
                attempt, stream_error, has_generated_content, buffered, model, wrapper, close_superseded
            ):
                yield item
        finally:
            await close_source()

    wrapper: Final = FallbackAwareAnthropicMessagesStream(stream_with_fallbacks(), source_iterator, attempt)
    return wrapper


async def _anthropic_recover_stream_error(
    attempt: StreamAttempt,
    stream_error: Exception,
    has_generated_content: bool,
    buffered: tuple[bytes, ...],
    model: str,
    wrapper: FallbackAwareAnthropicMessagesStream,
    close_superseded: Callable[[], Awaitable[None]],
) -> AsyncGenerator[bytes, None]:
    """Turns a source-iterator failure into a retry, a fallback, or the error reaching the caller."""
    if isinstance(stream_error, MidStreamFallbackError) and _anthropic_stream_should_decline_fallback(
        has_generated_content, stream_error
    ):
        for buffered_chunk in buffered:
            yield buffered_chunk
        if stream_error.original_exception is not None:
            raise stream_error.original_exception from stream_error
        raise stream_error
    fallback_error: Final = (
        stream_error
        if isinstance(stream_error, MidStreamFallbackError)
        else _stream_fallback_error_for_raised(attempt, stream_error, model, has_generated_content)
    )
    if fallback_error is None:
        raise stream_error
    await close_superseded()
    retried: Final = await attempt.retry(_mid_stream_retry_trigger(fallback_error))
    if isinstance(retried, BaseException):
        last_error: Final = (
            fallback_error
            if retried is _mid_stream_retry_trigger(fallback_error)
            else _stream_pre_content_error(
                retried if isinstance(retried, Exception) else fallback_error, model, fallback_error.llm_provider
            )
        )
        async for item in _anthropic_fallback_attempt(attempt, last_error, wrapper):
            yield item
        return
    response, attempted, budget = retried
    inner: Final = response
    if isinstance(inner, FallbackAwareAnthropicMessagesStream):
        inner.attempt.attempted_retries = attempted
        inner.attempt.committed_budget = budget
    async for item in _anthropic_yield_recovered(inner, wrapper, (attempted, budget)):
        yield item


async def _anthropic_yield_recovered(
    recovered: object,
    wrapper: FallbackAwareAnthropicMessagesStream,
    retry_counters: tuple[int, int] | None = None,
) -> AsyncGenerator[bytes, None]:
    """Hands a retry's or a fallback's response to the client through the wrapper, closing it after."""
    from litellm.llms.anthropic.messages.native.streaming_iterator import (
        aclose_if_supported,
        anthropic_messages_response_as_sse_events,
    )

    hidden_params, headers = _prepare_fallback_hidden_params(recovered)
    wrapper.merge_fallback_hidden_params(hidden_params, headers)
    wrapper.adopt_fallback_source(recovered)
    if retry_counters is not None:
        add_retry_headers_to_response(
            response=wrapper, attempted_retries=retry_counters[0], max_retries=retry_counters[1]
        )
    try:
        if hasattr(recovered, "__aiter__"):
            async for item in cast("AsyncIterator[bytes]", recovered):  # cast-ok: __aiter__ checked above
                wrapper.follow_source_attribution()
                yield item
            return
        # A complete response for a streaming request is replayed as the SSE lifecycle a stream sends.
        for event in anthropic_messages_response_as_sse_events(recovered):  # pyright: ignore[reportArgumentType]  # the non-streaming shape by elimination
            yield event
    finally:
        with anyio.CancelScope(shield=True), contextlib.suppress(BaseException):
            await aclose_if_supported(recovered)


async def _anthropic_fallback_attempt(
    attempt: StreamAttempt, e: MidStreamFallbackError, wrapper: FallbackAwareAnthropicMessagesStream
) -> AsyncGenerator[bytes, None]:
    """Re-enters the fallback walk for a ``/v1/messages`` stream error and yields what the fallback produces."""
    trigger: Final = _fallback_trigger(
        e, (litellm_exceptions.ContentPolicyViolationError, litellm_exceptions.ContextWindowExceededError)
    )
    try:
        fallback_response: Final = await attempt.fallback(trigger)
    except Exception as fallback_error:
        verbose_router_logger.error("Anthropic messages streaming fallback also failed: %s", fallback_error)
        if isinstance(trigger, MidStreamFallbackError) and trigger.original_exception is not None:
            raise trigger.original_exception from fallback_error
        raise trigger from fallback_error
    async for fallback_item in _anthropic_yield_recovered(fallback_response, wrapper):
        yield fallback_item


# --- Responses API ------------------------------------------------------------------------------


def _responses_stream_holds_event(item: object, held_event_count: int) -> bool:
    from litellm.responses.streaming_iterator import PRE_OUTPUT_LIFECYCLE_EVENT_TYPES

    if held_event_count >= MAX_HELD_PRE_OUTPUT_RESPONSES_EVENTS:
        return False
    return getattr(item, "type", None) in PRE_OUTPUT_LIFECYCLE_EVENT_TYPES


def _responses_event_announces_background_run(item: object) -> bool:
    return getattr(getattr(item, "response", None), "background", None) is True


def _mid_stream_error_before_output(error: MidStreamFallbackError) -> bool:
    return not error.generated_content or error.is_pre_first_chunk


def _extract_partial_responses_usage(source_iterator: "BaseResponsesAPIStreamingIterator") -> "ResponseAPIUsage | None":
    """Partial token usage of a Responses stream that failed, normalized to ``ResponseAPIUsage``."""
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.types.llms.openai import (
        ResponseAPIUsage,
        ResponseCompletedEvent,
        ResponseFailedEvent,
        ResponseIncompleteEvent,
    )

    if isinstance(source_iterator, LiteLLMCompletionStreamingIterator):
        chunks: Final = source_iterator.collected_chat_completion_chunks
        if chunks:
            try:
                from litellm.main import (
                    stream_chunk_builder,  # pyright: ignore[reportUnknownVariableType]  # untyped chunk list
                )

                built: Final = stream_chunk_builder(chunks=chunks)
                chat: Final[object | None] = getattr(built, "usage", None) if built is not None else None
                if chat is not None:
                    prompt: Final = int(getattr(chat, "prompt_tokens", 0) or 0)
                    completion: Final = int(getattr(chat, "completion_tokens", 0) or 0)
                    total: Final = int(getattr(chat, "total_tokens", prompt + completion) or (prompt + completion))
                    return ResponseAPIUsage(input_tokens=prompt, output_tokens=completion, total_tokens=total)
            except Exception as error:  # noqa: BLE001  # the builder is best-effort
                verbose_router_logger.debug("partial usage of a failed Responses stream: %s", error)
    completed: Final = source_iterator.completed_response
    if isinstance(completed, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent)):
        return completed.response.usage
    return None


def _combine_responses_fallback_usage(fallback_item: object, partial_usage: "ResponseAPIUsage") -> None:
    from litellm.types.llms.openai import (
        ResponseAPIUsage,
        ResponseCompletedEvent,
        ResponseFailedEvent,
        ResponseIncompleteEvent,
    )

    if not isinstance(fallback_item, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent)):
        return
    response: Final = fallback_item.response
    if response.usage is None:
        return
    fb: Final = response.usage
    response.usage = ResponseAPIUsage(
        input_tokens=(partial_usage.input_tokens or 0) + (fb.input_tokens or 0),
        output_tokens=(partial_usage.output_tokens or 0) + (fb.output_tokens or 0),
        total_tokens=(partial_usage.total_tokens or 0) + (fb.total_tokens or 0),
    )


def _params_sent_to_deployment(attempt: StreamAttempt, deployment_id: object) -> Mapping[str, object]:
    """The request as the deployment received it, merged in the order the attempt applies."""
    return attempt.core.params_sent_to_deployment(attempt.request, deployment_id)


def wrap_responses_stream(  # noqa: C901  # one function per Python wrapper it ports
    attempt: StreamAttempt, response: "BaseResponsesAPIStreamingIterator"
) -> "BaseResponsesAPIStreamingIterator":
    """``_aresponses_streaming_iterator``: pre-output lifecycle events are held while a fallback can take
    over; a mid-stream error falls back through the core, continuing after text."""
    import time
    from datetime import datetime, timezone

    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator, get_openai_response_types

    source_iterator: Final = response
    openai_types: Final = get_openai_response_types()
    terminal_event_types: Final = (
        openai_types.ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        openai_types.ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE,
        openai_types.ResponsesAPIStreamEvents.RESPONSE_FAILED,
    )

    class FallbackResponsesStreamWrapper(BaseResponsesAPIStreamingIterator):
        """Subclasses the base iterator only for isinstance checks; iteration is the generator's."""

        fallback_headers_adopted: bool = False

        def __init__(self, async_generator: AsyncGenerator[object, None]) -> None:  # pyright: ignore[reportMissingSuperCall]  # no httpx response of its own
            self._async_generator: Final = async_generator
            self._billing_source: object = source_iterator
            self.response = getattr(source_iterator, "response", None)  # pyright: ignore[reportAttributeAccessIssue]  # copied from the source
            self.model = getattr(source_iterator, "model", None)  # pyright: ignore[reportAttributeAccessIssue]  # see above
            self.logging_obj = getattr(
                source_iterator, "logging_obj", getattr(source_iterator, "litellm_logging_obj", None)
            )  # pyright: ignore[reportAttributeAccessIssue]  # see above
            self.finished = False
            self.responses_api_provider_config = getattr(source_iterator, "responses_api_provider_config", None)  # pyright: ignore[reportAttributeAccessIssue]  # see above
            self.completed_response = None
            self.start_time = getattr(source_iterator, "start_time", None) or datetime.now(timezone.utc).replace(
                tzinfo=None
            )
            self._failure_handled = False
            self._yielded_first_chunk = False
            self._generated_content = ""
            self._completed_response_cached = False
            self._completed_response_logged = False
            self._completed_response_cache_hit = None
            self._persist_completed_response_before_logging = True
            self._stream_created_time = time.time()
            self.litellm_metadata = getattr(source_iterator, "litellm_metadata", None)  # pyright: ignore[reportAttributeAccessIssue]  # see above
            self.custom_llm_provider = getattr(source_iterator, "custom_llm_provider", None)  # pyright: ignore[reportAttributeAccessIssue]  # see above
            self.request_data = getattr(source_iterator, "request_data", {}) or {}
            self.call_type = getattr(source_iterator, "call_type", None)  # pyright: ignore[reportAttributeAccessIssue]  # see above
            self._hidden_params = dict(getattr(source_iterator, "_hidden_params", None) or {})

        def adopt_fallback_headers(self, fallback_response: object) -> _Prepared:
            prepared: Final = _prepare_fallback_hidden_params(fallback_response)
            self._billing_source = fallback_response
            self._hidden_params = {**prepared[0], "additional_headers": prepared[1]}
            self.fallback_headers_adopted = True
            return prepared

        def partial_response_on_disconnect(self) -> "ResponseIncompleteEvent | None":
            from litellm.responses.streaming_iterator import partial_responses_incomplete_event_on_disconnect

            current: Final = partial_responses_incomplete_event_on_disconnect(self._billing_source)
            if self._billing_source is source_iterator:
                return current
            primary: Final = partial_responses_incomplete_event_on_disconnect(source_iterator)
            if current is None or primary is None:
                return current or primary
            if primary.response.usage is not None:
                _combine_responses_fallback_usage(current, primary.response.usage)
            return current

        def __aiter__(self) -> "FallbackResponsesStreamWrapper":
            return self

        @property
        def yielded_first_chunk(self) -> bool:
            return self._yielded_first_chunk

        async def __anext__(self) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # yields what the stream yields
            try:
                chunk: Final = await self._async_generator.__anext__()
            except StopAsyncIteration:
                if self.completed_response is None:
                    self.completed_response = getattr(source_iterator, "completed_response", None)
                raise
            if self.completed_response is None and getattr(chunk, "type", None) in terminal_event_types:
                self.completed_response = chunk  # pyright: ignore[reportAttributeAccessIssue]  # the terminal event
            self._yielded_first_chunk = True
            return chunk

        async def aclose(self) -> None:
            await self._async_generator.aclose()

    def background_stream_can_release(item: object) -> bool:
        from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator

        if (
            type(source_iterator) is not ResponsesAPIStreamingIterator
            or source_iterator.custom_llm_provider != "openai"
        ):
            return False
        if not _responses_event_announces_background_run(item):
            return False
        return not attempt.can_fall_back()

    async def stream_with_fallbacks() -> AsyncGenerator[object, None]:
        held: tuple[object, ...] = ()  # rebind-ok: flushed at first output, dropped on fallback
        response_id_exposed = False  # rebind-ok: set once, before an early-released event is yielded
        try:
            async for item in cast("AsyncIterator[object]", source_iterator):  # cast-ok: an async stream
                if not response_id_exposed and background_stream_can_release(item):
                    response_id_exposed = True
                if not response_id_exposed and _responses_stream_holds_event(item, len(held)):
                    held = (*held, item)
                    continue
                for held_event in held:
                    yield held_event
                held = ()
                yield item
            for held_event in held:
                yield held_event
        except (MidStreamFallbackError, litellm_exceptions.ContextWindowExceededError) as stream_error:
            e: Final = (
                _context_window_stream_fallback_error(
                    stream_error,
                    source_iterator.model or "",
                    source_iterator.custom_llm_provider or "",
                    wrapper.yielded_first_chunk,
                )
                if _is_context_window(stream_error)
                else cast("MidStreamFallbackError", stream_error)  # cast-ok: not a context window error
            )
            if response_id_exposed:
                _raise_original_mid_stream_error(e)
            async with contextlib.aclosing(
                _responses_fallback_attempt(attempt, e, source_iterator, wrapper.adopt_fallback_headers, held)
            ) as fallback_stream:
                async for fallback_item in fallback_stream:
                    yield fallback_item
        except Exception:
            for held_event in held:
                yield held_event
            raise
        finally:
            with anyio.CancelScope(shield=True):
                if hasattr(source_iterator, "aclose"):
                    try:
                        await source_iterator.aclose()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
                    except Exception as exc:  # noqa: BLE001  # closing is best-effort
                        verbose_router_logger.debug("stream_with_fallbacks(aresponses): error closing source: %s", exc)

    wrapper: Final = FallbackResponsesStreamWrapper(stream_with_fallbacks())
    return wrapper


def _responses_continuation_request(
    attempt: StreamAttempt, e: MidStreamFallbackError, source_iterator: "BaseResponsesAPIStreamingIterator"
) -> Mapping[str, object] | None:
    """The request a fallback continues from once a Responses stream broke after text, or None before
    output. Raises the primary's error when the output is constrained, a tool call began, a content
    filter cut it, or the output ceiling is spent."""
    from litellm.litellm_core_utils.token_counter import (
        token_counter,  # pyright: ignore[reportUnknownVariableType]  # untyped tokenizer argument
    )
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import (
        RESPONSES_CONTINUATION_KWARG,
        ResponsesContinuation,
        constrains_responses_continuation,
        continuation_output_ceilings,
        responses_continuation_input,
    )

    if _mid_stream_error_before_output(e):
        return None
    if source_iterator.started_tool_call or isinstance(
        e.original_exception, litellm_exceptions.ContentPolicyViolationError
    ):
        _raise_original_mid_stream_error(e)
    cut_deployment_id: Final = (get_hidden_params(source_iterator) or {}).get("model_id")
    if constrains_responses_continuation(_params_sent_to_deployment(attempt, cut_deployment_id)):
        _raise_original_mid_stream_error(e)
    request: Final = attempt.request
    earlier: Final = request.get(RESPONSES_CONTINUATION_KWARG)
    continuation: Final = (
        earlier.extended(e.generated_content)
        if isinstance(earlier, ResponsesContinuation)
        else ResponsesContinuation(request.get("input"), e.generated_content)
    )
    emitted_tokens: Final = token_counter(model=e.model or "", text=e.generated_content)
    reduced: Final = continuation_output_ceilings(request, emitted_tokens)
    if reduced is None:
        _raise_original_mid_stream_error(e)
    return {
        **request,
        **reduced,
        RESPONSES_CONTINUATION_KWARG: continuation,
        "input": responses_continuation_input(continuation),
    }


async def _responses_fallback_attempt(
    attempt: StreamAttempt,
    e: MidStreamFallbackError,
    source_iterator: "BaseResponsesAPIStreamingIterator",
    adopt_headers: Callable[[object], _Prepared],
    held: tuple[object, ...],
) -> AsyncGenerator[object, None]:
    """Re-enters the fallback walk for a mid-stream Responses error. The primary's held lifecycle events
    reach the client only when no fallback lands."""
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import ResponsesContinuationSeam

    continued: Final = _responses_continuation_request(attempt, e, source_iterator)
    seam: Final = ResponsesContinuationSeam(e.generated_content if continued is not None else "")
    partial_usage: Final = _extract_partial_responses_usage(source_iterator)
    trigger: Final = _fallback_trigger(
        e, (litellm_exceptions.ContentPolicyViolationError, litellm_exceptions.ContextWindowExceededError)
    )
    fallback_response: object = None  # rebind-ok: set once a fallback answered, closed at the end
    fallback_yielded = False  # rebind-ok: flipped on the first fallback item
    try:
        fallback_response = await attempt.fallback(trigger, continued)  # rebind-ok: rebound as the stream advances
        prepared: Final = adopt_headers(fallback_response)
        if hasattr(fallback_response, "__aiter__"):
            spliced: Final = cast(  # cast-ok: the fallback's events
                "AsyncIterator[object]",
                seam.splice(fallback_response),  # pyright: ignore[reportArgumentType]  # an async stream
            )
            async for fallback_item in spliced:
                _apply_fallback_hidden_params_to_item(fallback_item, prepared)
                if partial_usage is not None:
                    _combine_responses_fallback_usage(fallback_item, partial_usage)
                fallback_yielded = True
                yield fallback_item
        else:
            fallback_yielded = True  # rebind-ok: rebound as the stream advances
            yield fallback_response
    except Exception as fallback_error:
        verbose_router_logger.error("Responses streaming fallback also failed: %s", fallback_error)
        if not fallback_yielded:
            for held_event in held:
                yield held_event
        if fallback_response is None:
            if isinstance(trigger, MidStreamFallbackError) and trigger.original_exception is not None:
                raise trigger.original_exception from fallback_error
            raise trigger from fallback_error
        if isinstance(fallback_error, MidStreamFallbackError) and fallback_error.original_exception is not None:
            raise fallback_error.original_exception from fallback_error
        raise
    finally:
        if fallback_response is not None and hasattr(fallback_response, "aclose"):
            with anyio.CancelScope(shield=True):
                try:
                    await fallback_response.aclose()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
                except Exception as exc:  # noqa: BLE001  # closing is best-effort
                    verbose_router_logger.debug("stream_with_fallbacks(aresponses): error closing fallback: %s", exc)


# --- the attempt hook ---------------------------------------------------------------------------


async def open_attempt_stream(attempt: StreamAttempt, response: object) -> object:
    """What an attempt returns once its stream is wrapped: a chat stream's first chunk is read here, so
    a stream that fails before any token raises inside the attempt."""
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator

    if attempt.op == "acompletion" and isinstance(response, CustomStreamWrapper):
        if response.completion_stream is None and response.make_call is not None:  # pyright: ignore[reportUnknownMemberType]  # untyped attributes
            await response.fetch_stream()
        first_chunk: Final = await _aread_first_chunk(response)
        held_slot: Final = attempt.slot.pop_all() if attempt.slot is not None else contextlib.ExitStack()
        return wrap_chat_stream(attempt, response, first_chunk, held_slot)
    if not attempt.kwargs.get("stream"):
        return response
    if attempt.op == "aresponses" and isinstance(response, BaseResponsesAPIStreamingIterator):
        return wrap_responses_stream(attempt, response)
    if attempt.op in ("aanthropic_messages", "anthropic_messages") and hasattr(response, "__aiter__"):
        return wrap_anthropic_stream(attempt, cast("AsyncIterator[bytes]", response))  # cast-ok: a byte stream
    return response


def open_attempt_stream_sync(attempt: StreamAttempt, response: object) -> object:
    if attempt.op == "completion" and isinstance(response, CustomStreamWrapper):
        if response.completion_stream is None and response.make_call is not None:  # pyright: ignore[reportUnknownMemberType]  # untyped attributes
            response.fetch_sync_stream()
        return wrap_chat_stream_sync(attempt, response, _read_first_chunk(response))
    return response


def shape_attempt_kwargs(kwargs: Mapping[str, object], deployment: Mapping[str, object]) -> Mapping[str, object]:
    """A Responses continuation is shaped for the deployment the fallback landed on."""
    from litellm.router_utils.pre_call_checks.continuation_prefill_check import (
        RESPONSES_CONTINUATION_KWARG,
        responses_continuation_input_for,
    )

    if RESPONSES_CONTINUATION_KWARG not in kwargs:
        return kwargs
    shaped: Final = {key: value for key, value in kwargs.items() if key != RESPONSES_CONTINUATION_KWARG}
    shaped["input"] = responses_continuation_input_for(kwargs, deployment)
    return shaped


__all__ = [
    "MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS",
    "MAX_HELD_PRE_OUTPUT_RESPONSES_EVENTS",
    "FallbackAwareAnthropicMessagesStream",
    "FallbackAwareStreamWrapper",
    "StreamAttempt",
    "open_attempt_stream",
    "open_attempt_stream_sync",
    "shape_attempt_kwargs",
]
