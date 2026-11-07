"""Keeps a mid-stream continuation on deployments whose model supports assistant prefill and whose own params leave
the output unconstrained, and shapes a Responses API continuation for the deployment that serves it."""

import json
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from os.path import commonprefix
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.routing_common_utils import MID_STREAM_CONTINUATION_KWARG
from litellm.litellm_core_utils.rules import Rules
from litellm.types.llms.openai import AllMessageValues, OutputTextDeltaEvent
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage
from litellm.utils import supports_assistant_prefill


class _ContinuationMarker:
    """Only the router constructs one, so a JSON request body cannot forge the continuation flag."""


MID_STREAM_CONTINUATION_MARKER: Final = _ContinuationMarker()

_STR_KEYED_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_NO_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})
_UNCONSTRAINED_RESPONSE_FORMAT: Final[Mapping[str, str]] = MappingProxyType({"type": "text"})
_OUTPUT_CEILING_KEYS: Final = ("max_tokens", "max_completion_tokens", "max_output_tokens")
_QUOTED_ENDING_CHARS: Final = 100
RESPONSES_CONTINUATION_KWARG: Final = "_responses_continuation"


def _str_keyed(value: object) -> Mapping[str, object]:
    try:
        return _STR_KEYED_DICT_ADAPTER.validate_python(value)
    except ValidationError:
        return _NO_FIELDS


def _underlying_model(deployment: object) -> str | None:
    deployment_map: Final = _str_keyed(deployment)
    model_info: Final = _str_keyed(deployment_map.get("model_info"))
    litellm_params: Final = _str_keyed(deployment_map.get("litellm_params"))
    candidates: Final = (model_info.get("base_model"), litellm_params.get("base_model"), litellm_params.get("model"))
    return next((candidate for candidate in candidates if isinstance(candidate, str) and candidate), None)


def deployment_supports_prefill(deployment: object) -> bool:
    declared: Final = _str_keyed(_str_keyed(deployment).get("model_info")).get("supports_assistant_prefill")
    if isinstance(declared, bool):
        return declared
    model: Final = _underlying_model(deployment)
    return model is not None and supports_assistant_prefill(model=model)


def _forces_tool_call(tool_choice: object) -> bool:
    return tool_choice == "required" or isinstance(tool_choice, Mapping)


def _forces_responses_tool_call(tool_choice: object) -> bool:
    choice: Final = _str_keyed(tool_choice)
    leaves_text_open: Final = choice.get("type") == "allowed_tools" and choice.get("mode") == "auto"
    return _forces_tool_call(tool_choice) and not leaves_text_open


def _constrains(params: Mapping[str, object], forces_tool_call: Callable[[object], bool]) -> bool:
    sent: Final = {**params, **_str_keyed(params.get("extra_body"))}
    choice_count: Final = sent.get("n")
    response_format: Final = sent.get("response_format")
    text_format: Final = _str_keyed(sent.get("text")).get("format")
    return (
        (isinstance(choice_count, int) and choice_count > 1)
        or (response_format is not None and response_format != _UNCONSTRAINED_RESPONSE_FORMAT)
        or (text_format is not None and text_format != _UNCONSTRAINED_RESPONSE_FORMAT)
        or sent.get("text_format") is not None
        or forces_tool_call(sent.get("tool_choice"))
        or sent.get("merge_reasoning_content_in_choices") is True
    )


def constrains_continuation(params: Mapping[str, object]) -> bool:
    return _constrains(params, _forces_tool_call)


def constrains_responses_continuation(params: Mapping[str, object]) -> bool:
    return _constrains(params, _forces_responses_tool_call)


def deployment_can_continue(deployment: object) -> bool:
    litellm_params: Final = _str_keyed(_str_keyed(deployment).get("litellm_params"))
    return deployment_supports_prefill(deployment) and not constrains_continuation(litellm_params)


def continuation_deployments(
    healthy_deployments: list[dict[str, object]],  # mutable-ok: router deployment-list contract
    request_kwargs: Mapping[str, object] | None,
) -> list[dict[str, object]]:  # mutable-ok: returns a mutable deployment list
    marker: Final = (request_kwargs or {}).get(MID_STREAM_CONTINUATION_KWARG)
    if not isinstance(marker, _ContinuationMarker):
        return healthy_deployments
    return [deployment for deployment in healthy_deployments if deployment_can_continue(deployment)]


class ContinuationPrefillDeploymentCheck(CustomLogger):
    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]],  # mutable-ok: CustomLogger deployment-list contract
        messages: Sequence[AllMessageValues] | None,
        request_kwargs: Mapping[str, object] | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict[str, object]]:  # mutable-ok: returns a mutable deployment list
        return continuation_deployments(healthy_deployments, request_kwargs)


def _delta_text(chunk: ModelResponseStream) -> str:
    if not chunk.choices:
        return ""
    content: Final = chunk.choices[0].delta.content
    return content if isinstance(content, str) else ""


_CONTENT_BLOCKS_ADAPTER: Final = TypeAdapter(list[dict[str, object]])
_INPUT_ITEMS_ADAPTER: Final = TypeAdapter(list[object])


def _prefill_text(prefill_message: Mapping[str, object]) -> str:
    content: Final = prefill_message.get("content")
    if isinstance(content, str):
        return content
    try:
        blocks: Final = _CONTENT_BLOCKS_ADAPTER.validate_python(content)
    except ValidationError:
        return ""
    return "".join(
        text for block in blocks if block.get("type") == "text" and isinstance(text := block.get("text"), str)
    )


def resendable_whitespace(prefill: str) -> str:
    return prefill[len(prefill.rstrip()) :] if prefill.strip() else ""


def continued_text(prefill_message: Mapping[str, object], generated_content: str) -> str | None:
    prefill: Final = _prefill_text(prefill_message)
    if generated_content and len(generated_content) < len(prefill) and prefill.startswith(generated_content):
        return None
    echo: Final = len(prefill) if prefill and generated_content.startswith(prefill) else 0
    after_echo: Final = generated_content[echo:]
    return after_echo[len(commonprefix((after_echo, resendable_whitespace(prefill)))) :]


def _reduced_ceilings(params: Mapping[str, object], emitted_tokens: int) -> Mapping[str, int]:
    return MappingProxyType(
        {key: value - emitted_tokens for key in _OUTPUT_CEILING_KEYS if isinstance(value := params.get(key), int)}
    )


def continuation_output_ceilings(
    request_kwargs: Mapping[str, object], emitted_tokens: int
) -> Mapping[str, object] | None:
    """The caller's output ceilings, top-level and in extra_body, minus the tokens already emitted, or None once
    nothing is left."""
    extra_body: Final = _str_keyed(request_kwargs.get("extra_body"))
    top_level: Final = _reduced_ceilings(request_kwargs, emitted_tokens)
    in_extra_body: Final = _reduced_ceilings(extra_body, emitted_tokens)
    if any(ceiling <= 0 for ceiling in (*top_level.values(), *in_extra_body.values())):
        return None
    if not in_extra_body:
        return top_level
    return MappingProxyType({**top_level, "extra_body": {**extra_body, **in_extra_body}})


def _with_first_delta(chunk: ModelResponseStream, update: Mapping[str, object]) -> ModelResponseStream:
    choice: Final = chunk.choices[0]
    delta: Final = choice.delta.model_copy(update=update)
    return chunk.model_copy(update={"choices": [choice.model_copy(update={"delta": delta}), *chunk.choices[1:]]})


def _without_leading_text(chunk: ModelResponseStream, count: int) -> ModelResponseStream:
    if count <= 0:
        return chunk
    return _with_first_delta(chunk, {"content": _delta_text(chunk)[count:]})


def _without_role(chunk: ModelResponseStream) -> ModelResponseStream:
    if not chunk.choices or chunk.choices[0].delta.role is None:
        return chunk
    return _with_first_delta(chunk, {"role": None})


def _stream_identity(primary_response: object) -> Mapping[str, object]:
    if not isinstance(primary_response, ModelResponse):
        return _NO_FIELDS
    return MappingProxyType({"id": primary_response.id, "created": primary_response.created})


def _drop_leading_text(chunks: tuple[ModelResponseStream, ...], count: int) -> tuple[ModelResponseStream, ...]:
    lengths: Final = tuple(len(_delta_text(chunk)) for chunk in chunks)
    offsets: Final = tuple(accumulate(lengths, initial=0))[:-1]
    return tuple(
        _without_leading_text(chunk, min(length, count - offset))
        for chunk, offset, length in zip(chunks, offsets, lengths, strict=True)
    )


@dataclass(frozen=True, slots=True)
class ContinuedPartial:
    usage: Usage
    response_cost: float
    response_id: str


class ContinuationOutput:
    """Splices a fallback's continuation onto the stream the client already received. Drops a leading echo of the
    prefill, since some providers (Mistral) return the prefix inside the completion, and the prefill's trailing
    whitespace sent again by a provider that strips it (Anthropic), keeps the primary's chunk id, created time and single role delta, and
    runs litellm.post_call_rules over the joined text, since each stream wrapper only sees its own fragment. A
    continuation nested in another leaves the rules to the outermost one, the only one holding the whole text."""

    def __init__(
        self,
        emitted_text: str,
        prefill_message: Mapping[str, object],
        model: str,
        primary_response: object,
        checks_post_call_rules: bool = True,
    ) -> None:
        self._text = emitted_text
        self._checks_post_call_rules: Final = checks_post_call_rules
        self._prefill_text: Final = _prefill_text(prefill_message)
        self._model: Final = model
        self._identity: Final = _stream_identity(primary_response)
        self._rules: Final = Rules()
        self._held: tuple[ModelResponseStream, ...] = ()
        self._echo_settled = not self._prefill_text
        self._seam = resendable_whitespace(self._prefill_text)

    async def splice(self, stream: AsyncIterable[object]) -> AsyncIterator[object]:
        async for item in stream:
            if not isinstance(item, ModelResponseStream):
                yield item
                continue
            for released in self._observed(self._past_seam(self._released(item))):
                yield self._in_primary_stream(released)
        for released in self._observed(self._past_seam(self._flushed())):
            yield self._in_primary_stream(released)

    def _in_primary_stream(self, chunk: ModelResponseStream) -> ModelResponseStream:
        return _without_role(chunk.model_copy(update=self._identity) if self._identity else chunk)

    def _flushed(self) -> tuple[ModelResponseStream, ...]:
        held: Final = self._held
        self._held = ()
        self._echo_settled = True
        return held

    def _released(self, chunk: ModelResponseStream) -> tuple[ModelResponseStream, ...]:
        if self._echo_settled:
            return (chunk,)
        held: Final = (*self._held, chunk)
        text: Final = "".join(_delta_text(held_chunk) for held_chunk in held)
        if len(text) < len(self._prefill_text) and self._prefill_text.startswith(text):
            self._held = held
            return ()
        self._held = ()
        self._echo_settled = True
        if not text.startswith(self._prefill_text):
            return held
        return _drop_leading_text(held, len(self._prefill_text))

    def _past_seam(self, chunks: tuple[ModelResponseStream, ...]) -> tuple[ModelResponseStream, ...]:
        return tuple(self._without_seam(chunk) for chunk in chunks)

    def _without_seam(self, chunk: ModelResponseStream) -> ModelResponseStream:
        text: Final = _delta_text(chunk)
        matched: Final = len(commonprefix((text, self._seam)))
        self._seam = self._seam[matched:] if matched == len(text) else ""
        return _without_leading_text(chunk, matched)

    def _observed(self, chunks: tuple[ModelResponseStream, ...]) -> tuple[ModelResponseStream, ...]:
        if not self._checks_post_call_rules or not litellm.post_call_rules:
            return chunks
        for chunk in chunks:
            self._text += _delta_text(chunk)
            self._rules.post_call_rules(input=self._text, model=self._model)
        return chunks


@dataclass(frozen=True, slots=True)
class ResponsesContinuation:
    prior_input: object
    emitted_text: str

    def extended(self, generated_content: str) -> "ResponsesContinuation":
        return ResponsesContinuation(self.prior_input, f"{self.emitted_text}{generated_content}")


def deployment_takes_responses_prefill(deployment: object) -> bool:
    """Claude takes a trailing assistant turn as prefill with no `prefix` flag, which a Responses input item cannot
    carry."""
    model: Final = _underlying_model(deployment)
    return model is not None and "claude" in model.lower() and deployment_supports_prefill(deployment)


def _continuation_request(emitted_text: str) -> Mapping[str, object]:
    seam: Final = (
        "It already ends with whitespace, so do not add another space before your first word."
        if emitted_text[-1:].isspace()
        else "It ends without trailing whitespace, so begin with a space if your first word starts a new word."
    )
    ending: Final = json.dumps(emitted_text[-_QUOTED_ENDING_CHARS:], ensure_ascii=False)
    instruction: Final = (
        f"The previous assistant response was interrupted mid-stream. It ended with {ending}. Continue exactly where "
        "it stopped, so your first characters are the ones that would have come right after it, and do not repeat "
        f"any of its content. {seam} Your response must read as a seamless continuation."
    )
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": instruction}]}


def _input_items(request_input: object) -> tuple[object, ...]:
    if isinstance(request_input, str):
        return ({"type": "message", "role": "user", "content": [{"type": "input_text", "text": request_input}]},)
    try:
        return tuple(_INPUT_ITEMS_ADAPTER.validate_python(request_input))
    except ValidationError:
        return ()


def responses_continuation_input(continuation: ResponsesContinuation) -> Sequence[object]:
    """The prior input, the partial response as assistant history, and a user request to continue it."""
    partial: Final = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": continuation.emitted_text}],
    }
    return [*_input_items(continuation.prior_input), partial, _continuation_request(continuation.emitted_text)]


def responses_continuation_input_for(request_kwargs: Mapping[str, object], deployment: Mapping[str, object]) -> object:
    """The continuation input as the deployment receives it, without the request to continue where the deployment
    takes the partial response as prefill. It reads the input the router already stripped of reasoning this
    deployment cannot decrypt, never the continuation's prior input."""
    request_input: Final = request_kwargs.get("input")
    continuation: Final = request_kwargs.get(RESPONSES_CONTINUATION_KWARG)
    if not isinstance(continuation, ResponsesContinuation):
        return request_input
    items: Final = _input_items(request_input)
    if items[-1:] != (_continuation_request(continuation.emitted_text),) or not deployment_takes_responses_prefill(
        deployment
    ):
        return request_input
    return list(items[:-1])


_StreamItem = TypeVar("_StreamItem")


class ResponsesContinuationSeam:
    """Drops whitespace a fallback sends again at the start of its text when the client already has it, since
    Anthropic strips a prefill's trailing whitespace and the model then writes it back."""

    def __init__(self, emitted_text: str) -> None:
        self._seam = resendable_whitespace(emitted_text)

    async def splice(self, stream: AsyncIterable[_StreamItem]) -> AsyncIterator[_StreamItem]:
        async for item in stream:
            yield self._past(item)

    def _past(self, item: _StreamItem) -> _StreamItem:
        if not self._seam or not isinstance(item, OutputTextDeltaEvent):
            return item
        matched: Final = len(commonprefix((item.delta, self._seam)))
        self._seam = self._seam[matched:] if matched == len(item.delta) else ""
        return item.model_copy(update={"delta": item.delta[matched:]}) if matched else item
