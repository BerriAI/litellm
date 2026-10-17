"""Keeps a mid-stream continuation on deployments whose model supports assistant prefill and whose own params leave
the output unconstrained."""

from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from os.path import commonprefix
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.rules import Rules
from litellm.router_utils.common_utils import MID_STREAM_CONTINUATION_KWARG
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage
from litellm.utils import supports_assistant_prefill


class _ContinuationMarker:
    """Only the router constructs one, so a JSON request body cannot forge the continuation flag."""


MID_STREAM_CONTINUATION_MARKER: Final = _ContinuationMarker()

_STR_KEYED_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])
_NO_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})
_UNCONSTRAINED_RESPONSE_FORMAT: Final[Mapping[str, str]] = MappingProxyType({"type": "text"})
_OUTPUT_CEILING_KEYS: Final = ("max_tokens", "max_completion_tokens")


def _str_keyed(value: object) -> Mapping[str, object]:
    try:
        return _STR_KEYED_DICT_ADAPTER.validate_python(value)
    except ValidationError:
        return _NO_FIELDS


def deployment_supports_prefill(deployment: object) -> bool:
    deployment_map: Final = _str_keyed(deployment)
    model_info: Final = _str_keyed(deployment_map.get("model_info"))
    declared: Final = model_info.get("supports_assistant_prefill")
    if isinstance(declared, bool):
        return declared
    litellm_params: Final = _str_keyed(deployment_map.get("litellm_params"))
    candidates: Final = (model_info.get("base_model"), litellm_params.get("base_model"), litellm_params.get("model"))
    model: Final = next((candidate for candidate in candidates if isinstance(candidate, str) and candidate), None)
    return model is not None and supports_assistant_prefill(model=model)


def constrains_continuation(params: Mapping[str, object]) -> bool:
    sent: Final = {**params, **_str_keyed(params.get("extra_body"))}
    choice_count: Final = sent.get("n")
    response_format: Final = sent.get("response_format")
    tool_choice: Final = sent.get("tool_choice")
    return (
        (isinstance(choice_count, int) and choice_count > 1)
        or (response_format is not None and response_format != _UNCONSTRAINED_RESPONSE_FORMAT)
        or tool_choice == "required"
        or isinstance(tool_choice, Mapping)
        or sent.get("merge_reasoning_content_in_choices") is True
    )


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


def _resendable_whitespace(prefill: str) -> str:
    return prefill[len(prefill.rstrip()) :] if prefill.strip() else ""


def continued_text(prefill_message: Mapping[str, object], generated_content: str) -> str | None:
    prefill: Final = _prefill_text(prefill_message)
    if generated_content and len(generated_content) < len(prefill) and prefill.startswith(generated_content):
        return None
    echo: Final = len(prefill) if prefill and generated_content.startswith(prefill) else 0
    after_echo: Final = generated_content[echo:]
    return after_echo[len(commonprefix((after_echo, _resendable_whitespace(prefill)))) :]


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
        self._seam = _resendable_whitespace(self._prefill_text)

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
