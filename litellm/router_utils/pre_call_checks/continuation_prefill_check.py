"""Keeps a mid-stream continuation on deployments whose model supports assistant prefill."""

from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from itertools import accumulate
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.litellm_core_utils.rules import Rules
from litellm.router_utils.common_utils import MID_STREAM_CONTINUATION_KWARG
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponseStream
from litellm.utils import supports_assistant_prefill


class _ContinuationMarker:
    """Only the router constructs one, so a JSON request body cannot forge the continuation flag."""


MID_STREAM_CONTINUATION_MARKER: Final = _ContinuationMarker()

_STR_KEYED_DICT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _declared_prefill_support(deployment_map: Mapping[str, object]) -> bool | None:
    try:
        model_info: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment_map.get("model_info"))
    except ValidationError:
        return None
    declared: Final = model_info.get("supports_assistant_prefill")
    return declared if isinstance(declared, bool) else None


def _deployment_supports_prefill(deployment: object) -> bool:
    try:
        deployment_map: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment)
    except ValidationError:
        return False
    declared: Final = _declared_prefill_support(deployment_map)
    if declared is not None:
        return declared
    try:
        litellm_params: Final = _STR_KEYED_DICT_ADAPTER.validate_python(deployment_map.get("litellm_params"))
    except ValidationError:
        return False
    model: Final = litellm_params.get("model")
    return isinstance(model, str) and bool(model) and supports_assistant_prefill(model=model)


class ContinuationPrefillDeploymentCheck(CustomLogger):
    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]],  # mutable-ok: CustomLogger deployment-list contract
        messages: Sequence[AllMessageValues] | None,
        request_kwargs: Mapping[str, object] | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict[str, object]]:  # mutable-ok: returns a mutable deployment list
        marker: Final = (request_kwargs or {}).get(MID_STREAM_CONTINUATION_KWARG)
        if not isinstance(marker, _ContinuationMarker):
            return healthy_deployments
        eligible: Final = (deployment for deployment in healthy_deployments if _deployment_supports_prefill(deployment))
        return list(eligible)


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


def _without_leading_text(chunk: ModelResponseStream, count: int) -> ModelResponseStream:
    if count <= 0:
        return chunk
    choice: Final = chunk.choices[0]
    delta: Final = choice.delta.model_copy(update={"content": _delta_text(chunk)[count:]})
    return chunk.model_copy(update={"choices": [choice.model_copy(update={"delta": delta}), *chunk.choices[1:]]})


def _drop_leading_text(chunks: tuple[ModelResponseStream, ...], count: int) -> tuple[ModelResponseStream, ...]:
    lengths: Final = tuple(len(_delta_text(chunk)) for chunk in chunks)
    offsets: Final = tuple(accumulate(lengths, initial=0))[:-1]
    return tuple(
        _without_leading_text(chunk, min(length, count - offset))
        for chunk, offset, length in zip(chunks, offsets, lengths, strict=True)
    )


class ContinuationOutput:
    """Splices a fallback's continuation onto the text the client already received. Drops a leading echo of the
    prefill, since some providers (Mistral) return the prefix inside the completion, and runs litellm.post_call_rules
    over the joined text, since each stream wrapper only sees its own fragment."""

    def __init__(self, emitted_text: str, prefill_message: Mapping[str, object], model: str) -> None:
        self._text = emitted_text
        self._prefill_text: Final = _prefill_text(prefill_message)
        self._model: Final = model
        self._rules: Final = Rules()
        self._held: tuple[ModelResponseStream, ...] = ()
        self._echo_settled = not self._prefill_text

    async def splice(self, stream: AsyncIterable[object]) -> AsyncIterator[object]:
        async for item in stream:
            if not isinstance(item, ModelResponseStream):
                yield item
                continue
            for released in self._observed(self._released(item)):
                yield released
        for released in self._observed(self._flushed()):
            yield released

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

    def _observed(self, chunks: tuple[ModelResponseStream, ...]) -> tuple[ModelResponseStream, ...]:
        if not litellm.post_call_rules:
            return chunks
        for chunk in chunks:
            self._text += _delta_text(chunk)
            self._rules.post_call_rules(input=self._text, model=self._model)
        return chunks
