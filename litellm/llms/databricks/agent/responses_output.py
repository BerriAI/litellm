from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.databricks.common_utils import DatabricksException
from litellm.types.utils import GenericStreamingChunk, Usage


class _ContentPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    text: str | None = None


class OutputItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    id: str | None = None
    content: tuple[_ContentPart, ...] | None = None


class AgentUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    output_tokens: int
    total_tokens: int | None = None

    def to_usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.input_tokens,
            completion_tokens=self.output_tokens,
            total_tokens=self.total_tokens if self.total_tokens is not None else self.input_tokens + self.output_tokens,
        )


class AgentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    output: tuple[OutputItem, ...]
    custom_outputs: object = None
    usage: AgentUsage | None = None


class _StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str = ""
    item_id: str | None = None
    delta: str | None = None
    item: OutputItem | None = None
    message: object = None


_MESSAGE_SEPARATOR: Final = "\n\n"


def output_text(items: Sequence[OutputItem] | None) -> str:
    texts: Final = (_message_text(item) for item in items or ())
    return _MESSAGE_SEPARATOR.join(text for text in texts if text)


def _message_text(item: OutputItem) -> str:
    if item.type != "message":
        return ""
    return "".join(part.text or "" for part in item.content or () if part.type == "output_text")


def _text_chunk(text: str) -> GenericStreamingChunk:
    return GenericStreamingChunk(text=text, is_finished=False, finish_reason="", usage=None, index=0, tool_use=None)


class DatabricksAgentResponsesIterator(BaseModelResponseIterator):
    _streamed_item_ids: frozenset[str] = frozenset()
    _emitted_text: bool = False
    _last_text_item_id: str | None = None

    def chunk_parser(self, chunk: Mapping[str, object]) -> GenericStreamingChunk:
        event: Final = _StreamEvent.model_validate(chunk)
        if event.type == "response.output_text.delta":
            if event.item_id is not None:
                self._streamed_item_ids = self._streamed_item_ids | {event.item_id}
            return self._item_text_chunk(event.delta or "", event.item_id)
        if event.type == "response.output_item.done":
            if event.item is None or event.item.id in self._streamed_item_ids:
                return _text_chunk("")
            return self._item_text_chunk(output_text((event.item,)), event.item.id)
        if event.type == "error":
            raise DatabricksException(status_code=500, message=f"Databricks agent stream error: {event.message}")
        return _text_chunk("")

    def _item_text_chunk(self, text: str, item_id: str | None) -> GenericStreamingChunk:
        if not text:
            return _text_chunk("")
        separator: Final = _MESSAGE_SEPARATOR if self._emitted_text and item_id != self._last_text_item_id else ""
        self._emitted_text = True
        self._last_text_item_id = item_id
        return _text_chunk(f"{separator}{text}")
