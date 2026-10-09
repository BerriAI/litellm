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


def output_text(items: Sequence[OutputItem] | None) -> str:
    return "".join(_message_text(item) for item in items or ())


def _message_text(item: OutputItem) -> str:
    if item.type != "message":
        return ""
    return "".join(part.text or "" for part in item.content or () if part.type == "output_text")


def _text_chunk(text: str) -> GenericStreamingChunk:
    return GenericStreamingChunk(text=text, is_finished=False, finish_reason="", usage=None, index=0, tool_use=None)


class DatabricksAgentResponsesIterator(BaseModelResponseIterator):
    _streamed_item_ids: frozenset[str] = frozenset()

    def chunk_parser(self, chunk: Mapping[str, object]) -> GenericStreamingChunk:
        event: Final = _StreamEvent.model_validate(chunk)
        if event.type == "response.output_text.delta":
            if event.item_id is not None:
                self._streamed_item_ids = self._streamed_item_ids | {event.item_id}
            return _text_chunk(event.delta or "")
        if event.type == "response.output_item.done":
            if event.item is None or event.item.id in self._streamed_item_ids:
                return _text_chunk("")
            return _text_chunk(output_text((event.item,)))
        if event.type == "error":
            raise DatabricksException(status_code=500, message=f"Databricks agent stream error: {event.message}")
        return _text_chunk("")
