import codecs
import functools
import json
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter, ValidationError, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import StreamingChatCompletionChunk

STR_KEYED_MAPPING: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _SageMakerToken(TypedDict, total=False):
    text: ReadOnly[str | None]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _SageMakerChunkData(TypedDict, total=False):
    token: ReadOnly[_SageMakerToken | None]
    index: ReadOnly[int | None]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _SageMakerMessagesChunkData(TypedDict, total=False):
    choices: ReadOnly[Required[list[Mapping[str, object]]]]


class _SageMakerChunkDataRoot(RootModel[_SageMakerChunkData]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


class _SageMakerMessagesChunkDataRoot(RootModel[_SageMakerMessagesChunkData]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


_STREAM_FRAMING: Final = re.compile(r"(?:\s|data:|\[DONE\]|(?::|event:|id:|retry:)[^\n]*\n)*")
_JSON_DECODER: Final = json.JSONDecoder()
_RAW_DECODED: Final = TypeAdapter(tuple[object, int])
_NO_RECORD: Final = object()


def _framing_end(text: str) -> int:
    framing: Final = _STREAM_FRAMING.match(text)
    return framing.end() if framing else 0


class _StreamRecordBuffer:
    """
    SageMaker cuts a container's stream into PayloadParts at arbitrary points, so one part can hold
    several SSE events or JSON lines, or only part of one, down to half of a UTF-8 character
    """

    def __init__(self) -> None:
        self._decoder: Final = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending: str = ""

    def feed(self, data: bytes) -> tuple[object, ...]:
        self._pending += self._decoder.decode(data)
        return tuple(iter(self._pop_record, _NO_RECORD))

    def unparsed(self) -> str:
        return self._pending + self._decoder.decode(b"", final=True)

    def _pop_record(self) -> object:
        start: Final = _framing_end(self._pending)
        try:
            record, end = _RAW_DECODED.validate_python(_JSON_DECODER.raw_decode(self._pending, start))
        except json.JSONDecodeError:
            self._pending = self._pending[start:]
            return _NO_RECORD
        self._pending = self._pending[end:]
        return record


def _load_sagemaker_response_stream_shape():
    try:
        from botocore.loaders import Loader
        from botocore.model import ServiceModel

        loader: Final = Loader()
        service_dict: Final = STR_KEYED_MAPPING.validate_python(
            loader.load_service_model("sagemaker-runtime", "service-2")
        )
        return ServiceModel(service_dict).shape_for("InvokeEndpointWithResponseStreamOutput")
    except Exception as e:
        verbose_logger.warning(
            "litellm: could not load sagemaker-runtime response stream shape "
            "— SageMaker event-stream decoding will be unavailable. Error: %s",
            e,
        )
        return None


@functools.lru_cache(maxsize=1)
def get_sagemaker_response_stream_shape():
    """
    Lazily load and cache the sagemaker-runtime stream shape for the process.

    Avoids importing botocore (and logging warnings) unless SageMaker event-stream
    decoding is actually needed.
    """
    return _load_sagemaker_response_stream_shape()


def with_inference_component_hint(error_message: str) -> str:
    if "Inference Component Name header is required" in error_message:
        return f"{error_message}\n pass in via `litellm.embedding(..., model_id={{InferenceComponentName}})`"
    if "Inference Component Name header is not allowed" in error_message:
        return f"{error_message}\n remove `model_id` from this deployment, the endpoint has no inference components"
    return error_message


class _ClientErrorBody(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    message: str | None = Field(default=None, alias="Message")


class _ClientErrorMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    http_status_code: int = Field(default=500, alias="HTTPStatusCode")


class _ClientErrorResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    error: _ClientErrorBody = Field(default_factory=_ClientErrorBody, alias="Error")
    response_metadata: _ClientErrorMetadata = Field(default_factory=_ClientErrorMetadata, alias="ResponseMetadata")


def client_error_details(error: Exception) -> tuple[int, str]:
    try:
        response: Final = _ClientErrorResponse.model_validate(getattr(error, "response", None))
    except ValidationError:
        return 500, str(error)
    return response.response_metadata.http_status_code, response.error.message or str(error)


class SagemakerError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AWSEventStreamDecoder:
    def __init__(self, model: str, is_messages_api: bool | None = None) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.model = model
        self.parser = EventStreamJSONParser()
        self.content_blocks: list = []
        self.is_messages_api = is_messages_api

    def _chunk_parser_messages_api(self, chunk_data: _SageMakerMessagesChunkData) -> StreamingChatCompletionChunk:
        openai_chunk: Final = StreamingChatCompletionChunk(**{"model": self.model, **chunk_data})

        return openai_chunk

    def _chunk_parser(self, chunk_data: _SageMakerChunkData) -> GChunk:
        verbose_logger.debug("in sagemaker chunk parser, chunk_data %s", chunk_data)
        _token: Final = chunk_data.get("token", {}) or {}
        _index: Final = chunk_data.get("index", None) or 0
        is_finished: Final = False
        finish_reason: Final = ""

        _text: Final = _token.get("text", "")
        if _text == "<|endoftext|>":
            return GChunk(
                text="",
                index=_index,
                is_finished=True,
                finish_reason="stop",
                usage=None,
            )

        return GChunk(
            text=_text,
            index=_index,
            is_finished=is_finished,
            finish_reason=finish_reason,
            usage=None,
        )

    def _record_to_chunk(self, record: object) -> GChunk | StreamingChatCompletionChunk | None:
        if isinstance(record, Mapping) and "error" in record:
            verbose_logger.error("SageMaker endpoint streamed an error: %s", json.dumps(record))
            return None
        try:
            if self.is_messages_api:
                return self._chunk_parser_messages_api(
                    chunk_data=_SageMakerMessagesChunkDataRoot.model_validate(record).root
                )
            return self._chunk_parser(chunk_data=_SageMakerChunkDataRoot.model_validate(record).root)
        except ValidationError:
            verbose_logger.error("Skipping a SageMaker stream record that is not a chunk: %s", json.dumps(record))
            return None

    def _chunks(self, records: tuple[object, ...]) -> Iterator[GChunk | StreamingChatCompletionChunk]:
        return (chunk for chunk in map(self._record_to_chunk, records) if chunk is not None)

    def iter_bytes(self, iterator: Iterator[bytes]) -> Iterator[GChunk | StreamingChatCompletionChunk | None]:
        """Given an iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer: Final = EventStreamBuffer()
        records: Final = _StreamRecordBuffer()

        for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                if payload := self._parse_message_from_event(event):
                    yield from self._chunks(records.feed(payload))

        unparsed: Final = records.unparsed()
        if unparsed:
            verbose_logger.error("Warning: Unparseable JSON data remained: %s", unparsed)
            yield None

    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[GChunk | StreamingChatCompletionChunk | None]:
        """Given an async iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer: Final = EventStreamBuffer()
        records: Final = _StreamRecordBuffer()

        async for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                try:
                    if payload := self._parse_message_from_event(event):
                        verbose_logger.debug("sagemaker  parsed chunk bytes %s", payload)
                        for chunk in self._chunks(records.feed(payload)):
                            yield chunk
                except Exception as e:
                    verbose_logger.error("Error parsing message: %s. Attempting to combine with next event.", e)

        unparsed: Final = records.unparsed()
        if unparsed:
            verbose_logger.error("Warning: Unparseable JSON data remained: %s", unparsed)
            yield None

    def _parse_message_from_event(self, event) -> bytes | None:
        response_stream_shape: Final = get_sagemaker_response_stream_shape()
        if response_stream_shape is None:
            raise SagemakerError(
                status_code=500,
                message=(
                    "SageMaker event-stream shape could not be loaded from botocore. "
                    "Ensure botocore is correctly installed."
                ),
            )
        response_dict: Final = event.to_response_dict()
        parsed_response: Final = self.parser.parse(response_dict, response_stream_shape)

        if response_dict["status_code"] != 200:
            raise ValueError(f"Bad response code, expected 200: {response_dict}")

        if "chunk" in parsed_response:
            chunk = parsed_response.get("chunk")
            if not chunk:
                return None
            return chunk.get("bytes")
        else:
            chunk = response_dict.get("body")
            if not chunk:
                return None

            return chunk
