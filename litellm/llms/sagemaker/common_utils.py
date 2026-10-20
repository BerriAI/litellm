import codecs
import functools
import json
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from types import MappingProxyType
from typing import Final, Protocol

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
_NO_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})
_CLIENT_ERROR_TYPE_STATUS: Final = MappingProxyType(
    {
        "overloaded": 429,
        "validation": 422,
        "template_error": 422,
        "missing_template_variable": 422,
        "tool_error": 422,
    }
)
_CLIENT_ERROR_STATUSES: Final = frozenset(_CLIENT_ERROR_TYPE_STATUS.values())


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

    def close(self) -> "SagemakerError | str":
        unparsed: Final = self._pending + self._decoder.decode(b"", final=True)
        record_start: Final = unparsed[_framing_end(unparsed) :]
        if record_start.startswith(("{", "[")):
            return SagemakerError(
                status_code=500,
                message=f"SageMaker stream ended inside a record, {len(record_start)} characters into it",
            )
        return unparsed

    def _pop_record(self) -> object:
        start: Final = _framing_end(self._pending)
        try:
            record, end = _RAW_DECODED.validate_python(_JSON_DECODER.raw_decode(self._pending, start))
        except json.JSONDecodeError:
            self._pending = self._pending[start:]
            return _NO_RECORD
        self._pending = self._pending[end:]
        return record


def _fields(value: object) -> Mapping[str, object]:
    return STR_KEYED_MAPPING.validate_python(value) if isinstance(value, Mapping) else _NO_FIELDS


def _client_error_status(value: object) -> int | None:
    return value if isinstance(value, int) and value in _CLIENT_ERROR_STATUSES else None


def _streamed_error_status(record: Mapping[str, object]) -> int:
    detail: Final = _fields(record.get("error"))
    candidates: Final = (
        detail.get("code"),
        detail.get("http_status_code"),
        record.get("code"),
        _CLIENT_ERROR_TYPE_STATUS.get(str(record.get("error_type"))),
    )
    return next(filter(None, map(_client_error_status, candidates)), 500)


def _failed_generation(record: Mapping[str, object]) -> "SagemakerError | None":
    error_message: Final = _fields(record.get("token")).get("error_msg")
    if not error_message and _fields(record.get("details")).get("finish_reason") != "error":
        return None
    return SagemakerError(
        status_code=500,
        message=f"SageMaker endpoint streamed a failed generation: {error_message or 'no error message'}",
    )


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


class _EventStreamMessage(Protocol):
    def to_response_dict(self) -> Mapping[str, object]: ...


class SagemakerError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class _StreamExceptionBody(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, hide_input_in_errors=True)
    message: str = Field(default="", alias="Message")
    error_code: str | None = Field(default=None, alias="ErrorCode")


def _stream_exception_error(headers: Mapping[str, object], body: object) -> SagemakerError:
    name: Final = headers.get(":exception-type") or headers.get(":error-code") or "an error"
    try:
        detail: Final = _StreamExceptionBody.model_validate_json(body if isinstance(body, bytes) else b"")
    except ValidationError:
        return SagemakerError(
            status_code=500, message=f"SageMaker stream failed with {name}: {headers.get(':error-message', '')}"
        )
    code: Final = f" ({detail.error_code})" if detail.error_code else ""
    return SagemakerError(status_code=500, message=f"SageMaker stream failed with {name}{code}: {detail.message}")


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

    def _record_to_chunk(self, record: object) -> GChunk | StreamingChatCompletionChunk | SagemakerError | None:
        fields: Final = _fields(record)
        if fields.get("error") is not None or fields.get("object") == "error":
            return SagemakerError(
                status_code=_streamed_error_status(fields),
                message=f"SageMaker endpoint streamed an error: {json.dumps(record)}",
            )
        if failed_generation := _failed_generation(fields):
            return failed_generation
        try:
            if self.is_messages_api:
                return self._chunk_parser_messages_api(
                    chunk_data=_SageMakerMessagesChunkDataRoot.model_validate(record).root
                )
            return self._chunk_parser(chunk_data=_SageMakerChunkDataRoot.model_validate(record).root)
        except ValidationError:
            verbose_logger.error(
                "Skipping a SageMaker stream record that is not a chunk: %s with keys %s",
                type(record).__name__,
                sorted(fields),
            )
            return None

    def _chunks(self, records: tuple[object, ...]) -> Iterator[GChunk | StreamingChatCompletionChunk]:
        for outcome in map(self._record_to_chunk, records):
            if isinstance(outcome, SagemakerError):
                raise outcome
            if outcome is not None:
                yield outcome

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

        unparsed: Final = records.close()
        if isinstance(unparsed, SagemakerError):
            raise unparsed
        if unparsed:
            verbose_logger.error("Warning: Unparseable JSON data remained, %d characters of it", len(unparsed))
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
                if payload := self._parse_message_from_event(event):
                    for parsed_chunk in self._chunks(records.feed(payload)):
                        yield parsed_chunk

        unparsed: Final = records.close()
        if isinstance(unparsed, SagemakerError):
            raise unparsed
        if unparsed:
            verbose_logger.error("Warning: Unparseable JSON data remained, %d characters of it", len(unparsed))
            yield None

    def _parse_message_from_event(self, event: _EventStreamMessage) -> bytes | None:
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
        body: Final = response_dict.get("body")

        if response_dict["status_code"] != 200:
            raise _stream_exception_error(_fields(response_dict.get("headers")), body)

        if "chunk" in parsed_response:
            chunk = parsed_response.get("chunk")
            if not chunk:
                return None
            return chunk.get("bytes")
        return body if isinstance(body, bytes) and body else None
