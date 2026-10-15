from __future__ import annotations

import datetime
from collections.abc import AsyncIterator, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # Logging's litellm params are untyped

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.rust_bridge.execution import AsyncStream, Execution, SyncStream
from litellm.rust_bridge.host.callbacks_legacy_python import finalize
from litellm.types.utils import CallTypes

_HEAD: Final = TypeAdapter(Mapping[str, object])


class _CallLogger(Protocol):
    litellm_params: Mapping[str, object]
    model_call_details: MutableMapping[str, object]  # mutable-ok: the Logging object's own call details


@dataclass(frozen=True, slots=True)
class StreamSource:
    """What Python's Responses handler hands its stream iterator, for a stream the native route opened"""

    headers: Sequence[tuple[str, str]]
    model: str
    config: BaseResponsesAPIConfig
    logging_obj: Logging
    litellm_metadata: dict[str, object] | None  # mutable-ok: the iterator writes stream timing into it
    custom_llm_provider: str
    request_data: Mapping[str, object]
    call_kwargs: Mapping[str, object]


def _source(head: object) -> StreamSource:
    source: Final = _HEAD.validate_python(head).get("source")
    if not isinstance(source, StreamSource):
        raise TypeError("the native Responses stream head carries no stream source")
    return source


def _chunk(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"a native Responses stream chunk is bytes, got {type(value).__name__}")
    return value


class _AsyncNativeBytes(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncStream) -> None:
        self._stream: Final = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield _chunk(chunk)

    async def aclose(self) -> None:
        await self._stream.aclose()


class _SyncNativeBytes(httpx.SyncByteStream):
    def __init__(self, stream: SyncStream) -> None:
        self._stream: Final = stream

    def __iter__(self) -> Iterator[bytes]:
        return map(_chunk, self._stream)

    def close(self) -> None:
        self._stream.close()


def _response(source: StreamSource, stream: httpx.AsyncByteStream | httpx.SyncByteStream) -> httpx.Response:
    return httpx.Response(200, headers=list(source.headers), stream=stream)


def _open(source: StreamSource, stream: BaseResponsesAPIStreamingIterator, asynchronous: bool) -> None:
    """Python's route marks its logger async or sync, which picks the handlers the iterator's logging runs, and
    `@client` sets the stream's hidden params"""
    logger: Final = cast(_CallLogger, source.logging_obj)  # cast-ok: Logging's litellm params are untyped
    logger.litellm_params = {**logger.litellm_params, "aresponses": asynchronous}
    logger.model_call_details["litellm_params"] = logger.litellm_params
    opened_at: Final = datetime.datetime.now()  # noqa: DTZ005  # Logging start times are naive local datetimes
    finalize(stream, source.logging_obj, dict(source.call_kwargs), source.logging_obj.start_time, opened_at)


def open_async(execution: Execution, head: object) -> ResponsesAPIStreamingIterator:
    source: Final = _source(head)
    stream: Final = ResponsesAPIStreamingIterator(
        response=_response(source, _AsyncNativeBytes(AsyncStream(execution, head))),
        model=source.model,
        responses_api_provider_config=source.config,
        logging_obj=source.logging_obj,
        litellm_metadata=source.litellm_metadata,
        custom_llm_provider=source.custom_llm_provider,
        request_data=dict(source.request_data),
        call_type=CallTypes.responses.value,
    )
    _open(source, stream, asynchronous=True)
    return stream


def open_sync(execution: Execution, head: object) -> SyncResponsesAPIStreamingIterator:
    source: Final = _source(head)
    stream: Final = SyncResponsesAPIStreamingIterator(
        response=_response(source, _SyncNativeBytes(SyncStream(execution, head))),
        model=source.model,
        responses_api_provider_config=source.config,
        logging_obj=source.logging_obj,
        litellm_metadata=source.litellm_metadata,
        custom_llm_provider=source.custom_llm_provider,
        request_data=dict(source.request_data),
        call_type=CallTypes.responses.value,
    )
    _open(source, stream, asynchronous=False)
    return stream
