from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, Literal, Protocol

import httpx
from openai._streaming import SSEDecoder
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.exceptions import APIError, Timeout
from litellm.litellm_core_utils.thread_pool_executor import executor

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse


class _StreamLogging(Protocol):
    start_time: datetime

    async def dispatch_success_handlers(
        self, result: object, *, start_time: datetime, end_time: datetime, prefer_async_handlers: bool = False
    ) -> None: ...

    def failure_handler(
        self, exception: Exception, traceback_exception: str, start_time: datetime, end_time: datetime
    ) -> None: ...

    async def async_failure_handler(
        self, exception: Exception, traceback_exception: str, start_time: datetime, end_time: datetime
    ) -> None: ...


class SearchStreamEvent(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, hide_input_in_errors=True)
    type: str


class SearchTextDeltaEvent(SearchStreamEvent):
    delta: str


class SearchGroundingEvent(SearchStreamEvent):
    grounding: Sequence[Mapping[str, object]]


class SearchResultsEvent(SearchStreamEvent):
    results: Sequence[Mapping[str, object]]


class SearchResetEvent(SearchStreamEvent):
    streamReset: Literal[True]


class SearchDoneEvent(SearchStreamEvent):
    output: Mapping[str, object] | None
    searchTime: float


class SearchErrorEvent(SearchStreamEvent):
    error: Mapping[str, object]


_EVENT_MODELS: Final[Mapping[str, type[SearchStreamEvent]]] = {
    "text-delta": SearchTextDeltaEvent,
    "grounding": SearchGroundingEvent,
    "results": SearchResultsEvent,
    "stream-reset": SearchResetEvent,
    "done": SearchDoneEvent,
    "error": SearchErrorEvent,
}


class SearchStreamWrapper:
    def __init__(
        self,
        response: httpx.Response,
        provider_config: BaseSearchConfig,
        logging_obj: Logging,
        provider: str,
    ) -> None:
        self.response = response
        self.provider_config = provider_config
        self.logging_obj: _StreamLogging = logging_obj
        self._transform_logging_obj = logging_obj
        self.provider = provider
        self._hidden_params: Mapping[str, object] = {"additional_headers": dict(response.headers)}
        self._results: object = []
        self._request_metadata: Mapping[str, object] = {}
        self._finished = False
        self._logging_started = False
        self._success_task: asyncio.Task[None] | None = None
        self._iterator = self._iter_events()
        self._async_iterator = self._aiter_events()
        self.final_response: SearchResponse | None = None

    def __iter__(self) -> Iterator[SearchStreamEvent]:
        return self

    def __next__(self) -> SearchStreamEvent:
        return next(self._iterator)

    def __aiter__(self) -> AsyncIterator[SearchStreamEvent]:
        return self

    async def __anext__(self) -> SearchStreamEvent:
        return await anext(self._async_iterator)

    def _error(self, message: str, body: object | None = None) -> APIError:
        return APIError(
            status_code=502,
            message=message,
            llm_provider=self.provider,
            model=f"{self.provider}/search",
            request=self.response.request,
            body=body,
        )

    def _stream_error(self, error: Exception) -> Exception:
        if isinstance(error, APIError):
            return error
        if isinstance(error, httpx.TimeoutException):
            return Timeout(
                message="Search stream timed out", model=f"{self.provider}/search", llm_provider=self.provider
            )
        return self._error(f"Search stream failed: {type(error).__name__}")

    def _parse(self, data: str) -> SearchStreamEvent:
        if data == "[DONE]":
            raise self._error("Search stream ended without its done event")
        try:
            envelope: Final = SearchStreamEvent.model_validate_json(data)
            event: Final = _EVENT_MODELS.get(envelope.type, SearchStreamEvent).model_validate_json(data)
        except ValidationError as error:
            raise self._error("Malformed search stream event") from error
        payload: Final = TypeAdapter(Mapping[str, object]).validate_python(event.model_dump())
        if "requestId" in payload:
            self._request_metadata = {"requestId": payload["requestId"]}
        if event.type == "error":
            raise self._error(f"Search stream error: {payload.get('error')}", body=payload)
        if event.type == "results":
            self._results = payload.get("results", [])
        if event.type == "done":
            self._finished = True
            completed: Final = {key: value for key, value in payload.items() if key != "type"}
            self.final_response = self.provider_config.transform_search_response(
                httpx.Response(200, json={**self._request_metadata, "results": self._results, **completed}),
                self._transform_logging_obj,
            )
        return event

    def _iter_events(self) -> Generator[SearchStreamEvent, None, None]:
        try:
            for sse in SSEDecoder().iter_bytes(self.response.iter_bytes()):
                if not sse.data:
                    continue
                event = self._parse(sse.data)
                if self._finished:
                    self._log_success()
                    self.response.close()
                yield event
                if self._finished:
                    return
            raise self._error("Search stream disconnected before its done event")
        except GeneratorExit:
            self._log_success()
            raise
        except Exception as error:
            self._logging_started = True
            failure: Final = self._stream_error(error)
            self.logging_obj.failure_handler(
                failure, str(failure), self.logging_obj.start_time, datetime.now(timezone.utc).replace(tzinfo=None)
            )
            if failure is error:
                raise
            raise failure from error
        finally:
            self.response.close()

    async def _aiter_events(self) -> AsyncGenerator[SearchStreamEvent, None]:
        try:
            async for sse in SSEDecoder().aiter_bytes(self.response.aiter_bytes()):
                if not sse.data:
                    continue
                event = self._parse(sse.data)
                if self._finished:
                    await self._alog_success()
                    await self.response.aclose()
                yield event
                if self._finished:
                    return
            raise self._error("Search stream disconnected before its done event")
        except (GeneratorExit, asyncio.CancelledError):
            await self._alog_success()
            raise
        except Exception as error:
            self._logging_started = True
            failure: Final = self._stream_error(error)
            await self.logging_obj.async_failure_handler(
                failure, str(failure), self.logging_obj.start_time, datetime.now(timezone.utc).replace(tzinfo=None)
            )
            if failure is error:
                raise
            raise failure from error
        finally:
            await self.response.aclose()

    def _billing_response(self) -> SearchResponse:
        if self.final_response is not None:
            return self.final_response
        return self.provider_config.transform_search_response(
            httpx.Response(200, json={**self._request_metadata, "results": self._results}),
            self._transform_logging_obj,
        )

    async def _alog_success(self) -> None:
        if self._success_task is None:
            if self._logging_started:
                return
            self._logging_started = True
            self._success_task = asyncio.create_task(
                self.logging_obj.dispatch_success_handlers(
                    self._billing_response(),
                    start_time=self.logging_obj.start_time,
                    end_time=datetime.now(timezone.utc).replace(tzinfo=None),
                    prefer_async_handlers=True,
                )
            )
        await asyncio.shield(self._success_task)

    def _log_success(self) -> None:
        if self._logging_started:
            return
        self._logging_started = True
        end_time: Final = datetime.now(timezone.utc).replace(tzinfo=None)
        executor.submit(
            asyncio.run,
            self.logging_obj.dispatch_success_handlers(
                self._billing_response(), start_time=self.logging_obj.start_time, end_time=end_time
            ),
        )

    def get_final_response(self) -> SearchResponse:
        for _ in self:
            pass
        if self.final_response is None:
            raise self._error("Search stream has no final response")
        return self.final_response

    async def aget_final_response(self) -> SearchResponse:
        async for _ in self:
            pass
        if self.final_response is None:
            raise self._error("Search stream has no final response")
        return self.final_response

    def close(self) -> None:
        self._iterator.close()
        self.response.close()
        self._log_success()

    async def aclose(self) -> None:
        await self._async_iterator.aclose()
        await self.response.aclose()
        await self._alog_success()

    def __enter__(self) -> SearchStreamWrapper:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> SearchStreamWrapper:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
