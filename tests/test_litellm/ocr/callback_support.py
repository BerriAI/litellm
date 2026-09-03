import asyncio
import contextvars
import inspect
import json
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, TypedDict

from typing_extensions import ReadOnly

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse

MODEL: Final = "mistral/mistral-ocr-4-1"
DOCUMENT: Final = {"type": "document_url", "document_url": "https://example.com/document.pdf"}
RESPONSE: Final = (
    '{"pages":[{"index":0,"markdown":"callback-test"}],"model":"mistral-ocr-4-1","usage_info":{"pages_processed":1}}'
)
REQUEST_CONTEXT: Final = contextvars.ContextVar("ocr_callback_test_context", default="unset")


@dataclass(frozen=True, slots=True)
class CallbackEvent:
    name: str
    model: object
    call_id: object
    original_response: object
    input: object
    additional_args: str
    response_type: str
    start_time: datetime | None
    end_time: datetime | None
    context: str
    native_provider_hook: bool


class CallbackRecorder(CustomLogger):
    def __init__(self, asynchronous: bool, raises: str = "", name: str = "recorder", expected_calls: int = 1) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger exposes untyped keyword arguments
        self.asynchronous = asynchronous
        self.name = name
        self.expected_calls = expected_calls
        self.raises = raises
        self.events: tuple[CallbackEvent, ...] = ()
        self.done = threading.Event()
        self.lock = threading.Lock()

    def record(
        self,
        name: str,
        kwargs: Mapping[str, object],
        response: object = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        event: Final = CallbackEvent(
            name,
            kwargs.get("model"),
            kwargs.get("litellm_call_id"),
            kwargs.get("original_response"),
            kwargs.get("input"),
            json.dumps(kwargs.get("additional_args"), sort_keys=True, default=str),
            type(response).__name__,
            start_time,
            end_time,
            REQUEST_CONTEXT.get(),
            any(frame.filename == "litellm_callbacks.py" for frame in inspect.stack(context=0)),
        )
        with self.lock:
            self.events += (event,)
            terminals: Final = tuple(
                item.name for item in self.events if "success" in item.name or "failure" in item.name
            )
            if len(terminals) >= self.expected_calls * (2 if self.asynchronous and "failure" in name else 1):
                self.done.set()
        if name == self.raises:
            raise ValueError("intentional observer failure")

    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        assert model == kwargs.get("model")
        assert isinstance(messages, list)
        self.record("pre", kwargs)

    def log_post_api_call(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime | None
    ) -> None:
        self.record("post", kwargs, response_obj, start_time, end_time)

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("success", kwargs, response_obj, start_time, end_time)

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("failure", kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("async_success", kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.record("async_failure", kwargs, response_obj, start_time, end_time)

    async def wait(self) -> tuple[CallbackEvent, ...]:
        assert await asyncio.to_thread(self.done.wait, 5), tuple(event.name for event in self.events)
        return self.events


class OcrUpstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, status: int, body: str, stall: bool = False) -> None:
        super().__init__(("127.0.0.1", 0), OcrHandler)
        self.status = status
        self.body = body.encode()
        self.stall = stall
        self.release = threading.Event()
        self.started = threading.Event()
        self.requests: tuple[tuple[str, str], ...] = ()
        self.lock = threading.Lock()

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"


class OcrHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        assert isinstance(self.server, OcrUpstream)
        body: Final = self.rfile.read(int(self.headers["content-length"])).decode()
        with self.server.lock:
            self.server.requests += ((self.path, body),)
        self.server.started.set()
        if self.server.stall:
            self.server.release.wait(5)
            return
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.server.body)))
        self.end_headers()
        self.wfile.write(self.server.body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def ocr_upstream(status: int = 200, body: str = RESPONSE, stall: bool = False) -> Generator[OcrUpstream, None, None]:
    with OcrUpstream(status, body, stall) as server:
        thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.release.set()
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def assert_provider_request(server: OcrUpstream) -> None:
    assert len(server.requests) == 1
    path, body = server.requests[0]
    assert path == "/v1/ocr"
    assert json.loads(body) == {"model": "mistral-ocr-4-1", "document": DOCUMENT}


class OcrArguments(TypedDict):
    model: ReadOnly[str]
    document: ReadOnly[dict[str, str]]
    api_key: ReadOnly[str]
    api_base: ReadOnly[str]
    callbacks: ReadOnly[list[CustomLogger]]
    rust: ReadOnly[bool]
    num_retries: ReadOnly[int]
    timeout: ReadOnly[float]


def verify_installed_package() -> None:
    root: Final = Path(litellm.__file__).resolve()
    assert "site-packages" in root.parts, f"SDK must come from the installed wheel: {root}"


async def call_ocr(arguments: OcrArguments, asynchronous: bool) -> tuple[OCRResponse | None, Exception | None]:
    try:
        response: Final = (
            await litellm.aocr(**arguments) if asynchronous else await asyncio.to_thread(litellm.ocr, **arguments)
        )
        return OCRResponse.model_validate(response), None
    except Exception as error:
        return None, error
