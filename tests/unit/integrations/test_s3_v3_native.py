import asyncio
import base64
import copy
import json
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch
from urllib.parse import parse_qs

import pytest
from pydantic import BaseModel

import litellm
from litellm.integrations import s3_v3
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.host import request_scope
from litellm.integrations.custom_logger_v2.legacy import KwargsFields
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.selection import RequestSelection
from litellm.integrations.s3_v3 import S3Logger
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS, V2_ASYNC_LOGGING_LINE
from litellm.llms.bedrock import base_aws_llm
from litellm.types.integrations.custom_logger_v2 import Outcome

COMPARED_HEADERS: Final = (
    "content-type",
    "content-md5",
    "x-amz-content-sha256",
    "content-language",
    "content-disposition",
    "cache-control",
    "x-amz-server-side-encryption",
    "x-amz-server-side-encryption-aws-kms-key-id",
    "x-amz-security-token",
)
START: Final = datetime(2026, 11, 13, 9, 5, 7, 42)
GOLDEN_FILE: Final = Path(__file__).parent / "s3_v2_golden" / "python_s3_v2.json"


@dataclass(frozen=True)
class Golden:
    test: str
    values: Mapping[str, object]
    consumed: set[str] = field(default_factory=set)  # mutable-ok: the labels this test read

    def expected(self, label: str) -> object:
        key: Final = f"{self.test}::{label}"
        self.consumed.add(key)
        return self.values[key]

    def unconsumed(self) -> list[str]:
        return sorted(key for key in self.values if key.startswith(f"{self.test}::") and key not in self.consumed)


@pytest.fixture(scope="session")
def _golden_values() -> Mapping[str, object]:
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


@pytest.fixture
def golden(request: pytest.FixtureRequest, _golden_values: Mapping[str, object]) -> Iterator[Golden]:
    recorded: Final = Golden(request.node.name, _golden_values)
    yield recorded
    assert recorded.unconsumed() == []


@dataclass
class Received:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes


@dataclass
class FakeBucket:
    received: list[Received] = field(default_factory=list)
    objects: dict[str, bytes] = field(default_factory=dict)
    failures: list[tuple[str, int, str]] = field(default_factory=list)
    delays: list[tuple[str, float]] = field(default_factory=list)
    port: int = 0
    in_flight: int = 0
    peak: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def puts(self) -> list[Received]:
        return [request for request in self.received if request.method == "PUT"]


def _handler(bucket: FakeBucket) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            return

        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with bucket.lock:
                bucket.received.append(
                    Received("PUT", self.path, {k.lower(): v for k, v in self.headers.items()}, body)
                )
                bucket.in_flight += 1
                bucket.peak = max(bucket.peak, bucket.in_flight)
            try:
                self._answer_put(body)
            finally:
                with bucket.lock:
                    bucket.in_flight -= 1

        def _answer_put(self, body: bytes) -> None:
            delay: Final = next((d[1] for d in bucket.delays if re.search(d[0], self.path)), 0.0)
            if delay:
                time.sleep(delay)
            failure: Final = next((f for f in bucket.failures if re.search(f[0], self.path)), None)
            if failure is not None and failure[1] == 0:
                self.close_connection = True
                return
            if failure is not None:
                self._reply(failure[1], failure[2].encode())
                return
            bucket.objects[self.path] = body
            self._reply(200, b"")

        def do_GET(self) -> None:
            bucket.received.append(Received("GET", self.path, {k.lower(): v for k, v in self.headers.items()}, b""))
            stored: Final = bucket.objects.get(self.path)
            self._reply(404, b"<Error><Code>NoSuchKey</Code></Error>") if stored is None else self._reply(200, stored)

    return Handler


@pytest.fixture
def bucket() -> Iterator[FakeBucket]:
    fake: Final = FakeBucket()
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    fake.port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake
    server.shutdown()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("S3_PARTITION_GRANULARITY", raising=False)
    monkeypatch.delenv("S3_LOG_PROMPTS_ONLY", raising=False)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", None)
    yield


def _params(endpoint: str, extra: Mapping[str, object]) -> dict[str, object]:
    return {
        "s3_bucket_name": "logs",
        "s3_region_name": "us-east-1",
        "s3_endpoint_url": endpoint,
        "s3_aws_access_key_id": "AKIDEXAMPLE",
        "s3_aws_secret_access_key": "secret",
        "s3_aws_session_token": "session-token",
        **extra,
    }


async def _logger(monkeypatch: pytest.MonkeyPatch, bucket: FakeBucket, extra: Mapping[str, object]) -> S3Logger:
    monkeypatch.setattr(litellm, "s3_callback_params", _params(bucket.url, extra))
    return S3Logger()


def _attempt_ended_view(kwargs: Mapping[str, object], start_time: datetime) -> FieldView:
    args: Final = HookArgs(request=kwargs, start_time=start_time, end_time=start_time, outcome=Outcome.success)
    return FieldView(with_sources(KwargsFields(kwargs, LOGGING_KEYS), DERIVED_ATTEMPT_FIELDS, args))


async def _log_native(logger: S3Logger, kwargs: Mapping[str, object], start_time: datetime = START) -> None:
    line: Final = V2_ASYNC_LOGGING_LINE.set(True)
    try:
        await logger.on_attempt_ended(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # generated from the native manifest with its declared reads
            _attempt_ended_view(kwargs, start_time),
            SimpleNamespace(call=None, state=None),
        )
    finally:
        V2_ASYNC_LOGGING_LINE.reset(line)


async def _log(logger: S3Logger, payload: object) -> None:
    await _log_native(logger, {"standard_logging_object": copy.deepcopy(payload)})


def _signed_headers(request: Received) -> str:
    match: Final = re.search(r"SignedHeaders=([^,]+)", request.headers["authorization"])
    assert match is not None
    return match.group(1)


def _botocore_authorization(
    request: Received,
    port: int,
    credentials: tuple[str, str, str] = ("AKIDEXAMPLE", "secret", "session-token"),
    region: str = "us-east-1",
) -> str:
    from botocore.auth import S3SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    signed: Final = _signed_headers(request).split(";")
    unsigned: Final = {
        name: value
        for name, value in request.headers.items()
        if name in signed and name not in ("host", "x-amz-date", "x-amz-security-token")
    }
    aws_request: Final = AWSRequest(
        method=request.method, url=f"http://127.0.0.1:{port}{request.path}", data=request.body, headers=unsigned
    )
    at: Final = datetime.strptime(request.headers["x-amz-date"], "%Y%m%dT%H%M%SZ")
    with patch("botocore.auth.get_current_datetime", return_value=at):
        S3SigV4Auth(Credentials(*credentials), "s3", region).add_auth(aws_request)
    return aws_request.headers["Authorization"]


def _json_differences(python_value: object, rust_value: object, path: str = "$") -> list[str]:
    if isinstance(python_value, dict) and isinstance(rust_value, dict):
        return [
            difference
            for key in dict.fromkeys([*python_value, *rust_value])
            for difference in _json_differences(
                python_value.get(key, "<missing>"), rust_value.get(key, "<missing>"), f"{path}.{key}"
            )
        ]
    if isinstance(python_value, list) and isinstance(rust_value, list) and len(python_value) == len(rust_value):
        return [
            difference
            for index, (python_item, rust_item) in enumerate(zip(python_value, rust_value, strict=True))
            for difference in _json_differences(python_item, rust_item, f"{path}[{index}]")
        ]
    return [] if python_value == rust_value else [f"{path}: python {python_value!r} != rust {rust_value!r}"]


def _as_text(body: bytes) -> object:
    return json.loads(body, parse_float=str, parse_int=str, parse_constant=str)


def _body_differences(python_body: bytes, rust_body: bytes) -> str:
    try:
        differences: Final = _json_differences(_as_text(python_body), _as_text(rust_body))
    except ValueError:
        return "the bodies are not both JSON"
    at: Final = next(
        (index for index, pair in enumerate(zip(python_body, rust_body)) if pair[0] != pair[1]),
        min(len(python_body), len(rust_body)),
    )
    around: Final = slice(max(at - 40, 0), at + 40)
    return "; ".join(differences) or f"at byte {at}: python {python_body[around]!r} != rust {rust_body[around]!r}"


def _body(record: Mapping[str, object]) -> bytes:
    if "body_base64" in record:
        return base64.b64decode(str(record["body_base64"]))
    return str(record["body"]).encode("utf-8")


def _expected_puts(golden: Golden) -> list[dict[str, object]]:
    recorded: Final = golden.expected("puts")
    assert isinstance(recorded, list)
    return recorded


LIVE_BODY_FIELDS: Final = (
    "id",
    "litellm_call_id",
    "trace_id",
    "startTime",
    "endTime",
    "completionStartTime",
    "response_time",
)
LIVE_RESPONSE_FIELDS: Final = ("id", "created")
LIVE_HEADERS: Final = ("content-md5", "x-amz-content-sha256", "content-disposition")
LIVE_KEY: Final = re.compile(r"\d{4}-\d\d-\d\d/time-\d\d-\d\d-\d\d-\d{6}_[^/]+\.json$")
TRACEBACK_FRAME: Final = re.compile(r'^  File "(?P<path>[^"]+)", line \d+, in (?P<function>.+)$')
ENVIRONMENT_ROOTS: Final = (
    (re.compile(r"^.*/litellm/"), "litellm/"),
    (re.compile(r"^.*/site-packages/"), "<site-packages>/"),
    (re.compile(r"^.*/lib/python3\.\d+/"), "<stdlib>/"),
)


def _portable_path(path: str) -> str:
    return next((root.sub(name, path) for root, name in ENVIRONMENT_ROOTS if root.match(path)), path)


def _portable_traceback_line(line: str) -> str | None:
    frame: Final = TRACEBACK_FRAME.match(line)
    if frame is not None:
        return f"{_portable_path(frame['path'])} in {frame['function']}"
    return None if line.startswith(" ") else line


def _portable_traceback(text: str) -> str:
    return "\n".join(line for line in map(_portable_traceback_line, text.splitlines()) if line is not None)


def _with_portable_traceback(body: object) -> object:
    if not isinstance(body, dict):
        return body
    error: Final = body.get("error_information")
    if not isinstance(error, dict) or not isinstance(error.get("traceback"), str):
        return body
    return {**body, "error_information": {**error, "traceback": _portable_traceback(error["traceback"])}}


def _with_portable_put(put: object) -> object:
    if not isinstance(put, dict):
        return put
    return {**put, "body_json": _with_portable_traceback(put.get("body_json"))}


def _live_record(request: Received) -> dict[str, object]:
    body: Final = json.loads(request.body)
    assert isinstance(body, dict)
    for name in LIVE_BODY_FIELDS:
        if name in body:
            body[name] = "<live>"
    response: Final = body.get("response")
    if isinstance(response, dict):
        for name in LIVE_RESPONSE_FIELDS:
            if name in response:
                response[name] = "<live>"
    return {
        "path": LIVE_KEY.sub("<date>/time-<clock>_<id>.json", request.path),
        "headers": {name: "<live>" if name in LIVE_HEADERS else request.headers.get(name) for name in COMPARED_HEADERS},
        "signed_headers": _signed_headers(request),
        "body_keys": list(body),
        "body_json": _with_portable_traceback(body),
    }


def _assert_same_live_puts(golden: Golden, bucket: FakeBucket, count: int) -> None:
    python_puts: Final = golden.expected("live-puts")
    rust_puts: Final = [_live_record(put) for put in bucket.puts()]
    assert isinstance(python_puts, list)
    assert len(python_puts) == len(rust_puts) == count
    for python_put, rust_put in zip(python_puts, rust_puts, strict=True):
        assert rust_put == _with_portable_put(python_put)
    for put in bucket.puts():
        assert put.headers["authorization"] == _botocore_authorization(put, bucket.port)


def _assert_same_puts(golden: Golden, bucket: FakeBucket, count: int) -> None:
    python_puts: Final = _expected_puts(golden)
    rust_puts: Final = sorted(bucket.puts(), key=lambda r: r.path)
    assert len(python_puts) == len(rust_puts) == count
    for python_put, rust_put in zip(python_puts, rust_puts, strict=True):
        assert rust_put.path == python_put["path"]
        assert rust_put.body == _body(python_put), _body_differences(_body(python_put), rust_put.body)
        for name in COMPARED_HEADERS:
            assert rust_put.headers.get(name) == python_put["headers"][name], name
        assert _signed_headers(rust_put) == python_put["signed_headers"]
        assert rust_put.headers["authorization"] == _botocore_authorization(rust_put, bucket.port)


class _Model(BaseModel):
    name: str
    when: datetime


class _Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no text")


class _Opaque:
    def __str__(self) -> str:
        return "opaque\x00object"


def _payload(response_id: str = "chatcmpl-1", **overrides: object) -> dict[str, object]:
    return {
        "id": response_id,
        "call_type": "acompletion",
        "startTime": START.timestamp(),
        "endTime": START.timestamp() + 1.25,
        "response_cost": 0.000123,
        "metadata": {
            "user_api_key_team_alias": "team-α",
            "user_api_key_alias": "key/one",
            "user_api_key_hash": "88dc28",
        },
        "messages": [{"role": "user", "content": "hello «world» 😀"}],
        "response": {"choices": [{"message": {"content": "hi"}}]},
        **overrides,
    }


def _odd_payload() -> dict[str, object]:
    looped: dict[str, object] = {"name": "loop"}
    looped["self"] = looped
    deep: object = "bottom"
    for _ in range(120):
        deep = [deep]
    moment: Final = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    amount: Final = Decimal("1.50")
    return _payload(
        "odd",
        repeated={"first": moment, "second": moment, "amounts": [amount, amount]},
        numbers=(1, 2.5, 1e16, 1e-5, -0.0, float("nan"), float("inf"), 10**30, True),
        members={3, 1, 2},
        model=_Model(name="m", when=START),
        opaque=_Opaque(),
        nul="a\x00b",
        looped=looped,
        deep=deep,
        keys={"a\x00": 1, "a": 2, 3: "dropped"},
    )


PARAM_GRID: Final = [
    pytest.param({}, id="defaults"),
    pytest.param({"s3_partition_granularity": "hour", "s3_path": "base/logs/"}, id="hourly-path"),
    pytest.param({"s3_use_team_prefix": True, "s3_use_key_prefix": "false"}, id="team-and-key-prefix"),
    pytest.param({"s3_log_prompts_only": "yes"}, id="prompts-only"),
    pytest.param({"s3_sse_kms_key_id": "kms-1"}, id="sse-kms"),
    pytest.param({"s3_server_side_encryption": "AES256", "s3_sse_kms_key_id": "k"}, id="sse-aes-drops-key"),
    pytest.param({"s3_strip_base64_files": True}, id="strip-base64"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", PARAM_GRID)
async def test_both_loggers_put_the_same_objects(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, extra: Mapping[str, object]
) -> None:
    logger = await _logger(monkeypatch, bucket, extra)
    payloads: Final = [
        _payload(),
        _payload("x" * 400),
        _payload("cmpl:with/odd chars=+&#?%"),
        _payload(
            "with-file",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see data:image/png;base64,AAAA== inline"},
                        {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBER"}},
                    ],
                },
                "not a dict",
            ],
        ),
        _odd_payload(),
    ]
    for payload in payloads:
        await _log(logger, payload)
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, len(payloads))


@pytest.mark.asyncio
async def test_a_missing_payload_logs_nothing(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    await _log(logger, None)
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_payload(metadata=None), id="metadata-none"),
        pytest.param(_payload(metadata={"user_api_key_team_alias": 7}), id="alias-not-text"),
        pytest.param({k: v for k, v in _payload().items() if k != "id"}, id="no-id"),
        pytest.param(_payload(id=5), id="id-not-text"),
        pytest.param(_payload(metadata={"user_api_key_team_alias": ["team-a"]}), id="alias-a-list"),
        pytest.param(_payload(metadata={"user_api_key_team_alias": {"team": "a"}}), id="alias-a-dict"),
        pytest.param(_payload(metadata={"user_api_key_team_alias": 10**40}), id="alias-past-i128"),
        pytest.param(_payload(metadata={"user_api_key_team_alias": _Unprintable()}), id="alias-whose-str-raises"),
        pytest.param(["not", "a", "mapping"], id="payload-not-a-mapping"),
    ],
)
async def test_both_loggers_skip_and_count_the_same_bad_payloads(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_use_team_prefix": True})
    with patch.object(s3_v3._Host, "callback_failed") as rust_failures:
        await _log(logger, payload)
        await logger.flush_queue()
        await asyncio.sleep(0.2)
    assert _expected_puts(golden) == bucket.puts() == []
    assert golden.expected("queue") == []
    assert logger.sink.queued_keys() == []
    assert golden.expected("failures") == rust_failures.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("granularity", ["day", "hour"])
async def test_a_cold_storage_key_is_reused_when_it_matches(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, granularity: str
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_path": "cold"})
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    date: Final = "2026-11-13/" if granularity == "day" else "2026-11-13/09/"
    matching: Final = f"cold/{date}time-09-05-07-000042_chatcmpl-1.json"
    for response_id, key in (("chatcmpl-1", matching), ("chatcmpl-2", "elsewhere/x.json")):
        await _log(logger, _payload(response_id, metadata={"cold_storage_object_key": key}))
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 2)
    assert {r.path for r in bucket.puts()} == {
        f"/logs/{matching}",
        "/logs/cold/2026-11-13/time-09-05-07-000042_chatcmpl-2.json",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "extra"),
    [
        pytest.param([("_a\\.json", 503, "")], {}, id="partial-503-keeps-the-failed-one"),
        pytest.param([(".", 500, "")], {}, id="bucket-down-keeps-everything"),
        pytest.param([("_a\\.json", 400, "<Error><Code>EntityTooLarge</Code></Error>")], {}, id="terminal-is-dropped"),
        pytest.param(
            [("_a\\.json", 400, "<Error><Code>EntityTooLarge</Code></Error>")],
            {"s3_drop_on_terminal_error": "false"},
            id="terminal-kept-when-dropping-is-off",
        ),
        pytest.param([(".", 503, "")], {"s3_max_queue_size": 2}, id="queue-cap-trims-oldest"),
        pytest.param([("_a\\.json", 403, "<Error><Code>AccessDenied</Code></Error>")], {}, id="403-retried"),
    ],
)
async def test_both_loggers_keep_and_drop_the_same_failed_uploads(
    golden: Golden,
    bucket: FakeBucket,
    monkeypatch: pytest.MonkeyPatch,
    failures: list[tuple[str, int, str]],
    extra: Mapping[str, object],
) -> None:
    bucket.failures.extend(failures)
    logger = await _logger(monkeypatch, bucket, extra)
    for response_id in ("a", "b", "c"):
        await _log(logger, _payload(response_id))
    await logger.flush_queue()
    python_queue: Final = golden.expected("queue")
    assert logger.sink.queued_keys() == python_queue
    assert len(bucket.puts()) == golden.expected("put-count")
    bucket.failures.clear()
    await logger.flush_queue()
    assert logger.sink.queued_keys() == golden.expected("queue-after-recovery") == []
    assert sorted(r.path for r in bucket.puts()) == golden.expected("put-paths")


@pytest.mark.asyncio
async def test_batch_files_hold_the_same_lines(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_batch_file_upload": True, "s3_use_team_prefix": True})
    for response_id, team in (("a", "t1"), ("b", "t2"), ("c", "t1")):
        await _log(logger, _payload(response_id, metadata={"user_api_key_team_alias": team}))
    await logger.flush_queue()
    name: Final = re.compile(r"/batch_\d\d-\d\d-\d\d_[0-9a-f]{32}\.jsonl$")
    python_puts: Final = golden.expected("puts")
    rust_puts: Final = bucket.puts()
    assert isinstance(python_puts, list)
    assert len(python_puts) == len(rust_puts) == 2
    for python_put, rust_put in zip(python_puts, sorted(rust_puts, key=lambda r: r.path), strict=True):
        assert name.sub("", rust_put.path) == python_put["path"]
        assert rust_put.body == _body(python_put), _body_differences(_body(python_put), rust_put.body)
        assert rust_put.headers["content-type"] == python_put["headers"]["content-type"] == "application/x-ndjson"


@pytest.mark.asyncio
async def test_audit_events_and_line_items_land_at_the_same_keys(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_path": "audit-base"})
    audit: Final = {"id": "audit-1", "action": "created", "before_value": None, "updated_at": "2026-11-13"}
    await logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # a partial audit payload is enough here
    await logger.flush_queue()
    line_items: Final = {
        "batch_id": "batch/1",
        "part": 3,
        "finished_at": START.timestamp(),
        "lines": [{"custom_id": "x", "status": "ok", "text": "ünïcode"}],
    }
    await logger.async_log_batch_line_items(line_items)  # pyright: ignore[reportArgumentType]  # a partial line items payload is enough here
    clock: Final = re.compile(r"/\d\d-\d\d-\d\d_")
    date: Final = re.compile(r"/\d{4}-\d\d-\d\d/")
    python_puts: Final = golden.expected("puts")
    rust_puts: Final = bucket.puts()
    assert isinstance(python_puts, list)
    assert len(python_puts) == len(rust_puts) == 2
    for python_put, rust_put in zip(python_puts, rust_puts, strict=True):
        assert date.sub("/", clock.sub("/", rust_put.path)) == python_put["path"]
        assert rust_put.body == _body(python_put), _body_differences(_body(python_put), rust_put.body)
        for name in COMPARED_HEADERS:
            assert rust_put.headers.get(name) == python_put["headers"][name], name


@pytest.mark.asyncio
async def test_cold_storage_reads_return_the_same_object(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    await _log(logger, _payload())
    await logger.flush_queue()
    key: Final = "2026-11-13/time-09-05-07-000042_chatcmpl-1.json"
    python_object: Final = golden.expected("object")
    rust_object: Final = await logger.get_proxy_server_request_from_cold_storage_with_object_key(key)
    assert python_object is not None
    assert rust_object == python_object
    assert await logger.get_proxy_server_request_from_cold_storage_with_object_key("missing.json") is None
    python_get: Final = golden.expected("get")
    assert isinstance(python_get, dict)
    rust_get: Final = bucket.received[-1]
    assert rust_get.headers["x-amz-content-sha256"] == python_get["headers"]["x-amz-content-sha256"]
    assert _signed_headers(rust_get) == python_get["signed_headers"]


@pytest.mark.asyncio
async def test_the_native_logger_answers_the_python_helpers(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_path": "p", "s3_partition_granularity": "hour"})
    assert logger.s3_path == golden.expected("s3_path") == "p"
    assert logger.resolve_partition_granularity() == golden.expected("granularity") == "hour"
    selection: Final = type(logger).selection
    assert selection is not None
    assert selection.name == "s3_v2"
    assert json.loads(logger.sink.callback().manifest())["subscriptions"][0]["hook"] == "attempt_ended"


@pytest.mark.asyncio
async def test_stripping_base64_rewrites_the_shared_payload_the_same_way(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_strip_base64_files": True})
    original: Final = _payload(
        "with-file",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see data:image/png;base64,AAAA== inline"},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBER"}},
                ],
            }
        ],
    )
    rust_payload: Final = copy.deepcopy(original)
    await _log_native(logger, {"standard_logging_object": rust_payload})
    assert rust_payload == golden.expected("payload") != original
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 1)


@pytest.mark.asyncio
async def test_a_payload_changed_before_the_flush_is_stored_as_it_is_at_the_flush(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    payload: Final = _payload()
    await _log_native(logger, {"standard_logging_object": payload})
    payload["metadata"]["late"] = "written after the event"  # pyright: ignore[reportIndexIssue]  # metadata is the dict _payload builds
    payload["response_cost"] = 99.0
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 1)
    stored: Final = json.loads(bucket.puts()[0].body)
    assert stored["metadata"]["late"] == "written after the event"
    assert stored["response_cost"] == 0.000123
    assert logger.sink.stashed() == 0


def test_both_loggers_need_a_running_event_loop(golden: Golden, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "s3_callback_params", _params("http://127.0.0.1:9", {}))
    with pytest.raises(RuntimeError, match="no running event loop") as raised:
        S3Logger()
    assert str(raised.value) == golden.expected("error")


@pytest.mark.asyncio
async def test_the_native_logger_flushes_on_its_own_after_the_interval(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s3_v3, "DEFAULT_S3_FLUSH_INTERVAL_SECONDS", 0.05)
    logger = await _logger(monkeypatch, bucket, {})
    await _log_native(logger, {"standard_logging_object": _payload()})
    for _ in range(100):
        if bucket.puts():
            break
        await asyncio.sleep(0.02)
    assert [r.path for r in bucket.puts()] == ["/logs/2026-11-13/time-09-05-07-000042_chatcmpl-1.json"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment", "region"),
    [
        pytest.param({"AWS_REGION_NAME": "eu-central-1"}, "eu-central-1", id="region-name"),
        pytest.param({"AWS_REGION_NAME": "eu-central-1", "AWS_REGION": "ap-south-1"}, "ap-south-1", id="region-wins"),
        pytest.param({}, "us-west-2", id="default"),
        pytest.param({"AWS_REGION": "Bad_Region"}, None, id="invalid"),
    ],
)
async def test_without_a_region_param_both_sign_with_the_same_environment_region(
    golden: Golden,
    bucket: FakeBucket,
    monkeypatch: pytest.MonkeyPatch,
    environment: Mapping[str, str],
    region: str | None,
) -> None:
    for name in ("AWS_REGION_NAME", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    logger = await _logger(monkeypatch, bucket, {"s3_region_name": None})
    with patch.object(s3_v3._Host, "callback_failed") as rust_failures:
        await _log(logger, _payload())
        await logger.flush_queue()
        await asyncio.sleep(0.2)
    python_puts: Final = golden.expected("put-count")
    python_failure_count: Final = golden.expected("failures")
    if region is None:
        assert python_puts == len(bucket.puts()) == 0
        assert python_failure_count == rust_failures.call_count == 1
        return
    assert python_puts == len(bucket.puts()) == 1
    put: Final = bucket.puts()[0]
    assert put.headers["authorization"] == _botocore_authorization(put, bucket.port, region=region)


@dataclass
class FakeSts:
    calls: list[Mapping[str, list[str]]] = field(default_factory=list)
    port: int = 0


STS_REPLY: Final = (
    '<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">'
    "<AssumeRoleWithWebIdentityResult><Credentials><AccessKeyId>AKIDSTS</AccessKeyId>"
    "<SecretAccessKey>sts-secret</SecretAccessKey><SessionToken>sts-token</SessionToken>"
    "<Expiration>2099-01-01T00:00:00Z</Expiration></Credentials>"
    "<SubjectFromWebIdentityToken>subject</SubjectFromWebIdentityToken>"
    "<AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/logs/s3</Arn>"
    "<AssumedRoleId>AROA:s3</AssumedRoleId></AssumedRoleUser><PackedPolicySize>10</PackedPolicySize>"
    "</AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>"
)


@pytest.fixture
def sts() -> Iterator[FakeSts]:
    fake: Final = FakeSts()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            return

        def do_POST(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            fake.calls.append(parse_qs(body.decode()))
            reply: Final = STS_REPLY.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/xml")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake
    server.shutdown()


def _sts_session(sts: FakeSts, values: object) -> object:
    if not isinstance(values, list):
        return values
    return [str(value).replace(f"s3-{sts.port}", "s3-<sts-port>") for value in values]


@pytest.mark.asyncio
async def test_web_identity_assumes_the_role_with_the_same_session_policy(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, sts: FakeSts
) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_EXTERNAL_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("S3_V3_TEST_OIDC_TOKEN", "oidc-token-value")
    identity: Final = {
        "s3_aws_access_key_id": None,
        "s3_aws_secret_access_key": None,
        "s3_aws_session_token": None,
        "s3_aws_web_identity_token": "oidc/env/S3_V3_TEST_OIDC_TOKEN",
        "s3_aws_role_name": "arn:aws:iam::123456789012:role/logs",
        "s3_aws_session_name": f"s3-{sts.port}",
        "s3_aws_sts_endpoint": f"http://127.0.0.1:{sts.port}",
    }
    logger = await _logger(monkeypatch, bucket, identity)
    await _log(logger, _payload())
    await logger.flush_queue()
    compared: Final = ("Action", "RoleArn", "RoleSessionName", "WebIdentityToken", "DurationSeconds", "Policy")
    assert len(sts.calls) == 1
    python_call: Final = golden.expected("sts-call")
    assert isinstance(python_call, dict)
    rust_call: Final = sts.calls[-1]
    for name in compared:
        assert _sts_session(sts, rust_call.get(name)) == python_call[name], name
    assert rust_call["WebIdentityToken"] == ["oidc-token-value"]
    assert json.loads(rust_call["Policy"][0]) == json.loads(str(golden.expected("policy")))
    python_put: Final = golden.expected("put")
    assert isinstance(python_put, dict)
    assert python_put["headers"]["x-amz-security-token"] == "sts-token"
    put: Final = bucket.puts()[0]
    assert put.headers["x-amz-security-token"] == "sts-token"
    assert put.headers["authorization"] == _botocore_authorization(
        put, bucket.port, credentials=("AKIDSTS", "sts-secret", "sts-token")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [True, 7, "abc", ""])
async def test_a_web_identity_token_botocore_rejects_fails_both_before_sts(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, sts: FakeSts, token: object
) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_EXTERNAL_ID"):
        monkeypatch.delenv(name, raising=False)
    reference: Final = "s3-v3-test/oidc-token"

    def secret(name: str, default_value: object = None) -> object:
        return token if name == reference else default_value

    monkeypatch.setattr(base_aws_llm, "get_secret", secret)
    monkeypatch.setattr(s3_v3, "get_secret", secret)
    identity: Final = {
        "s3_aws_access_key_id": None,
        "s3_aws_secret_access_key": None,
        "s3_aws_session_token": None,
        "s3_aws_web_identity_token": reference,
        "s3_aws_role_name": "arn:aws:iam::123456789012:role/logs",
        "s3_aws_session_name": f"s3-rejected-{sts.port}",
        "s3_aws_sts_endpoint": f"http://127.0.0.1:{sts.port}",
    }
    logger = await _logger(monkeypatch, bucket, identity)
    with patch.object(s3_v3._Host, "callback_failed") as rust_failures:
        await _log(logger, _payload())
        await logger.flush_queue()
    assert sts.calls == []
    assert _expected_puts(golden) == bucket.puts() == []
    assert golden.expected("failures") == rust_failures.call_count
    assert rust_failures.call_count == 1
    assert logger.sink.queued_keys() == golden.expected("queue")
    python_error: Final = golden.expected("error")
    with pytest.raises(ValueError, match=r"^Parameter validation failed:") as rust_error:
        s3_v3._Host().web_identity_token(reference)
    assert str(rust_error.value) == python_error


@pytest.mark.asyncio
@pytest.mark.parametrize("global_disable", [False, True])
@pytest.mark.parametrize("no_log", [None, True, "true"])
@pytest.mark.parametrize("disabled_header", [None, "s3_v2", "langfuse, S3_V2"])
async def test_the_native_opt_out_matches_should_run_callback(
    golden: Golden,
    bucket: FakeBucket,
    monkeypatch: pytest.MonkeyPatch,
    global_disable: bool,
    no_log: object,
    disabled_header: str | None,
) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.proxy import proxy_server

    monkeypatch.setattr(litellm, "global_disable_no_log_param", global_disable)
    monkeypatch.setattr(litellm, "allow_dynamic_callback_disabling", True)
    monkeypatch.setattr(proxy_server, "premium_user", True)
    logger = await _logger(monkeypatch, bucket, {})
    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="call-1",
        start_time=START,
        function_id="fn-1",
    )
    litellm_params: Final = {
        **({} if no_log is None else {"no-log": no_log}),
        "proxy_server_request": {
            "headers": {} if disabled_header is None else {"x-litellm-disable-callbacks": disabled_header}
        },
    }
    expected: Final = golden.expected("should_run_callback")
    assert isinstance(expected, bool)
    kwargs: Final = {
        "litellm_params": litellm_params,
        "standard_callback_dynamic_params": logging_obj.standard_callback_dynamic_params,
        "standard_logging_object": _payload(),
    }
    assert s3_v3.legacy_should_log(kwargs) is expected
    await _log_native(logger, kwargs)
    await logger.flush_queue()
    assert len(bucket.puts()) == int(expected)


async def _until_queued(logger: S3Logger) -> None:
    for _ in range(100):
        if logger.sink.queued_keys():
            return
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_scope", [False, True])
@pytest.mark.parametrize("global_disable", [False, True])
@pytest.mark.parametrize("no_log", [False, True])
async def test_an_acompletion_reaches_both_loggers_alike(
    golden: Golden,
    bucket: FakeBucket,
    monkeypatch: pytest.MonkeyPatch,
    global_disable: bool,
    no_log: bool,
    proxy_scope: bool,
) -> None:
    monkeypatch.setattr(litellm, "global_disable_no_log_param", global_disable)
    logger = await _logger(monkeypatch, bucket, {"s3_use_team_prefix": True})
    monkeypatch.setattr(litellm, "callbacks", [logger])
    with request_scope(RequestSelection(no_log=no_log and not global_disable)) if proxy_scope else nullcontext():
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello «world»"}],
            mock_response="hi there",
            metadata={"user_api_key_team_alias": "team-a"},
            **({"no-log": True} if no_log else {}),
        )
    await _until_queued(logger)
    await logger.flush_queue()
    expected: Final = 0 if no_log and not global_disable else 1
    _assert_same_live_puts(golden, bucket, expected)
    assert all(put.path.startswith("/logs/team-a/") for put in bucket.puts())


class _Mutating:
    def __init__(self, target: dict[str, object]) -> None:
        self.target: Final = target

    def __str__(self) -> str:
        self.target[f"grown-{len(self.target)}"] = True
        return "mutating"


@pytest.mark.asyncio
async def test_a_payload_that_changes_while_it_is_rendered_is_kept_for_retry_by_both_loggers(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    racing: Final = _payload("racing")
    metadata: Final = racing["metadata"]
    assert isinstance(metadata, dict)
    metadata["mutating"] = _Mutating(metadata)  # pyright: ignore[reportUnknownArgumentType]  # the metadata dict _payload builds
    with patch.object(s3_v3._Host, "callback_failed") as rust_failures:
        await _log_native(logger, {"standard_logging_object": racing})
        await _log(logger, _payload("steady"))
        await logger.flush_queue()
        await asyncio.sleep(0.2)
    assert logger.sink.queued_keys() == golden.expected("queue")
    assert len(logger.sink.queued_keys()) == 1
    assert golden.expected("failures") == rust_failures.call_count
    assert rust_failures.call_count == 1
    _assert_same_puts(golden, bucket, 1)
    assert not logger.periodic_flush_task.done()


class _FlakySink:
    def __init__(self, failures: tuple[BaseException, ...]) -> None:
        self.failures: Final = list(failures)
        self.flushed = 0

    async def flush(self) -> None:
        self.flushed += 1
        if self.failures:
            raise self.failures.pop(0)


class _Panic(BaseException):
    pass


@pytest.mark.asyncio
async def test_the_periodic_flush_keeps_running_after_a_flush_fails() -> None:
    sink: Final = _FlakySink((RuntimeError("bucket gone"), _Panic("a PanicException is a BaseException too")))
    task: Final = asyncio.create_task(s3_v3._periodic_flush(sink, 0.01))  # pyright: ignore[reportArgumentType]  # the loop only calls flush
    for _ in range(100):
        if sink.flushed >= 3:
            break
        await asyncio.sleep(0.01)
    assert sink.flushed >= 3
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _settled(logger: S3Logger) -> None:
    for _ in range(3):
        await asyncio.sleep(0.3)
        await logger.flush_queue()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_sync_completion_reaches_both_loggers_alike(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    monkeypatch.setattr(litellm, "callbacks", [logger])

    def call() -> None:
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="hi there",
            stream=stream,
        )
        if stream:
            for _ in response:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
                pass

    await asyncio.to_thread(call)
    await _settled(logger)
    _assert_same_puts(golden, bucket, 0)


@pytest.mark.asyncio
async def test_a_sync_line_delivery_is_skipped_by_the_native_logger(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    await logger.run_rust(
        Hook.attempt_ended,
        frozenset({Field.deprecated_kwargs}),
        SimpleNamespace(deprecated_kwargs={"standard_logging_object": _payload()}),
        SimpleNamespace(call=None, state=None),  # pyright: ignore[reportArgumentType]  # run_rust reads only call and state
    )
    assert logger.sink.queued_keys() == []
    assert logger.sink.stashed() == 0


@pytest.mark.asyncio
async def test_readiness_and_the_callbacks_view_see_the_native_logger_as_s3_v2(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.proxy_server import _hidden_runtime_callback_names

    logger = await _logger(monkeypatch, bucket, {})
    python_view: Final = golden.expected("callbacks-view")
    assert python_view == {"str-names-s3-logger": True, "hidden": True}
    assert "S3Logger" in str(logger)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    name: Final = litellm.logging_callback_manager._get_callback_string(logger)
    assert name in _hidden_runtime_callback_names(frozenset({"s3_v2"}))


def _put_paths(bucket: FakeBucket) -> list[str]:
    return [request.path for request in bucket.puts()]


def _queues(golden: Golden, logger: S3Logger) -> tuple[object, list[str]]:
    python_queue: Final = golden.expected("queue")
    return python_queue, logger.sink.queued_keys()


@pytest.mark.asyncio
async def test_a_failed_acompletion_reaches_both_loggers_alike(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_use_key_prefix": True})
    monkeypatch.setattr(litellm, "callbacks", [logger])
    with pytest.raises(litellm.RateLimitError):
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="litellm.RateLimitError",
            metadata={"user_api_key_alias": "key-a"},
        )
    await _settled(logger)
    _assert_same_live_puts(golden, bucket, 1)
    assert _put_paths(bucket)[0].startswith("/logs/key-a/")
    assert json.loads(bucket.puts()[0].body)["status"] == "failure"


@pytest.mark.asyncio
async def test_a_streamed_acompletion_reaches_both_loggers_alike(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    monkeypatch.setattr(litellm, "callbacks", [logger])
    response: Final = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="hi there, streamed",
        stream=True,
    )
    async for _ in response:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
        pass
    await _settled(logger)
    _assert_same_live_puts(golden, bucket, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wait", "kept"), [pytest.param(0.0, True, id="young"), pytest.param(1.2, False, id="expired")]
)
async def test_both_loggers_drop_an_upload_retrying_past_the_max_retry_age(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, wait: float, kept: bool
) -> None:
    bucket.failures.append(("_a\\.json", 404, ""))
    logger = await _logger(monkeypatch, bucket, {"s3_max_retry_age_seconds": 1})
    for response_id in ("a", "b"):
        await _log(logger, _payload(response_id))
    await logger.flush_queue()
    await asyncio.sleep(wait)
    await _log(logger, _payload("c"))
    await logger.flush_queue()
    python_queue, rust_queue = _queues(golden, logger)
    assert rust_queue == python_queue
    assert bool(rust_queue) is kept


@pytest.mark.asyncio
async def test_both_loggers_upload_in_the_same_order_with_requeued_items_last(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    bucket.failures.extend([("_a\\.json", 404, ""), ("_b\\.json", 404, "")])
    logger = await _logger(monkeypatch, bucket, {"s3_max_concurrent_uploads": 1})
    for response_id in ("a", "b", "c"):
        await _log(logger, _payload(response_id))
    await logger.flush_queue()
    python_queue, rust_queue = _queues(golden, logger)
    assert rust_queue == python_queue
    bucket.failures.clear()
    for response_id in ("d", "e"):
        await _log(logger, _payload(response_id))
    await logger.flush_queue()
    assert _put_paths(bucket) == golden.expected("put-paths")
    assert [path.rsplit("_", 1)[-1] for path in _put_paths(bucket)] == [
        "a.json",
        "b.json",
        "c.json",
        "d.json",
        "e.json",
        "a.json",
        "b.json",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adaptive", "peak"), [pytest.param(True, 3, id="adaptive"), pytest.param(False, 1, id="fixed")]
)
async def test_both_loggers_widen_the_adaptive_upload_limit_alike(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, adaptive: bool, peak: int
) -> None:
    bucket.delays.append((".", 0.05))
    logger = await _logger(
        monkeypatch,
        bucket,
        {"s3_max_concurrent_uploads": 1, "s3_adaptive_concurrency": adaptive, "s3_max_adaptive_concurrency": 3},
    )
    for index in range(10):
        await _log(logger, _payload(f"r{index}"))
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 10)
    assert bucket.peak == golden.expected("peak") == peak


@pytest.mark.asyncio
@pytest.mark.parametrize(("batch_size", "puts"), [(1, 2), (2, 2), (3, 0)])
async def test_audit_events_flush_at_the_same_batch_size(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, batch_size: int, puts: int
) -> None:
    monkeypatch.setattr(s3_v3, "DEFAULT_S3_BATCH_SIZE", batch_size)
    logger = await _logger(monkeypatch, bucket, {})
    for number in (1, 2):
        audit = {"id": f"audit-{number}", "action": "created", "updated_at": "2026-11-13"}
        await logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # a partial audit payload is enough here
    python_bodies: Final = golden.expected("bodies")
    assert isinstance(python_bodies, list)
    assert len(bucket.puts()) == len(python_bodies) == puts
    assert sorted(r.body.decode() for r in bucket.puts()) == python_bodies


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["dropped-connection", "timeout"])
async def test_both_loggers_keep_uploads_that_fail_in_transport(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(s3_v3, "S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS", 0.3)
    if failure == "timeout":
        bucket.delays.append(("_a\\.json", 1.0))
    else:
        bucket.failures.append(("_a\\.json", 0, ""))
    logger = await _logger(monkeypatch, bucket, {})
    with patch.object(s3_v3._Host, "callback_failed") as rust_failures:
        for response_id in ("a", "b"):
            await _log(logger, _payload(response_id))
        await logger.flush_queue()
        await asyncio.sleep(0.2)
    python_queue, rust_queue = _queues(golden, logger)
    assert rust_queue == python_queue
    assert [key.rsplit("_", 1)[-1] for key in rust_queue] == ["a.json"]
    assert golden.expected("failures") == rust_failures.call_count
    assert rust_failures.call_count == 1
    attempts: Final = 3 if failure == "timeout" else 6
    python_paths: Final = golden.expected("put-paths")
    assert isinstance(python_paths, list)
    assert [str(path).rsplit("_", 1)[-1] for path in python_paths].count("a.json") == attempts
    assert [path.rsplit("_", 1)[-1] for path in _put_paths(bucket)].count("a.json") == attempts


_NO_METADATA: Final = object()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "puts"),
    [
        pytest.param(_NO_METADATA, 1, id="missing"),
        pytest.param(None, 1, id="none"),
        pytest.param("text", 0, id="text"),
        pytest.param(["listed"], 0, id="list"),
    ],
)
async def test_cold_storage_handles_odd_metadata_like_python(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, metadata: object, puts: int
) -> None:
    logger = await _logger(monkeypatch, bucket, {})
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    payload: Final = _payload()
    if metadata is _NO_METADATA:
        del payload["metadata"]
    else:
        payload["metadata"] = metadata
    with patch.object(s3_v3._Host, "callback_failed") as rust_failures:
        await _log(logger, payload)
        await logger.flush_queue()
        await asyncio.sleep(0.2)
    _assert_same_puts(golden, bucket, puts)
    assert golden.expected("failures") == rust_failures.call_count
    assert rust_failures.call_count == 1 - puts
    assert logger.sink.stashed() == 0


AWARE_START: Final = datetime(2026, 11, 13, 23, 30, 0, 5, tzinfo=timezone(timedelta(hours=5)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("start_time", "payload_start", "logging_start"),
    [
        pytest.param(AWARE_START, AWARE_START.timestamp(), AWARE_START, id="aware"),
        pytest.param(START, None, START, id="payload-without-start"),
        pytest.param(START, START.timestamp(), START + timedelta(seconds=3), id="handler-start-differs"),
    ],
)
async def test_both_loggers_name_the_object_from_the_same_start_time(
    golden: Golden,
    bucket: FakeBucket,
    monkeypatch: pytest.MonkeyPatch,
    start_time: datetime,
    payload_start: float | None,
    logging_start: datetime,
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_partition_granularity": "hour"})
    payload: Final = _payload()
    if payload_start is None:
        del payload["startTime"]
    else:
        payload["startTime"] = payload_start
    await _log_native(
        logger, {"standard_logging_object": copy.deepcopy(payload), "start_time": logging_start}, start_time
    )
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 1)
    stamp: Final = start_time.strftime("%Y-%m-%d/%H/time-%H-%M-%S-%f")
    assert _put_paths(bucket) == [f"/logs/{stamp}_chatcmpl-1.json"]


@pytest.mark.asyncio
async def test_cold_storage_reads_without_a_region_send_nothing_like_python(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_region_name": None})
    bucket.objects["/logs/k.json"] = b'{"stored": 1}'
    python_object: Final = golden.expected("object")
    rust_object: Final = await logger.get_proxy_server_request_from_cold_storage_with_object_key("k.json")
    assert rust_object is python_object is None
    assert golden.expected("received") == []
    assert bucket.received == []


@pytest.mark.asyncio
async def test_params_json_cannot_hold_still_build_the_native_logger(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.config import Config

    logger = await _logger(
        monkeypatch, bucket, {"s3_config": Config(retries={"max_attempts": 1}), "s3_max_concurrent_uploads": 10**30}
    )
    await _log(logger, _payload())
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 1)


class _AliasLookupFails(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:
        if key == "user_api_key_team_alias":
            raise RuntimeError("the alias lookup failed")
        return super().get(key, default)


@pytest.mark.asyncio
async def test_the_attempt_ended_hook_only_enqueues_while_s3_is_slow(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s3_v3, "DEFAULT_S3_BATCH_SIZE", 1)
    logger = await _logger(monkeypatch, bucket, {})
    bucket.delays.append((".", 3.0))
    started: Final = time.monotonic()
    await _log_native(logger, {"standard_logging_object": _payload()})
    hook_seconds: Final = time.monotonic() - started
    monkeypatch.setattr(litellm, "callbacks", [logger])
    await litellm.acompletion(model="gpt-4o", messages=[{"role": "user", "content": "hello"}], mock_response="hi there")
    for _ in range(100):
        if len(logger.sink.queued_keys()) == 2:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    assert hook_seconds < 1.0
    assert golden.expected("queue-length") == 1
    assert len(logger.sink.queued_keys()) == 2
    assert _expected_puts(golden) == bucket.puts() == []


@pytest.mark.asyncio
async def test_a_payload_whose_alias_lookup_fails_is_counted_and_released(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_use_team_prefix": True})
    with patch.object(s3_v3._Host, "callback_failed") as rust_failures:
        await _log(logger, _payload(metadata=_AliasLookupFails(team="t")))
        await logger.flush_queue()
        await asyncio.sleep(0.2)
    _assert_same_puts(golden, bucket, 0)
    assert golden.expected("failures") == rust_failures.call_count
    assert rust_failures.call_count == 1
    assert logger.sink.stashed() == 0


@pytest.mark.asyncio
async def test_prompts_only_adds_a_null_response_to_a_payload_without_one_like_python(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_log_prompts_only": True})
    payload: Final = {key: value for key, value in _payload().items() if key != "response"}
    await _log(logger, payload)
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 1)
    assert list(json.loads(bucket.puts()[0].body))[-1] == "response"


@pytest.mark.asyncio
async def test_the_dispatched_native_strips_the_messages_before_the_payload_is_copied(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.litellm_core_utils import litellm_logging

    logger = await _logger(monkeypatch, bucket, {"s3_strip_base64_files": True})
    payload: Final = _payload(
        "stripped",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "see data:image/png;base64,AAAA== inline"}]},
            "not a message",
        ],
    )
    kwargs: Final = {"standard_logging_object": copy.deepcopy(payload)}
    await litellm_logging._run_callbacks(
        Hook.attempt_ended,
        [logger],
        kwargs,
        lambda _callback, _kwargs: None,
        args=HookArgs(request=kwargs, start_time=START, end_time=START, outcome=Outcome.success),
    )
    await _until_queued(logger)
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 1)
    assert json.loads(bucket.puts()[0].body)["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "see [BASE64_REDACTED]"}]}
    ]


class _FalsyAlias:
    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:
        return "falsy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alias",
    [
        pytest.param([], id="empty-list"),
        pytest.param({}, id="empty-dict"),
        pytest.param(set(), id="empty-set"),
        pytest.param(_FalsyAlias(), id="falsy-object"),
        pytest.param(0.0, id="zero-float"),
        pytest.param("", id="empty-text"),
    ],
)
async def test_a_falsy_alias_of_any_type_adds_no_prefix_like_python(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, alias: object
) -> None:
    logger = await _logger(monkeypatch, bucket, {"s3_use_team_prefix": True, "s3_use_key_prefix": True})
    await _log(logger, _payload(metadata={"user_api_key_team_alias": alias}))
    await logger.flush_queue()
    _assert_same_puts(golden, bucket, 1)
    assert _put_paths(bucket) == ["/logs/2026-11-13/time-09-05-07-000042_chatcmpl-1.json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_excluded_payload_fields_are_left_out_like_python(
    golden: Golden, bucket: FakeBucket, monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", ["response", "metadata", "messages"])
    logger = await _logger(monkeypatch, bucket, {"s3_use_team_prefix": True})
    monkeypatch.setattr(litellm, "callbacks", [logger])
    with pytest.raises(litellm.RateLimitError) if failed else nullcontext():
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="litellm.RateLimitError" if failed else "hi there",
            metadata={"user_api_key_team_alias": "team-a"},
        )
    await _settled(logger)
    _assert_same_live_puts(golden, bucket, 1)


RECORDED_TRACEBACK: Final = (
    '  File "/home/user/wt/rust-s3/litellm/main.py", line 956, in acompletion\n'
    "    init_response: Final = await loop.run_in_executor(None, func_with_context)\n"
    "                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
    '  File "/usr/local/lib/python3.13/concurrent/futures/thread.py", line 59, in run\n'
    "    result = self.fn(*self.args, **self.kwargs)\n"
    '  File "/home/user/wt/rust-s3/litellm/main.py", line 1235, in mock_completion\n'
    "    _handle_mock_potential_exceptions(\n"
    "    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^\n"
    "    ...<3 lines>...\n"
    "    )\n"
)


@pytest.mark.parametrize(
    ("other", "same"),
    [
        (
            '  File "/home/runner/work/litellm/litellm/litellm/main.py", line 970, in acompletion\n'
            "    init_response: Final = await loop.run_in_executor(None, func_with_context)\n"
            '  File "/opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/concurrent/futures/thread.py",'
            " line 58, in run\n"
            "    result = self.fn(*self.args, **self.kwargs)\n"
            "             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
            '  File "/home/runner/work/litellm/litellm/litellm/main.py", line 1240, in mock_completion\n'
            "    _handle_mock_potential_exceptions(\n",
            True,
        ),
        (
            '  File "/home/runner/work/litellm/litellm/litellm/main.py", line 956, in acompletion\n'
            '  File "/usr/local/lib/python3.13/concurrent/futures/thread.py", line 59, in run\n'
            '  File "/home/runner/work/litellm/litellm/litellm/utils.py", line 1235, in mock_completion\n',
            False,
        ),
        (
            '  File "/home/runner/work/litellm/litellm/litellm/main.py", line 956, in acompletion\n'
            '  File "/usr/local/lib/python3.13/concurrent/futures/thread.py", line 59, in run\n'
            '  File "/home/runner/work/litellm/litellm/litellm/main.py", line 1235, in completion\n',
            False,
        ),
        (
            '  File "/home/runner/work/litellm/litellm/litellm/main.py", line 956, in acompletion\n'
            '  File "/home/runner/work/litellm/litellm/litellm/main.py", line 1235, in mock_completion\n',
            False,
        ),
    ],
    ids=["other-machine-and-python", "other-litellm-file", "other-function", "missing-frame"],
)
def test_a_recorded_traceback_matches_only_the_same_frames_from_another_environment(other: str, same: bool) -> None:
    assert (_portable_traceback(other) == _portable_traceback(RECORDED_TRACEBACK)) is same
    assert _portable_traceback(RECORDED_TRACEBACK) == (
        "litellm/main.py in acompletion\n<stdlib>/concurrent/futures/thread.py in run\nlitellm/main.py in mock_completion"
    )
