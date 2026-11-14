import asyncio
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
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch
from urllib.parse import parse_qs

import pytest
from pydantic import BaseModel

import litellm
from litellm.integrations import s3_v2, s3_v3
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.host import request_scope
from litellm.integrations.custom_logger_v2.legacy import KwargsFields
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.selection import RequestSelection
from litellm.integrations.s3_v2 import S3Logger
from litellm.integrations.s3_v3 import S3LoggerV3, native_s3_v2_enabled, new_s3_v2_logger
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS, V2_ASYNC_LOGGING_LINE
from litellm.llms.bedrock import base_aws_llm
from litellm.rust_bridge import _native
from litellm.types.integrations.custom_logger_v2 import Outcome

pytestmark = pytest.mark.skipif(not hasattr(_native, "NativeS3Log"), reason="the built _native has no NativeS3Log")

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
def buckets() -> Iterator[tuple[FakeBucket, FakeBucket]]:
    pair: Final = (FakeBucket(), FakeBucket())
    servers: Final = []
    for bucket in pair:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(bucket))
        bucket.port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    yield pair
    for server in servers:
        server.shutdown()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(s3_v2, "_backoff_seconds", lambda _attempt: 0.0)
    monkeypatch.delenv("S3_PARTITION_GRANULARITY", raising=False)
    monkeypatch.delenv("S3_LOG_PROMPTS_ONLY", raising=False)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", None)
    native_s3_v2_enabled.cache_clear()
    yield
    native_s3_v2_enabled.cache_clear()


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


async def _loggers(
    monkeypatch: pytest.MonkeyPatch, pair: tuple[FakeBucket, FakeBucket], extra: Mapping[str, object]
) -> tuple[S3Logger, S3LoggerV3]:
    monkeypatch.setattr(litellm, "s3_callback_params", _params(pair[0].url, extra))
    monkeypatch.setenv("LITELLM_NATIVE_S3_V2", "false")
    native_s3_v2_enabled.cache_clear()
    python_logger: Final = new_s3_v2_logger()
    monkeypatch.setattr(litellm, "s3_callback_params", _params(pair[1].url, extra))
    monkeypatch.setenv("LITELLM_NATIVE_S3_V2", "true")
    native_s3_v2_enabled.cache_clear()
    rust_logger: Final = new_s3_v2_logger()
    assert isinstance(python_logger, S3Logger)
    assert isinstance(rust_logger, S3LoggerV3)
    return python_logger, rust_logger


def _attempt_ended_view(kwargs: Mapping[str, object], start_time: datetime) -> FieldView:
    args: Final = HookArgs(request=kwargs, start_time=start_time, end_time=start_time, outcome=Outcome.success)
    return FieldView(with_sources(KwargsFields(kwargs, LOGGING_KEYS), DERIVED_ATTEMPT_FIELDS, args))


async def _log_native(rust_logger: S3LoggerV3, kwargs: Mapping[str, object], start_time: datetime = START) -> None:
    line: Final = V2_ASYNC_LOGGING_LINE.set(True)
    try:
        await rust_logger.on_attempt_ended(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # generated from the native manifest with its declared reads
            _attempt_ended_view(kwargs, start_time),
            SimpleNamespace(call=None, state=None),
        )
    finally:
        V2_ASYNC_LOGGING_LINE.reset(line)


async def _log_each(python_logger: S3Logger, rust_logger: S3LoggerV3, payloads: tuple[object, object]) -> None:
    await python_logger.async_log_success_event({"standard_logging_object": payloads[0]}, None, START, START)
    await _log_native(rust_logger, {"standard_logging_object": payloads[1]})


async def _log(python_logger: S3Logger, rust_logger: S3LoggerV3, payload: object) -> None:
    await _log_each(python_logger, rust_logger, (payload, copy.deepcopy(payload)))


async def _flush(python_logger: S3Logger, rust_logger: S3LoggerV3) -> None:
    await python_logger.flush_queue()
    await rust_logger.flush_queue()


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


def _assert_same_puts(pair: tuple[FakeBucket, FakeBucket], count: int) -> None:
    python_puts, rust_puts = pair[0].puts(), pair[1].puts()
    assert len(python_puts) == len(rust_puts) == count
    for python_put, rust_put in zip(
        sorted(python_puts, key=lambda r: r.path), sorted(rust_puts, key=lambda r: r.path), strict=True
    ):
        assert rust_put.path == python_put.path
        assert rust_put.body == python_put.body, _body_differences(python_put.body, rust_put.body)
        for name in COMPARED_HEADERS:
            assert rust_put.headers.get(name) == python_put.headers.get(name), name
        assert _signed_headers(rust_put) == _signed_headers(python_put)
        assert rust_put.headers["authorization"] == _botocore_authorization(rust_put, pair[1].port)
        assert python_put.headers["authorization"] == _botocore_authorization(python_put, pair[0].port)


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
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, extra: Mapping[str, object]
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, extra)
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
        await _log(python_logger, rust_logger, payload)
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, len(payloads))


@pytest.mark.asyncio
async def test_a_missing_payload_logs_nothing(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    await _log(python_logger, rust_logger, None)
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 0)


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
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, payload: object
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_use_team_prefix": True})
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log(python_logger, rust_logger, payload)
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    assert buckets[0].puts() == buckets[1].puts() == []
    assert python_logger.log_queue == []
    assert rust_logger.sink.queued_keys() == []
    assert python_failures.call_count == rust_failures.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("granularity", ["day", "hour"])
async def test_a_cold_storage_key_is_reused_when_it_matches(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, granularity: str
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_path": "cold"})
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    date: Final = "2026-11-13/" if granularity == "day" else "2026-11-13/09/"
    matching: Final = f"cold/{date}time-09-05-07-000042_chatcmpl-1.json"
    for response_id, key in (("chatcmpl-1", matching), ("chatcmpl-2", "elsewhere/x.json")):
        await _log(python_logger, rust_logger, _payload(response_id, metadata={"cold_storage_object_key": key}))
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 2)
    assert {r.path for r in buckets[1].puts()} == {
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
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    failures: list[tuple[str, int, str]],
    extra: Mapping[str, object],
) -> None:
    for bucket in buckets:
        bucket.failures.extend(failures)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, extra)
    for response_id in ("a", "b", "c"):
        await _log(python_logger, rust_logger, _payload(response_id))
    await _flush(python_logger, rust_logger)
    python_queue: Final = [element.s3_object_key for element in python_logger.log_queue]
    assert rust_logger.sink.queued_keys() == python_queue
    assert len(buckets[1].puts()) == len(buckets[0].puts())
    for bucket in buckets:
        bucket.failures.clear()
    await _flush(python_logger, rust_logger)
    assert rust_logger.sink.queued_keys() == [element.s3_object_key for element in python_logger.log_queue] == []
    assert sorted(r.path for r in buckets[1].puts()) == sorted(r.path for r in buckets[0].puts())


@pytest.mark.asyncio
async def test_batch_files_hold_the_same_lines(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(
        monkeypatch, buckets, {"s3_batch_file_upload": True, "s3_use_team_prefix": True}
    )
    for response_id, team in (("a", "t1"), ("b", "t2"), ("c", "t1")):
        await _log(python_logger, rust_logger, _payload(response_id, metadata={"user_api_key_team_alias": team}))
    await _flush(python_logger, rust_logger)
    python_puts, rust_puts = buckets[0].puts(), buckets[1].puts()
    assert len(python_puts) == len(rust_puts) == 2
    name: Final = re.compile(r"/batch_\d\d-\d\d-\d\d_[0-9a-f]{32}\.jsonl$")
    for python_put, rust_put in zip(
        sorted(python_puts, key=lambda r: r.path), sorted(rust_puts, key=lambda r: r.path), strict=True
    ):
        assert name.sub("", rust_put.path) == name.sub("", python_put.path)
        assert rust_put.body == python_put.body, _body_differences(python_put.body, rust_put.body)
        assert rust_put.headers["content-type"] == python_put.headers["content-type"] == "application/x-ndjson"


@pytest.mark.asyncio
async def test_audit_events_and_line_items_land_at_the_same_keys(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_path": "audit-base"})
    audit: Final = {"id": "audit-1", "action": "created", "before_value": None, "updated_at": "2026-11-13"}
    await python_logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # a partial audit payload is enough here
    await rust_logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # see above
    await _flush(python_logger, rust_logger)
    line_items: Final = {
        "batch_id": "batch/1",
        "part": 3,
        "finished_at": START.timestamp(),
        "lines": [{"custom_id": "x", "status": "ok", "text": "ünïcode"}],
    }
    await python_logger.async_log_batch_line_items(line_items)  # pyright: ignore[reportArgumentType]  # see above
    await rust_logger.async_log_batch_line_items(line_items)  # pyright: ignore[reportArgumentType]  # see above
    python_puts, rust_puts = buckets[0].puts(), buckets[1].puts()
    assert len(python_puts) == len(rust_puts) == 2
    clock: Final = re.compile(r"/\d\d-\d\d-\d\d_")
    for python_put, rust_put in zip(python_puts, rust_puts, strict=True):
        assert clock.sub("/", rust_put.path) == clock.sub("/", python_put.path)
        assert re.sub(r"/\d{4}-\d\d-\d\d/", "/", rust_put.path) == re.sub(r"/\d{4}-\d\d-\d\d/", "/", python_put.path)
        assert rust_put.body == python_put.body, _body_differences(python_put.body, rust_put.body)
        for name in COMPARED_HEADERS:
            assert rust_put.headers.get(name) == python_put.headers.get(name), name


@pytest.mark.asyncio
async def test_cold_storage_reads_return_the_same_object(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    await _log(python_logger, rust_logger, _payload())
    await _flush(python_logger, rust_logger)
    key: Final = "2026-11-13/time-09-05-07-000042_chatcmpl-1.json"
    python_object: Final = await python_logger.get_proxy_server_request_from_cold_storage_with_object_key(key)
    rust_object: Final = await rust_logger.get_proxy_server_request_from_cold_storage_with_object_key(key)
    assert python_object is not None
    assert rust_object == python_object
    assert await rust_logger.get_proxy_server_request_from_cold_storage_with_object_key("missing.json") is None
    python_get, rust_get = buckets[0].received[-1], buckets[1].received[-1]
    assert rust_get.headers["x-amz-content-sha256"] == python_get.headers["x-amz-content-sha256"]
    assert _signed_headers(rust_get) == _signed_headers(python_get)


@pytest.mark.asyncio
async def test_the_native_logger_answers_the_python_helpers(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(
        monkeypatch, buckets, {"s3_path": "p", "s3_partition_granularity": "hour"}
    )
    assert rust_logger.s3_path == python_logger.s3_path == "p"
    assert rust_logger.resolve_partition_granularity() == python_logger.resolve_partition_granularity() == "hour"
    selection: Final = type(rust_logger).selection
    assert selection is not None
    assert selection.name == "s3_v2"
    assert json.loads(rust_logger.sink.callback().manifest())["subscriptions"][0]["hook"] == "attempt_ended"


@pytest.mark.asyncio
async def test_stripping_base64_rewrites_the_shared_payload_the_same_way(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_strip_base64_files": True})
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
    python_payload, rust_payload = copy.deepcopy(original), copy.deepcopy(original)
    await _log_each(python_logger, rust_logger, (python_payload, rust_payload))
    assert rust_payload == python_payload != original
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)


@pytest.mark.asyncio
async def test_a_payload_changed_before_the_flush_is_stored_as_it_is_at_the_flush(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    python_payload, rust_payload = _payload(), _payload()
    await _log_each(python_logger, rust_logger, (python_payload, rust_payload))
    for payload in (python_payload, rust_payload):
        payload["metadata"]["late"] = "written after the event"  # pyright: ignore[reportIndexIssue]  # metadata is the dict _payload builds
        payload["response_cost"] = 99.0
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)
    stored: Final = json.loads(buckets[1].puts()[0].body)
    assert stored["metadata"]["late"] == "written after the event"
    assert stored["response_cost"] == 0.000123
    assert rust_logger.sink.stashed() == 0


def test_both_loggers_need_a_running_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "s3_callback_params", _params("http://127.0.0.1:9", {}))
    for flag in ("false", "true"):
        monkeypatch.setenv("LITELLM_NATIVE_S3_V2", flag)
        native_s3_v2_enabled.cache_clear()
        with pytest.raises(RuntimeError, match="no running event loop"):
            new_s3_v2_logger()


@pytest.mark.asyncio
async def test_the_native_logger_flushes_on_its_own_after_the_interval(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s3_v3, "DEFAULT_S3_FLUSH_INTERVAL_SECONDS", 0.05)
    _, rust_logger = await _loggers(monkeypatch, buckets, {})
    await _log_native(rust_logger, {"standard_logging_object": _payload()})
    for _ in range(100):
        if buckets[1].puts():
            break
        await asyncio.sleep(0.02)
    assert [r.path for r in buckets[1].puts()] == ["/logs/2026-11-13/time-09-05-07-000042_chatcmpl-1.json"]


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
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    environment: Mapping[str, str],
    region: str | None,
) -> None:
    for name in ("AWS_REGION_NAME", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_region_name": None})
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log(python_logger, rust_logger, _payload())
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    if region is None:
        assert buckets[0].puts() == buckets[1].puts() == []
        assert python_failures.call_count == rust_failures.call_count == 1
        return
    for bucket in buckets:
        put = bucket.puts()[0]
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


@pytest.mark.asyncio
async def test_web_identity_assumes_the_role_with_the_same_session_policy(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, sts: FakeSts
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
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, identity)
    await _log(python_logger, rust_logger, _payload())
    await _flush(python_logger, rust_logger)
    assert len(sts.calls) == 2
    python_call, rust_call = sts.calls
    for name in ("Action", "RoleArn", "RoleSessionName", "WebIdentityToken", "DurationSeconds", "Policy"):
        assert rust_call.get(name) == python_call.get(name), name
    assert rust_call["WebIdentityToken"] == ["oidc-token-value"]
    assert json.loads(rust_call["Policy"][0]) == json.loads(S3Logger._web_identity_session_policy_document())
    for bucket in buckets:
        put = bucket.puts()[0]
        assert put.headers["x-amz-security-token"] == "sts-token"
        assert put.headers["authorization"] == _botocore_authorization(
            put, bucket.port, credentials=("AKIDSTS", "sts-secret", "sts-token")
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("token", [True, 7, "abc", ""])
async def test_a_web_identity_token_botocore_rejects_fails_both_before_sts(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, sts: FakeSts, token: object
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
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, identity)
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
        patch.object(s3_v2.verbose_logger, "exception") as python_errors,
    ):
        await _log(python_logger, rust_logger, _payload())
        await _flush(python_logger, rust_logger)
    assert sts.calls == []
    assert buckets[0].puts() == buckets[1].puts() == []
    assert python_failures.call_count == rust_failures.call_count == 1
    assert rust_logger.sink.queued_keys() == [element.s3_object_key for element in python_logger.log_queue]
    python_error: Final = next(
        call.args[1] for call in python_errors.call_args_list if call.args[0] == "Error uploading to s3: %s"
    )
    with pytest.raises(ValueError, match=r"^Parameter validation failed:") as rust_error:
        s3_v3._Host().web_identity_token(reference)
    assert str(rust_error.value) == str(python_error)


@pytest.mark.asyncio
@pytest.mark.parametrize("global_disable", [False, True])
@pytest.mark.parametrize("no_log", [None, True, "true"])
@pytest.mark.parametrize("disabled_header", [None, "s3_v2", "langfuse, S3_V2"])
async def test_the_native_opt_out_matches_should_run_callback(
    buckets: tuple[FakeBucket, FakeBucket],
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
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
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
    expected: Final = logging_obj.should_run_callback(python_logger, litellm_params, "async_success_handler")
    kwargs: Final = {
        "litellm_params": litellm_params,
        "standard_callback_dynamic_params": logging_obj.standard_callback_dynamic_params,
        "standard_logging_object": _payload(),
    }
    assert s3_v3.legacy_should_log(kwargs) is expected
    await _log_native(rust_logger, kwargs)
    await rust_logger.flush_queue()
    assert len(buckets[1].puts()) == int(expected)


async def _until_queued(python_logger: S3Logger, rust_logger: S3LoggerV3) -> None:
    for _ in range(100):
        if python_logger.log_queue and rust_logger.sink.queued_keys():
            return
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_scope", [False, True])
@pytest.mark.parametrize("global_disable", [False, True])
@pytest.mark.parametrize("no_log", [False, True])
async def test_an_acompletion_reaches_both_loggers_alike(
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    global_disable: bool,
    no_log: bool,
    proxy_scope: bool,
) -> None:
    monkeypatch.setattr(litellm, "global_disable_no_log_param", global_disable)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_use_team_prefix": True})
    monkeypatch.setattr(litellm, "callbacks", [python_logger, rust_logger])
    with request_scope(RequestSelection(no_log=no_log and not global_disable)) if proxy_scope else nullcontext():
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello «world»"}],
            mock_response="hi there",
            metadata={"user_api_key_team_alias": "team-a"},
            **({"no-log": True} if no_log else {}),
        )
    await _until_queued(python_logger, rust_logger)
    await _flush(python_logger, rust_logger)
    expected: Final = 0 if no_log and not global_disable else 1
    _assert_same_puts(buckets, expected)
    assert all(put.path.startswith("/logs/team-a/") for put in buckets[1].puts())


class _Mutating:
    def __init__(self, target: dict[str, object]) -> None:
        self.target: Final = target

    def __str__(self) -> str:
        self.target[f"grown-{len(self.target)}"] = True
        return "mutating"


@pytest.mark.asyncio
async def test_a_payload_that_changes_while_it_is_rendered_is_kept_for_retry_by_both_loggers(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    payloads: Final = (_payload("racing"), _payload("racing"))
    for payload in payloads:
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        metadata["mutating"] = _Mutating(metadata)  # pyright: ignore[reportUnknownArgumentType]  # the metadata dict _payload builds
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log_each(python_logger, rust_logger, payloads)
        await _log(python_logger, rust_logger, _payload("steady"))
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    assert rust_logger.sink.queued_keys() == [element.s3_object_key for element in python_logger.log_queue]
    assert len(rust_logger.sink.queued_keys()) == 1
    assert python_failures.call_count == rust_failures.call_count == 1
    _assert_same_puts(buckets, 1)
    assert not rust_logger.periodic_flush_task.done()


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


async def _settled(pair: tuple[FakeBucket, FakeBucket], python_logger: S3Logger, rust_logger: S3LoggerV3) -> None:
    for _ in range(3):
        await asyncio.sleep(0.3)
        await _flush(python_logger, rust_logger)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_a_sync_completion_reaches_both_loggers_alike(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    monkeypatch.setattr(litellm, "callbacks", [python_logger, rust_logger])

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
    await _settled(buckets, python_logger, rust_logger)
    _assert_same_puts(buckets, 0)


@pytest.mark.asyncio
async def test_a_sync_line_delivery_is_skipped_by_the_native_logger(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, rust_logger = await _loggers(monkeypatch, buckets, {})
    await rust_logger.run_rust(
        Hook.attempt_ended,
        frozenset({Field.deprecated_kwargs}),
        SimpleNamespace(deprecated_kwargs={"standard_logging_object": _payload()}),
        SimpleNamespace(call=None, state=None),  # pyright: ignore[reportArgumentType]  # run_rust reads only call and state
    )
    assert rust_logger.sink.queued_keys() == []
    assert rust_logger.sink.stashed() == 0


@pytest.mark.asyncio
async def test_readiness_and_the_callbacks_view_see_the_native_logger_as_s3_v2(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.proxy_server import _hidden_runtime_callback_names

    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    for logger in (python_logger, rust_logger):
        assert "S3Logger" in str(logger)
        monkeypatch.setattr(litellm, "callbacks", [logger])
        name = litellm.logging_callback_manager._get_callback_string(logger)
        assert name in _hidden_runtime_callback_names(frozenset({"s3_v2"}))


def _put_paths(bucket: FakeBucket) -> list[str]:
    return [request.path for request in bucket.puts()]


def _queues(python_logger: S3Logger, rust_logger: S3LoggerV3) -> tuple[list[str], list[str]]:
    return [element.s3_object_key for element in python_logger.log_queue], rust_logger.sink.queued_keys()


@pytest.mark.asyncio
async def test_a_failed_acompletion_reaches_both_loggers_alike(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_use_key_prefix": True})
    monkeypatch.setattr(litellm, "callbacks", [python_logger, rust_logger])
    with pytest.raises(litellm.RateLimitError):
        await litellm.acompletion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            mock_response="litellm.RateLimitError",
            metadata={"user_api_key_alias": "key-a"},
        )
    await _settled(buckets, python_logger, rust_logger)
    _assert_same_puts(buckets, 1)
    assert _put_paths(buckets[1])[0].startswith("/logs/key-a/")
    assert json.loads(buckets[1].puts()[0].body)["status"] == "failure"


@pytest.mark.asyncio
async def test_a_streamed_acompletion_reaches_both_loggers_alike(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    monkeypatch.setattr(litellm, "callbacks", [python_logger, rust_logger])
    response: Final = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="hi there, streamed",
        stream=True,
    )
    async for _ in response:  # pyright: ignore[reportGeneralTypeIssues]  # a stream when stream=True
        pass
    await _settled(buckets, python_logger, rust_logger)
    _assert_same_puts(buckets, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wait", "kept"), [pytest.param(0.0, True, id="young"), pytest.param(1.2, False, id="expired")]
)
async def test_both_loggers_drop_an_upload_retrying_past_the_max_retry_age(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, wait: float, kept: bool
) -> None:
    for bucket in buckets:
        bucket.failures.append(("_a\\.json", 404, ""))
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_max_retry_age_seconds": 1})
    for response_id in ("a", "b"):
        await _log(python_logger, rust_logger, _payload(response_id))
    await _flush(python_logger, rust_logger)
    await asyncio.sleep(wait)
    await _log(python_logger, rust_logger, _payload("c"))
    await _flush(python_logger, rust_logger)
    python_queue, rust_queue = _queues(python_logger, rust_logger)
    assert rust_queue == python_queue
    assert bool(rust_queue) is kept


@pytest.mark.asyncio
async def test_both_loggers_upload_in_the_same_order_with_requeued_items_last(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    for bucket in buckets:
        bucket.failures.extend([("_a\\.json", 404, ""), ("_b\\.json", 404, "")])
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_max_concurrent_uploads": 1})
    for response_id in ("a", "b", "c"):
        await _log(python_logger, rust_logger, _payload(response_id))
    await _flush(python_logger, rust_logger)
    python_queue, rust_queue = _queues(python_logger, rust_logger)
    assert rust_queue == python_queue
    for bucket in buckets:
        bucket.failures.clear()
    for response_id in ("d", "e"):
        await _log(python_logger, rust_logger, _payload(response_id))
    await _flush(python_logger, rust_logger)
    assert _put_paths(buckets[1]) == _put_paths(buckets[0])
    assert [path.rsplit("_", 1)[-1] for path in _put_paths(buckets[1])] == [
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
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, adaptive: bool, peak: int
) -> None:
    for bucket in buckets:
        bucket.delays.append((".", 0.05))
    python_logger, rust_logger = await _loggers(
        monkeypatch,
        buckets,
        {"s3_max_concurrent_uploads": 1, "s3_adaptive_concurrency": adaptive, "s3_max_adaptive_concurrency": 3},
    )
    for index in range(10):
        await _log(python_logger, rust_logger, _payload(f"r{index}"))
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 10)
    assert buckets[1].peak == buckets[0].peak == peak


@pytest.mark.asyncio
@pytest.mark.parametrize(("batch_size", "puts"), [(1, 2), (2, 2), (3, 0)])
async def test_audit_events_flush_at_the_same_batch_size(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, batch_size: int, puts: int
) -> None:
    monkeypatch.setattr(s3_v3, "DEFAULT_S3_BATCH_SIZE", batch_size)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    python_logger.batch_size = batch_size
    for number in (1, 2):
        audit = {"id": f"audit-{number}", "action": "created", "updated_at": "2026-11-13"}
        await python_logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # a partial audit payload is enough here
        await rust_logger.async_log_audit_log_event(audit)  # pyright: ignore[reportArgumentType]  # see above
    assert len(buckets[1].puts()) == len(buckets[0].puts()) == puts
    assert sorted(r.body for r in buckets[1].puts()) == sorted(r.body for r in buckets[0].puts())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["dropped-connection", "timeout"])
async def test_both_loggers_keep_uploads_that_fail_in_transport(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import httpx

    monkeypatch.setattr(s3_v2, "_UPLOAD_ATTEMPT_TIMEOUT", httpx.Timeout(0.3))
    monkeypatch.setattr(s3_v3, "S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS", 0.3)
    for bucket in buckets:
        if failure == "timeout":
            bucket.delays.append(("_a\\.json", 1.0))
        else:
            bucket.failures.append(("_a\\.json", 0, ""))
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        for response_id in ("a", "b"):
            await _log(python_logger, rust_logger, _payload(response_id))
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    python_queue, rust_queue = _queues(python_logger, rust_logger)
    assert rust_queue == python_queue
    assert [key.rsplit("_", 1)[-1] for key in rust_queue] == ["a.json"]
    assert python_failures.call_count == rust_failures.call_count == 1
    attempts: Final = 3 if failure == "timeout" else 6
    for bucket in buckets:
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
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, metadata: object, puts: int
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    payload: Final = _payload()
    if metadata is _NO_METADATA:
        del payload["metadata"]
    else:
        payload["metadata"] = metadata
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log(python_logger, rust_logger, payload)
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    _assert_same_puts(buckets, puts)
    assert python_failures.call_count == rust_failures.call_count == 1 - puts
    assert rust_logger.sink.stashed() == 0


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
    buckets: tuple[FakeBucket, FakeBucket],
    monkeypatch: pytest.MonkeyPatch,
    start_time: datetime,
    payload_start: float | None,
    logging_start: datetime,
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_partition_granularity": "hour"})
    payload: Final = _payload()
    if payload_start is None:
        del payload["startTime"]
    else:
        payload["startTime"] = payload_start
    await python_logger.async_log_success_event({"standard_logging_object": payload}, None, start_time, start_time)
    await _log_native(
        rust_logger, {"standard_logging_object": copy.deepcopy(payload), "start_time": logging_start}, start_time
    )
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)
    stamp: Final = start_time.strftime("%Y-%m-%d/%H/time-%H-%M-%S-%f")
    assert _put_paths(buckets[1]) == [f"/logs/{stamp}_chatcmpl-1.json"]


@pytest.mark.asyncio
async def test_cold_storage_reads_without_a_region_send_nothing_like_python(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_region_name": None})
    for bucket in buckets:
        bucket.objects["/logs/k.json"] = b'{"stored": 1}'
    python_object: Final = await python_logger.get_proxy_server_request_from_cold_storage_with_object_key("k.json")
    rust_object: Final = await rust_logger.get_proxy_server_request_from_cold_storage_with_object_key("k.json")
    assert rust_object is python_object is None
    assert buckets[0].received == buckets[1].received == []


@pytest.mark.asyncio
async def test_params_json_cannot_hold_still_build_the_native_logger(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    from botocore.config import Config

    python_logger, rust_logger = await _loggers(
        monkeypatch, buckets, {"s3_config": Config(retries={"max_attempts": 1}), "s3_max_concurrent_uploads": 10**30}
    )
    await _log(python_logger, rust_logger, _payload())
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)


class _AliasLookupFails(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:
        if key == "user_api_key_team_alias":
            raise RuntimeError("the alias lookup failed")
        return super().get(key, default)


@pytest.mark.asyncio
async def test_the_attempt_ended_hook_only_enqueues_while_s3_is_slow(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(s3_v3, "DEFAULT_S3_BATCH_SIZE", 1)
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {})
    for bucket in buckets:
        bucket.delays.append((".", 3.0))
    started: Final = time.monotonic()
    await _log_native(rust_logger, {"standard_logging_object": _payload()})
    hook_seconds: Final = time.monotonic() - started
    monkeypatch.setattr(litellm, "callbacks", [python_logger, rust_logger])
    await litellm.acompletion(model="gpt-4o", messages=[{"role": "user", "content": "hello"}], mock_response="hi there")
    for _ in range(100):
        if python_logger.log_queue and len(rust_logger.sink.queued_keys()) == 2:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    assert hook_seconds < 1.0
    assert len(python_logger.log_queue) == 1
    assert len(rust_logger.sink.queued_keys()) == 2
    assert buckets[0].puts() == buckets[1].puts() == []


@pytest.mark.asyncio
async def test_a_payload_whose_alias_lookup_fails_is_counted_and_released(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_use_team_prefix": True})
    with (
        patch.object(S3Logger, "handle_callback_failure") as python_failures,
        patch.object(s3_v3._Host, "callback_failed") as rust_failures,
    ):
        await _log(python_logger, rust_logger, _payload(metadata=_AliasLookupFails(team="t")))
        await _flush(python_logger, rust_logger)
        await asyncio.sleep(0.2)
    _assert_same_puts(buckets, 0)
    assert python_failures.call_count == rust_failures.call_count == 1
    assert rust_logger.sink.stashed() == 0


@pytest.mark.asyncio
async def test_prompts_only_adds_a_null_response_to_a_payload_without_one_like_python(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_log_prompts_only": True})
    payload: Final = {key: value for key, value in _payload().items() if key != "response"}
    await _log(python_logger, rust_logger, payload)
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)
    assert list(json.loads(buckets[1].puts()[0].body))[-1] == "response"


@pytest.mark.asyncio
async def test_the_dispatched_native_strips_the_messages_before_the_payload_is_copied(
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.litellm_core_utils import litellm_logging

    python_logger, rust_logger = await _loggers(monkeypatch, buckets, {"s3_strip_base64_files": True})
    payload: Final = _payload(
        "stripped",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "see data:image/png;base64,AAAA== inline"}]},
            "not a message",
        ],
    )
    kwargs: Final = {"standard_logging_object": copy.deepcopy(payload)}
    await python_logger.async_log_success_event({"standard_logging_object": payload}, None, START, START)
    await litellm_logging._run_callbacks(
        Hook.attempt_ended,
        [rust_logger],
        kwargs,
        lambda _callback, _kwargs: None,
        args=HookArgs(request=kwargs, start_time=START, end_time=START, outcome=Outcome.success),
    )
    await _until_queued(python_logger, rust_logger)
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)
    assert json.loads(buckets[1].puts()[0].body)["messages"] == [
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
    buckets: tuple[FakeBucket, FakeBucket], monkeypatch: pytest.MonkeyPatch, alias: object
) -> None:
    python_logger, rust_logger = await _loggers(
        monkeypatch, buckets, {"s3_use_team_prefix": True, "s3_use_key_prefix": True}
    )
    await _log(python_logger, rust_logger, _payload(metadata={"user_api_key_team_alias": alias}))
    await _flush(python_logger, rust_logger)
    _assert_same_puts(buckets, 1)
    assert _put_paths(buckets[1]) == ["/logs/2026-11-13/time-09-05-07-000042_chatcmpl-1.json"]
