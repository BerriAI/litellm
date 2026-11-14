import asyncio
import json
import re
import sys
import textwrap
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.integrations.s3_v2 import S3Logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.integrations.s3_v2 import s3BatchLoggingElement
from collections import defaultdict


def _assert_jittered_backoff(sleep_mock: MagicMock, retries: int) -> None:
    """Each retry waits a jittered exponential backoff: attempt n sleeps somewhere in [2**n / 2, 2**n * 1.5)."""
    waits: Final = [sleep_call.args[0] for sleep_call in sleep_mock.call_args_list]
    assert len(waits) == retries, waits
    for attempt, wait in enumerate(waits):
        assert (1 << attempt) * 0.5 <= wait < (1 << attempt) * 1.5, (attempt, wait)


_SIGV4_ACCESS_KEY = re.compile(r"Credential=(AKIA\d+)/")


@pytest.fixture
def rotating_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """
    A real botocore profile whose credential_process hands out a new key generation on every call and
    expires inside the advisory refresh window, so RefreshableCredentials re-runs it on every property read.
    """
    counter = tmp_path / "generation"
    script = tmp_path / "rotate_credentials.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, sys
            from datetime import datetime, timedelta, timezone
            from pathlib import Path

            counter = Path({str(counter)!r})
            generation = int(counter.read_text()) if counter.exists() else 0
            counter.write_text(str(generation + 1))
            expiry = (datetime.now(timezone.utc) + timedelta(minutes=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
            json.dump(
                {{
                    "Version": 1,
                    "AccessKeyId": f"AKIA{{generation}}",
                    "SecretAccessKey": f"secret-{{generation}}",
                    "SessionToken": f"token-{{generation}}",
                    "Expiration": expiry,
                }},
                sys.stdout,
            )
            """
        )
    )
    profile = f"rotating-{uuid.uuid4().hex}"
    (tmp_path / "config").write_text(f"[profile {profile}]\ncredential_process = {sys.executable} {script}\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    return profile


def _generation(request: httpx.Request) -> tuple[str, str]:
    """(access key generation, session token generation) SigV4 baked into one request."""
    access_key = _SIGV4_ACCESS_KEY.search(request.headers["Authorization"])
    assert access_key is not None
    return access_key.group(1).removeprefix("AKIA"), request.headers["X-Amz-Security-Token"].removeprefix("token-")


@asynccontextmanager
async def _s3_logger_on_production_handler(profile: str, statuses: list[int | type[httpx.TransportError]]):
    """
    S3Logger wired to the real AsyncHTTPHandler over an httpx MockTransport that answers with the given
    statuses (or raises the given transport errors) in order, so the handler's own error mapping is exercised.
    """
    requests: list[httpx.Request] = []
    replies = iter(statuses)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = next(replies)
        if not isinstance(reply, int):
            raise reply("S3 attempt failed", request=request)
        return httpx.Response(reply, request=request, text="<Error><Code>SignatureDoesNotMatch</Code></Error>")

    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    logger = S3Logger(
        s3_bucket_name="test-bucket",
        s3_region_name="us-east-1",
        s3_aws_profile_name=profile,
        s3_flush_interval=3600,
    )
    logger.async_httpx_client = handler
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        yield logger, requests, mock_sleep
    await handler.client.aclose()


@pytest.mark.asyncio
async def test_async_upload_signs_with_one_frozen_credential_snapshot(rotating_profile: str, caplog):
    """
    RefreshableCredentials refreshes on every property read once inside the advisory window, so signing
    off the live object would mix the access key of one generation with the token of the next.
    """
    test_element = s3BatchLoggingElement(
        s3_object_key="2025-09-14/test-frozen.json",
        payload={"test": "frozen"},
        s3_object_download_filename="test-frozen.json",
    )
    async with _s3_logger_on_production_handler(rotating_profile, [200]) as (logger, requests, _):
        await logger.async_upload_data_to_s3(test_element)

    assert len(requests) == 1
    access_key_generation, token_generation = _generation(requests[0])
    assert access_key_generation == token_generation
    assert "Error uploading to s3" not in caplog.text


@pytest.mark.asyncio
async def test_async_upload_retries_403_with_fresh_credentials_and_signature(rotating_profile: str, caplog):
    """
    A 403 (SignatureDoesNotMatch after an IMDS rotation) must be retried, and the retry must fetch
    credentials again and carry a signature computed from that newer generation.
    """
    test_element = s3BatchLoggingElement(
        s3_object_key="2025-09-14/test-403.json",
        payload={"test": "403"},
        s3_object_download_filename="test-403.json",
    )
    async with _s3_logger_on_production_handler(rotating_profile, [403, 200]) as (logger, requests, mock_sleep):
        await logger.async_upload_data_to_s3(test_element)

    assert len(requests) == 2
    first_key, first_token = _generation(requests[0])
    second_key, second_token = _generation(requests[1])
    assert first_key == first_token
    assert second_key == second_token
    assert int(second_key) > int(first_key)
    assert requests[1].headers["Authorization"] != requests[0].headers["Authorization"]
    _assert_jittered_backoff(mock_sleep, 1)
    assert "Error uploading to s3" not in caplog.text


@pytest.mark.asyncio
async def test_async_upload_exhausts_403_retries_through_production_http_handler(rotating_profile: str, caplog):
    test_element = s3BatchLoggingElement(
        s3_object_key="2025-09-14/test-403-exhausted.json",
        payload={"test": "403-exhausted"},
        s3_object_download_filename="test-403-exhausted.json",
    )
    async with _s3_logger_on_production_handler(rotating_profile, [403, 403, 403]) as (logger, requests, mock_sleep):
        await logger.async_upload_data_to_s3(test_element)

    assert len(requests) == 3
    _assert_jittered_backoff(mock_sleep, 2)
    assert "Error uploading to s3" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("transient_status", [500, 503])
async def test_async_upload_recovers_from_transient_5xx_through_production_http_handler(
    transient_status: int, rotating_profile: str, caplog: pytest.LogCaptureFixture
):
    """
    AsyncHTTPHandler.put raises MaskedHTTPStatusError on 5xx instead of returning the response, so a retry
    loop that only inspects returned status codes never runs (#42868).
    """
    test_element = s3BatchLoggingElement(
        s3_object_key=f"2025-09-14/test-{transient_status}.json",
        payload={"test": str(transient_status)},
        s3_object_download_filename=f"test-{transient_status}.json",
    )
    async with _s3_logger_on_production_handler(rotating_profile, [transient_status, 200]) as (
        logger,
        requests,
        mock_sleep,
    ):
        uploaded = await logger.async_upload_data_to_s3(test_element)

    assert uploaded is True
    assert len(requests) == 2
    assert all(request.method == "PUT" for request in requests)
    assert requests[0].url == requests[1].url
    assert requests[0].content == requests[1].content
    _assert_jittered_backoff(mock_sleep, 1)
    assert "Error uploading to s3" not in caplog.text


@pytest.mark.asyncio
async def test_async_upload_is_single_attempted_on_404_through_production_http_handler(rotating_profile: str, caplog):
    test_element = s3BatchLoggingElement(
        s3_object_key="2025-09-14/test-404.json",
        payload={"test": "404"},
        s3_object_download_filename="test-404.json",
    )
    async with _s3_logger_on_production_handler(rotating_profile, [404]) as (logger, requests, mock_sleep):
        outcome = await logger.async_upload_data_to_s3(test_element)

    assert len(requests) == 1
    assert outcome is False
    mock_sleep.assert_not_awaited()
    assert "Error uploading to s3" in caplog.text


@pytest.mark.asyncio
async def test_async_upload_access_denied_403_is_retried_and_then_requeued(rotating_profile: str, caplog):
    test_element = s3BatchLoggingElement(
        s3_object_key="2025-09-14/test-403-denied.json",
        payload={"test": "403-denied"},
        s3_object_download_filename="test-403-denied.json",
    )
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(403, request=request, text="<Error><Code>AccessDenied</Code></Error>")

    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    logger = S3Logger(
        s3_bucket_name="test-bucket",
        s3_region_name="us-east-1",
        s3_aws_profile_name=rotating_profile,
        s3_flush_interval=3600,
    )
    logger.async_httpx_client = handler
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        outcome = await logger.async_upload_data_to_s3(test_element)
    await handler.client.aclose()

    assert outcome is False
    assert len(requests) == 3
    _assert_jittered_backoff(mock_sleep, 2)
    assert "Error uploading to s3" in caplog.text


@pytest.mark.asyncio
async def test_async_upload_retries_429_through_production_http_handler(rotating_profile: str, caplog):
    """S3 and S3-compatible stores answer throttling with 429 too, so it backs off and retries like a 503."""
    async with _s3_logger_on_production_handler(rotating_profile, [429, 200]) as (logger, requests, mock_sleep):
        uploaded = await logger.async_upload_data_to_s3(_element({"test": "429"}, "429"))

    assert uploaded is True
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    _assert_jittered_backoff(mock_sleep, 1)
    assert "Error uploading to s3" not in caplog.text


@pytest.mark.asyncio
async def test_async_upload_retries_a_timed_out_attempt_through_production_http_handler(rotating_profile: str, caplog):
    """AsyncHTTPHandler.put turns an httpx timeout into litellm.Timeout, which must count as a retryable attempt."""
    async with _s3_logger_on_production_handler(rotating_profile, [httpx.ReadTimeout, 200]) as (
        logger,
        requests,
        mock_sleep,
    ):
        uploaded = await logger.async_upload_data_to_s3(_element({"test": "timeout"}, "timeout"))

    assert uploaded is True
    assert len(requests) == 2
    _assert_jittered_backoff(mock_sleep, 1)
    assert "Error uploading to s3" not in caplog.text


@pytest.mark.asyncio
async def test_async_upload_sends_every_attempt_with_a_finite_deadline(rotating_profile: str) -> None:
    """
    The logger's handler is built with explicit params and no timeout, and the handler forwards that None to
    httpx as "no timeout", so without a per-attempt deadline a stalled S3 endpoint pins the upload forever.
    """
    async with _s3_logger_on_production_handler(rotating_profile, [503, 200]) as (logger, requests, _):
        await logger.async_upload_data_to_s3(_element({"test": "deadline"}, "deadline"))

    assert len(requests) == 2
    for request in requests:
        deadlines = request.extensions["timeout"]
        assert all(isinstance(limit, (int, float)) and limit > 0 for limit in deadlines.values()), deadlines


def test_sync_upload_retries_403_with_fresh_signature(rotating_profile: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("AWS_PROFILE", rotating_profile)
    logger = S3Logger(s3_bucket_name="test-bucket", s3_region_name="us-east-1", s3_flush_interval=3600)
    test_element = s3BatchLoggingElement(
        s3_object_key="2025-09-14/test-sync-403.json",
        payload={"test": "sync-403"},
        s3_object_download_filename="test-sync-403.json",
    )
    requests: list[httpx.Request] = []
    replies = iter([403, 200])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(next(replies), request=request, text="<Error><Code>SignatureDoesNotMatch</Code></Error>")

    handler = HTTPHandler()
    handler.client = httpx.Client(transport=httpx.MockTransport(respond))
    with (
        patch(  # test-quality-ok: sync upload builds its HTTPHandler per call, there is no injection seam for it
            "litellm.integrations.s3_v2.get_httpx_client", return_value=handler
        ),
        patch("time.sleep") as mock_sleep,
    ):
        logger.upload_data_to_s3(test_element)

    assert len(requests) == 2
    first_key, first_token = _generation(requests[0])
    second_key, second_token = _generation(requests[1])
    assert first_key == first_token
    assert second_key == second_token
    assert int(second_key) > int(first_key)
    _assert_jittered_backoff(mock_sleep, 1)
            # Either params is None or params={'ssl_verify': None} is acceptable


@pytest.fixture(autouse=True)
def patch_asyncio_create_task(request):
    """Prevent 'no running event loop' errors when S3Logger calls asyncio.create_task()."""
    if request.node.originalname in {
        "test_basic_s3_logging",
        "test_basic_s3_v2_logging",
        "test_basic_s3_v2_logging_failure",
    }:
        yield
        return
    with patch("asyncio.create_task"):
        yield


_PARTITION_START: Final = datetime(2026, 9, 29, 14, 5, 9, 123456)


def _element(payload: dict[str, object], key_suffix: str) -> s3BatchLoggingElement:
    return s3BatchLoggingElement(
        s3_object_key=f"2025-09-14/test-{key_suffix}.json",
        payload=payload,
        s3_object_download_filename=f"test-{key_suffix}.json",
    )

@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm state to the true defaults captured at conftest import time,
    then restores after the test. This prevents module-level mutations (e.g.
    `litellm.num_retries = 3` at the top of test_langfuse_e2e_test.py) from
    leaking across tests within the same xdist worker.
    """
    from litellm.litellm_core_utils import litellm_logging as ll_logging
    from litellm.proxy.management_helpers import audit_logs as ll_audit_logs

    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])
    yield
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    ll_logging._in_memory_loggers.clear()
    ll_audit_logs._audit_log_callback_cache.clear()
    for attr in _LIST_ATTRS:
        if attr in _DEFAULTS:
            default = _DEFAULTS[attr]
            setattr(litellm, attr, default.copy() if isinstance(default, list) else default)
    for attr in _SCALAR_ATTRS:
        if attr in _DEFAULTS:
            setattr(litellm, attr, _DEFAULTS[attr])

_LIST_ATTRS = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "service_callback",
    "pre_call_rules",
    "post_call_rules",
)

_SCALAR_ATTRS = (
    "set_verbose",
    "cache",
    "num_retries",
    "num_retries_per_request",
    "turn_off_message_logging",
    "redact_messages_in_exceptions",
    "redact_user_api_key_info",
    "s3_callback_params",
    "s3_audit_callback_params",
    "datadog_params",
    "vector_store_registry",
)

_DEFAULTS: Final = {
    attr: getattr(litellm, attr).copy() if isinstance(getattr(litellm, attr), list) else getattr(litellm, attr)
    for attr in (*_LIST_ATTRS, *_SCALAR_ATTRS)
    if hasattr(litellm, attr)
}

class _FakeS3Paginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, Bucket):
        keys = sorted(self.objects[Bucket])
        if not keys:
            return [{}]
        return [{"Contents": [{"Key": key} for key in keys]}]

class _FakeS3Client:
    def __init__(self):
        self.objects = defaultdict(dict)
        self.object_written = threading.Event()

    def clear(self):
        self.objects.clear()
        self.object_written.clear()

    def put_object(self, Bucket, Key, Body, **_kwargs):
        self.objects[Bucket][Key] = Body
        self.object_written.set()
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def delete_object(self, Bucket, Key):
        self.objects[Bucket].pop(Key, None)
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakeS3Paginator(self.objects)

    def list_objects(self, Bucket):
        keys = sorted(self.objects[Bucket])
        return {"Contents": [{"Key": key, "LastModified": 0} for key in keys]}

_FAKE_S3_CLIENT = _FakeS3Client()

class TestS3Logger(S3Logger):
    def __init__(self, *args, **kwargs):
        self.recorded_requests = {}
        self.logged_standard_logging_payload = None
        super().__init__(*args, **kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.recorded_requests[response_obj["id"]] = start_time
        self.logged_standard_logging_payload = kwargs["standard_logging_object"]
        return await super().async_log_success_event(kwargs, response_obj, start_time, end_time)
