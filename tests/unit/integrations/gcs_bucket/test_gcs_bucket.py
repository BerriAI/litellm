import asyncio
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Final
from unittest.mock import Mock

import google.auth
import google.auth.credentials
import google.auth.transport
import httpx
import litellm
import pytest
import respx

from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


mock_response_data: Final = {
    "id": "chatcmpl-9870a859d6df402795f75dc5fca5b2e0",
    "trace_id": None,
    "call_type": "acompletion",
    "cache_hit": None,
    "stream": True,
    "status": "success",
    "custom_llm_provider": "openai",
    "saved_cache_cost": 0.0,
    "startTime": 1739235379.683053,
    "endTime": 1739235379.84533,
    "completionStartTime": 1739235379.84533,
    "response_time": 0.1622769832611084,
    "model": "my-fake-model",
    "metadata": {
        "user_api_key_hash": "sk-test-mock-api-key-123",
        "user_api_key_alias": None,
        "user_api_key_team_id": None,
        "user_api_key_org_id": None,
        "user_api_key_user_id": "default_user_id",
        "user_api_key_team_alias": None,
        "spend_logs_metadata": None,
        "requester_ip_address": "127.0.0.1",
        "requester_metadata": {},
        "user_api_key_end_user_id": None,
        "prompt_management_metadata": None,
    },
    "cache_key": None,
    "response_cost": 3.7500000000000003e-05,
    "total_tokens": 21,
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "request_tags": [],
    "end_user": "",
    "api_base": "https://exampleopenaiendpoint-production.up.railway.app",
    "model_group": "fake-openai-endpoint",
    "model_id": "b68d56d76b0c24ac9462ab69541e90886342508212210116e300441155f37865",
    "requester_ip_address": "127.0.0.1",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "very gm to u"}]}],
    "response": {
        "id": "chatcmpl-9870a859d6df402795f75dc5fca5b2e0",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0301",
        "object": "chat.completion",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "\n\nHello there, how may I assist you today?",
                    "role": "assistant",
                    "tool_calls": None,
                    "function_call": None,
                    "refusal": None,
                },
            }
        ],
        "usage": {
            "completion_tokens": 12,
            "prompt_tokens": 9,
            "total_tokens": 21,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
        "service_tier": None,
    },
    "model_parameters": {"stream": False, "max_retries": 0, "extra_body": {}},
    "hidden_params": {
        "model_id": "b68d56d76b0c24ac9462ab69541e90886342508212210116e300441155f37865",
        "cache_key": None,
        "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
        "response_cost": 3.7500000000000003e-05,
        "additional_headers": {},
        "litellm_overhead_time_ms": 2.126,
    },
    "model_map_information": {
        "model_map_key": "gpt-3.5-turbo-0301",
        "model_map_value": {},
    },
    "error_str": None,
    "error_information": {"error_code": "", "error_class": "", "llm_provider": ""},
    "response_cost_failure_debug_info": None,
    "guardrail_information": None,
}


class _StaticGoogleCredentials(google.auth.credentials.Credentials):
    def refresh(self, request: google.auth.transport.Request) -> None:
        self.token = "test-access-token"


def _google_default_credentials(scopes: Sequence[str]) -> tuple[_StaticGoogleCredentials, str]:
    return _StaticGoogleCredentials(), "test-project"


def _gcs_logger_storing_payload_on(stored_date: str | None, monkeypatch: pytest.MonkeyPatch) -> GCSBucketLogger:
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr(google.auth, "default", _google_default_credentials)

    def storage(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Authorization") != "Bearer test-access-token":
            return httpx.Response(401)
        if not str(request.url).startswith("https://storage.googleapis.com/storage/v1/b/test-bucket/o/"):
            return httpx.Response(404)
        if stored_date is None or stored_date not in str(request.url):
            return httpx.Response(404, text="No such object")
        return httpx.Response(200, content=json.dumps(mock_response_data).encode("utf-8"))

    gcs_logger: Final = GCSBucketLogger(bucket_name="test-bucket")
    gcs_logger.async_httpx_client = AsyncHTTPHandler(transport=httpx.MockTransport(storage))
    return gcs_logger


@pytest.mark.asyncio
async def test_get_payload_current_day(monkeypatch):
    gcs_logger: Final = _gcs_logger_storing_payload_on("2024-01-01", monkeypatch)
    start_time: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id: Final = mock_response_data["id"]

    payload: Final = await gcs_logger.get_request_response_payload(request_id, start_time, None)
    assert payload == mock_response_data


@pytest.mark.asyncio
async def test_get_payload_next_day(monkeypatch):
    gcs_logger: Final = _gcs_logger_storing_payload_on("2024-01-02", monkeypatch)
    start_time: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id: Final = mock_response_data["id"]

    payload: Final = await gcs_logger.get_request_response_payload(request_id, start_time, None)
    assert payload == mock_response_data


@pytest.mark.asyncio
async def test_get_payload_previous_day(monkeypatch):
    gcs_logger: Final = _gcs_logger_storing_payload_on("2023-12-31", monkeypatch)
    start_time: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id: Final = mock_response_data["id"]

    payload: Final = await gcs_logger.get_request_response_payload(request_id, start_time, None)
    assert payload == mock_response_data


@pytest.mark.asyncio
async def test_get_payload_not_found(monkeypatch):
    gcs_logger: Final = _gcs_logger_storing_payload_on(None, monkeypatch)
    start_time: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id: Final = mock_response_data["id"]

    payload: Final = await gcs_logger.get_request_response_payload(request_id, start_time, None)
    assert payload is None


def _gcs_logger_with_upload_status(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> tuple[GCSBucketLogger, Mock]:
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr(google.auth, "default", _google_default_credentials)
    monkeypatch.setenv("GCS_FLUSH_INTERVAL", "3600")
    uploads: Final = Mock()

    def upload(request: httpx.Request) -> httpx.Response:
        uploads(request=request)
        return httpx.Response(status_code)

    logger: Final = GCSBucketLogger(bucket_name="test-bucket")
    logger.async_httpx_client = AsyncHTTPHandler(transport=httpx.MockTransport(upload))
    return logger, uploads


@pytest.mark.asyncio
async def test_aaabasic_gcs_logger_stores_exact_success_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    logger, uploads = _gcs_logger_with_upload_status(monkeypatch, 200)
    messages: Final = [{"role": "user", "content": "gcs payload marker"}]
    payload: Final = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "response": {
            "choices": [{"message": {"role": "assistant", "content": "GCS response text"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "metadata": {"requester_metadata": {"marker": "gcs-payload"}},
        "error_str": None,
        "id": "gcs-success",
    }
    await logger.async_log_success_event(
        {"standard_logging_object": payload, "litellm_params": {"metadata": payload["metadata"]}},
        payload["response"],
        None,
        None,
    )
    await GLOBAL_LOGGING_WORKER.flush()
    await logger.flush_queue()

    request: Final = uploads.call_args.kwargs["request"]
    payload: Final = json.loads(request.content.decode().splitlines()[0])
    assert request.url.host == "storage.googleapis.com"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == messages
    assert payload["response"]["choices"][0]["message"]["content"] == "GCS response text"
    assert payload["prompt_tokens"] == 2
    assert payload["completion_tokens"] == 3
    assert payload["metadata"]["requester_metadata"] == {"marker": "gcs-payload"}


@pytest.mark.asyncio
async def test_basic_gcs_logger_failure_stores_error_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    logger, uploads = _gcs_logger_with_upload_status(monkeypatch, 200)
    messages: Final = [{"role": "user", "content": "failed gcs request"}]
    failure_payload: Final = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "response": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error_str": "provider failure",
        "id": "gcs-failure",
    }
    await logger.async_log_failure_event(
        {"standard_logging_object": failure_payload},
        {},
        None,
        None,
    )
    await GLOBAL_LOGGING_WORKER.flush()
    await logger.flush_queue()

    request: Final = uploads.call_args.kwargs["request"]
    payload: Final = json.loads(request.content.decode().splitlines()[0])
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == messages
    assert payload["error_str"] == "provider failure"
    assert payload["response"] is None


@pytest.mark.asyncio
async def test_unbatched_gcs_logs_upload_individual_success_and_failure_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr(google.auth, "default", _google_default_credentials)
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    monkeypatch.setenv("GCS_USE_BATCHED_LOGGING", "false")
    monkeypatch.setenv("GCS_FLUSH_INTERVAL", "3600")
    uploaded_requests: Final[asyncio.Queue[httpx.Request]] = asyncio.Queue()

    def capture_upload(request: httpx.Request) -> httpx.Response:
        uploaded_requests.put_nowait(request)
        return httpx.Response(200)

    with respx.mock(base_url="https://storage.googleapis.com", assert_all_called=False) as router:
        upload_route: Final = router.post("/upload/storage/v1/b/test-bucket/o").mock(side_effect=capture_upload)
        logger: Final = GCSBucketLogger(bucket_name="test-bucket")
        logger.async_httpx_client = AsyncHTTPHandler()
        success_payload: Final = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "success-marker"}],
            "response": {"id": "gcs-success", "choices": [{"message": {"content": "stored response"}}]},
            "error_str": None,
            "id": "gcs-success",
        }
        failure_payload: Final = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "failure-marker"}],
            "response": None,
            "error_str": "provider failure",
            "id": "gcs-failure",
        }

        await logger.async_log_success_event(
            {"standard_logging_object": success_payload, "litellm_params": {"metadata": {}}},
            success_payload["response"],
            None,
            None,
        )
        await logger.async_log_failure_event(
            {"standard_logging_object": failure_payload},
            {},
            None,
            None,
        )
        logger.flush_interval = 0
        flush_task: Final = asyncio.create_task(logger.periodic_flush())
        try:
            await asyncio.wait_for(uploaded_requests.get(), timeout=5)
            await asyncio.wait_for(uploaded_requests.get(), timeout=5)

            assert upload_route.call_count == 2
            success_request: Final = upload_route.calls[0].request
            failure_request: Final = upload_route.calls[1].request
            success_object_name: Final = success_request.url.params["name"]
            failure_object_name: Final = failure_request.url.params["name"]
            success_date, success_id = success_object_name.split("/", maxsplit=1)
            failure_date, failure_id = failure_object_name.split("/", maxsplit=1)
            failure_uuid: Final = failure_id.removeprefix("failure-")

            assert datetime.strptime(success_date, "%Y-%m-%d").strftime("%Y-%m-%d") == success_date
            assert success_object_name == f"{success_date}/gcs-success"
            assert success_id == "gcs-success"
            assert failure_date == success_date
            assert len(failure_uuid) == 32
            assert all(character in "0123456789abcdef" for character in failure_uuid)
            assert failure_object_name == f"{success_date}/failure-{failure_uuid}"
            assert json.loads(success_request.content) == success_payload
            assert json.loads(failure_request.content) == failure_payload
        finally:
            flush_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await flush_task
