import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Final

import google.auth
import google.auth.credentials
import google.auth.transport
import httpx
import pytest

from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


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
    "api_base": "https://exampleopenaiendpoint.example.invalid",
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
        "api_base": "https://exampleopenaiendpoint.example.invalid/",
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
    assert payload is not None
    assert payload["id"] == request_id


@pytest.mark.asyncio
async def test_get_payload_next_day(monkeypatch):
    gcs_logger: Final = _gcs_logger_storing_payload_on("2024-01-02", monkeypatch)
    start_time: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id: Final = mock_response_data["id"]

    payload: Final = await gcs_logger.get_request_response_payload(request_id, start_time, None)
    assert payload is not None
    assert payload["id"] == request_id


@pytest.mark.asyncio
async def test_get_payload_previous_day(monkeypatch):
    gcs_logger: Final = _gcs_logger_storing_payload_on("2023-12-31", monkeypatch)
    start_time: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id: Final = mock_response_data["id"]

    payload: Final = await gcs_logger.get_request_response_payload(request_id, start_time, None)
    assert payload is not None
    assert payload["id"] == request_id


@pytest.mark.asyncio
async def test_get_payload_not_found(monkeypatch):
    gcs_logger: Final = _gcs_logger_storing_payload_on(None, monkeypatch)
    start_time: Final = datetime(2024, 1, 1, tzinfo=timezone.utc)
    request_id: Final = mock_response_data["id"]

    payload: Final = await gcs_logger.get_request_response_payload(request_id, start_time, None)
    assert payload is None
