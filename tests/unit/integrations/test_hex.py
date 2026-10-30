import json
from datetime import datetime, timezone
from typing import Final

import pytest
import httpx

import litellm
from litellm.integrations.hex import HexLogger
from litellm.integrations.s3_v2 import S3Logger
from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry
from litellm.litellm_core_utils.litellm_logging import (
    build_custom_logger_compatible_class,
    create_dummy_standard_logging_payload,
    get_custom_logger_compatible_class,
)
from litellm.proxy.management_endpoints.callback_management_endpoints import get_callback_configs
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import StandardLoggingPayload

START: Final = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def spend_record() -> StandardLoggingPayload:
    payload: Final = create_dummy_standard_logging_payload()
    payload.update(
        id="request-1",
        startTime=START.timestamp(),
        endTime=START.timestamp() + 1,
        response_cost=0.125,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        response_time=1.0,
        messages=[{"role": "user", "content": "private prompt"}],
        response="private response",
        error_str="private provider error",
    )
    payload["metadata"]["requester_metadata"] = {"private": "must not be exported"}
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["success", "failure"])
async def test_hex_exports_spend_rows_without_request_content(
    spend_record: StandardLoggingPayload, status: str
) -> None:
    logger: Final = HexLogger(bucket_name="hex-analytics")
    if status == "failure":
        spend_record["status"] = "failure"
        await logger.async_log_failure_event({"standard_logging_object": spend_record}, None, START, START)
    else:
        await logger.async_log_success_event({"standard_logging_object": spend_record}, None, START, START)

    assert len(logger.log_queue) == 1
    element: Final = logger.log_queue[0]
    assert element.payload == {
        "id": "request-1",
        "start_time": START.timestamp(),
        "end_time": START.timestamp() + 1,
        "model": "gpt-3.5-turbo",
        "model_group": "openai-gpt",
        "custom_llm_provider": "openai",
        "call_type": "completion",
        "status": status,
        "spend": 0.125,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "response_time": 1.0,
        "cache_hit": False,
        "team_id": "test_team",
        "organization_id": None,
        "user_id": "test_user",
        "key_alias": "test_alias",
    }
    assert element.s3_object_key == "litellm-spend/2026-10-30/time-12-00-00-000000_request-1.json"
    assert "private" not in logger._prepare_put(element).json_string
    assert spend_record["messages"] == [{"role": "user", "content": "private prompt"}]


@pytest.mark.asyncio
async def test_hex_skips_events_without_a_spend_payload() -> None:
    logger: Final = HexLogger(bucket_name="hex-analytics")
    await logger.async_log_success_event({}, None, START, START)
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_hex_retains_a_rejected_spend_record_until_the_next_flush(
    monkeypatch: pytest.MonkeyPatch, spend_record: StandardLoggingPayload
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-signing-key")
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(400 if len(requests) == 1 else 200)

    logger: Final = HexLogger(bucket_name="hex-analytics", endpoint_url="https://s3.example.test")
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    logger.async_httpx_client = handler
    try:
        await logger.async_log_success_event({"standard_logging_object": spend_record}, None, START, START)
        await logger.flush_queue()
        assert len(logger.log_queue) == 1
        await logger.flush_queue()
        assert logger.log_queue == []
        assert len(requests) == 2
        assert requests[0].content == requests[1].content
        assert str(requests[1].url) == (
            "https://s3.example.test/hex-analytics/litellm-spend/2026-10-30/time-12-00-00-000000_request-1.json"
        )
        assert json.loads(requests[1].content)["spend"] == 0.125
        assert b"private" not in requests[1].content
    finally:
        await handler.client.aclose()


def test_hex_requires_a_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEX_S3_BUCKET_NAME", raising=False)
    with pytest.raises(ValueError, match="HEX_S3_BUCKET_NAME"):
        HexLogger()


@pytest.mark.asyncio
async def test_hex_uses_its_own_destination_and_reuses_the_registered_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEX_S3_BUCKET_NAME", "hex-analytics")
    monkeypatch.setenv("HEX_S3_PATH", "analytics")
    monkeypatch.setenv("HEX_S3_REGION_NAME", "eu-west-1")
    monkeypatch.setattr(litellm, "s3_callback_params", {"s3_bucket_name": "other-bucket", "s3_path": "other-path"})

    hex_logger: Final = build_custom_logger_compatible_class("hex", None, None)
    s3_logger: Final = build_custom_logger_compatible_class("s3_v2", None, None)
    assert isinstance(hex_logger, HexLogger)
    assert type(s3_logger) is S3Logger
    assert (hex_logger.s3_bucket_name, hex_logger.s3_path, hex_logger.s3_region_name) == (
        "hex-analytics",
        "analytics",
        "eu-west-1",
    )
    assert s3_logger.s3_bucket_name == "other-bucket"
    assert build_custom_logger_compatible_class("hex", None, None) is hex_logger
    assert get_custom_logger_compatible_class("hex") is hex_logger
    assert get_custom_logger_compatible_class("s3_v2") is s3_logger
    assert CustomLoggerRegistry.get_callback_str_from_class_type(HexLogger) == "hex"


@pytest.mark.asyncio
async def test_hex_settings_list_the_required_bucket() -> None:
    configs: Final = await get_callback_configs()
    assert isinstance(configs, list)
    config: Final = next(item for item in configs if isinstance(item, dict) and item.get("id") == "hex")
    assert config["displayName"] == "Hex"
    assert config["supports_key_team_logging"] is False
    assert config["dynamic_params"]["HEX_S3_BUCKET_NAME"]["required"] is True
    assert "API_KEY" not in json.dumps(config)
