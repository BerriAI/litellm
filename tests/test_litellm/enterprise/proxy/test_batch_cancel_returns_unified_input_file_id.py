import base64
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Response
from starlette.requests import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import LiteLLMBatch


def _make_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/batches/test/cancel",
            "headers": [],
            "query_string": b"",
        }
    )


def _make_unified_batch_id(raw_batch_id: str, model_id: str) -> str:
    decoded = f"litellm_proxy;model_id:{model_id};llm_batch_id:{raw_batch_id}"
    return base64.urlsafe_b64encode(decoded.encode()).decode().rstrip("=")


@pytest.mark.asyncio
async def test_cancel_batch_returns_unified_input_file_id_for_managed_batch():
    from litellm.proxy.batches_endpoints.endpoints import cancel_batch

    raw_batch_id = "batch-raw-123"
    raw_input_file_id = "file-input-raw-123"
    model_id = "model-deploy-xyz"
    unified_batch_id = _make_unified_batch_id(
        raw_batch_id=raw_batch_id, model_id=model_id
    )
    unified_input_file_id = "bGl0ZWxsbV9wcm94eTp1bmlmaWVkLWlucHV0LWlk"

    response = LiteLLMBatch(
        id=raw_batch_id,
        completion_window="24h",
        created_at=1700000000,
        endpoint="/v1/chat/completions",
        input_file_id=raw_input_file_id,
        object="batch",
        status="cancelling",
        output_file_id=None,
    )
    response._hidden_params = {}

    mock_router = MagicMock()
    mock_router.acancel_batch = AsyncMock(return_value=response)

    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.get_proxy_hook.return_value = MagicMock()
    mock_proxy_logging_obj.post_call_success_hook = AsyncMock(return_value=response)
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock()
    mock_proxy_logging_obj.update_request_status = AsyncMock()

    managed_file_record = MagicMock()
    managed_file_record.unified_file_id = unified_input_file_id

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_managedfiletable.find_first = AsyncMock(
        return_value=managed_file_record
    )

    processed_data = {"batch_id": raw_batch_id, "model": model_id}
    fake_proxy_server = ModuleType("litellm.proxy.proxy_server")
    fake_proxy_server.add_litellm_data_to_request = AsyncMock(
        side_effect=lambda data, **_: data
    )
    fake_proxy_server.general_settings = {}
    fake_proxy_server.llm_router = mock_router
    fake_proxy_server.prisma_client = mock_prisma
    fake_proxy_server.proxy_config = MagicMock()
    fake_proxy_server.proxy_logging_obj = mock_proxy_logging_obj
    fake_proxy_server.version = "test-version"

    with patch(
        "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing.common_processing_pre_call_logic",
        AsyncMock(return_value=(processed_data, MagicMock())),
    ), patch(
        "litellm.proxy.batches_endpoints.endpoints.ProxyBaseLLMRequestProcessing.get_custom_headers",
        return_value={},
    ), patch(
        "litellm.proxy.batches_endpoints.endpoints.update_batch_in_database",
        AsyncMock(),
    ), patch.dict(
        sys.modules, {"litellm.proxy.proxy_server": fake_proxy_server}
    ):
        result = await cancel_batch(
            request=_make_request(),
            batch_id=unified_batch_id,
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="sk-test",
                user_id="test-user",
                parent_otel_span=None,
            ),
        )

    assert result.input_file_id == unified_input_file_id
    mock_prisma.db.litellm_managedfiletable.find_first.assert_called_once_with(
        where={"flat_model_file_ids": {"has": raw_input_file_id}}
    )
