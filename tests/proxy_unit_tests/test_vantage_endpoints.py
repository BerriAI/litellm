from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.spend_tracking import vantage_endpoints
from litellm.types.proxy.vantage_endpoints import VantageExportRequest


@pytest.mark.asyncio
async def test_vantage_export_preserves_upstream_http_status(monkeypatch):
    request = httpx.Request(
        "POST",
        "https://api.vantage.sh/costs",
        content=b"usage-data",
    )
    response = httpx.Response(
        422,
        request=request,
        content=b'{"error":"ServiceName is required"}',
    )
    upstream_error = httpx.HTTPStatusError(
        "Unprocessable Entity",
        request=request,
        response=response,
    )
    masked_error = MaskedHTTPStatusError(
        upstream_error,
        message='{"error":"ServiceName is required"}',
        text='{"error":"ServiceName is required"}',
    )

    fake_logger = AsyncMock()
    fake_logger.export_usage_data.side_effect = masked_error
    monkeypatch.setattr(
        vantage_endpoints,
        "_get_registered_vantage_logger",
        lambda: fake_logger,
    )

    with pytest.raises(HTTPException) as exc_info:
        await vantage_endpoints.vantage_export(
            request=VantageExportRequest(),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-test",
            ),
        )

    assert exc_info.value.status_code == 422
    assert "ServiceName is required" in exc_info.value.detail["error"]
