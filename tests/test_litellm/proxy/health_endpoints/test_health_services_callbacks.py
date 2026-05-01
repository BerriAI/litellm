from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.health_endpoints._health_endpoints import health_services_endpoint
from litellm.proxy._types import ProxyException


@pytest.mark.asyncio
async def test_health_services_endpoint_accepts_s3_callback():
    with (
        patch("litellm.success_callback", ["s3"]),
        patch("litellm.acompletion", new=AsyncMock(return_value={})),
    ):
        result = await health_services_endpoint(service="s3")

    assert result["status"] == "success"
    assert "s3" in result["message"]


@pytest.mark.asyncio
async def test_health_services_endpoint_rejects_unconfigured_s3_callback():
    with patch("litellm.success_callback", []):
        with pytest.raises(ProxyException) as exc_info:
            await health_services_endpoint(service="s3")

    assert exc_info.value.code == "422"
    assert "litellm_settings.success_callback" in str(exc_info.value.message)
