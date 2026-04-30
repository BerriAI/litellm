from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.health_endpoints._health_endpoints import health_services_endpoint


@pytest.mark.asyncio
async def test_health_services_endpoint_accepts_s3_callback():
    with (
        patch("litellm.success_callback", ["s3"]),
        patch("litellm.acompletion", new=AsyncMock(return_value={})),
    ):
        result = await health_services_endpoint(service="s3")

    assert result["status"] == "success"
    assert "s3" in result["message"]
