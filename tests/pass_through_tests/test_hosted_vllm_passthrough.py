import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from litellm.passthrough.main import allm_passthrough_route
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders
from litellm.llms.vllm.passthrough.transformation import (
    VLLMPassthroughConfig,
)


def test_get_provider_passthrough_config_for_hosted_vllm_returns_vllm_config():
    # When requesting passthrough config for HOSTED_VLLM
    cfg = ProviderConfigManager.get_provider_passthrough_config(
        model="hosted_vllm/my-deployment",
        provider=LlmProviders.HOSTED_VLLM,
    )

    # Then we should get a VLLMPassthroughConfig instance
    assert isinstance(cfg, VLLMPassthroughConfig)


@pytest.mark.asyncio
async def test_allm_passthrough_route_with_hosted_vllm_model_does_not_raise():
    # Given a hosted_vllm model and an async http client
    client = AsyncHTTPHandler()

    # Mock the provider resolution to ensure we use hosted_vllm and provide api_base
    with patch(
        "litellm.passthrough.main.get_llm_provider",
        return_value=(
            "my-deployment",  # normalized model name
            "hosted_vllm",  # provider
            "fake-api-key",  # api key (not required for vllm)
            "http://localhost:8090",  # api base
        ),
    ):
        # Mock the underlying AsyncClient.send to avoid real network I/O
        fake_request = httpx.Request(
            method="POST", url="http://localhost:8090/v1/chat/completions"
        )
        fake_response = httpx.Response(
            status_code=200,
            content=b"{\n  \"ok\": true\n}",
            request=fake_request,
            headers={"content-type": "application/json"},
        )

        with patch.object(
            client.client, "send", new=AsyncMock(return_value=fake_response)
        ):
            # When calling the async passthrough route with a hosted_vllm/* model
            response = await allm_passthrough_route(
                method="POST",
                endpoint="v1/chat/completions",
                model="hosted_vllm/my-deployment",
                api_base="http://localhost:8090",
                json={
                    "model": "anything",  # will be replaced internally with normalized model
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                client=client,
            )

            # Then it should not raise and return a successful httpx.Response
            assert isinstance(response, httpx.Response)
            assert response.status_code == 200
