import asyncio
import os
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from fastapi.testclient import TestClient

from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo
from litellm.proxy.proxy_server import app, initialize


@pytest.fixture(scope="function")
def fake_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_api_key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://fake-openai-api-base")
    monkeypatch.setenv("AZURE_AI_API_BASE", "http://fake-azure-api-base")
    monkeypatch.setenv("AZURE_AI_API_KEY", "fake_azure_api_key")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake_azure_openai_api_key")
    monkeypatch.setenv("AZURE_SWEDEN_API_BASE", "http://fake-azure-sweden-api-base")
    monkeypatch.setenv("AZURE_SWEDEN_API_KEY", "fake_azure_sweden_api_key")
    monkeypatch.setenv("REDIS_HOST", "localhost")


@pytest.fixture(scope="function")
def client_no_auth(fake_env_vars):
    from litellm.proxy.proxy_server import cleanup_router_config_variables

    original_disable_aiohttp = litellm.disable_aiohttp_transport
    litellm.disable_aiohttp_transport = True
    litellm.in_memory_llm_clients_cache.flush_cache()
    cleanup_router_config_variables()

    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = os.path.join(filepath, "test_configs", "test_config_no_auth.yaml")
    asyncio.run(initialize(config=config_fp, debug=True))

    try:
        yield TestClient(app)
    finally:
        litellm.disable_aiohttp_transport = original_disable_aiohttp
        litellm.in_memory_llm_clients_cache.flush_cache()


def test_proxy_reducto_ocr_json_passthrough(client_no_auth):
    mocked_response = OCRResponse(
        pages=[OCRPage(index=0, markdown="Proxy OCR")],
        model="parse-v3",
        usage_info=OCRUsageInfo(pages_processed=1, credits=1),
    )

    with patch(
        "litellm.proxy.proxy_server.llm_router.aocr",
        new=AsyncMock(return_value=mocked_response),
    ) as mock_aocr:
        response = client_no_auth.post(
            "/v1/ocr",
            json={
                "model": "reducto/parse-v3",
                "document": {
                    "type": "document_url",
                    "document_url": "reducto://proxy.pdf",
                },
                "api_key": "proxy-key",
                "api_base": "https://platform.reducto.ai",
            },
        )

    assert response.status_code == 200
    assert mock_aocr.await_count == 1
    assert mock_aocr.await_args.kwargs["model"] == "reducto/parse-v3"
    assert mock_aocr.await_args.kwargs["document"] == {
        "type": "document_url",
        "document_url": "reducto://proxy.pdf",
    }
    assert mock_aocr.await_args.kwargs["api_key"] == "proxy-key"
    assert mock_aocr.await_args.kwargs["api_base"] == "https://platform.reducto.ai"

    response_body = response.json()
    assert response_body["object"] == "ocr"
    assert response_body["usage_info"]["credits"] == 1
    assert response_body["pages"][0]["markdown"] == "Proxy OCR"
