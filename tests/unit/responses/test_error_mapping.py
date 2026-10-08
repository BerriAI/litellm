from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError


@pytest.mark.asyncio
async def test_aresponses_unknown_model_surfaces_not_found(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=r".*api\.openai\.com/v1/responses.*").mock(
        return_value=httpx.Response(
            404, json={"error": {"message": "model not found", "type": "invalid_request_error", "code": "404"}}
        )
    )
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.aresponses(model="openai/non-existent-model", input="hi", api_key="fake-key")
    assert exc_info.value.status_code == 404


def test_aresponses_unsupported_temperature_raises_unsupported_params(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.responses(
            model="openai/responses/gpt-5.6-sol",
            input="hi",
            temperature=2000,
            api_key="fake-key",
        )


@pytest.mark.asyncio
async def test_acancel_responses_invalid_id_surfaces_status_error(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=r".*responses/invalid_response_id_12345/cancel.*").mock(
        return_value=httpx.Response(
            400, json={"error": {"message": "No such response", "type": "invalid_request_error", "code": "400"}}
        )
    )
    with pytest.raises(litellm.BadRequestError) as exc_info:
        await litellm.acancel_responses(
            response_id="invalid_response_id_12345",
            custom_llm_provider="openai",
            model="openai/gpt-4o",
            api_key="fake-key",
        )
    assert exc_info.value.status_code == 400
