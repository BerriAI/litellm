import base64
import io
import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
import datetime
from datetime import timezone

from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.types.utils import ImageResponse

_LEGACY_DIR: Final = Path(__file__).parents[2] / "image_gen_tests"
_ISHAAN_BYTES: Final = (_LEGACY_DIR / "ishaan_github.png").read_bytes()
_LITELLM_SITE_BYTES: Final = (_LEGACY_DIR / "litellm_site.png").read_bytes()
_EDIT_RESPONSE: Final = {
    "created": 1589478378,
    "data": [
        {
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
    ],
    "usage": {
        "total_tokens": 1100,
        "input_tokens": 100,
        "input_tokens_details": {"image_tokens": 50, "text_tokens": 50},
        "output_tokens": 1000,
    },
}


@pytest.mark.asyncio
async def test_openai_image_edit_accepts_bytesio_images(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=r"https://api\.openai\.com/v1/images/edits.*").mock(
        return_value=httpx.Response(200, json=_EDIT_RESPONSE)
    )
    result: Final = await litellm.aimage_edit(
        prompt="combine the reference images",
        model="gpt-image-1",
        image=[io.BytesIO(_ISHAAN_BYTES), io.BytesIO(_LITELLM_SITE_BYTES)],
        api_key="fake-key",
    )
    ImageResponse.model_validate(result)
    assert result.data


@pytest.mark.asyncio
async def test_openai_image_edit_accepts_mixed_bytes_and_bytesio(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=r"https://api\.openai\.com/v1/images/edits.*").mock(
        return_value=httpx.Response(200, json=_EDIT_RESPONSE)
    )
    result: Final = await litellm.aimage_edit(
        prompt="Create a cohesive artistic style across all images",
        model="gpt-image-1",
        image=[_ISHAAN_BYTES, io.BytesIO(_LITELLM_SITE_BYTES)],
        api_key="fake-key",
    )
    ImageResponse.model_validate(result)
    assert result is not None
    assert result.data is not None
    assert len(result.data) > 0


@pytest.mark.asyncio
async def test_azure_image_edit_logs_deployment_model_and_positive_cost(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    edit_route: Final = respx_mock.post(url__regex=r".*images/edits.*").mock(
        return_value=httpx.Response(200, json=_EDIT_RESPONSE)
    )
    result: Final = await litellm.aimage_edit(
        prompt="combine the reference images",
        model="azure/CUSTOM_AZURE_DEPLOYMENT_NAME",
        base_model="azure/gpt-image-1",
        image=[_ISHAAN_BYTES, _LITELLM_SITE_BYTES],
        api_key="fake-key",
        api_base="https://fake.openai.azure.com",
    )
    assert edit_route.called
    ImageResponse.model_validate(result)

    payload: Final = get_logging_payload(
        kwargs={
            "model": "azure/CUSTOM_AZURE_DEPLOYMENT_NAME",
            "custom_llm_provider": "azure",
            "litellm_params": {
                "metadata": {},
                "model": "azure/gpt-image-1",
                "custom_llm_provider": "azure",
            },
            "response_cost": result._hidden_params["response_cost"],  # pyright: ignore[reportPrivateUsage]  # cost is only surfaced on _hidden_params
        },
        response_obj=result,
        start_time=datetime.datetime.now(timezone.utc),
        end_time=datetime.datetime.now(timezone.utc),
    )
    assert payload["model"] == "azure/CUSTOM_AZURE_DEPLOYMENT_NAME"
    assert payload["custom_llm_provider"] == "azure"
    assert payload["spend"] == result._hidden_params["response_cost"]  # pyright: ignore[reportPrivateUsage]  # cost is only surfaced on _hidden_params
    assert payload["spend"] > 0
