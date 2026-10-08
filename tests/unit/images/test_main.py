import json
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import CallTypes


def test_image_generation_keeps_an_internal_prefixed_kwarg_out_of_the_provider_request(
    respx_mock: respx.MockRouter,
) -> None:
    api_base: Final = "http://localhost:12346/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/images/generations.*").mock(
        return_value=httpx.Response(status_code=200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})
    )

    litellm.image_generation(
        model="openai/gpt-image-1",
        prompt="a red circle",
        api_base=api_base,
        api_key="fake_openai_api_key",
        _litellm_undeclared_sentinel="internal",
    )

    assert mock_route.called
    sent: Final = json.loads(respx_mock.calls[0].request.content)
    assert "_litellm_undeclared_sentinel" not in sent, sent
    assert sent["prompt"] == "a red circle"


def test_image_edit_prices_a_vertex_deployment_at_its_configured_location(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_base: Final = "http://localhost:12347/generateContent"
    respx_mock.post(api_base).mock(
        return_value=httpx.Response(
            status_code=200,
            json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1n"}}]}}]},
        )
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/gemini-fake-regional-edit-model",
        {
            "litellm_provider": "vertex_ai-language-models",
            "mode": "image_generation",
            "output_cost_per_image": 0.04,
            "regional_endpoint_uplift_multiplier": 1.1,
        },
    )

    def cost_at(location: str) -> float:
        logging_obj: Final = Logging(
            model="gemini-fake-regional-edit-model",
            messages=[],
            stream=False,
            call_type=CallTypes.image_edit.value,
            start_time=datetime.now(),
            litellm_call_id=f"vertex-edit-{location}",
            function_id="f",
        )
        response: Final = litellm.image_edit(
            model="vertex_ai/gemini-fake-regional-edit-model",
            image=b"\x89PNG\r\n\x1a\nfakepng",
            prompt="make the circle blue",
            api_base=api_base,
            vertex_location=location,
            litellm_logging_obj=logging_obj,
        )
        return logging_obj.response_cost_calculator(result=response)

    assert cost_at("global") == pytest.approx(0.04)
    assert cost_at("us-central1") == pytest.approx(0.044)
