import json
from collections.abc import Callable
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server

_PROJECT: Final = "scripted-project"
_LOCATION: Final = "global"
_BACKEND: Final = "gemini-scripted-image-tokens"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models"
_PROMPT: Final = "scripted image token spend"
_PNG_B64: Final = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
_PROMPT_TOKENS: Final = 12
_TEXT_OUTPUT_TOKENS: Final = 10
_IMAGE_OUTPUT_TOKENS: Final = 1290
_THOUGHTS_TOKENS: Final = 40
_INPUT_RATE: Final = 2e-6
_TEXT_OUTPUT_RATE: Final = 1e-5
_IMAGE_OUTPUT_RATE: Final = 4e-5
_PRIORITY_INPUT_RATE: Final = 4e-6
_PRIORITY_OUTPUT_RATE: Final = 2e-5
_EXPECTED_STANDARD_SPEND: Final = (
    _PROMPT_TOKENS * _INPUT_RATE
    + (_TEXT_OUTPUT_TOKENS + _THOUGHTS_TOKENS) * _TEXT_OUTPUT_RATE
    + _IMAGE_OUTPUT_TOKENS * _IMAGE_OUTPUT_RATE
)
_EXPECTED_PRIORITY_SPEND: Final = (
    _PROMPT_TOKENS * _PRIORITY_INPUT_RATE
    + (_TEXT_OUTPUT_TOKENS + _THOUGHTS_TOKENS) * _PRIORITY_OUTPUT_RATE
    + _IMAGE_OUTPUT_TOKENS * _IMAGE_OUTPUT_RATE
)


def _image_reply(traffic_type: str | None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == (
            "POST",
            f"{_MODEL_PATH}/{_BACKEND}:generateContent",
        ), (request.method, request.target)
        assert _PROMPT in request.body.decode(), request.body
        usage_metadata: Final[dict[str, object]] = {
            "promptTokenCount": _PROMPT_TOKENS,
            "candidatesTokenCount": _TEXT_OUTPUT_TOKENS + _IMAGE_OUTPUT_TOKENS,
            "thoughtsTokenCount": _THOUGHTS_TOKENS,
            "totalTokenCount": _PROMPT_TOKENS + _TEXT_OUTPUT_TOKENS + _IMAGE_OUTPUT_TOKENS + _THOUGHTS_TOKENS,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": _PROMPT_TOKENS}],
            "candidatesTokensDetails": [
                {"modality": "TEXT", "tokenCount": _TEXT_OUTPUT_TOKENS},
                {"modality": "IMAGE", "tokenCount": _IMAGE_OUTPUT_TOKENS},
            ],
            **({"trafficType": traffic_type} if traffic_type is not None else {}),
        }
        return Reply(
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"inlineData": {"mimeType": "image/png", "data": _PNG_B64}},
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": usage_metadata,
                }
            ).encode()
        )

    return respond


def _deployment(gateway: Gateway, scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"vertex_ai/{_BACKEND}",
        api_base=f"{wire.url}{_MODEL_PATH}/{_BACKEND}:generateContent",
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        input_cost_per_token=_INPUT_RATE,
        output_cost_per_token=_TEXT_OUTPUT_RATE,
        output_cost_per_image_token=_IMAGE_OUTPUT_RATE,
        input_cost_per_token_priority=_PRIORITY_INPUT_RATE,
        output_cost_per_token_priority=_PRIORITY_OUTPUT_RATE,
    )


def _spend_for_key(key: str) -> float:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return float(str(rows[0]["spend"]))


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("traffic_type", "expected_spend"),
    [
        pytest.param(None, _EXPECTED_STANDARD_SPEND, id="standard"),
        pytest.param("ON_DEMAND_PRIORITY", _EXPECTED_PRIORITY_SPEND, id="on-demand-priority"),
    ],
)
def test_vertex_gemini_image_spend_prices_text_thinking_and_served_tier(
    gateway: Gateway,
    traffic_type: str | None,
    expected_spend: float,
) -> None:
    with wire_server(_image_reply(traffic_type)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT},
            key=key,
        )

        assert response.status_code == 200, response.text
        assert _spend_for_key(key) == pytest.approx(expected_spend)
