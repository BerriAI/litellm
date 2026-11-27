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
_PRICE_MAP_BACKEND: Final = "gemini-3.1-flash-image"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models"
_PROMPT: Final = "scripted image token spend"
_PNG_B64: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
_PROMPT_TOKENS: Final = 12
_TEXT_OUTPUT_TOKENS: Final = 10
_IMAGE_OUTPUT_TOKENS: Final = 1290
_THOUGHTS_TOKENS: Final = 40
_INPUT_RATE: Final = 2e-6
_TEXT_OUTPUT_RATE: Final = 1e-5
_IMAGE_OUTPUT_RATE: Final = 4e-5
_PRIORITY_INPUT_RATE: Final = 4e-6
_PRIORITY_OUTPUT_RATE: Final = 2e-5
_PRIORITY_IMAGE_RATE: Final = 8e-5
_EXPECTED_STANDARD_SPEND: Final = (
    _PROMPT_TOKENS * _INPUT_RATE
    + (_TEXT_OUTPUT_TOKENS + _THOUGHTS_TOKENS) * _TEXT_OUTPUT_RATE
    + _IMAGE_OUTPUT_TOKENS * _IMAGE_OUTPUT_RATE
)
_EXPECTED_PRIORITY_SPEND: Final = (
    _PROMPT_TOKENS * _PRIORITY_INPUT_RATE
    + (_TEXT_OUTPUT_TOKENS + _THOUGHTS_TOKENS) * _PRIORITY_OUTPUT_RATE
    + _IMAGE_OUTPUT_TOKENS * _PRIORITY_IMAGE_RATE
)
_EXPECTED_TEXT_ONLY_SPEND: Final = _PROMPT_TOKENS * _INPUT_RATE + 25 * _TEXT_OUTPUT_RATE
_HEALTH_PROBE_KEYS: Final = (
    "litellm-internal-health-check",
    sha256(b"litellm-internal-health-check").hexdigest(),
)


def _image_reply(
    traffic_type: str | None, text_only: bool, breakdown: bool, prompt: str = _PROMPT, backend: str = _BACKEND
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == (
            "POST",
            f"{_MODEL_PATH}/{backend}:generateContent",
        ), (request.method, request.target)
        assert prompt in request.body.decode(), request.body
        usage_metadata: Final[dict[str, object]] = {
            "promptTokenCount": _PROMPT_TOKENS,
            "candidatesTokenCount": 25 if text_only else _TEXT_OUTPUT_TOKENS + _IMAGE_OUTPUT_TOKENS,
            "thoughtsTokenCount": 0 if text_only else _THOUGHTS_TOKENS,
            "totalTokenCount": _PROMPT_TOKENS
            + (25 if text_only else _TEXT_OUTPUT_TOKENS + _IMAGE_OUTPUT_TOKENS + _THOUGHTS_TOKENS),
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": _PROMPT_TOKENS}],
            **(
                {
                    "candidatesTokensDetails": [
                        {"modality": "TEXT", "tokenCount": 25 if text_only else _TEXT_OUTPUT_TOKENS},
                        *([] if text_only else [{"modality": "IMAGE", "tokenCount": _IMAGE_OUTPUT_TOKENS}]),
                    ]
                }
                if breakdown
                else {}
            ),
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
                                    {"text": "No image generated"}
                                    if text_only
                                    else {"inlineData": {"mimeType": "image/png", "data": _PNG_B64}},
                                ],
                            },
                            "finishReason": "NO_IMAGE" if text_only else "STOP",
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
        output_cost_per_image_token_priority=_PRIORITY_IMAGE_RATE,
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
    ("traffic_type", "text_only", "breakdown", "expected_status", "expected_spend"),
    [
        pytest.param(None, False, True, 200, _EXPECTED_STANDARD_SPEND, id="standard"),
        pytest.param("ON_DEMAND_PRIORITY", False, True, 200, _EXPECTED_PRIORITY_SPEND, id="on-demand-priority"),
        pytest.param(None, True, True, 422, _EXPECTED_TEXT_ONLY_SPEND, id="text-only"),
        pytest.param(None, True, False, 422, _EXPECTED_TEXT_ONLY_SPEND, id="text-only-no-breakdown"),
        pytest.param(
            None,
            False,
            False,
            200,
            _PROMPT_TOKENS * _INPUT_RATE
            + _THOUGHTS_TOKENS * _TEXT_OUTPUT_RATE
            + (_TEXT_OUTPUT_TOKENS + _IMAGE_OUTPUT_TOKENS) * _IMAGE_OUTPUT_RATE,
            id="image-no-breakdown",
        ),
    ],
)
def test_vertex_gemini_image_spend_prices_text_thinking_and_served_tier(
    gateway: Gateway,
    traffic_type: str | None,
    text_only: bool,
    breakdown: bool,
    expected_status: int,
    expected_spend: float,
) -> None:
    with wire_server(_image_reply(traffic_type, text_only, breakdown)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": model, "prompt": _PROMPT},
            key=key,
        )

        assert response.status_code == expected_status, response.text
        assert _spend_for_key(key) == pytest.approx(expected_spend)


@pytest.mark.timeout(150)
def test_a_text_only_image_health_probe_records_what_the_same_answer_bills_a_caller(gateway: Gateway) -> None:
    with (
        wire_server(_image_reply(None, True, True, prompt="test", backend=_PRICE_MAP_BACKEND)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"vertex_ai/{_PRICE_MAP_BACKEND}",
            api_base=f"{wire.url}{_MODEL_PATH}/{_PRICE_MAP_BACKEND}:generateContent",
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
        )
        key: Final = scenario.key(models=[model])
        caller: Final = gateway.request("POST", "/v1/images/generations", {"model": model, "prompt": "test"}, key=key)
        health: Final = gateway.request("GET", "/health", params={"model": model})

        assert caller.status_code == 422, caller.text
        assert health.json()["healthy_count"] == 1, health.text
        probe_rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, spend FROM "LiteLLM_SpendLogs" WHERE api_key = ANY(%s) AND api_base LIKE %s',
                (list(_HEALTH_PROBE_KEYS), f"{wire.url}%"),
            ),
            lambda found: len(found) == 1,
            seconds=70,
        )
        caller_spend: Final = _spend_for_key(key)
        assert caller_spend > 0
        assert (probe_rows[0]["status"], float(str(probe_rows[0]["spend"]))) == ("failure", pytest.approx(caller_spend))
