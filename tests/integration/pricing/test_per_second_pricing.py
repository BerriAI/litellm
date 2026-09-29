import json
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import SseResponse

RATE: Final = 0.5
FRAME_DELAY_MS: Final = 300
CONTENT: Final = ("one", " two", " three", " four")
PRICING_FIELDS: Final = frozenset({"cost_per_second", "input_cost_per_second", "output_cost_per_second"})
PER_SECOND_CONFIGURATIONS: Final[tuple[tuple[str, Mapping[str, JsonValue]], ...]] = (
    ("new_field", {"cost_per_second": RATE}),
    ("legacy_input", {"input_cost_per_second": RATE}),
    ("legacy_output", {"output_cost_per_second": RATE}),
    ("legacy_both", {"input_cost_per_second": RATE, "output_cost_per_second": 0.25}),
    (
        "all_three",
        {"cost_per_second": RATE, "input_cost_per_second": 0.25, "output_cost_per_second": 0.125},
    ),
)


def _sse_chunk(delta: dict[str, JsonValue], finish_reason: str | None) -> str:
    payload: Final = {
        "id": "$REQUEST_ID",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "integration-per-second",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}"


def _sse_frames() -> tuple[str, ...]:
    content_frames: Final = tuple(_sse_chunk({"content": content}, None) for content in CONTENT)
    usage_payload: Final = {
        "id": "$REQUEST_ID",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "integration-per-second",
        "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
    }
    usage_frame: Final = f"data: {json.dumps(usage_payload)}"
    return (*content_frames, _sse_chunk({}, "stop"), usage_frame, "data: [DONE]")


def _stream_content(event: dict[str, JsonValue]) -> str:
    choices: Final = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta: Final = object_value(object_value(choices[0])["delta"])
    content: Final = delta.get("content")
    return content if isinstance(content, str) else ""


def _clear_observations(upstream: httpx.Client) -> None:
    response: Final = upstream.get("/__observations")
    assert response.status_code == 200, response.text


def _observed_request_body(upstream: httpx.Client) -> dict[str, JsonValue]:
    observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
    assert isinstance(observations, list)
    assert len(observations) == 1
    return object_value(object_value(observations[0])["body"])


@pytest.mark.parametrize(
    ("pricing_case", "pricing"),
    PER_SECOND_CONFIGURATIONS,
    ids=("new_field", "legacy_input", "legacy_output", "legacy_both", "all_three"),
)
def test_chat_per_second_pricing_is_charged_once_and_not_forwarded(
    gateway: Gateway, pricing_case: str, pricing: Mapping[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"per-second-{pricing_case}-{uuid.uuid4().hex}"
        key: Final = scenario.key()
        model: Final = scenario.model(
            model=f"openai/integration-per-second-{uuid.uuid4().hex}",
            api_key=scenario_id,
            api_base=f"{gateway.upstream_url.rstrip('/')}/v1",
            **pricing,
        )
        with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
            _clear_observations(upstream)
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "price this request"}]},
                key=key,
            )
            body: Final = _observed_request_body(upstream)
            assert response.status_code == 200, f"{pricing_case}: {response.text}"
            response_cost: Final = float(response.headers.get("x-litellm-response-cost", "0"))
            duration_ms: Final = float(response.headers.get("x-litellm-response-duration-ms", "0"))
            assert response_cost > 0, f"{pricing_case}: cost={response_cost}, duration_ms={duration_ms}, body={body}"
            assert response_cost == pytest.approx(RATE * duration_ms / 1000, rel=1e-3), (
                f"{pricing_case}: cost={response_cost}, duration_ms={duration_ms}, body={body}"
            )
            assert not PRICING_FIELDS.intersection(body), body

        request_id: Final = string_value(object_value(response.json())["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (request_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(response_cost, rel=1e-3)


@pytest.mark.parametrize(
    ("pricing_case", "pricing"),
    PER_SECOND_CONFIGURATIONS,
    ids=("new_field", "legacy_input", "legacy_output", "legacy_both", "all_three"),
)
def test_streaming_chat_per_second_pricing_covers_the_full_stream(
    gateway: Gateway, pricing_case: str, pricing: Mapping[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"per-second-stream-{pricing_case}-{uuid.uuid4().hex}"
        frames: Final = _sse_frames()
        handle: Final = register_scenario(
            scenario_id,
            SseResponse(content_type="text/event-stream", frames=frames, frame_delay_ms=FRAME_DELAY_MS),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        key: Final = scenario.key()
        model: Final = scenario.model(
            model=f"openai/integration-per-second-{uuid.uuid4().hex}",
            api_key=scenario_id,
            api_base=handle.api_base(),
            **pricing,
        )
        with httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
            _clear_observations(upstream)
            with gateway.client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "price this streamed request"}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                headers={"Authorization": f"Bearer {key}"},
                ) as response:
                    stream_lines: Final = tuple(response.iter_lines())
                    assert response.status_code == 200, "\n".join(stream_lines)
            body: Final = _observed_request_body(upstream)

        events: Final = tuple(
            JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in stream_lines
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        assert len(events) == len(frames) - 1, events
        assert "".join(_stream_content(event) for event in events) == "".join(CONTENT), events
        usage: Final = object_value(events[-1]["usage"])
        assert usage["total_tokens"] == 40, events[-1]
        request_id: Final = string_value(events[0]["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, request_duration_ms, '
                'CAST(EXTRACT(EPOCH FROM ("endTime" - "startTime")) * 1000 AS DOUBLE PRECISION) '
                'AS elapsed_duration_ms '
                'FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (request_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        spend: Final = float(str(rows[0]["spend"]))
        request_duration_ms: Final = float(str(rows[0]["request_duration_ms"]))
        elapsed_duration_ms: Final = float(str(rows[0]["elapsed_duration_ms"]))
        assert spend == pytest.approx(RATE * request_duration_ms / 1000, rel=5e-2), (
            f"spend={spend}, request_duration_ms={request_duration_ms}, "
            f"endTime-startTime duration_ms={elapsed_duration_ms}, body={body}"
        )
        total_frame_delay_seconds: Final = (len(frames) - 1) * FRAME_DELAY_MS / 1000
        assert spend >= RATE * total_frame_delay_seconds * 0.95, (
            f"spend={spend}, total frame delay={total_frame_delay_seconds}s, body={body}"
        )
        assert not PRICING_FIELDS.intersection(body), body
