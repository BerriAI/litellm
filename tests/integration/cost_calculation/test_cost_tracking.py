"""Cost tracking coverage for literal integration request and response data."""

from __future__ import annotations

import io
import json
from hashlib import sha256
import struct
from typing import Final, cast
import wave
import zlib

import httpx
import pytest

from integration._support.client import JSON_OBJECT, Gateway
from integration.cost_calculation.conftest import (
    CostBreakdown,
    approx_equal,
    assert_total_is_sum_of_components,
    poll_cost_row,
    poll_failure_row,
    register_scenario_deployment,
)
from integration.cost_calculation.cost_tracking_case import (
    BinaryResponse,
    CASES,
    CostTrackingTestCase,
    ExactExpected,
    FailureExpected,
    RecountExpected,
    data_errors,
)

if _data_errors := data_errors():
    raise ValueError("\n".join(_data_errors))


_CASES: Final = tuple(
    pytest.param(case, marks=pytest.mark.covers(case.covers), id=case.name)
    for case in CASES
)


def _wav_bytes(seconds: float) -> bytes:
    frame_count: Final = round(16000 * seconds)
    output: Final = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * frame_count)
    return output.getvalue()


def _png_bytes() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _multipart_request(gateway: Gateway, case: CostTrackingTestCase, model_name: str, key: str) -> httpx.Response:
    assert case.upload is not None
    fields: Final = {
        field: value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        for field, value in {**case.request, "model": model_name}.items()
    }
    if case.upload.kind == "wav":
        files: Final = {"file": ("audio.wav", _wav_bytes(case.upload.seconds), "audio/wav")}
    else:
        files = {"image": ("image.png", _png_bytes(), "image/png")}
    return gateway.request_multipart(case.endpoint, fields, files, key=key)


def _assert_stream_has_no_error(response_text: str) -> None:
    for line in response_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            continue
        parsed = JSON_OBJECT.validate_json(payload)
        assert "error" not in parsed, f"stream carried an error event: {parsed}"


def _assert_breakdown(
    case: CostTrackingTestCase,
    expected: ExactExpected,
    breakdown: CostBreakdown,
    response: httpx.Response,
) -> None:
    assert breakdown.input_cost is not None and approx_equal(breakdown.input_cost, expected.input_cost), (
        f"{case.name}: input_cost {breakdown.input_cost} != expected {expected.input_cost}"
    )
    assert breakdown.output_cost is not None and approx_equal(breakdown.output_cost, expected.output_cost), (
        f"{case.name}: output_cost {breakdown.output_cost} != expected {expected.output_cost}"
    )
    for field, header_name, actual_component, expected_component in (
        (
            "cache_read_cost",
            "x-litellm-response-cost-cache-read",
            breakdown.cache_read_cost,
            expected.cache_read_cost,
        ),
        (
            "cache_creation_cost",
            "x-litellm-response-cost-cache-creation",
            breakdown.cache_creation_cost,
            expected.cache_creation_cost,
        ),
        (
            "reasoning_cost",
            "x-litellm-response-cost-reasoning",
            breakdown.reasoning_cost,
            expected.reasoning_cost,
        ),
        (
            "tool_usage_cost",
            "x-litellm-response-cost-tool-usage",
            breakdown.tool_usage_cost,
            expected.tool_usage_cost,
        ),
    ):
        if expected_component is None:
            continue
        assert actual_component is not None and approx_equal(actual_component, expected_component), (
            f"{case.name}: {field} {actual_component} != expected {expected_component}"
        )
        if case.response.content_type == "application/json":
            header: Final = response.headers.get(header_name)
            assert header is not None and approx_equal(float(header), expected_component), (
                f"{case.name}: {header_name} {header} != expected {expected_component}"
            )
    if case.response.content_type == "application/json" and any(
        component is not None
        for component in (
            expected.cache_read_cost,
            expected.cache_creation_cost,
            expected.reasoning_cost,
            expected.tool_usage_cost,
        )
    ):
        input_header: Final = response.headers.get("x-litellm-response-cost-input")
        output_header: Final = response.headers.get("x-litellm-response-cost-output")
        expected_input_header: Final = expected.input_cost - (
            expected.cache_read_cost or 0.0
        ) - (expected.cache_creation_cost or 0.0)
        assert input_header is not None and approx_equal(float(input_header), expected_input_header), (
            f"{case.name}: x-litellm-response-cost-input {input_header} != expected {expected_input_header}"
        )
        assert output_header is not None and approx_equal(float(output_header), expected.output_cost), (
            f"{case.name}: x-litellm-response-cost-output {output_header} != expected {expected.output_cost}"
        )


@pytest.mark.parametrize("case", _CASES)
def test_case_bills_expected_cost(gateway: Gateway, case: CostTrackingTestCase) -> None:
    marker: Final = sha256(case.name.encode()).hexdigest()[:12]
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        model_name: Final = register_scenario_deployment(scenario, case, marker, key)
        response: Final = (
            _multipart_request(gateway, case, model_name, key)
            if case.upload is not None
            else gateway.request("POST", case.endpoint, {**case.request, "model": model_name}, key=key)
        )
        if isinstance(case.expected, FailureExpected):
            assert response.status_code == case.expected.failure.status, (
                f"{case.name}: proxy returned {response.status_code}, expected {case.expected.failure.status}: "
                f"{response.text[:400]}"
            )
            response_cost: Final = response.headers.get("x-litellm-response-cost")
            assert response_cost is None or approx_equal(float(response_cost), 0.0), (
                f"{case.name}: failure response cost was {response_cost}"
            )
            row: Final = poll_failure_row(key)
            assert row.spend == 0, f"{case.name}: failure spend was {row.spend}"
            return
        assert response.is_success, f"{case.name}: proxy returned {response.status_code}: {response.text[:400]}"
        if case.response.content_type == "text/event-stream":
            _assert_stream_has_no_error(response.text)
        row: Final = poll_cost_row(key)
        if isinstance(case.expected, RecountExpected):
            assert row.prompt_tokens is not None and row.prompt_tokens > 0, (
                f"{case.name}: recount case counted no input tokens: prompt_tokens={row.prompt_tokens}"
            )
            assert row.completion_tokens is not None and row.completion_tokens > 0, (
                f"{case.name}: recount case counted no output tokens: completion_tokens={row.completion_tokens}"
            )
            recount: Final = row.prompt_tokens * case.expected.recount.input_cost_per_token + (
                row.completion_tokens * case.expected.recount.output_cost_per_token
            )
            assert row.spend is not None and approx_equal(row.spend, recount), (
                f"{case.name}: spend {row.spend} != recount {recount} at map rates"
            )
            breakdown: Final = row.breakdown
            assert breakdown is not None, f"{case.name}: no cost_breakdown persisted"
            assert_total_is_sum_of_components(row, breakdown, case.name)
            return
        expected: Final = case.expected
        assert isinstance(expected, ExactExpected)
        if isinstance(case.response, BinaryResponse):
            header: Final = response.headers.get("x-litellm-response-cost")
            if header is not None:
                assert approx_equal(float(header), expected.spend), (
                    f"{case.name}: x-litellm-response-cost {header} != expected {expected.spend}"
                )
        elif case.response.content_type == "application/json":
            header: Final = cast(str | None, response.headers.get("x-litellm-response-cost"))
            assert header is not None and approx_equal(float(header), expected.spend), (
                f"{case.name}: x-litellm-response-cost {header} != expected {expected.spend}"
            )
        assert row.spend is not None and approx_equal(row.spend, expected.spend), (
            f"{case.name}: spend {row.spend} != expected {expected.spend} "
            f"(breakdown {row.breakdown.model_dump() if row.breakdown is not None else None})"
        )
        breakdown: Final = row.breakdown
        if expected.breakdown_persisted:
            assert breakdown is not None, f"{case.name}: no cost_breakdown persisted"
        if breakdown is not None:
            _assert_breakdown(case, expected, breakdown, response)
        assert row.prompt_tokens == expected.prompt_tokens, (
            f"{case.name}: prompt_tokens {row.prompt_tokens} != expected {expected.prompt_tokens}"
        )
        assert row.completion_tokens == expected.completion_tokens, (
            f"{case.name}: completion_tokens {row.completion_tokens} != expected {expected.completion_tokens}"
        )
        if breakdown is not None:
            assert_total_is_sum_of_components(row, breakdown, case.name)
