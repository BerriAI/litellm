"""Cost tracking coverage for literal integration request and response data."""

from __future__ import annotations

import io
import json
import struct
import time
import uuid
import wave
import zlib
from collections.abc import Mapping
from hashlib import sha256
from itertools import islice
from typing import Final, cast

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, string_value
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.assertions import assert_exact, assert_recount
from integration.cost_calculation.conftest import (
    approx_equal,
    poll_cost_row,
    poll_failure_row,
    poll_rollups,
    poll_rows,
    read_rows_now,
    register_scenario_deployment,
)
from integration.cost_calculation.cost_tracking_case import (
    CASES,
    BinaryResponse,
    CostTrackingTestCase,
    ExactExpected,
    FailureExpected,
    RecountExpected,
    data_errors,
)
from pydantic import JsonValue

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
        assert (
            "error" not in parsed and parsed.get("type") not in {"error", "response.failed"}
        ), f"stream carried an error event: {parsed}"


def _replace_model(value: JsonValue, model_name: str) -> JsonValue:
    if isinstance(value, str):
        return value.replace("$MODEL", model_name)
    if isinstance(value, list):
        return [_replace_model(item, model_name) for item in value]
    if isinstance(value, dict):
        return {key: _replace_model(item, model_name) for key, item in value.items()}
    return value


def _prime_prior_response(
    gateway: Gateway, request_path: str, request_values: Mapping[str, JsonValue], key: str
) -> str:
    primed: Final = gateway.request(
        "POST",
        request_path,
        {field: value for field, value in request_values.items() if field != "previous_response_id"},
        key=key,
    )
    assert primed.is_success, f"priming response failed: {primed.status_code}: {primed.text[:400]}"
    return string_value(JSON_OBJECT.validate_json(primed.content)["id"])


@pytest.mark.parametrize("case", _CASES)
def test_case_bills_expected_cost(gateway: Gateway, case: CostTrackingTestCase) -> None:
    marker: Final = sha256(case.name.encode()).hexdigest()[:12]
    with gateway.scenario() as scenario:
        expected: Final = case.expected
        team_id: Final = scenario.team() if isinstance(expected, ExactExpected) and expected.rollups else None
        user_id: Final = (
            scenario.user(team_id=team_id)
            if team_id is not None
            else None
        )
        key: Final = (
            scenario.key(team_id=team_id, user_id=user_id)
            if team_id is not None and user_id is not None
            else scenario.key()
        )
        passthrough_provider: Final = case.passthrough_provider
        scenario_id: Final = f"sc-{marker}-{sha256(key.encode()).hexdigest()[:12]}"
        scenario_handle: Final = (
            register_scenario(scenario_id, case.response)
            if passthrough_provider in {"gemini", "anthropic"}
            else None
        )
        if scenario_handle is not None:
            scenario.cleanups.callback(delete_scenario, scenario_handle)
        deployment: Final = (
            register_scenario_deployment(scenario, case, marker, key)
            if passthrough_provider not in {"gemini", "anthropic"}
            else None
        )
        fallback_deployment: Final = (
            register_scenario_deployment(
                scenario,
                case,
                marker,
                key,
                response=case.fallback_from,
                marker_suffix="-fb",
            )
            if case.fallback_from is not None
            else None
        )
        model_name: Final = (
            case.model
            if passthrough_provider in {"gemini", "anthropic"}
            else deployment.model_name if deployment is not None else None
        )
        assert model_name is not None
        request_model: Final = (
            case.model.rsplit("/", 1)[-1]
            if passthrough_provider in {"gemini", "anthropic"}
            else fallback_deployment.model_name if fallback_deployment is not None else model_name
        )
        base_request_values: Final = (
            _replace_model(case.request, request_model)
            if passthrough_provider is not None
            else {**case.request, "model": model_name}
        )
        end_user_id: Final = (
            f"end-user-{uuid.uuid4()}"
            if isinstance(expected, ExactExpected) and expected.rollups
            else None
        )
        request_headers: Final = (
            {
                "x-pass-x-scripted-scenario": scenario_id,
                **(
                    {"x-goog-api-key": key}
                    if passthrough_provider == "gemini"
                    else {}
                ),
            }
            if passthrough_provider is not None
            else {}
        )
        request_path: Final = (
            case.endpoint.replace("$MODEL", request_model)
            if passthrough_provider is not None
            else case.endpoint
        )
        prior_response_id: Final = (
            _prime_prior_response(gateway, request_path, base_request_values, key)
            if case.chains_prior_response
            else None
        )
        request_body: Final = JSON_OBJECT.validate_python(
            {
                **base_request_values,
                **(
                    {"model": fallback_deployment.model_name, "fallbacks": [model_name]}
                    if fallback_deployment is not None
                    else {}
                ),
                **(
                    {"user": end_user_id, "cache": {"no-cache": True}}
                    if end_user_id is not None
                    else {}
                ),
                **({"previous_response_id": prior_response_id} if prior_response_id is not None else {}),
            }
        )
        if case.disconnect_after_frames is not None:
            with gateway.client.stream(
                "POST",
                request_path,
                json=request_body,
                headers={"Authorization": f"Bearer {key}", **request_headers},
            ) as stream_response:
                frames: Final = tuple(
                    islice(
                        (line for line in stream_response.iter_lines() if line.startswith("data:")),
                        case.disconnect_after_frames,
                    )
                )
                assert len(frames) == case.disconnect_after_frames
            row: Final = poll_cost_row(key)
            assert isinstance(expected, RecountExpected)
            assert row.status == "success", f"{case.name}: disconnect row status was {row.status}"
            assert_recount(case.name, expected, row)
            return
        responses: Final = tuple(
            (
                _multipart_request(gateway, case, model_name, key)
                if case.upload is not None
                else gateway.request("POST", request_path, request_body, key=key, headers=request_headers)
            )
            for _ in range(3 if isinstance(expected, ExactExpected) and expected.rollups else 1)
        )
        response: Final = responses[0]
        if isinstance(expected, FailureExpected):
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
        rows: Final = poll_rows(key, len(responses) + (prior_response_id is not None))
        if isinstance(expected, RecountExpected):
            row: Final = rows[0]
            assert_recount(case.name, expected, row)
            return
        assert isinstance(expected, ExactExpected)
        if fallback_deployment is not None:
            assert deployment is not None
            time.sleep(3)
            settled_rows: Final = read_rows_now(key)
            assert len(settled_rows) == 1
            assert settled_rows[0].status == "success"
            assert settled_rows[0].model_id == deployment.identity
        if isinstance(case.response, BinaryResponse):
            header: Final = response.headers.get("x-litellm-response-cost")
            if header is not None:
                assert approx_equal(float(header), expected.spend), (
                    f"{case.name}: x-litellm-response-cost {header} != expected {expected.spend}"
                )
        elif case.response.content_type == "application/json":
            header: Final = cast(str | None, response.headers.get("x-litellm-response-cost"))
            if expected.cost_header and expected.spend != 0:
                assert header is not None and approx_equal(float(header), expected.spend), (
                    f"{case.name}: x-litellm-response-cost {header} != expected {expected.spend}"
                )
            elif header is not None:
                assert approx_equal(float(header), expected.spend), (
                    f"{case.name}: x-litellm-response-cost {header} != expected {expected.spend}"
                )
        for row in rows:
            assert_exact(case.name, case.response.content_type, expected, row, response)
        if expected.rollups:
            assert deployment is not None and team_id is not None and user_id is not None
            assert end_user_id is not None
            target_spend: Final = expected.spend * 3
            target_requests: Final = 3
            rollups: Final = poll_rollups(
                key,
                team_id,
                user_id,
                end_user_id,
                target_spend,
                target_requests,
            )
            assert approx_equal(rollups.key_spend, target_spend)
            assert approx_equal(rollups.team_spend, target_spend)
            assert approx_equal(rollups.user_spend, target_spend)
            assert approx_equal(rollups.end_user_spend, target_spend)
            assert approx_equal(rollups.daily_user.spend, target_spend)
            assert approx_equal(rollups.daily_team.spend, target_spend)
            assert rollups.daily_user.prompt_tokens == expected.prompt_tokens * 3
            assert rollups.daily_user.completion_tokens == expected.completion_tokens * 3
            assert rollups.daily_user.api_requests == 3
            assert rollups.daily_team.prompt_tokens == expected.prompt_tokens * 3
            assert rollups.daily_team.completion_tokens == expected.completion_tokens * 3
            assert rollups.daily_team.api_requests == 3
