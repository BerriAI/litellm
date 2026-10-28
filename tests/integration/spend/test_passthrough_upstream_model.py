import json
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, JsonValue, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_REPORTED_MODEL: Final = "support-summarizer"
_REPORTED_COST: Final = 0.0123
_REPORTED_TOKENS: Final = 4210
_USAGE_HEADERS: Final = {
    "x-litellm-response-cost": str(_REPORTED_COST),
    "x-litellm-total-tokens": str(_REPORTED_TOKENS),
}
_MODEL_HEADER: Final = {"x-litellm-upstream-model": _REPORTED_MODEL}


def _upstream(
    marker: str, headers: Mapping[str, str], *, status: int = 200, stream: bool = False
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        assert json.loads(request.body)["ticket_id"] == marker, request
        if stream:
            return Reply(
                status=status,
                content_type="text/event-stream",
                chunks=(f'data: {{"delta": "{marker}"}}\n\n'.encode(), b"data: [DONE]\n\n"),
                headers=headers,
            )
        return Reply(status=status, body=json.dumps({"summary": marker}).encode(), headers=headers)

    return respond


def _endpoint(gateway: Gateway, scenario: Scenario, marker: str, target: str, **fields: JsonValue) -> str:
    path: Final = f"/support-ai-{marker}"
    created: Final = gateway.post(
        "/config/pass_through_endpoint", {"path": path, "target": target, "auth": True, **fields}
    )
    endpoint_id: Final = object_value(TypeAdapter(list[JsonValue]).validate_python(created["endpoints"])[0])["id"]
    scenario.cleanups.callback(
        lambda: gateway.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": str(endpoint_id)})
    )
    return path


def _spend_row(key: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model, spend, total_tokens, status FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND call_type=%s',
            (sha256(key.encode()).hexdigest(), "pass_through_endpoint"),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def _row_values(row: Mapping[str, JsonValue]) -> tuple[JsonValue, float, JsonValue, JsonValue]:
    spend: Final = row["spend"]
    assert isinstance(spend, (int, float)), row
    return row["model"], float(spend), row["total_tokens"], row["status"]


def test_upstream_reported_model_names_the_spend_row_daily_rollup_and_usage_breakdown(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream(marker, {**_USAGE_HEADERS, **_MODEL_HEADER})) as wire, gateway.scenario() as scenario:
        path: Final = _endpoint(gateway, scenario, marker, wire.url + "/v1/summarize")
        team: Final = scenario.team()
        key: Final = scenario.key(team_id=team, allowed_passthrough_routes=[path])

        response: Final = gateway.request("POST", path, {"ticket_id": marker}, key=key)
        assert response.status_code == 200, response.text
        assert response.json() == {"summary": marker}, response.text
        assert response.headers["x-litellm-upstream-model"] == _REPORTED_MODEL, response.headers

        assert _row_values(_spend_row(key)) == (
            _REPORTED_MODEL,
            pytest.approx(_REPORTED_COST),
            _REPORTED_TOKENS,
            "success",
        )
        daily: Final = eventually(
            lambda: read_rows('SELECT model, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert (daily[0]["model"], daily[0]["spend"]) == (_REPORTED_MODEL, pytest.approx(_REPORTED_COST)), daily

        today: Final = datetime.now(timezone.utc)
        activity: Final = gateway.request(
            "GET",
            "/team/daily/activity",
            params={
                "team_ids": team,
                "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
                "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            },
        )
        assert activity.status_code == 200, activity.text
        results: Final = object_value(activity.json())["results"]
        assert isinstance(results, list) and len(results) == 1, activity.text
        models: Final = object_value(object_value(object_value(results[0])["breakdown"])["models"])
        assert set(models) == {_REPORTED_MODEL}, activity.text
        assert object_value(object_value(models[_REPORTED_MODEL])["metrics"])["spend"] == pytest.approx(
            _REPORTED_COST
        ), activity.text


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_upstream_reported_model_wins_over_the_request_body_model(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(_upstream(marker, {**_USAGE_HEADERS, **_MODEL_HEADER}, stream=stream)) as wire,
        gateway.scenario() as scenario,
    ):
        path: Final = _endpoint(gateway, scenario, marker, wire.url + "/v1/summarize")
        key: Final = scenario.key(allowed_passthrough_routes=[path])

        response: Final = gateway.request("POST", path, {"ticket_id": marker, "model": "gpt-4o"}, key=key)
        assert response.status_code == 200, response.text
        assert marker in response.text, response.text

        assert _row_values(_spend_row(key)) == (
            _REPORTED_MODEL,
            pytest.approx(_REPORTED_COST),
            _REPORTED_TOKENS,
            "success",
        )


def test_upstream_error_row_carries_the_reported_model_and_cost(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(_upstream(marker, {**_USAGE_HEADERS, **_MODEL_HEADER}, status=500)) as wire,
        gateway.scenario() as scenario,
    ):
        path: Final = _endpoint(gateway, scenario, marker, wire.url + "/v1/summarize")
        key: Final = scenario.key(allowed_passthrough_routes=[path])

        response: Final = gateway.request("POST", path, {"ticket_id": marker, "model": "gpt-4o"}, key=key)
        assert response.status_code == 500, response.text

        assert _row_values(_spend_row(key)) == (
            _REPORTED_MODEL,
            pytest.approx(_REPORTED_COST),
            _REPORTED_TOKENS,
            "failure",
        )


def test_model_header_alone_keeps_the_configured_cost_per_request(gateway: Gateway) -> None:
    """The name is a label: an upstream that reports only a model does not own
    the cost, so the endpoint's flat cost_per_request still applies."""
    marker: Final = uuid.uuid4().hex
    with wire_server(_upstream(marker, _MODEL_HEADER)) as wire, gateway.scenario() as scenario:
        path: Final = _endpoint(gateway, scenario, marker, wire.url + "/v1/summarize", cost_per_request=0.5)
        key: Final = scenario.key(allowed_passthrough_routes=[path])

        response: Final = gateway.request("POST", path, {"ticket_id": marker}, key=key)
        assert response.status_code == 200, response.text

        model, spend, _, status = _row_values(_spend_row(key))
        assert (model, spend, status) == (_REPORTED_MODEL, pytest.approx(0.5), "success")


@pytest.mark.parametrize(
    ("body_model", "header_value", "expected_model"),
    [(None, None, "unknown"), ("gpt-4o", None, "gpt-4o"), ("gpt-4o", "   ", "gpt-4o")],
    ids=["no_header_no_body_model", "no_header_body_model", "blank_header_body_model"],
)
def test_without_a_usable_model_header_the_row_keeps_the_request_model(
    gateway: Gateway, body_model: str | None, header_value: str | None, expected_model: str
) -> None:
    marker: Final = uuid.uuid4().hex
    headers: Final = {**_USAGE_HEADERS, **({} if header_value is None else {"x-litellm-upstream-model": header_value})}
    body: Final[dict[str, JsonValue]] = {"ticket_id": marker, **({} if body_model is None else {"model": body_model})}
    with wire_server(_upstream(marker, headers)) as wire, gateway.scenario() as scenario:
        path: Final = _endpoint(gateway, scenario, marker, wire.url + "/v1/summarize")
        key: Final = scenario.key(allowed_passthrough_routes=[path])

        response: Final = gateway.request("POST", path, body, key=key)
        assert response.status_code == 200, response.text

        assert _row_values(_spend_row(key)) == (
            expected_model,
            pytest.approx(_REPORTED_COST),
            _REPORTED_TOKENS,
            "success",
        )


@pytest.mark.parametrize(
    ("header_value", "expected_model"),
    [
        (f"{_REPORTED_MODEL}, other-name", _REPORTED_MODEL),
        ("m" * 5000, "m" * 256),
    ],
    ids=["repeated_header_keeps_first", "long_name_cut_to_256"],
)
@pytest.mark.parametrize("status", [200, 500], ids=["success", "failure"])
def test_spend_row_keeps_a_bounded_first_model_name(
    gateway: Gateway, header_value: str, expected_model: str, status: int
) -> None:
    marker: Final = uuid.uuid4().hex
    headers: Final = {**_USAGE_HEADERS, "x-litellm-upstream-model": header_value}
    with wire_server(_upstream(marker, headers, status=status)) as wire, gateway.scenario() as scenario:
        path: Final = _endpoint(gateway, scenario, marker, wire.url + "/v1/summarize")
        key: Final = scenario.key(allowed_passthrough_routes=[path])

        response: Final = gateway.request("POST", path, {"ticket_id": marker}, key=key)
        assert response.status_code == status, response.text

        assert _row_values(_spend_row(key)) == (
            expected_model,
            pytest.approx(_REPORTED_COST),
            _REPORTED_TOKENS,
            "success" if status == 200 else "failure",
        )
