import json
import re
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from pydantic import BaseModel, JsonValue


class _Budget(BaseModel):
    budget_id: str
    max_budget: float | None


class _CustomerInfo(BaseModel):
    user_id: str
    blocked: bool
    alias: str | None
    spend: float
    litellm_budget_table: _Budget | None


def _new_customer(gateway: Gateway, user_id: str, *, path: str = "/customer/new", **fields: JsonValue) -> None:
    response: Final = gateway.request("POST", path, {"user_id": user_id, **fields})
    assert response.status_code == 200, response.text


def _customer_info(gateway: Gateway, user_id: str) -> _CustomerInfo:
    response: Final = gateway.request("GET", "/customer/info", params={"end_user_id": user_id})
    assert response.status_code == 200, response.text
    return _CustomerInfo.model_validate_json(response.content)


def _chat(gateway: Gateway, model: str, key: str, end_user: str, *, stream: bool = False) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": f"end user budget {uuid.uuid4().hex}"}],
        "user": end_user,
    }
    if stream:
        body["stream"] = True
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def _observed(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = response.json()["requests"]
    assert isinstance(requests, list)
    return requests


def _expected_observation(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "path": "/v1/chat/completions",
        "authorization": "Bearer integration-provider-key",
        "body": body,
        "method": "POST",
        "api_key": "",
    }


def _sent_body(response: httpx.Response, end_user: str) -> dict[str, JsonValue]:
    sent: Final = object_value(json.loads(response.request.content))
    expected: Final[dict[str, JsonValue]] = {"messages": sent["messages"], "model": "gpt-4o-mini", "user": end_user}
    if sent.get("stream") is True:
        expected["stream"] = True
        expected["stream_options"] = {"include_usage": True}
    return expected


def _end_user_row(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT spend, blocked FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (user_id,))


def _daily_rows(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT model, model_group, spend, api_requests, successful_requests FROM "LiteLLM_DailyEndUserSpend" WHERE end_user_id=%s',
        (user_id,),
    )


_REFUSAL_MESSAGE: Final = re.compile(
    r"^ExceededBudget: End User=(?P<user>\S+) over budget\. Spend=(?P<spend>[0-9.eE+-]+), Budget=(?P<budget>[0-9.eE+-]+)$"
)


def _assert_budget_refusal(response: httpx.Response, end_user: str, budget: float) -> None:
    error: Final = object_value(response.json()["error"])
    assert error["type"] == "budget_exceeded"
    assert error["param"] is None
    assert error["code"] == "422"
    matched: Final = _REFUSAL_MESSAGE.match(str(error["message"]))
    assert matched is not None, error["message"]
    assert matched.group("user") == end_user
    assert float(matched.group("spend")) == pytest.approx(0.06)
    assert float(matched.group("budget")) == budget


def test_customer_budget_refuses_calls_and_writes_daily_spend(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        info: Final = _customer_info(gateway, end_user)
        assert info.blocked is False
        assert info.litellm_budget_table is not None and info.litellm_budget_table.max_budget == 0.05, info
        joined: Final = read_rows(
            'SELECT b.max_budget FROM "LiteLLM_EndUserTable" e JOIN "LiteLLM_BudgetTable" b'
            " ON e.budget_id = b.budget_id WHERE e.user_id = %s",
            (end_user,),
        )
        assert len(joined) == 1 and float(str(joined[0]["max_budget"])) == 0.05, joined

        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        served: Final = _chat(gateway, model, key, end_user)
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [_expected_observation(_sent_body(served, end_user))]

        charged: Final = eventually(
            lambda: _end_user_row(end_user),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.05,
            seconds=70,
        )
        assert float(str(charged[0]["spend"])) == pytest.approx(20 * 0.001 + 20 * 0.002)
        daily: Final = eventually(lambda: _daily_rows(end_user), lambda rows: len(rows) == 1, seconds=70)
        assert daily[0]["model"] == "openai/gpt-4o-mini"
        assert daily[0]["model_group"] == model
        assert float(str(daily[0]["spend"])) == pytest.approx(0.06)
        assert int(str(daily[0]["api_requests"])) == 1
        assert int(str(daily[0]["successful_requests"])) == 1

        denied: Final = _chat(gateway, model, key, end_user)
        assert denied.status_code == 422, denied.text
        _assert_budget_refusal(denied, end_user, 0.05)
        assert _observed(upstream) == []


def test_customer_budget_update_takes_effect_without_restart(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, path="/end_user/new", max_budget=1.0)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        served: Final = _chat(gateway, model, key, end_user)
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [_expected_observation(_sent_body(served, end_user))]

        updated: Final = gateway.request("POST", "/customer/update", {"user_id": end_user, "max_budget": 0.01})
        assert updated.status_code == 200, updated.text
        info: Final = _customer_info(gateway, end_user)
        assert info.litellm_budget_table is not None and info.litellm_budget_table.max_budget == 0.01, info
        budget_row: Final = read_rows(
            'SELECT b.max_budget FROM "LiteLLM_EndUserTable" e JOIN "LiteLLM_BudgetTable" b'
            " ON e.budget_id = b.budget_id WHERE e.user_id = %s",
            (end_user,),
        )
        assert len(budget_row) == 1 and float(str(budget_row[0]["max_budget"])) == 0.01, budget_row
        eventually(
            lambda: _end_user_row(end_user),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.05,
            seconds=70,
        )
        denied: Final = _chat(gateway, model, key, end_user)
        assert denied.status_code == 422, denied.text
        _assert_budget_refusal(denied, end_user, 0.01)
        assert _observed(upstream) == []


def test_streamed_call_is_refused_as_json_once_over_budget(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        _observed(upstream)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        _new_customer(gateway, end_user, max_budget=0.05)
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        served: Final = _chat(gateway, model, key, end_user, stream=True)
        assert served.status_code == 200, served.text
        assert served.headers["content-type"].startswith("text/event-stream")
        assert 'data: {"id":"chatcmpl-' in served.text and "data: [DONE]" in served.text, served.text
        assert _observed(upstream) == [_expected_observation(_sent_body(served, end_user))]
        eventually(
            lambda: _end_user_row(end_user),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.05,
            seconds=70,
        )

        denied: Final = _chat(gateway, model, key, end_user, stream=True)
        assert denied.status_code == 422, denied.text
        assert denied.headers["content-type"].startswith("application/json")
        assert "data:" not in denied.text, denied.text
        _assert_budget_refusal(denied, end_user, 0.05)
        assert _observed(upstream) == []
