from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from hashlib import sha256
from typing import Final, Literal

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.provider import PROVIDER_URL, SharedProvider
from tests.integration._support.wire import Reply

_CALL_COST: Final = 0.02
_TINY_BUDGET: Final = 0.0000000005
_LIMIT_FIELDS: Final = ("max_budget", "rpm_limit", "tpm_limit")


def _completion() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )


def _priced_model(scenario: Scenario) -> str:
    return scenario.model(api_base=f"{PROVIDER_URL}/v1", input_cost_per_token=0.001, output_cost_per_token=0.002)


def _ask(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"budget {uuid.uuid4().hex}"}]},
        key=key,
    )


def _served(gateway: Gateway, provider: SharedProvider, model: str, key: str) -> None:
    provider.expect(_completion())
    response: Final = _ask(gateway, model, key)
    assert response.status_code == 200, response.text
    assert len(provider.received()) == 1


def _spend_reaches(table: Literal["key", "team"], identity: str, amount: float) -> None:
    query: Final = (
        'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s'
        if table == "key"
        else 'SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s'
    )
    eventually(
        lambda: read_rows(query, (identity,)),
        lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= amount,
        seconds=70,
    )


_COST_AND_LIMIT: Final = re.compile(r"Current cost: ([^\s,]+), Max budget: ([^\s,]+)")


def _budget_refusal(
    gateway: Gateway, provider: SharedProvider, model: str, key: str, *, spent: float, limit: float
) -> str:
    refused: Final = _ask(gateway, model, key)
    assert refused.status_code == 422, f"{refused.status_code} {refused.text}"
    error: Final = object_value(JSON_OBJECT.validate_json(refused.content)["error"])
    assert error["type"] == "budget_exceeded", refused.text
    assert error["code"] == "422", refused.text
    message: Final = string_value(error["message"])
    assert "Budget has been exceeded!" in message, refused.text
    figures: Final = _COST_AND_LIMIT.search(message)
    assert figures is not None, message
    assert float(figures[1]) == pytest.approx(spent), message
    assert float(figures[2]) == pytest.approx(limit), message
    assert provider.received() == ()
    return message


def _hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def test_a_key_with_a_tiny_budget_serves_once_and_then_answers_budget_exceeded(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _priced_model(scenario)
        key: Final = scenario.key(models=[model], max_budget=_TINY_BUDGET)
        _served(gateway, provider, model, key)
        _spend_reaches("key", _hashed(key), _CALL_COST)
        _budget_refusal(gateway, provider, model, key, spent=_CALL_COST, limit=_TINY_BUDGET)


def test_a_key_with_a_zero_budget_is_refused_before_the_provider_is_called(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _priced_model(scenario)
        key: Final = scenario.key(models=[model], max_budget=0)
        _budget_refusal(gateway, provider, model, key, spent=0.0, limit=0.0)


def test_a_key_with_room_for_two_calls_serves_both_before_answering_budget_exceeded(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _priced_model(scenario)
        limit: Final = _CALL_COST * 1.5
        key: Final = scenario.key(models=[model], max_budget=limit)
        _served(gateway, provider, model, key)
        _spend_reaches("key", _hashed(key), _CALL_COST)
        _served(gateway, provider, model, key)
        _spend_reaches("key", _hashed(key), _CALL_COST * 2)
        _budget_refusal(gateway, provider, model, key, spent=_CALL_COST * 2, limit=limit)


def test_a_team_key_serves_once_and_then_answers_the_team_budget_envelope(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        model: Final = _priced_model(scenario)
        team: Final = scenario.team(models=[model], max_budget=_TINY_BUDGET)
        key: Final = scenario.key(team_id=team, models=[model])
        _served(gateway, provider, model, key)
        _spend_reaches("team", team, _CALL_COST)
        message: Final = _budget_refusal(gateway, provider, model, key, spent=_CALL_COST, limit=_TINY_BUDGET)
        assert f"Team={team}" in message, message


def _limits(record: dict[str, JsonValue]) -> Sequence[JsonValue]:
    return [record[field] for field in _LIMIT_FIELDS]


@pytest.mark.parametrize("field", _LIMIT_FIELDS)
def test_a_key_limit_is_set_by_update_and_reset_to_null(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key(max_budget=None, rpm_limit=None, tpm_limit=None)
        raised: Final = gateway.post("/key/update", {"key": key, field: 10})
        assert raised[field] == 10, raised
        assert [value for name, value in zip(_LIMIT_FIELDS, _limits(raised)) if name != field] == [None, None]
        cleared: Final = gateway.post("/key/update", {"key": key, field: None})
        assert _limits(cleared) == [None, None, None], cleared
        saved: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert _limits(saved) == [None, None, None], saved


@pytest.mark.parametrize("field", _LIMIT_FIELDS)
def test_a_team_limit_is_set_by_update_and_reset_to_null(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(max_budget=None, rpm_limit=None, tpm_limit=None)
        raised: Final = object_value(gateway.post("/team/update", {"team_id": team, field: 10})["data"])
        assert raised[field] == 10, raised
        cleared: Final = object_value(gateway.post("/team/update", {"team_id": team, field: None})["data"])
        assert _limits(cleared) == [None, None, None], cleared
        saved: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])
        assert _limits(saved) == [None, None, None], saved
