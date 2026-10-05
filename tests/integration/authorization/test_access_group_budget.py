"""A model access group's shared budget: set and read back, enforced on real chat traffic, and cleared.

The scripted provider bills every call at 10 prompt and 10 completion tokens, so at the rates below one
call costs 0.00011 and exhausts the 0.0001 budget the group is given.
"""

import json
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel

_INPUT_RATE: Final = 0.000001
_OUTPUT_RATE: Final = 0.00001
_MAX_BUDGET: Final = 0.0001
_CALL_COST: Final = 10 * _INPUT_RATE + 10 * _OUTPUT_RATE
_PROVIDER_MODEL: Final = "group-budget-probe"


class _Budget(BaseModel):
    budget_id: str
    max_budget: float | None
    soft_budget: float | None
    budget_duration: str | None


class _GroupBudget(BaseModel):
    access_group: str
    spend: float
    budget: _Budget | None


class _DeletedBudget(BaseModel):
    access_group: str
    budget_deleted: bool
    message: str


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _Completion(BaseModel):
    id: str
    usage: _Usage


class _ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None
    code: str


class _Error(BaseModel):
    error: _ErrorBody


@dataclass(frozen=True, slots=True)
class _Group:
    name: str
    model: str
    key: str


def _is_model_discovery(request: Request) -> bool:
    return (request.method, request.target) == ("GET", "/v1/models")


def _provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(entry for entry in wire.drain() if not _is_model_discovery(entry))


def _billed(request: Request) -> Reply:
    if _is_model_discovery(request):
        return Reply(body=b'{"object":"list","data":[]}')
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 0,
                "model": _PROVIDER_MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "billed"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        ).encode()
    )


@pytest.fixture(params=("reserved", "unreserved"))
def proxy(request: pytest.FixtureRequest, gateway: Gateway, tmp_path: Path) -> Iterator[Gateway]:
    """The shared proxy reserves budget per request; the owned one runs with ``disable_budget_reservation``."""
    if request.param == "reserved":
        yield gateway
        return
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["disable_budget_reservation"] = True
    path: Final = tmp_path / "unreserved.yaml"
    path.write_text(yaml.safe_dump(config))
    with owned_proxy(gateway, tmp_path, {}, config=path) as owned:
        yield owned


def _budgeted_group(scenario: Scenario, wire: Wire) -> _Group:
    """A database model in a fresh access group, a key granted only that group, and a 0.0001 group budget."""
    name: Final = f"integration-group-{uuid.uuid4().hex}"
    model: Final = scenario.model(
        model=f"openai/{_PROVIDER_MODEL}",
        api_base=f"{wire.url}/v1",
        input_cost_per_token=_INPUT_RATE,
        output_cost_per_token=_OUTPUT_RATE,
        model_info={"access_groups": [name]},
    )
    key: Final = scenario.key(models=[name])
    gateway: Final = scenario.gateway
    created: Final = gateway.request(
        "PUT", f"/access_group/{name}/budget", {"max_budget": _MAX_BUDGET, "budget_duration": "30d"}
    )
    assert created.status_code == 200, created.text
    stored: Final = _GroupBudget.model_validate_json(created.content)
    assert stored.budget is not None, created.text
    scenario.cleanups.callback(scenario.delete_budget, stored.budget.budget_id)
    scenario.cleanups.callback(gateway.request, "DELETE", f"/access_group/{name}/budget")
    assert stored == _GroupBudget(
        access_group=name,
        spend=0.0,
        budget=_Budget(
            budget_id=stored.budget.budget_id, max_budget=_MAX_BUDGET, soft_budget=None, budget_duration="30d"
        ),
    ), created.text
    read_back: Final = gateway.request("GET", f"/access_group/{name}/budget")
    assert read_back.status_code == 200, read_back.text
    assert _GroupBudget.model_validate_json(read_back.content) == stored, read_back.text
    assert _budget_rows(name) == [
        {"budget_id": stored.budget.budget_id, "spend": 0.0, "max_budget": _MAX_BUDGET, "budget_duration": "30d"}
    ]
    return _Group(name, model, key)


def _budget_rows(group: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT g.budget_id, g.spend, b.max_budget, b.budget_duration FROM "LiteLLM_ModelAccessGroupBudgetTable" g '
        'LEFT JOIN "LiteLLM_BudgetTable" b ON b.budget_id = g.budget_id WHERE g.access_group_name = %s',
        (group,),
    )


def _chat(gateway: Gateway, group: _Group, text: str, max_tokens: int) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": group.model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens},
        key=group.key,
    )


def _sent(text: str, max_tokens: int) -> dict[str, object]:
    return {"model": _PROVIDER_MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens}


def _assert_served(response: httpx.Response, wire: Wire, text: str, max_tokens: int) -> None:
    assert response.status_code == 200, response.text
    assert _Completion.model_validate_json(response.content).usage == _Usage(
        prompt_tokens=10, completion_tokens=10, total_tokens=20
    ), response.text
    received: Final = _provider_calls(wire)
    assert [json.loads(entry.body) for entry in received] == [_sent(text, max_tokens)], response.text
    assert [entry.headers["authorization"] for entry in received] == ["Bearer integration-provider-key"]


def _assert_refused_for_budget(response: httpx.Response, wire: Wire, message: str) -> None:
    assert response.status_code == 422, response.text
    assert _Error.model_validate_json(response.content).error == _ErrorBody(
        message=message, type="budget_exceeded", param=None, code="422"
    ), response.text
    assert _provider_calls(wire) == (), f"A refused request reached the provider: {response.text}"


@pytest.mark.timeout(240)
def test_access_group_budget_refuses_traffic_once_spent_and_serves_again_after_delete(proxy: Gateway) -> None:
    with wire_server(_billed) as wire, proxy.scenario() as scenario:
        group: Final = _budgeted_group(scenario, wire)
        _assert_served(_chat(proxy, group, "first call under budget", 1), wire, "first call under budget", 1)
        spent: Final = eventually(
            lambda: _GroupBudget.model_validate_json(proxy.request("GET", f"/access_group/{group.name}/budget").content),
            lambda value: value.spend > 0,
            seconds=30,
        )
        assert spent.spend == pytest.approx(_CALL_COST), spent
        assert _budget_rows(group.name)[0]["spend"] == pytest.approx(_CALL_COST)

        _assert_refused_for_budget(
            _chat(proxy, group, "call over budget", 1),
            wire,
            f"Budget has been exceeded! Model access group={group.name} Current cost: {_CALL_COST}, Max budget: {_MAX_BUDGET}",
        )

        deleted: Final = proxy.request("DELETE", f"/access_group/{group.name}/budget")
        assert deleted.status_code == 200, deleted.text
        assert _DeletedBudget.model_validate_json(deleted.content) == _DeletedBudget(
            access_group=group.name,
            budget_deleted=True,
            message=f"Budget for access group '{group.name}' deleted successfully",
        ), deleted.text
        cleared: Final = proxy.request("GET", f"/access_group/{group.name}/budget")
        assert _GroupBudget.model_validate_json(cleared.content) == _GroupBudget(
            access_group=group.name, spend=0.0, budget=None
        ), cleared.text
        assert _budget_rows(group.name) == []
        _assert_served(_chat(proxy, group, "call after delete", 1), wire, "call after delete", 1)


@pytest.mark.timeout(120)
def test_reservation_refuses_a_concurrent_call_while_an_in_flight_call_holds_the_group_budget(
    gateway: Gateway,
) -> None:
    arrived: Final = threading.Event()
    release: Final = threading.Event()

    def held(request: Request) -> Reply:
        if b"held call" in request.body:
            arrived.set()
            assert release.wait(timeout=30), "The held call was never released"
        return _billed(request)

    with wire_server(held) as wire, gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=1) as pool:
        group: Final = _budgeted_group(scenario, wire)
        in_flight: Final = pool.submit(_chat, gateway, group, "held call", 1000)
        try:
            assert arrived.wait(timeout=30), "The held call never reached the provider"
            concurrent: Final = _chat(gateway, group, "concurrent call", 1)
        finally:
            release.set()
        _assert_served(in_flight.result(timeout=30), wire, "held call", 1000)
        _assert_refused_for_budget(
            concurrent,
            wire,
            f"Budget has been exceeded! Model access group={group.name} Current cost: {_MAX_BUDGET}, Max budget: {_MAX_BUDGET}",
        )
