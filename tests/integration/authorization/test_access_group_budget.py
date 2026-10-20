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
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_INPUT_RATE: Final = 0.000001
_OUTPUT_RATE: Final = 0.00001
_MAX_BUDGET: Final = 0.0001
_CALL_COST: Final = 10 * _INPUT_RATE + 10 * _OUTPUT_RATE
_PROVIDER_MODEL: Final = "group-budget-probe"
Grant = Literal["key", "team", "organization", "project"]


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


class _StreamDelta(BaseModel):
    content: str | None = None


class _StreamChoice(BaseModel):
    delta: _StreamDelta


class _StreamChunk(BaseModel):
    choices: list[_StreamChoice]
    usage: _Usage | None = None


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


@dataclass(frozen=True, slots=True)
class _BudgetProxy:
    gateway: Gateway
    reserved: bool


def _is_model_discovery(request: Request) -> bool:
    return (request.method, request.target) == ("GET", "/v1/models")


def _provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(entry for entry in wire.drain() if not _is_model_discovery(entry))


def _billed(request: Request) -> Reply:
    if _is_model_discovery(request):
        return Reply(body=b'{"object":"list","data":[]}')
    assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
    body: Final = json.loads(request.body)
    if body.get("stream") is True:
        identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
        frames: Final = (
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": _PROVIDER_MODEL,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "billed"}, "finish_reason": None}],
            },
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": _PROVIDER_MODEL,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": _PROVIDER_MODEL,
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )
        chunks: Final = tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames) + (b"data: [DONE]\n\n",)
        return Reply(content_type="text/event-stream", chunks=chunks)
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


@pytest.fixture(
    params=("reserved", "unreserved"),
    ids=("reservation", "no-reservation"),
    scope="module",
)
def proxy(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_BudgetProxy]:
    """The shared proxy reserves budget per request; the owned one runs with ``disable_budget_reservation``."""
    with gateway_from_environment() as gateway:
        if request.param == "reserved":
            yield _BudgetProxy(gateway=gateway, reserved=True)
            return
        directory: Final = tmp_path_factory.mktemp("access-group-budget-unreserved")
        loaded_config: Final = TypeAdapter(dict[str, object]).validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        general_settings: Final = TypeAdapter(dict[str, object]).validate_python(
            loaded_config.get("general_settings", {})
        )
        config: Final = {
            **loaded_config,
            "general_settings": {**general_settings, "disable_budget_reservation": True},
        }
        path: Final = directory / "unreserved.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, directory, {}, config=path) as owned:
            yield _BudgetProxy(gateway=owned, reserved=False)


def _group_key(scenario: Scenario, name: str, grant: Grant) -> str:
    match grant:
        case "key":
            return scenario.key(models=[name])
        case "team":
            team_id: Final = scenario.team(models=[name])
            return scenario.key(team_id=team_id, models=[])
        case "organization":
            organization: Final = scenario.organization(models=[name])
            return scenario.key(organization_id=organization, models=[])
        case "project":
            project_team: Final = scenario.team()
            project_id: Final = scenario.project(project_team, models=[name])
            return scenario.key(team_id=project_team, project_id=project_id, models=[])


def _budgeted_group(scenario: Scenario, wire: Wire, grant: Grant = "key") -> _Group:
    """A database model in a fresh access group with one grant and a 0.0001 group budget."""
    name: Final = f"integration-group-{uuid.uuid4().hex}"
    model: Final = scenario.model(
        model=f"openai/{_PROVIDER_MODEL}",
        api_base=f"{wire.url}/v1",
        input_cost_per_token=_INPUT_RATE,
        output_cost_per_token=_OUTPUT_RATE,
        model_info={"access_groups": [name]},
    )
    key: Final = _group_key(scenario, name, grant)
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


def _chat(gateway: Gateway, group: _Group, text: str, max_tokens: int, spelling: str = "json") -> httpx.Response:
    body: Final = {
        "model": group.model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        **({"stream": True, "stream_options": {"include_usage": True}} if spelling == "stream" else {}),
    }
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        body,
        key=group.key,
    )


def _sent(text: str, max_tokens: int, spelling: str = "json") -> dict[str, object]:
    return {
        "model": _PROVIDER_MODEL,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        **({"stream": True, "stream_options": {"include_usage": True}} if spelling == "stream" else {}),
    }


def _assert_served(response: httpx.Response, wire: Wire, text: str, max_tokens: int, spelling: str = "json") -> None:
    assert response.status_code == 200, response.text
    if spelling == "stream":
        lines: Final = tuple(line for line in response.text.splitlines() if line)
        frames: Final = tuple(
            _StreamChunk.model_validate_json(line.removeprefix("data: "))
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        assert "".join(choice.delta.content or "" for frame in frames for choice in frame.choices) == "billed", (
            response.text
        )
        assert frames[-1].usage == _Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20), response.text
        assert lines[-1] == "data: [DONE]", response.text
    else:
        assert _Completion.model_validate_json(response.content).usage == _Usage(
            prompt_tokens=10, completion_tokens=10, total_tokens=20
        ), response.text
    received: Final = _provider_calls(wire)
    assert [json.loads(entry.body) for entry in received] == [_sent(text, max_tokens, spelling)], response.text
    assert [entry.headers["authorization"] for entry in received] == ["Bearer integration-provider-key"]


def _assert_refused_for_budget(response: httpx.Response, wire: Wire, message: str) -> None:
    assert response.status_code == 422, response.text
    assert _Error.model_validate_json(response.content).error == _ErrorBody(
        message=message, type="budget_exceeded", param=None, code="422"
    ), response.text
    assert _provider_calls(wire) == (), f"A refused request reached the provider: {response.text}"


@pytest.mark.timeout(240)
@pytest.mark.parametrize("spelling", ("json", "stream"))
def test_access_group_budget_refuses_traffic_once_spent_and_serves_again_after_delete(
    proxy: _BudgetProxy, spelling: str
) -> None:
    gateway: Final = proxy.gateway
    arrived: Final = threading.Event()
    release: Final = threading.Event()

    def held(request: Request) -> Reply:
        if proxy.reserved and spelling == "stream" and b"first call under budget" in request.body:
            arrived.set()
            assert release.wait(timeout=30), "The held call was never released"
        return _billed(request)

    with wire_server(held) as wire, gateway.scenario() as scenario:
        group: Final = _budgeted_group(scenario, wire)
        if proxy.reserved and spelling == "stream":
            with ThreadPoolExecutor(max_workers=1) as pool:
                in_flight: Final = pool.submit(_chat, gateway, group, "first call under budget", 1000, spelling)
                try:
                    assert arrived.wait(timeout=30), "The held call never reached the provider"
                    concurrent: Final = _chat(gateway, group, "concurrent call", 1, spelling)
                finally:
                    release.set()
            _assert_served(in_flight.result(timeout=30), wire, "first call under budget", 1000, spelling)
            _assert_refused_for_budget(
                concurrent,
                wire,
                f"Budget has been exceeded! Model access group={group.name} Current cost: {_MAX_BUDGET}, "
                f"Max budget: {_MAX_BUDGET}",
            )
        else:
            _assert_served(
                _chat(gateway, group, "first call under budget", 1, spelling),
                wire,
                "first call under budget",
                1,
                spelling,
            )
        eventually(
            lambda: _GroupBudget.model_validate_json(
                gateway.request("GET", f"/access_group/{group.name}/budget").content
            ),
            lambda value: value.spend == pytest.approx(_CALL_COST),
            seconds=60,
        )
        assert _budget_rows(group.name)[0]["spend"] == pytest.approx(_CALL_COST)

        _assert_refused_for_budget(
            _chat(gateway, group, "call over budget", 1, spelling),
            wire,
            f"Budget has been exceeded! Model access group={group.name} Current cost: {_CALL_COST}, Max budget: {_MAX_BUDGET}",
        )

        deleted: Final = gateway.request("DELETE", f"/access_group/{group.name}/budget")
        assert deleted.status_code == 200, deleted.text
        assert _DeletedBudget.model_validate_json(deleted.content) == _DeletedBudget(
            access_group=group.name,
            budget_deleted=True,
            message=f"Budget for access group '{group.name}' deleted successfully",
        ), deleted.text
        cleared: Final = gateway.request("GET", f"/access_group/{group.name}/budget")
        assert _GroupBudget.model_validate_json(cleared.content) == _GroupBudget(
            access_group=group.name, spend=0.0, budget=None
        ), cleared.text
        assert _budget_rows(group.name) == []
        _assert_served(
            _chat(gateway, group, "call after delete", 1, spelling),
            wire,
            "call after delete",
            1,
            spelling,
        )


@pytest.mark.timeout(120)
@pytest.mark.parametrize("grant", ("team", "organization", "project"))
def test_access_group_budget_refuses_a_key_granted_the_group_through_its_scope(
    proxy: _BudgetProxy, grant: Literal["team", "organization", "project"]
) -> None:
    gateway: Final = proxy.gateway
    text: Final = f"group granted through {grant}"
    with wire_server(_billed) as wire, gateway.scenario() as scenario:
        group: Final = _budgeted_group(scenario, wire, grant)
        _assert_served(_chat(gateway, group, text, 1), wire, text, 1)
        eventually(
            lambda: _GroupBudget.model_validate_json(
                gateway.request("GET", f"/access_group/{group.name}/budget").content
            ),
            lambda value: value.spend == pytest.approx(_CALL_COST),
            seconds=60,
        )
        assert _provider_calls(wire) == (), "The served request was not drained before the refusal"
        _assert_refused_for_budget(
            _chat(gateway, group, f"{text} over budget", 1),
            wire,
            f"Budget has been exceeded! Model access group={group.name} Current cost: {_CALL_COST}, "
            f"Max budget: {_MAX_BUDGET}",
        )
        assert _budget_rows(group.name)[0]["spend"] == pytest.approx(_CALL_COST)


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


def test_yaml_deployment_accounting_includes_router_subcalls_and_survives_restart(tmp_path: Path) -> None:
    group: Final = f"yaml-pool-{uuid.uuid4().hex}"
    target: Final = f"target-{uuid.uuid4().hex}"
    judge_name: Final = f"judge-{uuid.uuid4().hex}"
    smart: Final = f"smart-{uuid.uuid4().hex}"

    def classifier(request: Request) -> Reply:
        response: Final = _billed(request)
        if request.method == "GET":
            return response
        body: Final = json.loads(response.body)
        return Reply(
            body=json.dumps(
                {
                    **body,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": '{"tier":"COMPLEX"}'},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
        )

    def receive_alert(request: Request) -> Reply:
        assert request.method == "POST"
        return Reply(body=b"{}")

    with (
        wire_server(_billed) as upstream,
        wire_server(classifier) as judge,
        wire_server(receive_alert) as alerts,
        httpx.Client() as client,
    ):
        seed: Final = Gateway(client, "sk-local-group-budget-test", upstream.url)
        deployments: Final = [
            {
                "model_name": name,
                "litellm_params": {
                    "model": f"openai/{_PROVIDER_MODEL}",
                    "api_base": f"{url}/v1",
                    "api_key": "synthetic",
                    "input_cost_per_token": _INPUT_RATE,
                    "output_cost_per_token": _OUTPUT_RATE,
                },
                "model_info": {"id": name, "access_groups": [group], "access_group_spend_attribution": "deployment"},
            }
            for name, url in ((target, upstream.url), (judge_name, judge.url))
        ]
        config: Final = {
            "model_list": [
                *deployments,
                {
                    "model_name": smart,
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "complexity_router_config": {
                            "classifier_type": "llm",
                            "classifier_llm_config": {"model": judge_name, "timeout_ms": 20000},
                            "tiers": {"SIMPLE": target, "MEDIUM": target, "COMPLEX": target, "REASONING": target},
                        },
                    },
                },
            ],
            "general_settings": {
                "database_url": "os.environ/DATABASE_URL",
                "master_key": seed.key,
                "store_model_in_db": True,
                "alerting": ["webhook"],
                "alert_types": ["budget_alerts"],
            },
            "litellm_settings": {"timezone": "UTC"},
        }
        path: Final = tmp_path / "yaml-groups.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(seed, tmp_path, {"WEBHOOK_URL": alerts.url}, config=path) as candidate:
            created: Final = candidate.request(
                "PUT", f"/access_group/{group}/budget", {"soft_budget": _CALL_COST / 2, "budget_duration": "1mo"}
            )
            assert created.status_code == 200, created.text
            budget: Final = created.json()["budget"]
            assert budget["max_budget"] is None
            assert budget["budget_reset_at"][8:10] == "01"
            with candidate.scenario() as scenario:
                key: Final = scenario.key(models=["*"])
                for stream in (False, True):
                    response: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": smart,
                            "stream": stream,
                            "messages": [{"role": "user", "content": f"route {uuid.uuid4().hex}"}],
                        },
                        key=key,
                    )
                    assert response.status_code == 200, response.text
                    assert "billed" in response.text

                def call(index: int) -> int:
                    result: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": target,
                            "messages": [{"role": "user", "content": str(index)}],
                        },
                        key=key,
                    )
                    assert result.status_code == 200, result.text
                    return result.status_code

                with ThreadPoolExecutor(max_workers=4) as executor:
                    assert tuple(executor.map(call, range(4))) == (200, 200, 200, 200)
                assert len(_provider_calls(upstream)) == 6
                assert len(_provider_calls(judge)) == 2
                spent: Final = eventually(
                    lambda: candidate.request("GET", f"/access_group/{group}/budget").json()["spend"],
                    lambda value: value >= 8 * _CALL_COST - 1e-10,
                    seconds=70,
                )
                assert spent == pytest.approx(8 * _CALL_COST)
                notifications: Final = tuple(json.loads(request.body) for request in alerts.drain())
                assert any(
                    event.get("event") == "soft_budget_crossed"
                    and event.get("key_alias") == group
                    and event.get("event_group") == "model_access_group"
                    for event in notifications
                ), notifications
        with owned_proxy(seed, tmp_path, {}, config=path) as restarted:
            persisted: Final = restarted.request("GET", f"/access_group/{group}/budget")
            assert persisted.status_code == 200, persisted.text
            assert persisted.json()["spend"] == pytest.approx(8 * _CALL_COST)
            assert persisted.json()["budget"]["budget_id"] == budget["budget_id"]
            assert restarted.request("DELETE", f"/access_group/{group}/budget").status_code == 200
            assert restarted.request("POST", "/budget/delete", {"id": budget["budget_id"]}).status_code == 200


def test_yaml_image_group_counts_identical_generations_once_each(tmp_path: Path) -> None:
    group: Final = f"image-pool-{uuid.uuid4().hex}"
    name: Final = f"images-{uuid.uuid4().hex}"
    price: Final = 0.25

    def image(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.target == "/images/generations"
        body: Final = json.loads(request.body)
        assert "access_group_spend_attribution" not in body
        return Reply(body=b'{"created":1700000000,"data":[{"b64_json":"aW1n"}]}')

    with wire_server(image) as upstream, httpx.Client() as client:
        seed: Final = Gateway(client, "sk-local-image-budget-test", upstream.url)
        config: Final = {
            "model_list": [
                {
                    "model_name": name,
                    "litellm_params": {
                        "model": "openai/dall-e-3",
                        "api_base": upstream.url,
                        "api_key": "synthetic",
                        "output_cost_per_image": price,
                    },
                    "model_info": {"access_groups": [group], "access_group_spend_attribution": "deployment"},
                }
            ],
            "general_settings": {"database_url": "os.environ/DATABASE_URL", "master_key": seed.key},
        }
        path: Final = tmp_path / "image-group.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(seed, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            created: Final = candidate.request("PUT", f"/access_group/{group}/budget", {"soft_budget": 0.1})
            assert created.status_code == 200, created.text
            budget_id: Final = created.json()["budget"]["budget_id"]
            scenario.cleanups.callback(scenario.delete_budget, budget_id)
            scenario.cleanups.callback(candidate.request, "DELETE", f"/access_group/{group}/budget")
            key: Final = scenario.key(models=["*"])
            body: Final = {"model": name, "prompt": "a scripted sea otter", "size": "1024x1024", "n": 1}
            for _ in range(2):
                response: Final = candidate.request("POST", "/v1/images/generations", body, key=key)
                assert response.status_code == 200, response.text
            spent: Final = eventually(
                lambda: candidate.request("GET", f"/access_group/{group}/budget").json()["spend"],
                lambda value: value >= 2 * price,
                seconds=70,
            )
            assert spent == pytest.approx(2 * price)
            assert len(tuple(request for request in upstream.drain() if request.method == "POST")) == 2
