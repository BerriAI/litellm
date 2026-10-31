import json
import os
import re
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import OwnedState, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, JsonValue, TypeAdapter
from redis import Redis

pytestmark: Final = pytest.mark.shares_worker_state

_CHAT_BODY: Final = TypeAdapter(dict[str, JsonValue])
_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
_EXPECTED_MODEL_DISCOVERY: Final = ("GET", "/v1/models")
_EXPECTED_CHAT: Final = ("POST", "/v1/chat/completions")


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


_FULL_USAGE: Final = _Usage(prompt_tokens=20, completion_tokens=20, total_tokens=40)
_POST_RESET_USAGE: Final = _Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)


class _AssistantMessage(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _AssistantMessage


class _ChatResponse(BaseModel):
    id: str
    choices: tuple[_Choice, ...]
    usage: _Usage


class _ModelInfoEntry(BaseModel):
    model_name: str


class _ModelInfoResponse(BaseModel):
    data: tuple[_ModelInfoEntry, ...]


class _BudgetError(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _BudgetError


class _ResetResponse(BaseModel):
    key_hash: str
    spend: float
    previous_spend: float


class _KeyInfo(BaseModel):
    spend: float
    max_budget: float


class _KeyInfoResponse(BaseModel):
    info: _KeyInfo


class _SpendRow(BaseModel):
    spend: float


class _UserMessage(BaseModel):
    content: str


class _ChatRequest(BaseModel):
    model: str
    max_tokens: int
    messages: tuple[_UserMessage, ...]


def _chat_body(model: str, prompt: str, max_tokens: int = 20) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _wire_reply(identity: str, usage: _Usage) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "integration-keys-auth-reset",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "wire response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage.model_dump(),
            }
        ).encode()
    )


def _assert_wire_requests(wire: Wire, expected: tuple[tuple[str, str], ...]) -> None:
    observed: Final = tuple(
        sorted(
            (request.method, request.target)
            for request in wire.drain()
            if (request.method, request.target) != _EXPECTED_MODEL_DISCOVERY
        )
    )
    assert observed == tuple(sorted(expected)), observed


def _wire_chat(gateway: Gateway, model: str, key: str, prompt: str, max_tokens: int = 20) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", _chat_body(model, prompt, max_tokens), key=key)


def _assert_wire_success(response: httpx.Response, expected_usage: _Usage) -> _ChatResponse:
    assert response.status_code == 200, response.text
    payload: Final = _ChatResponse.model_validate_json(response.content)
    assert payload.choices[0].message.content == "wire response", response.text
    assert payload.usage == expected_usage, response.text
    return payload


def _model_is_available(gateway: Gateway, model: str) -> bool:
    response: Final = gateway.request("GET", "/model/info")
    if response.status_code != 200:
        return False
    models: Final = _ModelInfoResponse.model_validate_json(response.content)
    return any(entry.model_name == model for entry in models.data)


def _spend(gateway: Gateway, key_hash: str) -> float:
    rows: Final = TypeAdapter(tuple[_SpendRow, ...]).validate_python(
        read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (key_hash,))
    )
    assert len(rows) == 1
    return rows[0].spend


def _assert_budget_refusal(gateway: Gateway, model: str, key: str, prompt: str) -> None:
    response: Final = _wire_chat(gateway, model, key, prompt)
    assert response.status_code == 422, response.text
    assert _ErrorResponse.model_validate_json(response.content).error.type == "budget_exceeded", response.text


def test_an_exhausted_key_is_refused_inference_but_can_still_read_its_own_info(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.06)
        assert (
            object_value(gateway.chat(model, key=key, text=f"spend {uuid.uuid4().hex}")["usage"])["total_tokens"] == 40
        )
        digest: Final = sha256(key.encode()).hexdigest()
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=70,
        )
        upstream.delete("/__observations").raise_for_status()
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"over budget {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied.status_code == 422, denied.text
        error: Final = denied.json()["error"]
        assert error["type"] == "budget_exceeded"
        assert "Budget has been exceeded!" in error["message"]
        assert upstream.get("/__observations").json()["requests"] == []
        info: Final = gateway.request("GET", "/key/info", key=key, params={"key": key})
        assert info.status_code == 200, info.text
        own: Final = object_value(info.json()["info"])
        assert float(str(own["spend"])) == pytest.approx(0.06)
        assert own["max_budget"] == 0.06


def test_soft_budget_alert_reads_spend_from_another_proxy(gateway: Gateway, tmp_path: Path) -> None:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = {
        **base,
        "general_settings": {
            **object_value(base["general_settings"]),
            "alerting": ["slack"],
            "alert_types": ["budget_alerts"],
        },
    }
    path: Final = tmp_path / "soft-budget-alerting.yaml"
    path.write_text(yaml.safe_dump(config))
    email: Final = "budget-owner@example.test"
    alias: Final = f"soft-budget-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario, wire_server(lambda _: Reply(body=b"ok", content_type="text/plain")) as alerts:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        user_id: Final = scenario.user(user_email=email)
        key: Final = scenario.key(user_id=user_id, models=[model], key_alias=alias, soft_budget=0.1)
        digest: Final = sha256(key.encode()).hexdigest()
        overrides: Final = {
            "SLACK_WEBHOOK_URL": f"{alerts.url}/alerts",
            "PROXY_BASE_URL": "http://proxy.integration.test",
        }
        with owned_proxy(gateway, tmp_path, overrides, config=path, state=OwnedState.SHARED_WITH_WORKER) as candidate:
            first: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "warm the alerting proxy"}]},
                key=key,
            )
            assert first.status_code == 200, first.text
            assert float(first.headers["x-litellm-response-cost"]) == pytest.approx(20 * 0.001 + 20 * 0.002)
            eventually(lambda: _spend(gateway, digest), lambda spend: spend == pytest.approx(0.06), seconds=70)
            second: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "cross the budget on the other proxy"}]},
                key=key,
            )
            assert second.status_code == 200, second.text
            eventually(lambda: _spend(gateway, digest), lambda spend: spend == pytest.approx(0.12), seconds=70)
            assert alerts.received.qsize() == 0
            third: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "check the shared spend"}]},
                key=key,
            )
            assert third.status_code == 200, third.text
            eventually(alerts.received.qsize, lambda count: count >= 1, seconds=40)
            requests: Final = alerts.drain()
            assert len(requests) == 1, requests
            assert (requests[0].method, requests[0].target) == ("POST", "/alerts")
            payload: Final = JSON_OBJECT.validate_json(requests[0].body)
            message: Final = string_value(payload["text"])
            normalized: Final = re.sub(r"Timestamp: `\d{2}:\d{2}:\d{2}`", "Timestamp: `<clock>`", message)
            assert {**payload, "text": normalized} == {
                "text": (
                    "Alert type: `budget_alerts`\nLevel: `High`\nTimestamp: `<clock>`\n\n"
                    "Message: Soft Budget Crossed: Total Soft Budget:`0.1`\n\n"
                    f"*spend:* `0.12`\n*soft_budget:* `0.1`\n*user_id:* `{user_id}`\n"
                    f"*user_email:* `{email}`\n*key_alias:* `{alias}`\n*event_group:* `key`\n"
                    "\n\nProxy URL: `http://proxy.integration.test`"
                )
            }


def _bounded_chat(gateway: Gateway, model: str, key: str, content: str | None = None) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": content or f"key recovery {uuid.uuid4().hex}"}],
        },
        key=key,
    )


def _observed(upstream: httpx.Client) -> list[JsonValue]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = object_value(response.json())["requests"]
    assert isinstance(requests, list), response.text
    return requests


def _upstream_chat(content: str) -> dict[str, JsonValue]:
    return {
        "path": "/v1/chat/completions",
        "authorization": "Bearer integration-provider-key",
        "body": {"model": "gpt-4o-mini", "max_tokens": 20, "messages": [{"role": "user", "content": content}]},
        "method": "POST",
        "api_key": "",
    }


def _served_chat(gateway: Gateway, upstream: httpx.Client, model: str, key: str) -> None:
    content: Final = f"tier served {uuid.uuid4().hex}"
    upstream.delete("/__observations").raise_for_status()
    served: Final = _bounded_chat(gateway, model, key, content)
    assert served.status_code == 200, served.text
    assert float(served.headers["x-litellm-response-cost"]) == 0.06, dict(served.headers)
    assert _observed(upstream) == [_upstream_chat(content)], served.text


def test_raising_a_spent_keys_budget_restores_serving(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.06)
        first: Final = _bounded_chat(gateway, model, key)
        assert first.status_code == 200, first.text
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=70,
        )
        eventually(lambda: _bounded_chat(gateway, model, key), lambda response: response.status_code != 200, seconds=30)
        upstream.delete("/__observations").raise_for_status()
        denied: Final = _bounded_chat(gateway, model, key)
        assert denied.status_code == 422, denied.text
        assert object_value(denied.json()["error"])["type"] == "budget_exceeded"
        assert upstream.get("/__observations").json()["requests"] == []
        gateway.post("/key/update", {"key": key, "max_budget": 1.0})
        served: Final = tuple(_bounded_chat(gateway, model, key) for _ in range(3))
        assert [response.status_code for response in served] == [200, 200, 200], [response.text for response in served]
        assert len(upstream.get("/__observations").json()["requests"]) == 3


@pytest.mark.parametrize("spelling", ("plaintext", "hashed"))
def test_reset_spend_lifts_an_exhausted_key_on_both_workers_for_both_path_spellings(
    gateway: Gateway, tmp_path: Path, spelling: str
) -> None:
    backend: Final = "integration-keys-auth-reset"
    prompts: Final = tuple(f"reset spend case {index} {uuid.uuid4().hex}" for index in range(11))
    accepted_after_reset: Final = frozenset((prompts[3], prompts[4], prompts[7], prompts[8]))

    def respond(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target == "/v1/models", request
            assert request.headers["authorization"] == "Bearer synthetic-openai-key"
            assert request.body == b""
            return Reply(
                body=json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": backend,
                                "object": "model",
                                "created": 0,
                                "owned_by": "integration",
                            }
                        ],
                    }
                ).encode()
            )
        assert request.method == "POST"
        assert request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-openai-key"
        body: Final = _CHAT_BODY.validate_json(request.body)
        parsed: Final = _ChatRequest.model_validate_json(request.body)
        prompt: Final = parsed.messages[0].content
        max_tokens: Final = 10 if prompt in accepted_after_reset else 20
        assert parsed.model == backend
        assert parsed.max_tokens == max_tokens
        assert len(parsed.messages) == 1 and prompt in prompts, request.body
        assert body == _chat_body(backend, prompt, max_tokens), body
        usage: Final = _POST_RESET_USAGE if prompt in accepted_after_reset else _FULL_USAGE
        return _wire_reply("chatcmpl-" + uuid.uuid4().hex, usage)

    with (
        wire_server(respond) as wire,
        gateway.scenario() as scenario,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
    ):
        model: Final = scenario.model(
            model=f"openai/{backend}",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        subscriber_baseline: Final = cache.pubsub_numsub(_CHANNEL)[0][1]
        with owned_proxy(gateway, tmp_path / "second-worker", {}) as second:
            eventually(
                lambda: cache.pubsub_numsub(_CHANNEL)[0][1],
                lambda count: count >= subscriber_baseline + 1,
                seconds=30,
            )
            eventually(lambda: _model_is_available(second, model), bool, seconds=30)
            key: Final = scenario.key(models=[model], max_budget=0.06)
            key_hash: Final = sha256(key.encode()).hexdigest()
            first_spend: Final = _assert_wire_success(_wire_chat(gateway, model, key, prompts[0]), _FULL_USAGE)
            assert first_spend.usage.total_tokens == 40
            exhausted_spend: Final = eventually(
                lambda: _spend(gateway, key_hash), lambda spend: spend >= 0.06, seconds=70
            )
            assert exhausted_spend == pytest.approx(0.06)
            _assert_wire_requests(wire, (_EXPECTED_CHAT,))
            _assert_budget_refusal(gateway, model, key, prompts[1])
            _assert_budget_refusal(second, model, key, prompts[2])
            assert wire.drain() == (), "exhausted-key requests reached the upstream"

            identifier: Final = key if spelling == "plaintext" else key_hash
            for cycle, reset_to in enumerate((0, 0.02)):
                prior_spend = _spend(gateway, key_hash)
                reset = gateway.request("POST", f"/key/{identifier}/reset_spend", {"reset_to": reset_to})
                assert reset.status_code == 200, reset.text
                reset_result = _ResetResponse.model_validate_json(reset.content)
                assert reset_result.key_hash == key_hash, reset.text
                assert reset_result.spend == pytest.approx(reset_to), reset.text
                assert reset_result.previous_spend == pytest.approx(prior_spend), reset.text
                info_after_reset = gateway.request("GET", "/key/info", params={"key": key})
                assert info_after_reset.status_code == 200, info_after_reset.text
                assert _KeyInfoResponse.model_validate_json(info_after_reset.content).info.spend == pytest.approx(
                    reset_to
                ), info_after_reset.text
                assert _spend(gateway, key_hash) == pytest.approx(reset_to)
                peer_info_after_reset = eventually(
                    lambda: second.request("GET", "/key/info", params={"key": key}),
                    lambda response, reset_to=reset_to: (
                        response.status_code == 200
                        and _KeyInfoResponse.model_validate_json(response.content).info.spend == pytest.approx(reset_to)
                    ),
                    seconds=10,
                )
                assert _KeyInfoResponse.model_validate_json(peer_info_after_reset.content).info.spend == pytest.approx(
                    reset_to
                ), peer_info_after_reset.text
                workers = (gateway, second) if spelling == "plaintext" else (second, gateway)
                first_chat = _wire_chat(workers[0], model, key, prompts[3 + cycle * 4], max_tokens=10)
                _assert_wire_success(first_chat, _POST_RESET_USAGE)
                first_reported_spend = float(first_chat.headers["x-litellm-key-spend"])
                assert first_reported_spend == pytest.approx(reset_to + 0.03), first_chat.headers
                eventually(
                    lambda: float(cache.get(f"spend:key:{key_hash}") or 0),
                    lambda counter, reset_to=reset_to: counter == pytest.approx(reset_to + 0.03),
                    seconds=30,
                )
                second_chat = _wire_chat(workers[1], model, key, prompts[4 + cycle * 4], max_tokens=10)
                _assert_wire_success(second_chat, _POST_RESET_USAGE)
                second_reported_spend = float(second_chat.headers["x-litellm-key-spend"])
                assert reset_to + 0.03 <= second_reported_spend <= reset_to + 0.06, second_chat.headers
                re_exhausted_spend = eventually(
                    lambda: _spend(gateway, key_hash),
                    lambda spend, reset_to=reset_to: spend >= reset_to + 0.06,
                    seconds=70,
                )
                assert re_exhausted_spend == pytest.approx(reset_to + 0.06)
                _assert_wire_requests(wire, (_EXPECTED_CHAT, _EXPECTED_CHAT))
                _assert_budget_refusal(gateway, model, key, prompts[5 + cycle * 4])
                _assert_budget_refusal(second, model, key, prompts[6 + cycle * 4])
                assert wire.drain() == (), "re-exhausted-key requests reached the upstream"


def _denied_with_budget_exceeded(gateway: Gateway, model: str, key: str) -> None:
    denied: Final = _bounded_chat(gateway, model, key)
    assert denied.status_code == 422, denied.text
    assert object_value(denied.json()["error"])["type"] == "budget_exceeded"


@pytest.mark.timeout(149)
def test_raising_a_spent_tier_budget_restores_serving(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        budget_id: Final = scenario.budget(budget_id=f"tier-{uuid.uuid4().hex}", max_budget=0.06, budget_duration="30d")
        key: Final = scenario.key(models=[model], budget_id=budget_id)
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert info["budget_id"] == budget_id, info
        tier_rows: Final = read_rows(
            'SELECT max_budget, budget_duration FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
        )
        assert len(tier_rows) == 1
        assert float(str(tier_rows[0]["max_budget"])) == 0.06
        assert tier_rows[0]["budget_duration"] == "30d"
        _served_chat(gateway, upstream, model, key)
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=70,
        )
        upstream.delete("/__observations").raise_for_status()
        _denied_with_budget_exceeded(gateway, model, key)
        assert upstream.get("/__observations").json()["requests"] == []
        gateway.post("/budget/update", {"budget_id": budget_id, "max_budget": 1.0})
        updated_rows: Final = read_rows(
            'SELECT max_budget FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
        )
        assert float(str(updated_rows[0]["max_budget"])) == 1.0, updated_rows
        restored_content: Final = f"tier restored {uuid.uuid4().hex}"
        served: Final = eventually(
            lambda: _bounded_chat(gateway, model, key, restored_content),
            lambda response: response.status_code == 200,
            seconds=125,
        )
        assert served.status_code == 200, served.text
        assert _observed(upstream) == [_upstream_chat(restored_content)], served.text


@pytest.mark.timeout(149)
def test_lowering_a_tier_budget_below_spend_blocks_the_key(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        budget_id: Final = scenario.budget(
            budget_id=f"tier-{uuid.uuid4().hex}", max_budget=1000.0, budget_duration="30d"
        )
        key: Final = scenario.key(models=[model], budget_id=budget_id)
        _served_chat(gateway, upstream, model, key)
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) >= 0.06,
            seconds=30,
        )
        upstream.delete("/__observations").raise_for_status()
        probe: Final = gateway.request("POST", "/v1/chat/completions", {"model": model}, key=key)
        assert probe.status_code == 400, probe.text
        assert object_value(probe.json()["error"]) == {
            "message": "/chat/completions: Missing required parameter: 'messages'.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "400",
        }, probe.text
        assert upstream.get("/__observations").json()["requests"] == []
        gateway.post("/budget/update", {"budget_id": budget_id, "max_budget": 0.01})
        updated_rows: Final = read_rows(
            'SELECT max_budget FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
        )
        assert len(updated_rows) == 1
        assert float(str(updated_rows[0]["max_budget"])) == 0.01, updated_rows
        upstream.delete("/__observations").raise_for_status()
        lowered_probe: Final = eventually(
            lambda: gateway.request("POST", "/v1/chat/completions", {"model": model}, key=key),
            lambda response: (
                response.status_code == 422 and object_value(response.json()["error"])["type"] == "budget_exceeded"
            ),
            seconds=125,
        )
        assert "Max budget: 0.01" in lowered_probe.text, lowered_probe.text
        denied: Final = _bounded_chat(gateway, model, key)
        assert denied.status_code == 422, denied.text
        assert object_value(denied.json()["error"])["type"] == "budget_exceeded", denied.text
        assert "Max budget: 0.01" in denied.text, denied.text
        assert upstream.get("/__observations").json()["requests"] == []


@pytest.mark.timeout(149)
def test_key_linked_to_a_tier_by_update_is_refused_once_streamed_spend_exhausts_it(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        budget_id: Final = scenario.budget(budget_id=f"tier-{uuid.uuid4().hex}", max_budget=0.06, budget_duration="30d")
        key: Final = scenario.key(models=[model])
        gateway.post("/key/update", {"key": key, "budget_id": budget_id})
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert info["budget_id"] == budget_id, info

        content: Final = f"tier streamed {uuid.uuid4().hex}"
        body: Final = {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }
        upstream.delete("/__observations").raise_for_status()
        served: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert served.status_code == 200, served.text
        events: Final = tuple(
            line.removeprefix("data: ") for line in served.text.splitlines() if line.startswith("data: ")
        )
        assert events[-1] == "[DONE]", served.text
        chunks: Final = tuple(ChatCompletionChunk.model_validate_json(event) for event in events[:-1])
        assert {chunk.object for chunk in chunks} == {"chat.completion.chunk"}, served.text
        assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices), served.text
        observations: Final = _observed(upstream)
        assert len(observations) == 1, observations
        upstream_body: Final = object_value(object_value(observations[0])["body"])
        assert upstream_body["stream"] is True, observations

        denied: Final = eventually(
            lambda: gateway.request("POST", "/v1/chat/completions", body, key=key),
            lambda response: (
                response.status_code == 422 and object_value(response.json()["error"])["type"] == "budget_exceeded"
            ),
            seconds=125,
        )
        assert denied.status_code == 422, denied.text
        assert object_value(denied.json()["error"])["type"] == "budget_exceeded", denied.text
        assert "Max budget: 0.06" in denied.text, denied.text
        upstream.delete("/__observations").raise_for_status()
        fresh_denied: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert fresh_denied.status_code == 422, fresh_denied.text
        assert object_value(fresh_denied.json()["error"])["type"] == "budget_exceeded", fresh_denied.text
        assert "Max budget: 0.06" in fresh_denied.text, fresh_denied.text
        assert _observed(upstream) == []
