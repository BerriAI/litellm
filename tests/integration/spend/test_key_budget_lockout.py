import json
import os
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, JsonValue, TypeAdapter
from redis import Redis

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
    observed: Final = tuple(sorted((request.method, request.target) for request in wire.drain()))
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
        upstream.get("/__observations").raise_for_status()
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


def _bounded_chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": f"key recovery {uuid.uuid4().hex}"}],
        },
        key=key,
    )


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
        upstream.get("/__observations").raise_for_status()
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
            _assert_wire_requests(wire, (_EXPECTED_MODEL_DISCOVERY, _EXPECTED_CHAT))
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
                first_chat = _wire_chat(
                    workers[0], model, key, prompts[3 + cycle * 4], max_tokens=10
                )
                _assert_wire_success(first_chat, _POST_RESET_USAGE)
                first_reported_spend = float(first_chat.headers["x-litellm-key-spend"])
                assert first_reported_spend == pytest.approx(reset_to + 0.03), first_chat.headers
                eventually(
                    lambda: float(cache.get(f"spend:key:{key_hash}") or 0),
                    lambda counter, reset_to=reset_to: counter == pytest.approx(reset_to + 0.03),
                    seconds=30,
                )
                second_chat = _wire_chat(
                    workers[1], model, key, prompts[4 + cycle * 4], max_tokens=10
                )
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
