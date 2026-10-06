import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.database import read_rows, scratch_database
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue


def _tier_peer(model: str, prompts: tuple[str, ...]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=b'{"object":"list","data":[]}')
        assert request.method == "POST", request.method
        assert request.target == "/v1/chat/completions", request.target
        body: Final = JSON_OBJECT.validate_json(request.body)
        expected_bodies: Final = tuple(
            {"model": model, "messages": [{"role": "user", "content": prompt}]}
            for prompt in prompts
        )
        assert body in expected_bodies, request.body.decode()
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-" + uuid.uuid4().hex[:8],
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "synthetic tier response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
                }
            ).encode()
        )

    return respond


def _deployment(
    alias: str,
    model: str,
    wire: Wire,
    input_cost_per_token: float,
    output_cost_per_token: float,
) -> dict[str, JsonValue]:
    return {
        "model_name": alias,
        "litellm_params": {
            "model": f"openai/{model}",
            "api_base": f"{wire.url}/v1",
            "api_key": "synthetic-auto-router-key",
        },
        "model_info": {
            "id": f"deployment-{alias}",
            "input_cost_per_token": input_cost_per_token,
            "output_cost_per_token": output_cost_per_token,
        },
    }


def _session_response(
    gateway: Gateway,
    key: str,
    session_id: str,
    database_url: str | None = None,
    stored_session_id: str | None = None,
) -> httpx.Response:
    if database_url is None or stored_session_id is None:
        return eventually(
            lambda: gateway.request(
                "GET",
                "/auto_router/session",
                params={"session_id": session_id},
                key=key,
            ),
            lambda response: response.status_code == 200,
            seconds=70,
        )

    key_hash: Final = hashlib.sha256(key.encode()).hexdigest()

    def read_response_state() -> tuple[httpx.Response, bool]:
        response: Final = gateway.request(
            "GET",
            "/auto_router/session",
            params={"session_id": session_id},
            key=key,
        )
        rows: Final = read_rows(
            'SELECT session_id FROM "LiteLLM_AutoRouterSession" WHERE api_key = %s AND session_id = %s',
            (key_hash, stored_session_id),
            database_url=database_url,
        )
        return response, bool(rows)

    response_state: Final = eventually(
        read_response_state,
        lambda state: state[0].status_code == 200 or state[1],
        seconds=70,
    )
    return response_state[0]


def _assert_rollup_row(
    api_key: str,
    database_url: str,
    router_name: str,
    expected_session_id: str,
) -> None:
    key_hash: Final = hashlib.sha256(api_key.encode()).hexdigest()
    rows: Final = read_rows(
        'SELECT session_id, router_name, router_type, last_model, turns, spend, saved_spend, '
        'savings_estimated_turns, savings_estimated_actual_spend, savings_estimated_saved_spend, '
        'savings_estimated_baseline_models, baseline_models, tier_turns '
        'FROM "LiteLLM_AutoRouterSession" WHERE api_key = %s AND session_id = %s AND router_name = %s',
        (key_hash, expected_session_id, router_name),
        database_url=database_url,
    )
    assert len(rows) == 1, rows
    assert rows[0] == {
        "session_id": expected_session_id,
        "router_name": router_name,
        "router_type": "complexity",
        "last_model": "openai/integration-simple",
        "turns": 1,
        "spend": pytest.approx(0.000015),
        "saved_spend": pytest.approx(0.000135),
        "savings_estimated_turns": 1,
        "savings_estimated_actual_spend": pytest.approx(0.000015),
        "savings_estimated_saved_spend": pytest.approx(0.000135),
        "savings_estimated_baseline_models": {"openai/integration-baseline": 1},
        "baseline_models": {"openai/integration-baseline": 1},
        "tier_turns": {"SIMPLE": 1},
    }


def test_auto_router_session_tracks_key_scoped_spend_and_bounds_long_session_ids(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    normal_session_id: Final = "auto-session-" + uuid.uuid4().hex
    long_session_id: Final = "s" * 300
    router_name: Final = "auto-router-" + uuid.uuid4().hex[:8]
    normal_prompt: Final = f"integration-auto-simple {normal_session_id}"
    long_prompt: Final = f"integration-auto-simple {long_session_id}"

    with (
        scratch_database() as database_url,
        wire_server(_tier_peer("integration-simple", (normal_prompt, long_prompt))) as simple,
        wire_server(_tier_peer("integration-medium", (normal_prompt,))) as medium,
        wire_server(_tier_peer("integration-complex", (normal_prompt,))) as complex_tier,
        wire_server(_tier_peer("integration-baseline", (normal_prompt,))) as baseline,
    ):
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            _deployment("integration-simple", "integration-simple", simple, 0.000001, 0.000002),
            _deployment("integration-medium", "integration-medium", medium, 0.000002, 0.000004),
            _deployment("integration-complex", "integration-complex", complex_tier, 0.000004, 0.000008),
            _deployment("integration-baseline", "integration-baseline", baseline, 0.00001, 0.00002),
            {
                "model_name": router_name,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "heuristic",
                        "keyword_tier_rules": [{"keywords": ["integration-auto-simple"], "tier": "SIMPLE"}],
                        "tiers": {
                            "SIMPLE": "integration-simple",
                            "MEDIUM": "integration-medium",
                            "COMPLEX": "integration-complex",
                            "REASONING": "integration-baseline",
                        },
                    },
                },
            },
        ]
        config["router_settings"] = {**config["router_settings"], "num_retries": 0, "disable_cooldowns": True}
        config_path: Final = tmp_path / "auto_router_management.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with owned_proxy(gateway, tmp_path, {"DATABASE_URL": database_url}, config=config_path, workers=1) as candidate:
            with candidate.scenario() as scenario:
                key: Final = scenario.key()
                other_key: Final = scenario.key()
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": router_name, "messages": [{"role": "user", "content": normal_prompt}]},
                    key=key,
                    headers={"x-litellm-session-id": normal_session_id},
                )
                assert response.status_code == 200, response.text
                assert response.json()["choices"][0]["message"]["content"] == "synthetic tier response", response.text

                session: Final = _session_response(candidate, key, normal_session_id)
                assert session.json() == {
                    "session_id": normal_session_id,
                    "router_name": router_name,
                    "router_type": "complexity",
                    "turns": 1,
                    "last_model": "openai/integration-simple",
                    "spend": pytest.approx(0.000015),
                    "savings_estimated_turns": 1,
                    "savings_estimated_actual_spend": pytest.approx(0.000015),
                    "saved_spend": pytest.approx(0.000135),
                    "baseline_spend": pytest.approx(0.00015),
                    "savings_estimated_baseline_spend": pytest.approx(0.00015),
                    "baseline_model": "openai/integration-baseline",
                    "baseline_models": {"openai/integration-baseline": 1},
                }, session.text
                _assert_rollup_row(key, database_url, router_name, normal_session_id)

                hidden: Final = candidate.request(
                    "GET",
                    "/auto_router/session",
                    params={"session_id": normal_session_id},
                    key=other_key,
                )
                assert hidden.status_code == 404, hidden.text
                assert hidden.json() == {
                    "detail": f"No auto-routed turns recorded for session {normal_session_id!r} under this key"
                }, hidden.text

                long_response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": router_name, "messages": [{"role": "user", "content": long_prompt}]},
                    key=key,
                    headers={"x-litellm-session-id": long_session_id},
                )
                assert long_response.status_code == 200, long_response.text
                assert (
                    long_response.json()["choices"][0]["message"]["content"] == "synthetic tier response"
                ), long_response.text

                bounded_session_id: Final = "sha256:" + hashlib.sha256(long_session_id.encode()).hexdigest()
                long_session: Final = _session_response(
                    candidate,
                    key,
                    long_session_id,
                    database_url,
                    bounded_session_id,
                )
                assert long_session.json() == {
                    "session_id": long_session_id,
                    "router_name": router_name,
                    "router_type": "complexity",
                    "turns": 1,
                    "last_model": "openai/integration-simple",
                    "spend": pytest.approx(0.000015),
                    "savings_estimated_turns": 1,
                    "savings_estimated_actual_spend": pytest.approx(0.000015),
                    "saved_spend": pytest.approx(0.000135),
                    "baseline_spend": pytest.approx(0.00015),
                    "savings_estimated_baseline_spend": pytest.approx(0.00015),
                    "baseline_model": "openai/integration-baseline",
                    "baseline_models": {"openai/integration-baseline": 1},
                }, long_session.text
                _assert_rollup_row(key, database_url, router_name, bounded_session_id)

        simple_calls: Final = tuple(request for request in simple.drain() if request.method == "POST")
        assert tuple(
            (
                request.method,
                request.target,
                json.loads(request.body)["model"],
                json.loads(request.body)["messages"],
            )
            for request in simple_calls
        ) == (
            ("POST", "/v1/chat/completions", "integration-simple", [{"role": "user", "content": normal_prompt}]),
            ("POST", "/v1/chat/completions", "integration-simple", [{"role": "user", "content": long_prompt}]),
        )
        assert tuple(request for request in medium.drain() if request.method == "POST") == ()
        assert tuple(request for request in complex_tier.drain() if request.method == "POST") == ()
        assert tuple(request for request in baseline.drain() if request.method == "POST") == ()
