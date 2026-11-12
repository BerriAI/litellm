import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

OPENAI_MODEL: Final = "gpt-4.1-mini"
MODEL_TPM: Final = 1000
TEAM_TPM: Final = 100
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_team_model_limit_floor",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "floor"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
).encode()


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
    assert request.headers["authorization"] == "Bearer synthetic-openai-key"
    return Reply(body=UPSTREAM_REPLY)


def _proxy(tmp_path_factory: pytest.TempPathFactory, saturation_threshold: float) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp(f"team-floor-{saturation_threshold}")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3"],
        "priority_reservation": {"interactive": 0.6, "batch": 0.4},
        "priority_reservation_settings": {
            "default_priority": 1.0,
            "saturation_threshold": saturation_threshold,
            "team_model_limit_is_floor": True,
        },
    }
    path: Final = directory / "team_floor.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        gateway_from_environment() as rig,
        wire_server(_respond) as wire,
        owned_proxy(rig, directory, {}, config=path) as owned,
    ):
        yield owned, wire.url


@pytest.fixture(scope="module")
def idle_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    yield from _proxy(tmp_path_factory, saturation_threshold=0.5)


@pytest.fixture(scope="module")
def saturated_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    yield from _proxy(tmp_path_factory, saturation_threshold=0.0)


def _team_calls(owned: Gateway, upstream: str, calls: int) -> tuple[httpx.Response, ...]:
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        key: Final = scenario.key(team_id=scenario.team(metadata={"model_tpm_limit": {model: TEAM_TPM}}))
        return tuple(
            owned.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"team floor {uuid.uuid4().hex}"}]},
                key=key,
            )
            for _ in range(calls)
        )


def test_team_borrows_idle_model_capacity_past_its_model_tpm_limit(idle_model: tuple[Gateway, str]) -> None:
    owned, upstream = idle_model
    responses: Final = _team_calls(owned, upstream, calls=8)
    assert [response.status_code for response in responses] == [200] * 8, [r.text for r in responses]


def test_team_is_held_to_its_model_tpm_limit_once_the_model_saturates(saturated_model: tuple[Gateway, str]) -> None:
    owned, upstream = saturated_model
    responses: Final = _team_calls(owned, upstream, calls=8)
    blocked: Final = tuple(response for response in responses if response.status_code == 429)
    assert blocked, [r.text for r in responses]
    assert responses[0].status_code == 200, responses[0].text
    first: Final = blocked[0]
    assert first.headers.get("rate_limited_by") == "team_model_floor", dict(first.headers)
    assert "Team share reached" in first.json()["error"]["message"], first.text
