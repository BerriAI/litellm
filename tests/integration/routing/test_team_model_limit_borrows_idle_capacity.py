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
MODEL_RPM: Final = 1000
TEAM_TPM: Final = 100
TEAM_RPM: Final = 3
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_team_model_limit_borrowing",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "borrowed"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
).encode()


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
    assert request.headers["authorization"] == "Bearer synthetic-openai-key"
    return Reply(body=UPSTREAM_REPLY)


def _proxy(tmp_path_factory: pytest.TempPathFactory, saturation_threshold: float) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp(f"team-borrowing-{saturation_threshold}")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3"],
        "priority_reservation": {"interactive": 0.6, "batch": 0.4},
        "priority_reservation_settings": {
            "default_priority": 1.0,
            "saturation_threshold": saturation_threshold,
            "team_model_limit_borrows_idle_capacity_from_other_teams": True,
        },
    }
    path: Final = directory / "team_model_limit_borrowing.yaml"
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


def _team_calls(
    owned: Gateway, upstream: str, calls: int, limit: str = "model_tpm_limit"
) -> tuple[httpx.Response, ...]:
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}",
            api_base=f"{upstream}/v1",
            api_key="synthetic-openai-key",
            tpm=MODEL_TPM,
            rpm=MODEL_RPM,
        )
        key: Final = scenario.key(
            team_id=scenario.team(metadata={limit: {model: TEAM_TPM if limit == "model_tpm_limit" else TEAM_RPM}})
        )
        return tuple(
            owned.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"team limit borrowing {uuid.uuid4().hex}"}]},
                key=key,
            )
            for _ in range(calls)
        )


@pytest.mark.parametrize("limit", ["model_tpm_limit", "model_rpm_limit"])
def test_team_borrows_idle_model_capacity_past_its_model_limit(idle_model: tuple[Gateway, str], limit: str) -> None:
    owned, upstream = idle_model
    responses: Final = _team_calls(owned, upstream, calls=8, limit=limit)
    assert [response.status_code for response in responses] == [200] * 8, [r.text for r in responses]


@pytest.mark.parametrize(
    "limit, rate_limit_type", [("model_tpm_limit", "tokens"), ("model_rpm_limit", "requests")], ids=["tpm", "rpm"]
)
def test_team_is_held_to_its_model_limit_once_the_model_saturates(
    saturated_model: tuple[Gateway, str], limit: str, rate_limit_type: str
) -> None:
    owned, upstream = saturated_model
    responses: Final = _team_calls(owned, upstream, calls=8, limit=limit)
    blocked: Final = tuple(response for response in responses if response.status_code == 429)
    assert blocked, [r.text for r in responses]
    assert responses[0].status_code == 200, responses[0].text
    first: Final = blocked[0]
    assert first.headers.get("rate_limited_by") == "model_per_team_saturated", dict(first.headers)
    assert first.headers.get("rate_limit_type") == rate_limit_type, dict(first.headers)
    assert f"Team {limit} reached" in first.json()["error"]["message"], first.text
    assert "no idle capacity to borrow" in first.json()["error"]["message"], first.text
