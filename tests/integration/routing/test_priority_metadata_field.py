import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

OPENAI_MODEL: Final = "gpt-4.1-mini"
MODEL_RPM: Final = 10
SHARES: Final = {"interactive": 0.6, "batch": 0.4}
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_priority_metadata_field",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "priority class control"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
).encode()


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
    assert request.headers["authorization"] == "Bearer synthetic-openai-key"
    return Reply(body=UPSTREAM_REPLY)


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp("priority-class")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3"],
        "priority_reservation": SHARES,
        "priority_reservation_settings": {
            "default_priority": 0,
            "saturation_threshold": 0,
            "priority_metadata_field": "priority_class",
        },
    }
    path: Final = directory / "priority_class.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        gateway_from_environment() as rig,
        wire_server(_respond) as wire,
        owned_proxy(rig, directory, {}, config=path) as owned,
    ):
        yield owned, wire.url


def _tagged_key(scenario: Scenario, tagged: str, priority_class: str) -> str:
    if tagged == "team":
        return scenario.key(team_id=scenario.team(metadata={"priority_class": priority_class}))
    return scenario.key(metadata={"priority_class": priority_class})


@pytest.mark.parametrize(("tagged", "priority_class"), (("team", "batch"), ("key", "interactive")))
def test_priority_class_metadata_selects_the_priority_reservation_pool(
    candidate: tuple[Gateway, str], tagged: str, priority_class: str
) -> None:
    owned, upstream = candidate
    reserved: Final = int(MODEL_RPM * SHARES[priority_class])
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", rpm=MODEL_RPM
        )
        key: Final = _tagged_key(scenario, tagged, priority_class)

        def attempt() -> httpx.Response:
            return owned.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"priority class {uuid.uuid4().hex}"}]},
                key=key,
            )

        served: Final = tuple(attempt() for _ in range(reserved))
        assert [response.status_code for response in served] == [200] * reserved, [r.text for r in served]
        assert {response.headers.get("x-litellm-priority") for response in served} == {priority_class}, [
            dict(r.headers) for r in served
        ]
        blocked: Final = eventually(attempt, lambda response: response.status_code == 429, seconds=20)
        assert blocked.headers.get("x-litellm-priority") == priority_class, blocked.headers
        detail: Final = (
            f"Priority-based rate limit exceeded. Model: {model}, Priority: {priority_class}, "
            f"Rate limit type: requests, Model TPM: not configured, Model RPM: {MODEL_RPM}, Remaining: 0"
        )
        assert blocked.json()["error"]["message"].startswith(detail), blocked.text
