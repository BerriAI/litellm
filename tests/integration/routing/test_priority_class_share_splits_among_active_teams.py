import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

OPENAI_MODEL: Final = "gpt-4.1-mini"
MODEL_TPM: Final = 1000
TOKENS_PER_CALL: Final = 30
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_priority_class_share",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "shared"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": TOKENS_PER_CALL},
    }
).encode()


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
    return Reply(body=UPSTREAM_REPLY)


@pytest.fixture(scope="module")
def saturated_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp("priority-class-share")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3"],
        "priority_reservation": {"interactive": 0.6, "batch": 0.4},
        "priority_reservation_settings": {
            "default_priority": 1.0,
            "saturation_threshold": 0.0,
            "priority_class_share_config": {"split_equally_among_active": ["team"]},
        },
    }
    path: Final = directory / "priority_class_share.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        gateway_from_environment() as rig,
        wire_server(_respond) as wire,
        owned_proxy(rig, directory, {}, config=path) as owned,
    ):
        yield owned, wire.url


def _team_key(scenario: Scenario, priority: str) -> str:
    return scenario.key(team_id=scenario.team(metadata={"priority": priority}))


def _call(owned: Gateway, model: str, key: str) -> httpx.Response:
    return owned.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"class share {uuid.uuid4().hex}"}]},
        key=key,
    )


def _spend_until_blocked(owned: Gateway, model: str, key: str, max_calls: int) -> tuple[httpx.Response, ...]:
    def calls() -> Iterator[httpx.Response]:
        for _ in range(max_calls):
            response: Final = _call(owned, model, key)
            yield response
            if response.status_code == 429:
                return

    return tuple(calls())


def test_second_interactive_team_keeps_its_half_of_the_class_share(saturated_model: tuple[Gateway, str]) -> None:
    """Two interactive teams, batch idle: the class share is the whole model and each team is held to half of it."""
    owned, upstream = saturated_model
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        first: Final = _team_key(scenario, "interactive")
        second: Final = _team_key(scenario, "interactive")
        assert _call(owned, model, second).status_code == 200

        responses: Final = _spend_until_blocked(owned, model, first, max_calls=MODEL_TPM // TOKENS_PER_CALL + 1)
        blocked: Final = responses[-1]
        assert blocked.status_code == 429, blocked.text
        assert blocked.headers.get("rate_limited_by") == "priority_class_share_saturated", dict(blocked.headers)
        served: Final = (len(responses) - 1) * TOKENS_PER_CALL
        assert MODEL_TPM // 2 <= served < MODEL_TPM // 2 + 2 * TOKENS_PER_CALL, served

        still_served: Final = _call(owned, model, second)
        assert still_served.status_code == 200, still_served.text


def test_only_active_class_gets_the_whole_model(saturated_model: tuple[Gateway, str]) -> None:
    """One batch team alone on the model is held to 100% of the model, not to batch's 40%."""
    owned, upstream = saturated_model
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        batch: Final = _team_key(scenario, "batch")
        responses: Final = _spend_until_blocked(owned, model, batch, max_calls=MODEL_TPM // TOKENS_PER_CALL + 1)
        served: Final = sum(1 for response in responses if response.status_code == 200) * TOKENS_PER_CALL
        assert served >= MODEL_TPM - 2 * TOKENS_PER_CALL, [r.text for r in responses]


def test_class_weights_renormalize_when_batch_wakes_up(saturated_model: tuple[Gateway, str]) -> None:
    """Interactive and batch both active: interactive is held to 60% of the model and batch to 40%."""
    owned, upstream = saturated_model
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        interactive: Final = _team_key(scenario, "interactive")
        batch: Final = _team_key(scenario, "batch")
        assert _call(owned, model, batch).status_code == 200

        responses: Final = _spend_until_blocked(owned, model, interactive, max_calls=MODEL_TPM // TOKENS_PER_CALL + 1)
        assert responses[-1].status_code == 429, responses[-1].text
        served: Final = (len(responses) - 1) * TOKENS_PER_CALL
        assert int(MODEL_TPM * 0.6) <= served < int(MODEL_TPM * 0.6) + 2 * TOKENS_PER_CALL, served

        batch_responses: Final = _spend_until_blocked(owned, model, batch, max_calls=MODEL_TPM // TOKENS_PER_CALL + 1)
        batch_served: Final = sum(1 for response in batch_responses if response.status_code == 200) * TOKENS_PER_CALL
        assert batch_served >= int(MODEL_TPM * 0.4) - 2 * TOKENS_PER_CALL, [r.text for r in batch_responses]
        assert batch_responses[-1].status_code == 429, batch_responses[-1].text
