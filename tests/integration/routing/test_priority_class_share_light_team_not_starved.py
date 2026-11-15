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
TEAM_SHARE: Final = MODEL_TPM // 2
TOKENS_PER_CALL: Final = 30
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_priority_class_share_borrowed",
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
def model_saturating_at_forty_percent(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp("priority-class-share-borrowed")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3"],
        "priority_reservation": {"interactive": 0.6, "batch": 0.4},
        "priority_reservation_settings": {
            "default_priority": 1.0,
            "saturation_threshold": 0.4,
            "saturation_check_cache_ttl": 0,
            "priority_class_share_splits_equally_among_active_teams": True,
        },
    }
    path: Final = directory / "priority_class_share_borrowed.yaml"
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


def test_team_that_borrowed_before_saturation_cannot_starve_a_teammate_within_its_share(
    model_saturating_at_forty_percent: tuple[Gateway, str],
) -> None:
    """Two interactive teams, batch idle, each owed half the model. The greedy one sends from an idle model until it
    is refused, and the light one must still be served up to its own half."""
    owned, upstream = model_saturating_at_forty_percent
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        greedy: Final = _team_key(scenario, "interactive")
        light: Final = _team_key(scenario, "interactive")
        assert _call(owned, model, light).status_code == 200

        greedy_responses: Final = _spend_until_blocked(owned, model, greedy, max_calls=MODEL_TPM // TOKENS_PER_CALL + 1)
        blocked: Final = greedy_responses[-1]
        assert blocked.status_code == 429, blocked.text
        assert blocked.headers.get("rate_limited_by") == "priority_class_share_saturated", blocked.text
        greedy_served: Final = (len(greedy_responses) - 1) * TOKENS_PER_CALL
        assert TEAM_SHARE <= greedy_served < TEAM_SHARE + 2 * TOKENS_PER_CALL, greedy_served

        light_responses: Final = _spend_until_blocked(owned, model, light, max_calls=MODEL_TPM // TOKENS_PER_CALL + 1)
        light_served: Final = (1 + sum(1 for r in light_responses if r.status_code == 200)) * TOKENS_PER_CALL
        assert light_served >= TEAM_SHARE - 2 * TOKENS_PER_CALL, (light_served, light_responses[-1].text)
