import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

OPENAI_MODEL: Final = "gpt-4.1-mini"
MODEL_TPM: Final = 1000
TEAM_TPM: Final = 100
SATURATION_CACHE_TTL: Final = 1
TOKENS_PER_CALL: Final = 30
CALLS_TO_SATURATE: Final = MODEL_TPM // 2 // TOKENS_PER_CALL + 1
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_borrower_regains_share",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "recovered"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": TOKENS_PER_CALL},
    }
).encode()


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/v1/chat/completions"), request.target
    assert request.headers["authorization"] == "Bearer synthetic-openai-key"
    return Reply(body=UPSTREAM_REPLY)


@pytest.fixture(scope="module")
def saturated_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp("recovery-window")
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration: Final = {
        **base,
        "litellm_settings": {
            **base["litellm_settings"],
            "callbacks": ["dynamic_rate_limiter_v3"],
            "priority_reservation": {"interactive": 0.6, "batch": 0.4},
            "priority_reservation_settings": {
                "default_priority": 1.0,
                "saturation_threshold": 0.5,
                "team_model_limit_borrows_idle_capacity_from_other_teams": True,
                "saturation_check_cache_ttl": SATURATION_CACHE_TTL,
            },
        },
    }
    path: Final = directory / "borrower_regains_share.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        gateway_from_environment() as rig,
        wire_server(_respond) as wire,
        owned_proxy(rig, directory, {}, config=path) as owned,
    ):
        yield owned, wire.url


def _call(owned: Gateway, model: str, key: str) -> httpx.Response:
    return owned.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"borrower regains share {uuid.uuid4().hex}"}]},
        key=key,
    )


def test_a_borrower_gets_its_share_back_once_the_model_saturates_and_a_newcomer_is_served_beside_it(
    saturated_model: tuple[Gateway, str],
) -> None:
    owned, upstream = saturated_model
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        borrower: Final = scenario.key(team_id=scenario.team(metadata={"model_tpm_limit": {model: TEAM_TPM}}))
        newcomer: Final = scenario.key(team_id=scenario.team(metadata={"model_tpm_limit": {model: TEAM_TPM}}))

        borrowed: Final = tuple(_call(owned, model, borrower) for _ in range(CALLS_TO_SATURATE))
        assert all(r.status_code == 200 for r in borrowed), [r.status_code for r in borrowed]
        time.sleep(SATURATION_CACHE_TTL + 0.5)

        forgiven: Final = _call(owned, model, borrower)
        assert forgiven.status_code == 200, forgiven.text
        served: Final = _call(owned, model, newcomer)
        assert served.status_code == 200, served.text

        held: Final = eventually(lambda: _call(owned, model, borrower), lambda r: r.status_code == 429, seconds=10)
        assert held.headers.get("rate_limited_by") == "model_per_team_saturated", dict(held.headers)
        still_served: Final = _call(owned, model, newcomer)
    assert still_served.status_code == 200, still_served.text
