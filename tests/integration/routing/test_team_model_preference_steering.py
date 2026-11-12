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

OPENAI_MODEL: Final = "gpt-5.6"
PREFERRED_RPM: Final = 2
ALTERNATE_RPM: Final = 1000
SATURATION_THRESHOLD: Final = 0.5
STEERED_FROM_HEADER: Final = "x-litellm-model-steered-from"


def _reply() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": OPENAI_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "steered"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
    ).encode()


@pytest.fixture(scope="module")
def steering_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp("model-preference-steering")
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration: Final = {
        **base,
        "litellm_settings": {
            **base["litellm_settings"],
            "callbacks": ["dynamic_rate_limiter_v3"],
            "priority_reservation_settings": {
                "saturation_threshold": SATURATION_THRESHOLD,
                "saturation_check_cache_ttl": 1,
            },
        },
    }
    path: Final = directory / "model_preference_steering.yaml"
    path.write_text(yaml.safe_dump(configuration))

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions", request.target
        return Reply(body=_reply())

    with (
        gateway_from_environment() as rig,
        wire_server(respond) as wire,
        owned_proxy(rig, directory, {}, config=path) as owned,
    ):
        yield owned, wire.url


def _deployment(owned: Gateway, scenario: Scenario, upstream: str, rpm: int) -> tuple[str, str]:
    name: Final = scenario.model(
        model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-key", rpm=rpm
    )
    rows: Final = owned.rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s', (name,))
    return name, str(rows[0]["model_id"])


def _chat(owned: Gateway, key: str, model: str) -> httpx.Response:
    return owned.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"model preference steering {uuid.uuid4().hex}"}]},
        key=key,
    )


def _spend_row(owned: Gateway, response: httpx.Response) -> list[dict[str, object]]:
    return owned.rows(
        'SELECT model_group FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (str(response.json()["id"]),)
    )


def test_flexible_team_is_steered_to_its_permitted_alternate_once_the_preferred_model_saturates(
    steering_proxy: tuple[Gateway, str],
) -> None:
    owned, upstream = steering_proxy
    with owned.scenario() as scenario:
        preferred, preferred_id = _deployment(owned, scenario, upstream, PREFERRED_RPM)
        alternate, alternate_id = _deployment(owned, scenario, upstream, ALTERNATE_RPM)
        rigid_key: Final = scenario.key(team_id=scenario.team(models=[preferred, alternate]))
        flexible_key: Final = scenario.key(
            team_id=scenario.team(
                models=[preferred, alternate],
                metadata={"model_preferences": [{"model": preferred, "alternates": [alternate]}]},
            )
        )

        warmup: Final = _chat(owned, rigid_key, preferred)
        assert warmup.status_code == 200, warmup.text
        assert warmup.headers["x-litellm-model-id"] == preferred_id
        assert STEERED_FROM_HEADER not in warmup.headers

        steered: Final = eventually(
            lambda: _chat(owned, flexible_key, preferred),
            lambda response: STEERED_FROM_HEADER in response.headers,
            seconds=15,
            return_last_on_timeout=True,
        )
        assert steered.status_code == 200, steered.text
        assert steered.headers[STEERED_FROM_HEADER] == preferred
        assert steered.headers["x-litellm-model-id"] == alternate_id

        rigid: Final = _chat(owned, rigid_key, preferred)
        assert rigid.status_code == 200, rigid.text
        assert rigid.headers["x-litellm-model-id"] == preferred_id
        assert STEERED_FROM_HEADER not in rigid.headers

        charged: Final = eventually(lambda: _spend_row(owned, steered), lambda rows: len(rows) == 1, seconds=30)
        assert charged[0]["model_group"] == alternate


def test_alternate_the_team_may_not_call_is_never_used(steering_proxy: tuple[Gateway, str]) -> None:
    owned, upstream = steering_proxy
    with owned.scenario() as scenario:
        preferred, _ = _deployment(owned, scenario, upstream, PREFERRED_RPM)
        forbidden, _ = _deployment(owned, scenario, upstream, ALTERNATE_RPM)
        key: Final = scenario.key(
            team_id=scenario.team(
                models=[preferred],
                metadata={"model_preferences": [{"model": preferred, "alternates": [forbidden]}]},
            )
        )

        first: Final = _chat(owned, key, preferred)
        assert first.status_code == 200, first.text
        second: Final = eventually(
            lambda: _chat(owned, key, preferred), lambda response: response.status_code != 200, seconds=15
        )
        assert second.status_code == 429, second.text
        assert STEERED_FROM_HEADER not in second.headers
