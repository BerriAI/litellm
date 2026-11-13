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
FALLBACK_RPM: Final = 1000
SATURATION_THRESHOLD: Final = 0.5
FALLBACK_REASON_HEADER: Final = "x-litellm-fallback-reason"
SATURATION_REASON: Final = "rate_limit:model_saturation_fallback"


def _reply() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl_{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1700000000,
            "model": OPENAI_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "served"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
    ).encode()


@pytest.fixture(scope="module")
def saturation_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp("fallback-on-saturation")
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
    path: Final = directory / "fallback_on_saturation.yaml"
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
        {"model": model, "messages": [{"role": "user", "content": f"fallback on saturation {uuid.uuid4().hex}"}]},
        key=key,
    )


def _spend_row(owned: Gateway, response: httpx.Response) -> list[dict[str, object]]:
    return owned.rows(
        'SELECT model_group FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (str(response.json()["id"]),)
    )


def test_a_team_with_fallback_on_saturation_is_served_by_its_fallback_once_the_model_saturates(
    saturation_proxy: tuple[Gateway, str],
) -> None:
    owned, upstream = saturation_proxy
    with owned.scenario() as scenario:
        preferred, preferred_id = _deployment(owned, scenario, upstream, PREFERRED_RPM)
        fallback, fallback_id = _deployment(owned, scenario, upstream, FALLBACK_RPM)
        rigid_key: Final = scenario.key(team_id=scenario.team(models=[preferred, fallback]))
        flexible_key: Final = scenario.key(
            team_id=scenario.team(
                models=[preferred, fallback],
                router_settings={"fallbacks": [{preferred: [fallback]}], "fallback_on_saturation": True},
            )
        )

        warmup: Final = _chat(owned, rigid_key, preferred)
        assert warmup.status_code == 200, warmup.text
        assert warmup.headers["x-litellm-model-id"] == preferred_id
        assert FALLBACK_REASON_HEADER not in warmup.headers

        fell_back: Final = eventually(
            lambda: _chat(owned, flexible_key, preferred),
            lambda response: response.headers.get(FALLBACK_REASON_HEADER) == SATURATION_REASON,
            seconds=15,
            return_last_on_timeout=True,
        )
        assert fell_back.status_code == 200, fell_back.text
        assert fell_back.headers[FALLBACK_REASON_HEADER] == SATURATION_REASON
        assert fell_back.headers["x-litellm-attempted-fallbacks"] == "1"
        assert fell_back.headers["x-litellm-model-id"] == fallback_id

        rigid: Final = _chat(owned, rigid_key, preferred)
        assert rigid.status_code == 200, rigid.text
        assert rigid.headers["x-litellm-model-id"] == preferred_id
        assert FALLBACK_REASON_HEADER not in rigid.headers

        charged: Final = eventually(lambda: _spend_row(owned, fell_back), lambda rows: len(rows) == 1, seconds=30)
        assert charged[0]["model_group"] == fallback


def test_a_fallback_the_team_may_not_call_is_never_used(saturation_proxy: tuple[Gateway, str]) -> None:
    owned, upstream = saturation_proxy
    with owned.scenario() as scenario:
        preferred, preferred_id = _deployment(owned, scenario, upstream, PREFERRED_RPM)
        forbidden, _ = _deployment(owned, scenario, upstream, FALLBACK_RPM)
        key: Final = scenario.key(
            team_id=scenario.team(
                models=[preferred],
                router_settings={"fallbacks": [{preferred: [forbidden]}], "fallback_on_saturation": True},
            )
        )

        def served_by_preferred_or_refused() -> httpx.Response:
            response: Final = _chat(owned, key, preferred)
            assert FALLBACK_REASON_HEADER not in response.headers, response.headers
            if response.status_code == 200:
                assert response.headers["x-litellm-model-id"] == preferred_id
            return response

        refused: Final = eventually(
            served_by_preferred_or_refused,
            lambda response: response.status_code != 200,
            seconds=15,
            return_last_on_timeout=True,
        )
        assert refused.status_code == 429, refused.text
