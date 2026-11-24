"""The dynamic rate limiter publishes each team's share of a model, what the team has used above that share, and
every refusal with the same reason the 429 carries in its rate_limited_by header."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Reply, Request, wire_server

OPENAI_MODEL: Final = "gpt-4.1-mini"
MODEL_TPM: Final = 1000
TEAM_TPM: Final = 100
REPLY_TOKENS: Final = 60
SATURATION_THRESHOLD: Final = 0.4
IDLE_CALLS: Final = TEAM_TPM // REPLY_TOKENS + 1
WARMUP_CALLS: Final = int(MODEL_TPM * SATURATION_THRESHOLD) // REPLY_TOKENS + 1
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_dynamic_limiter_metrics",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "metered"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": REPLY_TOKENS - 20, "total_tokens": REPLY_TOKENS},
    }
).encode()


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/v1/chat/completions"), request.target
    return Reply(body=UPSTREAM_REPLY)


@pytest.fixture(scope="module")
def saturated_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp("dynamic-limiter-metrics")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3", "prometheus"],
        "priority_reservation": {"interactive": 0.6, "batch": 0.4},
        "priority_reservation_settings": {
            "default_priority": 1.0,
            "saturation_threshold": SATURATION_THRESHOLD,
            "saturation_check_cache_ttl": 0,
            "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
        },
    }
    path: Final = directory / "dynamic_limiter_metrics.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        gateway_from_environment() as rig,
        wire_server(_respond) as wire,
        owned_proxy(rig, directory, {}, config=path) as owned,
    ):
        yield owned, wire.url


def _team_series(samples: Sequence[Sample], name: str, team: str) -> Mapping[str, float]:
    """The metric's samples for one team, keyed by the label values that are not the team itself."""
    return {
        json.dumps({k: v for k, v in sample.labels.items() if k != "team"}, sort_keys=True): sample.value
        for sample in samples
        if sample.name == name and sample.labels.get("team") == team
    }


def _call(owned: Gateway, model: str, key: str) -> httpx.Response:
    return owned.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"hi {uuid.uuid4().hex[:12]}"}]},
        key=key,
    )


def _settled_calls(owned: Gateway, model: str, team: str, key: str, count: int) -> None:
    for index in range(count):
        response: Final = _call(owned, model, key)
        assert response.status_code == 200, response.text
        eventually(
            lambda: sum(_team_series(scrape(owned), "litellm_total_tokens_metric_total", team).values()),
            lambda tokens, expected=(index + 1) * REPLY_TOKENS: tokens == expected,
            seconds=30,
        )


def test_a_team_over_its_share_shows_up_as_borrowed_tokens_and_a_matching_throttle_reason(
    saturated_model: tuple[Gateway, str],
) -> None:
    owned, upstream = saturated_model
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        team: Final = scenario.team(metadata={"model_tpm_limit": {model: TEAM_TPM}})
        key: Final = scenario.key(team_id=team)
        _settled_calls(owned, model, team, key, IDLE_CALLS)
        assert IDLE_CALLS * REPLY_TOKENS > TEAM_TPM
        warmup: Final = scenario.team(metadata={"priority": "batch"})
        _settled_calls(owned, model, warmup, scenario.key(team_id=warmup), WARMUP_CALLS)
        responses: Final = tuple(_call(owned, model, key) for _ in range(IDLE_CALLS + 1))
        blocked: Final = tuple(response for response in responses if response.status_code == 429)
        expected_borrowed: Final = IDLE_CALLS * REPLY_TOKENS - TEAM_TPM
        samples: Final = eventually(
            lambda: scrape(owned),
            lambda scraped: (
                tuple(_team_series(scraped, "litellm_dynamic_limiter_borrowed_tokens", team).values())
                == (expected_borrowed,)
                and tuple(_team_series(scraped, "litellm_dynamic_limiter_throttled_total", team).values())
                == (len(blocked),)
            ),
            seconds=30,
            return_last_on_timeout=True,
        )

    assert responses[0].status_code == 200, responses[0].text
    assert tuple(response.status_code for response in responses) == (200,) * IDLE_CALLS + (429,), [
        r.text for r in responses
    ]
    reason: Final = blocked[0].headers.get("rate_limited_by")
    assert reason == "model_per_team_saturated", dict(blocked[0].headers)

    share: Final = _team_series(samples, "litellm_dynamic_limiter_share_tokens", team)
    borrowed: Final = _team_series(samples, "litellm_dynamic_limiter_borrowed_tokens", team)
    throttled: Final = _team_series(samples, "litellm_dynamic_limiter_throttled_total", team)
    assert share, [s for s in samples if s.name.startswith("litellm_dynamic_limiter")]
    ((share_labels, share_tokens),) = share.items()
    labels: Final[dict[str, str]] = json.loads(share_labels)
    assert labels["model_group"] == model and labels["priority"] == "default", labels
    assert share_tokens == TEAM_TPM
    assert borrowed[share_labels] == expected_borrowed, borrowed
    assert borrowed[share_labels] > 0, borrowed
    ((throttle_labels, refused),) = throttled.items()
    throttle: Final[dict[str, str]] = json.loads(throttle_labels)
    assert throttle["reason"] == reason and throttle["rate_limit_type"] == "tokens", throttle
    assert refused == len(blocked)


def test_a_team_under_its_share_borrows_nothing(saturated_model: tuple[Gateway, str]) -> None:
    owned, upstream = saturated_model
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        team: Final = scenario.team(metadata={"model_tpm_limit": {model: TEAM_TPM}})
        warmup: Final = scenario.team(metadata={"priority": "batch"})
        _settled_calls(owned, model, warmup, scenario.key(team_id=warmup), WARMUP_CALLS)
        response: Final[httpx.Response] = owned.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"one call {uuid.uuid4().hex}"}]},
            key=scenario.key(team_id=team),
        )
        samples: Final = eventually(
            lambda: scrape(owned),
            lambda scraped: bool(_team_series(scraped, "litellm_dynamic_limiter_share_tokens", team)),
            seconds=30,
            return_last_on_timeout=True,
        )
    assert response.status_code == 200, response.text
    assert tuple(_team_series(samples, "litellm_dynamic_limiter_share_tokens", team).values()) == (TEAM_TPM,)
    assert tuple(_team_series(samples, "litellm_dynamic_limiter_borrowed_tokens", team).values()) == (0,)
    assert not _team_series(samples, "litellm_dynamic_limiter_throttled_total", team)


def test_a_team_served_while_the_model_is_idle_publishes_its_share_with_nothing_borrowed(
    saturated_model: tuple[Gateway, str],
) -> None:
    owned, upstream = saturated_model
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        team: Final = scenario.team(metadata={"model_tpm_limit": {model: TEAM_TPM}})
        _settled_calls(owned, model, team, scenario.key(team_id=team), IDLE_CALLS)
        samples: Final = eventually(
            lambda: scrape(owned),
            lambda scraped: bool(_team_series(scraped, "litellm_dynamic_limiter_share_tokens", team)),
            seconds=30,
            return_last_on_timeout=True,
        )
    assert IDLE_CALLS * REPLY_TOKENS > TEAM_TPM
    assert IDLE_CALLS * REPLY_TOKENS < MODEL_TPM * SATURATION_THRESHOLD
    assert tuple(_team_series(samples, "litellm_dynamic_limiter_share_tokens", team).values()) == (TEAM_TPM,)
    assert tuple(_team_series(samples, "litellm_dynamic_limiter_borrowed_tokens", team).values()) == (0,)
    assert not _team_series(samples, "litellm_dynamic_limiter_throttled_total", team)
