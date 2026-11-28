"""How a provider's `retry-after` sets the router's wait before a retry and the length of a cooldown.

The Python host reads only the `retry-after` header, as whole seconds or an HTTP date
(`retry_after_seconds` in `litellm/router_rust/core.py:536`, `_get_retry_after_from_exception_header` in
`litellm/utils.py:8581`); Azure's `retry-after-ms` is never read by the router. The core then uses it twice.
Before a retry in a group with one deployment, a value above 0 and at most 60 seconds replaces the
exponential backoff (`time_to_sleep` in `litellm/rust/crates/router-core/src/call.rs:1196`), and a longer
one is ignored in favour of the 0.5 second backoff. As a cooldown length, any value is used ahead of the
router's `cooldown_time` with no cap (`cooldown_for_call`, `health.rs:278`), and a value of 0 makes the
cooldown zero, which skips the cooldown entirely (`health.rs:369`).

Proxy one (`num_retries: 1`, `disable_cooldowns: true`) holds lone deployments for the wait tests. Proxy
two (`num_retries: 0`, `cooldown_time: 60`) holds weighted failing deployments with healthy siblings for
the cooldown tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_reliab_support import (
    ANTHROPIC_RATE_TEXT,
    GOOGLE_EXHAUSTED_TEXT,
    Rig,
    anthropic_deployment,
    anthropic_error,
    azure_deployment,
    azure_rate_limit,
    burst,
    chat,
    failing_then_healthy,
    gemini_deployment,
    google_error,
    http_date_in,
    openai_deployment,
    openai_rate_limit,
    reliab_rig,
)
from pydantic import JsonValue

from tests.integration._support.client import eventually
from tests.integration._support.wire import Reply

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

JITTER: Final = 0.75
SLACK: Final = 3.0


def _wait_models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        anthropic_deployment("wait-anthropic", "wait-anthropic", base),
        azure_deployment("wait-azure", "wait-azure", base),
        openai_deployment("wait-openai", "wait-openai", base),
        gemini_deployment("wait-gemini", "wait-gemini", base),
        azure_deployment("wait-azure-ms", "wait-azure-ms", base),
        openai_deployment("wait-long", "wait-long", base),
        anthropic_deployment("wait-date", "wait-date", base),
    )


def _cool_models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        *failing_then_healthy("anthropic", "cool-zero", "cool-zero-bad", "cool-zero-ok", base),
        *failing_then_healthy("azure", "cool-short", "cool-short-bad", "cool-short-ok", base),
        *failing_then_healthy("azure", "cool-ms", "cool-ms-bad", "cool-ms-ok", base),
    )


@pytest.fixture(scope="module")
def waits(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {"num_retries": 1, "disable_cooldowns": True}
    with reliab_rig(tmp_path_factory.mktemp("dec1-reliab-retry-after-wait"), _wait_models, settings) as value:
        yield value


@pytest.fixture(scope="module")
def cools(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {"num_retries": 0, "cooldown_time": 60}
    with reliab_rig(tmp_path_factory.mktemp("dec1-reliab-retry-after-cool"), _cool_models, settings) as value:
        yield value


@pytest.mark.parametrize(
    ("group", "failure"),
    [
        ("wait-anthropic", anthropic_error(429, "rate_limit_error", ANTHROPIC_RATE_TEXT, {"retry-after": "2"})),
        ("wait-azure", azure_rate_limit({"retry-after": "2"})),
        ("wait-openai", openai_rate_limit("rate_limit_exceeded", {"retry-after": "2"})),
        ("wait-gemini", google_error(429, "RESOURCE_EXHAUSTED", GOOGLE_EXHAUSTED_TEXT, {"retry-after": "2"})),
    ],
)
def test_a_providers_retry_after_of_two_seconds_sets_the_wait_before_the_retry(
    waits: Rig, group: str, failure: Reply
) -> None:
    """The lone deployment answers its provider's 429 with `retry-after: 2` once, then 200.

    The caller gets HTTP 200 after two attempts, and the second must arrive at least 2 seconds after the
    first and at most 2 + 0.75 jitter + 3 slack. The plain backoff would be 0.5 to 1.25 seconds, so a port that
    lost the header for any provider waits too little. UNVERIFIED for Gemini and Anthropic: assumes the
    mapped exception keeps the upstream headers through `litellm_response_headers`
    (`exception_mapping_utils.py:2991`).
    """
    label: Final = group
    waits.peer.plan(label, failure)
    response: Final = chat(waits.gateway, group)
    assert response.status_code == 200, response.text
    (gap,) = waits.peer.gaps(label)
    assert 2.0 <= gap < 2.0 + JITTER + SLACK


def test_azure_retry_after_ms_alone_is_ignored_by_the_router_backoff(waits: Rig) -> None:
    """The lone Azure deployment answers 429 with only `retry-after-ms: 4000` once, then 200.

    The router reads `retry-after` alone, so the wait is the first backoff step, 0.5 seconds plus at most 0.75
    jitter: the gap must be at least 0.5 and below 4 seconds. A port that started honouring
    `retry-after-ms` would wait four seconds.
    """
    waits.peer.plan("wait-azure-ms", azure_rate_limit({"retry-after-ms": "4000"}))
    response: Final = chat(waits.gateway, "wait-azure-ms")
    assert response.status_code == 200, response.text
    (gap,) = waits.peer.gaps("wait-azure-ms")
    assert 0.5 <= gap < 4.0


def test_a_retry_after_above_sixty_seconds_is_ignored_for_the_wait(waits: Rig) -> None:
    """The lone OpenAI deployment answers 429 `rate_limit_exceeded` with `retry-after: 120` once, then 200.

    `calculate_retry_after` only obeys values up to 60 seconds, so the router falls back to the 0.5 second
    backoff: the caller gets 200 and the gap is under 0.5 + 0.75 + 3 seconds, far from two minutes.
    """
    waits.peer.plan("wait-long", openai_rate_limit("rate_limit_exceeded", {"retry-after": "120"}))
    response: Final = chat(waits.gateway, "wait-long")
    assert response.status_code == 200, response.text
    (gap,) = waits.peer.gaps("wait-long")
    assert 0.5 <= gap < 0.5 + JITTER + SLACK


def test_a_retry_after_http_date_sets_the_wait_before_the_retry(waits: Rig) -> None:
    """The lone Anthropic deployment answers 429 with `retry-after` as an HTTP date four seconds ahead, then 200.

    The date form is parsed (`email.utils.parsedate_tz`), so the wait must be at least 2 seconds (the date has
    one second resolution and the request takes time to arrive) and below 4 + 0.75 + 3.
    """
    waits.peer.plan(
        "wait-date", anthropic_error(429, "rate_limit_error", ANTHROPIC_RATE_TEXT, {"retry-after": http_date_in(4)})
    )
    response: Final = chat(waits.gateway, "wait-date")
    assert response.status_code == 200, response.text
    (gap,) = waits.peer.gaps("wait-date")
    assert 2.0 <= gap < 4.0 + JITTER + SLACK


def test_a_retry_after_of_zero_on_a_429_skips_the_cooldown(cools: Rig) -> None:
    """The weighted Anthropic deployment answers 429 with `retry-after: 0` every time; its sibling is healthy.

    The header becomes a zero cooldown, which `_should_run_cooldown_logic` skips, so the deployment that would
    otherwise be cooled on its first 429 keeps every request: four requests all return 429 and all four reach
    Anthropic.
    """
    cools.peer.always(
        "cool-zero-bad", anthropic_error(429, "rate_limit_error", ANTHROPIC_RATE_TEXT, {"retry-after": "0"})
    )
    assert burst(cools.gateway, "cool-zero", 4) == (429, 429, 429, 429)
    assert cools.peer.count("cool-zero-bad") == 4
    assert cools.peer.count("cool-zero-ok") == 0


def test_an_azure_retry_after_shorter_than_cooldown_time_ends_the_cooldown_early(cools: Rig) -> None:
    """The weighted Azure deployment answers 429 with `retry-after: 2` once, then 200; router `cooldown_time` is 60.

    The first request returns 429 and the second 200 from the sibling. The provider's two seconds are used as
    the cooldown, so polling must see the Azure deployment serve again well inside 60 seconds, after exactly
    two requests to it.
    """
    cools.peer.plan("cool-short-bad", azure_rate_limit({"retry-after": "2"}))
    assert burst(cools.gateway, "cool-short", 2) == (429, 200)
    eventually(
        lambda: chat(cools.gateway, "cool-short"),
        lambda response: response.status_code == 200 and cools.peer.count("cool-short-bad") == 2,
        seconds=20,
    )


def test_azure_retry_after_ms_does_not_shorten_the_cooldown(cools: Rig) -> None:
    """The weighted Azure deployment answers 429 with only `retry-after-ms: 1000` once, then 200.

    `retry-after-ms` is not read, so the router's 60 second `cooldown_time` applies: twenty requests over the
    following seconds are all served by the sibling (statuses 429 then 200s) and the Azure deployment
    receives just the first one. UNVERIFIED timing: the twenty requests must span more than one second,
    which they do on any runner that needs over 50 ms per proxied call.
    """
    cools.peer.plan("cool-ms-bad", azure_rate_limit({"retry-after-ms": "1000"}))
    first: Final = burst(cools.gateway, "cool-ms", 1)
    assert first == (429,)
    rest: Final = burst(cools.gateway, "cool-ms", 20)
    assert rest == (200,) * 20
    assert cools.peer.count("cool-ms-bad") == 1
