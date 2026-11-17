"""Contract: budget alert events across soft and hard budgets and across resets, and two proxies on one Redis.

Alerts. Every auth of a key with a `max_budget` posts a `token_budget` alert with the key's counter spend
(`virtual_key_max_budget_check`, `litellm/proxy/auth/auth_checks.py:7071-7124`). `SlackAlerting.budget_alerts`
(`litellm/integrations/SlackAlerting/slack_alerting.py:538-637`) picks ONE event per alert from
`_get_event_and_event_message` (`:642-678`): the soft budget check runs first and the hard budget checks overwrite
it, so with both budgets set the event is `budget_crossed`, else `threshold_crossed` (5% or 15% left), else
`soft_budget_crossed`, and the message keeps the soft budget text in front of the hard one. Each event is sent once
per entity and event name, remembered under `budget_alerts:<event>:<id>` for `alerting_args.budget_alert_ttl`
seconds (default 86400, `litellm/types/integrations/slack_alerting.py:47`), a fixed TTL that knows nothing about
`budget_duration`.

Two instances. Holds, resets and reseeds all go through the shared Redis counters with atomic writes (SET NX for a
reseed, `spend_counter_reseed.py:286-296`), so what one proxy does is what the other enforces. Holds are sized from
`max_tokens`, so the overshoot past a budget is bounded by the real cost of the calls admitted, not by the holds.

Price: one served call costs 0.02 (`hold_model`, 5 completion tokens at 0.004).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, held
from tests.integration._support.database import write_rows
from tests.integration._support.wire import Reply, Wire, wire_server
from tests.integration.spend.dec1_budgets2_support import owned
from tests.integration.spend.dec1_budgets4_support import (
    FAST_RESET,
    RESET_SECONDS,
    Due,
    make_due,
    wait_for_row_reset,
)
from tests.integration.spend.dec1_budgets_support import (
    CHAT_COST,
    LANDING_SECONDS,
    background,
    call,
    counter,
    counter_settles,
    held_provider,
    hold_model,
    key_counter,
    redis_client,
    refused,
)
from tests.integration.spend.dec1_spend_support import hashed, scripted_upstream, served, spend_reaches

pytestmark: Final = pytest.mark.shares_worker_state


@contextmanager
def _alerting(
    gateway: Gateway, directory: Path, *, ttl: int | None = None, fast_reset: bool = False
) -> Iterator[tuple[Gateway, Wire]]:
    """An owned proxy that posts budget alerts to a loopback webhook (`alerting: ["webhook"]`, `WEBHOOK_URL`)."""
    settings: Final[dict[str, JsonValue]] = {"alerting": ["webhook"]}
    if ttl is not None:
        settings["alerting_args"] = {"budget_alert_ttl": ttl}
    with (
        wire_server(lambda _: Reply(body=b"{}")) as webhook,
        owned(
            gateway,
            directory,
            general_settings=settings,
            environment={"WEBHOOK_URL": webhook.url, **(FAST_RESET if fast_reset else {})},
        ) as proxy,
    ):
        yield proxy, webhook


def _events_for(webhook: Wire, seen: list[dict[str, JsonValue]], token: str, event: str) -> list[dict[str, JsonValue]]:
    seen.extend(JSON_OBJECT.validate_json(request.body) for request in webhook.drain() if request.method == "POST")
    return [body for body in seen if body.get("event") == event and body.get("token") == token]


@pytest.mark.timeout(300)
def test_a_key_past_its_soft_budget_and_far_from_its_max_gets_soft_budget_crossed_from_the_counter(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key with `soft_budget` 0.01 and `max_budget` 1.0 is served once (counter 0.02) and then once more. The second
    auth's `token_budget` alert reads the counter, finds the soft budget crossed and the hard budget far away, and
    posts one `soft_budget_crossed` event with `event_message` exactly ``Key Budget: Total Soft Budget:`0.01` ``
    without the trailing space (`slack_alerting.py:663-666`). This path does not wait for the stale cached key spend
    the `soft_budget` alert reads (BUG skips in `test_dec1_spend_budget_lifecycle.py`), so only keys without a
    `max_budget` are left with the late alert."""
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (proxy, webhook), proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=0.01, max_budget=1.0)
        served(call(proxy, model, key, max_tokens=5))
        spend_reaches(proxy, "key", hashed(key), CHAT_COST)
        served(call(proxy, model, key, max_tokens=5))
        events: Final = eventually(
            lambda: _events_for(webhook, seen, hashed(key), "soft_budget_crossed"),
            lambda found: len(found) >= 1,
            seconds=LANDING_SECONDS,
        )
        assert events[0]["event_message"] == "Key Budget: Total Soft Budget:`0.01`", events[0]
        assert events[0]["soft_budget"] == pytest.approx(0.01), events[0]


@pytest.mark.timeout(300)
def test_inside_the_last_five_percent_the_threshold_event_replaces_the_soft_budget_event(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A key with `soft_budget` 0.01 and `max_budget` 0.021 is served once (spend 0.02, 4.8% left) and called again.
    The alert is `threshold_crossed`, not `soft_budget_crossed`, and its `event_message` concatenates both texts
    exactly: ``Key Budget: Total Soft Budget:`0.01`5% or less of budget remaining`` (`slack_alerting.py:663-678`).
    No `soft_budget_crossed` event for the key is posted from this path. A port that sends both events, or
    separates the texts, changes what operators receive."""
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (proxy, webhook), proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=0.01, max_budget=0.021)
        served(call(proxy, model, key, max_tokens=5))
        spend_reaches(proxy, "key", hashed(key), CHAT_COST)
        call(proxy, model, key, max_tokens=5)
        events: Final = eventually(
            lambda: _events_for(webhook, seen, hashed(key), "threshold_crossed"),
            lambda found: len(found) >= 1,
            seconds=LANDING_SECONDS,
        )
        assert events[0]["event_message"] == "Key Budget: Total Soft Budget:`0.01`5% or less of budget remaining"


@pytest.mark.timeout(300)
def test_past_both_budgets_the_budget_crossed_event_carries_both_amounts(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `soft_budget` 0.01 and `max_budget` 0.02 is served once and refused on the next call. The webhook
    gets one `budget_crossed` event for the key whose `event_message` is exactly
    ``Key Budget: Total Soft Budget:`0.01`Budget Crossed\\n Total Budget:`0.02` ``, with `soft_budget` 0.01 and
    `max_budget` 0.02 (`slack_alerting.py:663-672`, `WebhookEvent` fields `:604-623`)."""
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (proxy, webhook), proxy.scenario() as scenario:
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=0.01, max_budget=CHAT_COST)
        served(call(proxy, model, key, max_tokens=5))
        spend_reaches(proxy, "key", hashed(key), CHAT_COST)
        refused(call(proxy, model, key, max_tokens=5))
        events: Final = eventually(
            lambda: _events_for(webhook, seen, hashed(key), "budget_crossed"),
            lambda found: len(found) >= 1,
            seconds=LANDING_SECONDS,
        )
        assert events[0]["event_message"] == "Key Budget: Total Soft Budget:`0.01`Budget Crossed\n Total Budget:`0.02`"
        assert (events[0]["soft_budget"], events[0]["max_budget"]) == (pytest.approx(0.01), pytest.approx(CHAT_COST))


@pytest.mark.timeout(400)
def test_a_budget_crossed_alert_is_not_repeated_in_the_next_budget_period(gateway: Gateway, tmp_path: Path) -> None:
    """A `1d` key with `max_budget` 0.02 crosses its budget and posts one `budget_crossed` event. Its budget is then
    reset (spend 0, counter deleted), it is served again and crosses the budget a second time. No second
    `budget_crossed` event for the key reaches the webhook: the dedupe entry `budget_alerts:budget_crossed:<token>`
    lives for `budget_alert_ttl` (24 hours) whatever the key's period (`slack_alerting.py:602-636`). Pinned as is;
    listed as a surprise because a daily budget that is crossed every day alerts at most once per 24 hours of wall
    clock, and an hourly one only once a day."""
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with (
        scripted_upstream() as wire,
        _alerting(gateway, tmp_path, fast_reset=True) as (proxy, webhook),
        proxy.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=CHAT_COST, budget_duration="1d")
        served(call(proxy, model, key, max_tokens=5))
        spend_reaches(proxy, "key", hashed(key), CHAT_COST)
        refused(call(proxy, model, key, max_tokens=5))
        eventually(
            lambda: _events_for(webhook, seen, hashed(key), "budget_crossed"),
            lambda found: len(found) == 1,
            seconds=LANDING_SECONDS,
        )
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(proxy, Due.KEY, hashed(key))
        eventually(lambda: call(proxy, model, key, max_tokens=5), lambda r: r.status_code == 200, seconds=RESET_SECONDS)
        spend_reaches(proxy, "key", hashed(key), CHAT_COST)
        refused(
            eventually(
                lambda: call(proxy, model, key, max_tokens=5), lambda r: r.status_code == 422, seconds=LANDING_SECONDS
            )
        )
        after: Final = held(
            lambda: _events_for(webhook, seen, hashed(key), "budget_crossed"), lambda found: len(found) == 1, holding=5
        )
        assert len(after) == 1, after


@pytest.mark.timeout(300)
def test_a_short_budget_alert_ttl_lets_a_still_crossed_key_alert_again(gateway: Gateway, tmp_path: Path) -> None:
    """With `general_settings.alerting_args.budget_alert_ttl: 3`, a key with `max_budget` 0.02 that keeps being
    refused posts a second `budget_crossed` event once the 3 second dedupe entry expired
    (`alerting_args.budget_alert_ttl`, `slack_alerting.py:632-636`, read from `general_settings`,
    `proxy_server.py:8054`). The TTL is the only knob for re-alerting."""
    seen: Final[list[dict[str, JsonValue]]] = []  # mutable-ok: accumulates webhook deliveries across polls
    with (
        scripted_upstream() as wire,
        _alerting(gateway, tmp_path, ttl=3) as (proxy, webhook),
        proxy.scenario() as scenario,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=CHAT_COST)
        served(call(proxy, model, key, max_tokens=5))
        spend_reaches(proxy, "key", hashed(key), CHAT_COST)

        def crossed() -> list[dict[str, JsonValue]]:
            refused(call(proxy, model, key, max_tokens=5))
            return _events_for(webhook, seen, hashed(key), "budget_crossed")

        assert len(eventually(crossed, lambda found: len(found) >= 2, seconds=LANDING_SECONDS)) >= 2


@pytest.mark.timeout(400)
def test_a_reset_on_one_instance_readmits_the_key_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """A `1d` key with `max_budget` 0.02 is spent through the suite proxy and refused there. The reset runs on a
    second proxy with a fast reset job (stored spend 0, `spend:key:<hash>` deleted). The suite proxy, which never ran
    the reset, then serves the key: its spend comes from the shared counter, reseeded from the reset row, and its
    stale cached key spend (0.02) only triggers a table re-read that confirms 0 (`proxy_server.py:3545-3566`)."""
    with (
        scripted_upstream() as wire,
        gateway.scenario() as scenario,
        owned(gateway, tmp_path, environment=FAST_RESET) as resetter,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=CHAT_COST, budget_duration="1d")
        served(call(gateway, model, key, max_tokens=5))
        spend_reaches(gateway, "key", hashed(key), CHAT_COST)
        refused(call(gateway, model, key, max_tokens=5))
        make_due(Due.KEY, hashed(key))
        wait_for_row_reset(resetter, Due.KEY, hashed(key))
        readmitted: Final = eventually(
            lambda: call(gateway, model, key, max_tokens=5), lambda r: r.status_code == 200, seconds=RESET_SECONDS
        )
        assert readmitted.status_code == 200, readmitted.text


@pytest.mark.timeout(300)
def test_concurrent_reseeds_on_two_instances_seed_the_counter_once(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `max_budget` 1.0 has stored spend 0.5 and no counter. One held call with `max_tokens: 10` starts on
    each proxy at the same time; both find the counter missing and reseed it. With both calls in flight the counter
    reads exactly 0.58 (0.5 once plus two holds of 0.04), not 1.08: the seed is SET NX, so the loser reads the
    winner's value (`spend_counter_reseed.py:286-296`)."""
    with (
        held_provider() as provider,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
        owned(gateway, tmp_path) as other,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        write_rows('UPDATE "LiteLLM_VerificationToken" SET spend = 0.5 WHERE token = %s', (hashed(key),))
        cache.delete(key_counter(key))
        calls: Final = [
            run(lambda proxy=proxy: call(proxy, model, key, hold=True, max_tokens=10)) for proxy in (gateway, other)
        ]
        provider.wait_for(2)
        assert counter(cache, key_counter(key)) == pytest.approx(0.58)
        provider.release()
        assert [future.result(timeout=60).status_code for future in calls] == [200, 200]


@pytest.mark.timeout(300)
def test_the_overshoot_across_two_instances_is_bounded_by_the_admitted_calls(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `max_budget` 0.01 receives four concurrent held calls with `max_tokens: 1` (hold 0.004 each), two
    through each proxy, while the provider answers 5 completion tokens (0.02 each). Exactly three are admitted: two
    full holds and a third shrunk to the 0.002 left (`engine.rs::over_budget`, partial fit), and the fourth is
    refused with HTTP 422. While they wait the counter reads 0.01; after they settle it reads 0.06, three real costs.
    The overshoot is bounded by admitted calls times their real cost, because the hold trusts `max_tokens`."""
    with (
        held_provider() as provider,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
        owned(gateway, tmp_path) as other,
    ):
        model: Final = hold_model(scenario, provider)
        key: Final = scenario.key(models=[model], max_budget=0.01)
        futures: Final = [
            run(lambda proxy=proxy: call(proxy, model, key, hold=True, max_tokens=1))
            for proxy in (gateway, other, gateway, other)
        ]
        provider.wait_for(3)
        eventually(lambda: sum(future.done() for future in futures), lambda done: done >= 1, seconds=60)
        assert counter(cache, key_counter(key)) == pytest.approx(0.01)
        provider.release()
        assert sorted(future.result(timeout=60).status_code for future in futures) == [200, 200, 200, 422]
        assert counter_settles(cache, key_counter(key), 3 * CHAT_COST) == pytest.approx(3 * CHAT_COST)


@pytest.mark.timeout(300)
def test_after_an_overshoot_both_instances_refuse_the_key(gateway: Gateway, tmp_path: Path) -> None:
    """After a key with `max_budget` 0.01 was overshot to 0.02 by one call with `max_tokens: 1` on the suite proxy,
    the next call is refused with HTTP 422 on the suite proxy and on a second proxy alike, each naming
    `Current cost: 0.02`: the shared counter holds the settled overshoot, not the hold."""
    with (
        scripted_upstream() as wire,
        gateway.scenario() as scenario,
        redis_client() as cache,
        owned(gateway, tmp_path) as other,
    ):
        model: Final = hold_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.01)
        served(call(gateway, model, key, max_tokens=1))
        counter_settles(cache, key_counter(key), CHAT_COST)
        for proxy in (gateway, other):
            assert "Current cost: 0.02" in refused(call(proxy, model, key, max_tokens=1))
