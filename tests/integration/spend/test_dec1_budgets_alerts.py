"""Contract: budget alerts posted to a webhook as spend approaches and crosses budgets.

With `general_settings.alerting: ["webhook"]` and `WEBHOOK_URL` set, budget checks in auth call
`ProxyLogging.budget_alerts`, which posts one JSON `WebhookEvent` per (event, entity) and then remembers it for
`budget_alert_ttl` (24 hours) so the same alert is not sent again
(`litellm/integrations/SlackAlerting/slack_alerting.py::budget_alerts`). The event is chosen from the spend the
check read and the entity's budgets:

    spend >= max_budget                 budget_crossed     "<Entity> Budget: Budget Crossed\\n Total Budget:`<max>`"
    5% or less of max_budget left       threshold_crossed  "<Entity> Budget: 5% or less of budget remaining"
    15% or less of max_budget left      threshold_crossed  "<Entity> Budget: 15% or less of budget remaining"

Keys and users are evaluated on every request, so they get threshold alerts. Teams, organizations and projects
only alert from the branch that refuses the request, so they never get a threshold alert, only `budget_crossed`.
The prefixes are `Key Budget: `, `User Budget: `, `Team Budget: `, `Organization Budget: ` and
`Project Budget: ` (`budget_alert_types.py`). Without `budget_alerts` in `alert_types` nothing is sent.

The key soft budget and key `budget_crossed` events are pinned in `test_dec1_spend_budget_lifecycle.py`. One
served call costs 0.02.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, held
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Wire, wire_server
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    TINY_BUDGET,
    ask,
    chat_model,
    hashed,
    proxy_config,
    scripted_upstream,
    served,
    spend_reaches,
)

pytestmark: Final = pytest.mark.shares_worker_state


@contextmanager
def _alerting(
    gateway: Gateway, directory: Path, general_settings: Mapping[str, JsonValue] | None = None
) -> Iterator[tuple[Gateway, Wire]]:
    config: Final = proxy_config(directory, general_settings={"alerting": ["webhook"], **(general_settings or {})})
    with (
        wire_server(lambda _: Reply(body=b"{}")) as webhook,
        owned_proxy(gateway, directory, {"WEBHOOK_URL": webhook.url}, config=config) as owned,
    ):
        yield owned, webhook


class _Inbox:
    def __init__(self, webhook: Wire) -> None:
        self.webhook = webhook
        self.seen: list[dict[str, JsonValue]] = []  # mutable-ok: accumulates deliveries across polls

    def events(self, event: str) -> list[dict[str, JsonValue]]:
        self.seen.extend(
            JSON_OBJECT.validate_json(request.body) for request in self.webhook.drain() if request.method == "POST"
        )
        return [body for body in self.seen if body.get("event") == event]

    def first(self, event: str, **match: JsonValue) -> dict[str, JsonValue]:
        found: Final = eventually(
            lambda: [body for body in self.events(event) if all(body.get(k) == v for k, v in match.items())],
            lambda bodies: len(bodies) >= 1,
            seconds=LANDING_SECONDS,
        )
        return found[0]


@pytest.mark.timeout(300)
def test_a_key_with_five_percent_left_gets_one_threshold_crossed_alert(gateway: Gateway, tmp_path: Path) -> None:
    """A key has `max_budget` 0.021. After one call (0.02) 4.8% of the budget is left. The next request is still
    served and posts `event` `threshold_crossed`, `event_group` `key`, `token` the hashed key and `event_message`
    `Key Budget: 5% or less of budget remaining`."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.021)
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        served(ask(owned, model, key))
        event: Final = inbox.first("threshold_crossed", token=hashed(key))
        assert event["event_group"] == "key", event
        assert event["event_message"] == "Key Budget: 5% or less of budget remaining", event
        assert event["max_budget"] == pytest.approx(0.021), event


@pytest.mark.timeout(300)
def test_a_key_with_fifteen_percent_left_gets_the_fifteen_percent_message(gateway: Gateway, tmp_path: Path) -> None:
    """A key has `max_budget` 0.0225. After one call 11.1% is left, which is above 5% and at most 15%, so the next
    request posts `threshold_crossed` with `event_message` `Key Budget: 15% or less of budget remaining`."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.0225)
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        served(ask(owned, model, key))
        event: Final = inbox.first("threshold_crossed", token=hashed(key))
        assert event["event_message"] == "Key Budget: 15% or less of budget remaining", event


@pytest.mark.timeout(300)
def test_a_key_far_from_its_budget_posts_no_budget_alert(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `max_budget` 1.0 makes three calls (0.06, 94% left). No `threshold_crossed` or `budget_crossed`
    event is posted for its token in the following three seconds."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=1.0)
        for _ in range(3):
            served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), 3 * CHAT_COST)
        served(ask(owned, model, key))
        assert (
            held(
                lambda: [
                    body
                    for event in ("threshold_crossed", "budget_crossed")
                    for body in inbox.events(event)
                    if body.get("token") == hashed(key)
                ],
                lambda bodies: bodies == [],
                holding=3,
            )
            == []
        )


@pytest.mark.timeout(300)
def test_a_threshold_alert_is_sent_once_per_key_not_per_request(gateway: Gateway, tmp_path: Path) -> None:
    """After the first `threshold_crossed` event for a key with 4.8% left, four more requests (each still served,
    the budget has 0.001 left and each hold is shrunk to it) do not post a second `threshold_crossed` for that
    key: the alert is remembered under `budget_alerts:threshold_crossed:<hashed key>`.
    UNVERIFIED: the later calls push spend past the budget, so a `budget_crossed` event is allowed; only a second
    `threshold_crossed` is forbidden."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.021)
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        served(ask(owned, model, key))
        inbox.first("threshold_crossed", token=hashed(key))
        for _ in range(4):
            ask(owned, model, key)
        assert (
            len(
                held(
                    lambda: [body for body in inbox.events("threshold_crossed") if body.get("token") == hashed(key)],
                    lambda bodies: len(bodies) == 1,
                    holding=3,
                )
            )
            == 1
        )


@pytest.mark.timeout(300)
def test_a_team_over_its_budget_posts_budget_crossed_with_the_team_prefix(gateway: Gateway, tmp_path: Path) -> None:
    """A team with `max_budget` 0.0000001 is refused after one call. The refusal posts `event` `budget_crossed`,
    `event_group` `team`, `team_id` the team and `event_message` exactly
    ``Team Budget: Budget Crossed\\n Total Budget:`1e-07` ``."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(max_budget=TINY_BUDGET, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(owned, model, key))
        spend_reaches(owned, "team", team, CHAT_COST)
        assert ask(owned, model, key).status_code == 422
        event: Final = inbox.first("budget_crossed", team_id=team, event_group="team")
        assert event["event_message"] == "Team Budget: Budget Crossed\n Total Budget:`1e-07`", event


@pytest.mark.timeout(300)
def test_a_team_close_to_its_budget_gets_no_threshold_alert(gateway: Gateway, tmp_path: Path) -> None:
    """A team has `max_budget` 0.021; after one call 4.8% is left and the team's key is still served. Unlike keys
    and users, the team check only alerts from the refusal branch (`auth_checks.py::team_max_budget_check`), so
    no `threshold_crossed` event with that `team_id` and `event_group` `team` is posted."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(max_budget=0.021, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(owned, model, key))
        spend_reaches(owned, "team", team, CHAT_COST)
        served(ask(owned, model, key))
        assert (
            held(
                lambda: [
                    body
                    for body in inbox.events("threshold_crossed")
                    if body.get("team_id") == team and body.get("event_group") == "team"
                ],
                lambda bodies: bodies == [],
                holding=3,
            )
            == []
        )


@pytest.mark.timeout(300)
def test_a_user_with_five_percent_left_gets_a_user_threshold_alert(gateway: Gateway, tmp_path: Path) -> None:
    """An internal user has `max_budget` 0.021 and a personal key. After one call the user's next request posts
    `threshold_crossed` with `event_group` `user`, `user_id` the user and `event_message` `User Budget: 5% or
    less of budget remaining`."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        user: Final = scenario.user(max_budget=0.021)
        key: Final = scenario.key(user_id=user, models=[model])
        served(ask(owned, model, key))
        spend_reaches(owned, "user", user, CHAT_COST)
        served(ask(owned, model, key))
        event: Final = inbox.first("threshold_crossed", user_id=user, event_group="user")
        assert event["event_message"] == "User Budget: 5% or less of budget remaining", event


@pytest.mark.timeout(300)
def test_an_organization_over_its_budget_posts_budget_crossed_with_the_organization_prefix(
    gateway: Gateway, tmp_path: Path
) -> None:
    """An organization with `max_budget` 0.0000001 refuses its team's key after one call. The refusal posts
    `budget_crossed` with `event_group` `organization`, `organization_id` the org and an `event_message` that
    starts `Organization Budget: Budget Crossed`."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        organization: Final = scenario.organization(max_budget=TINY_BUDGET, models=[model])
        team: Final = scenario.team(organization_id=organization, models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        served(ask(owned, model, key))
        spend_reaches(owned, "org", organization, CHAT_COST)
        assert ask(owned, model, key).status_code == 422
        event: Final = inbox.first("budget_crossed", organization_id=organization, event_group="organization")
        assert str(event["event_message"]).startswith("Organization Budget: Budget Crossed"), event


@pytest.mark.timeout(300)
def test_a_project_over_its_budget_posts_budget_crossed_with_the_project_prefix(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A project with `max_budget` 0.0000001 refuses its key after one call and posts `budget_crossed` with
    `event_group` `project` and an `event_message` that starts `Project Budget: Budget Crossed`.
    UNVERIFIED: assumes `/project/new` stores `max_budget` and keys accept `project_id` (see the boundaries file)."""
    with scripted_upstream() as wire, _alerting(gateway, tmp_path) as (owned, webhook), owned.scenario() as scenario:
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, max_budget=TINY_BUDGET, models=[model])
        key: Final = scenario.key(team_id=team, project_id=project, models=[model])
        served(ask(owned, model, key))
        eventually(lambda: ask(owned, model, key).status_code, lambda status: status == 422, seconds=LANDING_SECONDS)
        event: Final = inbox.first("budget_crossed", event_group="project")
        assert str(event["event_message"]).startswith("Project Budget: Budget Crossed"), event


@pytest.mark.timeout(300)
def test_no_budget_alert_is_sent_when_budget_alerts_is_not_an_enabled_alert_type(
    gateway: Gateway, tmp_path: Path
) -> None:
    """With `alerting: ["webhook"]` but `alert_types: ["llm_exceptions"]`, a key that crosses its budget is
    refused as usual, but no `budget_crossed` or `threshold_crossed` event reaches the webhook."""
    with (
        scripted_upstream() as wire,
        _alerting(gateway, tmp_path, {"alert_types": ["llm_exceptions"]}) as (owned, webhook),
        owned.scenario() as scenario,
    ):
        inbox: Final = _Inbox(webhook)
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=TINY_BUDGET)
        served(ask(owned, model, key))
        spend_reaches(owned, "key", hashed(key), CHAT_COST)
        assert ask(owned, model, key).status_code == 422
        assert (
            held(
                lambda: inbox.events("budget_crossed") + inbox.events("threshold_crossed"),
                lambda bodies: bodies == [],
                holding=3,
            )
            == []
        )
