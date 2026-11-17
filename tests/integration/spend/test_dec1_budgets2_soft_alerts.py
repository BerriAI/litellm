"""Contract: team soft budget emails and the key `projected_limit_exceeded` alert.

Team soft budget emails. When a team's spend reaches its `soft_budget`, auth (`auth_checks.py::_team_soft_budget_check`)
sends one email to the addresses in the team's `metadata.soft_budget_alerting_emails` (a list, or a comma separated
string). This works even when `general_settings.alerting` is not set (`ProxyLogging.budget_alerts`,
`is_soft_budget_with_alert_emails`), and is sent at most once per team per `EMAIL_BUDGET_ALERT_TTL`
(`email_budget_alerts:soft_budget_crossed:<team_id>`). The subject is
`LiteLLM: Soft Budget Crossed - Total Soft Budget: $<soft_budget>` and the body is
`TEAM_SOFT_BUDGET_ALERT_EMAIL_TEMPLATE`. A team without those addresses gets no email. The emails go through the
proxy's real SMTP client to a scripted SMTP peer (`_support/mail.py::smtp_sink`).

Projected limit. After each charged call `proxy_server.py::update_cache` projects the key's spend to the end of the
month (`utils.py::get_projected_spend_over_limit`: no `budget_duration` means
`spend + spend / max(day_of_month - 1, 1) * days_left_in_month`) and, when that exceeds the key's `soft_budget`,
posts a `projected_limit_exceeded` webhook event with `event_message` `Key Budget: Projected Limit Exceeded`
(`budget_alert_types.py::ProjectedLimitExceededAlert`). The soft budget is passed as `max_budget`, so a spend within
15% of it turns the event into `threshold_crossed` instead.

One served call costs 0.02 (`dec1_spend_support.chat_model`).
"""

from __future__ import annotations

import calendar
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, held
from tests.integration._support.mail import Delivery, Mailbox, smtp_sink
from tests.integration._support.wire import Reply, Wire, wire_server
from tests.integration.spend.dec1_budgets2_support import owned
from tests.integration.spend.dec1_spend_support import (
    CHAT_COST,
    LANDING_SECONDS,
    ask,
    chat_model,
    hashed,
    scripted_upstream,
    served,
)

pytestmark: Final = pytest.mark.shares_worker_state

SOFT: Final = 0.01
SUBJECT: Final = "LiteLLM: Soft Budget Crossed - Total Soft Budget: $0.01"


@contextmanager
def _mail_proxy(gateway: Gateway, directory: Path) -> Iterator[tuple[Gateway, Mailbox]]:
    """An owned proxy with SMTP configured and no `alerting` setting at all."""
    with smtp_sink() as mailbox:
        environment: Final = {
            "SMTP_HOST": mailbox.host,
            "SMTP_PORT": str(mailbox.port),
            "SMTP_TLS": "False",
            "SMTP_SENDER_EMAIL": "alerts@integration.test",
        }
        with owned(gateway, directory, environment=environment) as proxy:
            yield proxy, mailbox


def _until_mail(proxy: Gateway, model: str, key: str, mailbox: Mailbox) -> tuple[Delivery, ...]:
    """Keep calling until the soft budget email arrives; the team spend auth reads lands asynchronously."""

    def attempt() -> tuple[Delivery, ...]:
        served(ask(proxy, model, key))
        return mailbox.deliveries()

    return eventually(attempt, lambda found: len(found) >= 1, seconds=LANDING_SECONDS)


def _addresses() -> tuple[str, str]:
    suffix: Final = uuid.uuid4().hex[:10]
    return f"finance-{suffix}@integration.test", f"owner-{suffix}@integration.test"


@pytest.mark.timeout(300)
def test_a_team_over_its_soft_budget_emails_every_listed_address_once(gateway: Gateway, tmp_path: Path) -> None:
    """A team with `soft_budget` 0.01 and `metadata.soft_budget_alerting_emails` [<finance>, <owner>] calls until its
    spend (0.02 a call) reaches the soft budget. One email arrives, addressed to exactly those two recipients, with
    subject `LiteLLM: Soft Budget Crossed - Total Soft Budget: $0.01`. No `alerting` is configured on the proxy."""
    finance, owner = _addresses()
    with scripted_upstream() as wire, _mail_proxy(gateway, tmp_path) as (proxy, mailbox), proxy.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team_id: Final = scenario.team(
            models=[model], soft_budget=SOFT, metadata={"soft_budget_alerting_emails": [finance, owner]}
        )
        key: Final = scenario.key(team_id=team_id, models=[model])
        deliveries: Final = _until_mail(proxy, model, key, mailbox)
        assert [set(delivery.recipients) for delivery in deliveries] == [{finance, owner}]
        assert deliveries[0].subject == SUBJECT


@pytest.mark.timeout(300)
def test_a_comma_separated_address_string_is_split_into_recipients(gateway: Gateway, tmp_path: Path) -> None:
    """`metadata.soft_budget_alerting_emails` is the string `"<finance>, <owner>"`. The soft budget email goes to both
    addresses, trimmed, exactly as if a list had been given."""
    finance, owner = _addresses()
    with scripted_upstream() as wire, _mail_proxy(gateway, tmp_path) as (proxy, mailbox), proxy.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team_id: Final = scenario.team(
            models=[model], soft_budget=SOFT, metadata={"soft_budget_alerting_emails": f"{finance}, {owner}"}
        )
        key: Final = scenario.key(team_id=team_id, models=[model])
        deliveries: Final = _until_mail(proxy, model, key, mailbox)
        assert set(deliveries[0].recipients) == {finance, owner}


@pytest.mark.timeout(300)
def test_later_requests_over_the_soft_budget_send_no_second_email(gateway: Gateway, tmp_path: Path) -> None:
    """After the first soft budget email, three more calls on the team (all over its soft budget) leave the mailbox
    at exactly one delivery for the next three seconds: the alert is claimed once per team for 24 hours."""
    finance, _ = _addresses()
    with scripted_upstream() as wire, _mail_proxy(gateway, tmp_path) as (proxy, mailbox), proxy.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team_id: Final = scenario.team(
            models=[model], soft_budget=SOFT, metadata={"soft_budget_alerting_emails": [finance]}
        )
        key: Final = scenario.key(team_id=team_id, models=[model])
        _until_mail(proxy, model, key, mailbox)
        for _ in range(3):
            served(ask(proxy, model, key))
        assert len(held(mailbox.deliveries, lambda found: len(found) == 1, holding=3)) == 1


@pytest.mark.timeout(300)
def test_the_soft_budget_email_names_the_team_and_both_amounts(gateway: Gateway, tmp_path: Path) -> None:
    """The HTML body of the soft budget email contains the team alias, `<b>Soft Budget:</b> $0.01` and
    `<b>Maximum Budget:</b> $1.0` for a team with `max_budget` 1.0 (`send_team_soft_budget_alert_email`)."""
    finance, _ = _addresses()
    alias: Final = f"dec1-b2-soft-{uuid.uuid4().hex[:8]}"
    with scripted_upstream() as wire, _mail_proxy(gateway, tmp_path) as (proxy, mailbox), proxy.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team_id: Final = scenario.team(
            models=[model],
            team_alias=alias,
            soft_budget=SOFT,
            max_budget=1.0,
            metadata={"soft_budget_alerting_emails": [finance]},
        )
        key: Final = scenario.key(team_id=team_id, models=[model])
        html: Final = _until_mail(proxy, model, key, mailbox)[0].html
        assert alias in html
        assert "<b>Soft Budget:</b> $0.01" in html
        assert "<b>Maximum Budget:</b> $1.0" in html


@pytest.mark.timeout(300)
def test_a_team_without_alert_addresses_gets_no_soft_budget_email(gateway: Gateway, tmp_path: Path) -> None:
    """A team with `soft_budget` 0.01 but no `soft_budget_alerting_emails` makes three calls (0.06, well over). The
    mailbox stays empty for three seconds: without `alerting` configured, only the team's own addresses get mail."""
    with scripted_upstream() as wire, _mail_proxy(gateway, tmp_path) as (proxy, mailbox), proxy.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team_id: Final = scenario.team(models=[model], soft_budget=SOFT)
        key: Final = scenario.key(team_id=team_id, models=[model])
        for _ in range(3):
            served(ask(proxy, model, key))
        assert held(mailbox.deliveries, lambda found: found == (), holding=3) == ()


@pytest.mark.timeout(300)
def test_a_team_under_its_soft_budget_gets_no_email(gateway: Gateway, tmp_path: Path) -> None:
    """A team with `soft_budget` 1.0 and alert addresses makes two calls (0.04). No email arrives in the following
    three seconds."""
    finance, _ = _addresses()
    with scripted_upstream() as wire, _mail_proxy(gateway, tmp_path) as (proxy, mailbox), proxy.scenario() as scenario:
        model: Final = chat_model(scenario, wire)
        team_id: Final = scenario.team(
            models=[model], soft_budget=1.0, metadata={"soft_budget_alerting_emails": [finance]}
        )
        key: Final = scenario.key(team_id=team_id, models=[model])
        for _ in range(2):
            served(ask(proxy, model, key))
        assert held(mailbox.deliveries, lambda found: found == (), holding=3) == ()


def _projected_soft_budget(today: date) -> float | None:
    """A key soft budget that one 0.02 call is projected to exceed today but that stays more than 15% above 0.02, so
    the event is not turned into `threshold_crossed`; None when no such budget exists on this day of the month."""
    days_left: Final = calendar.monthrange(today.year, today.month)[1] - today.day
    projected: Final = CHAT_COST + CHAT_COST / max(today.day - 1, 1) * days_left
    candidate: Final = (CHAT_COST / 0.85 + projected) / 2
    return candidate if CHAT_COST / 0.85 < candidate < projected else None


@contextmanager
def _webhook_proxy(gateway: Gateway, directory: Path) -> Iterator[tuple[Gateway, Wire]]:
    with (
        wire_server(lambda _: Reply(body=b"{}")) as webhook,
        owned(
            gateway, directory, general_settings={"alerting": ["webhook"]}, environment={"WEBHOOK_URL": webhook.url}
        ) as proxy,
    ):
        yield proxy, webhook


def _events(webhook: Wire, seen: list[dict[str, JsonValue]], event: str) -> list[dict[str, JsonValue]]:
    seen.extend(JSON_OBJECT.validate_json(request.body) for request in webhook.drain() if request.method == "POST")
    return [body for body in seen if body.get("event") == event]


@pytest.mark.timeout(300)
def test_a_key_projected_past_its_soft_budget_posts_projected_limit_exceeded(gateway: Gateway, tmp_path: Path) -> None:
    """A key's `soft_budget` is chosen from today's date so that one 0.02 call projects past it by month end while
    the spend stays more than 15% below it. After one call a webhook event arrives with `event`
    `projected_limit_exceeded`, `event_group` `key`, `token` the hashed key, `event_message`
    `Key Budget: Projected Limit Exceeded`, `max_budget` the soft budget and a `projected_spend` above it. On the
    last days of a month no such budget exists and the test is skipped."""
    soft: Final = _projected_soft_budget(datetime.now(timezone.utc).date())
    if soft is None:
        pytest.skip("no soft budget can be projected past by one call this late in the month")
    seen: Final[list[dict[str, JsonValue]]] = []
    with (
        scripted_upstream() as wire,
        _webhook_proxy(gateway, tmp_path) as (proxy, webhook),
        proxy.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=soft)
        served(ask(proxy, model, key))
        event: Final = eventually(
            lambda: [
                body for body in _events(webhook, seen, "projected_limit_exceeded") if body.get("token") == hashed(key)
            ],
            lambda found: len(found) >= 1,
            seconds=LANDING_SECONDS,
        )[0]
        assert event["event_group"] == "key", event
        assert event["event_message"] == "Key Budget: Projected Limit Exceeded", event
        assert event["max_budget"] == pytest.approx(soft), event
        projected = event["projected_spend"]
        assert isinstance(projected, (int, float)) and projected > soft, event


@pytest.mark.timeout(300)
def test_projected_limit_exceeded_is_posted_once_per_key(gateway: Gateway, tmp_path: Path) -> None:
    """Same key and soft budget. After the first `projected_limit_exceeded` event, a second call posts no second
    one within three seconds: the key enters `soft_budget_cooldown` and the alert is cached for 24 hours under
    `budget_alerts:projected_limit_exceeded:<token>`. UNVERIFIED: the second projection may already sit within 15% of
    the soft budget, which would post `threshold_crossed`; this test only counts `projected_limit_exceeded`."""
    soft: Final = _projected_soft_budget(datetime.now(timezone.utc).date())
    if soft is None:
        pytest.skip("no soft budget can be projected past by one call this late in the month")
    seen: Final[list[dict[str, JsonValue]]] = []
    with (
        scripted_upstream() as wire,
        _webhook_proxy(gateway, tmp_path) as (proxy, webhook),
        proxy.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=soft)
        served(ask(proxy, model, key))

        def mine() -> list[dict[str, JsonValue]]:
            return [
                body for body in _events(webhook, seen, "projected_limit_exceeded") if body.get("token") == hashed(key)
            ]

        eventually(mine, lambda found: len(found) == 1, seconds=LANDING_SECONDS)
        served(ask(proxy, model, key))
        assert len(held(mine, lambda found: len(found) == 1, holding=3)) == 1


@pytest.mark.timeout(300)
def test_a_key_far_below_any_projection_posts_no_projected_alert(gateway: Gateway, tmp_path: Path) -> None:
    """A key with `soft_budget` 100.0 makes one 0.02 call. No `projected_limit_exceeded` event is posted for its token
    within three seconds: one call cannot be projected to 100 by month end on any day."""
    seen: Final[list[dict[str, JsonValue]]] = []
    with (
        scripted_upstream() as wire,
        _webhook_proxy(gateway, tmp_path) as (proxy, webhook),
        proxy.scenario() as scenario,
    ):
        model: Final = chat_model(scenario, wire)
        key: Final = scenario.key(models=[model], soft_budget=100.0)
        served(ask(proxy, model, key))
        found: Final = held(
            lambda: [
                body for body in _events(webhook, seen, "projected_limit_exceeded") if body.get("token") == hashed(key)
            ],
            lambda events: events == [],
            holding=3,
        )
        assert found == []
