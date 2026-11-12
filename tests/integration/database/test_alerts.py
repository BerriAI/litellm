import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
import pytest_asyncio
from psycopg import sql

from litellm.models.alert import (
    Alert,
    AlertDetection,
    AlertPage,
    AlertResourceType,
    AlertType,
    AlertView,
    AlertViewPage,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.alert_repository import AlertRepository
from tests.integration._support.client import Gateway
from tests.integration._support.database import read_rows, write_rows


def saved_alert(content: bytes) -> Alert:
    view: Final = AlertView.model_validate_json(content)
    return Alert.model_validate(view.model_dump(exclude={"resource_name", "history"}))


def saved_page(content: bytes) -> AlertPage:
    page: Final = AlertViewPage.model_validate_json(content)
    return AlertPage(
        data=tuple(saved_alert(alert.model_dump_json().encode()) for alert in page.data), next_cursor=page.next_cursor
    )


@dataclass(frozen=True, slots=True)
class Alerts:
    repository: AlertRepository
    resource_id: str

    def detection(self, kind: AlertType = "spend", day: int = 0) -> AlertDetection:
        start: Final = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
        evidence: Final = (
            {"deprecation_date": "2026-03-01", "replacement_model": "example/replacement"}
            if kind == "model_deprecation"
            else {
                "value": 0.5 if kind == "reliability" else 20,
                "baseline": 0.1 if kind == "reliability" else 2,
                "unit": {"spend": "usd", "traffic": "requests", "latency": "milliseconds", "reliability": "ratio"}[
                    kind
                ],
                "sample_size": 100,
            }
        )
        return AlertDetection.model_validate(
            {
                "type": kind,
                "resource_type": "model",
                "resource_id": self.resource_id,
                "window_start": start,
                "window_end": start + timedelta(days=1),
                "evidence": evidence,
            }
        )


@pytest_asyncio.fixture(loop_scope="function")
async def alerts() -> AsyncIterator[Alerts]:
    resource_id: Final = f"alerts-test-{uuid.uuid4().hex}"
    client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(UserApiKeyCache()))
    await client.connect()
    try:
        yield Alerts(AlertRepository(client), resource_id)
    finally:
        try:
            write_rows('DELETE FROM "LiteLLM_Alert" WHERE resource_id=%s', (resource_id,))
        finally:
            await client.disconnect()


@pytest.mark.parametrize("kind", ("spend", "traffic", "latency", "reliability", "model_deprecation"))
@pytest.mark.asyncio
async def test_each_alert_type_round_trips_through_api_and_status_changes(
    alerts: Alerts, gateway: Gateway, kind: AlertType
) -> None:
    detection: Final = alerts.detection(kind)
    saved: Final = await alerts.repository.record(detection)
    response: Final = gateway.request("GET", f"/alerts/{saved.id}")
    assert response.status_code == 200, response.text
    assert saved_alert(response.content) == saved
    assert saved.status == "open"
    assert saved.model_dump(include=set(AlertDetection.model_fields)) == detection.model_dump()
    for status in ("resolved", "dismissed", "open"):
        update: Final = gateway.request("PATCH", f"/alerts/{saved.id}", {"status": status})
        assert update.status_code == 200, update.text
        changed: Final = saved_alert(update.content)
        assert changed.status == status
        assert changed.updated_by
        assert changed.evidence == saved.evidence
        assert changed.created_at == saved.created_at
        assert changed.updated_at >= saved.updated_at
        assert await alerts.repository.get(int(saved.id)) == changed


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("spend", "traffic", "latency", "reliability"))
async def test_gateway_alert_round_trips_without_colliding_with_resource_alerts(
    alerts: Alerts, gateway: Gateway, kind: AlertType
) -> None:
    detection: Final = AlertDetection.model_validate(
        {**alerts.detection(kind).model_dump(), "resource_type": "gateway", "resource_id": "gateway"}
    )
    try:
        saved: Final = await alerts.repository.record(detection)
        scoped: Final = await alerts.repository.record(detection.model_copy(update={"resource_type": "model"}))
        assert scoped.id != saved.id
        detail: Final = gateway.request("GET", f"/alerts/{saved.id}")
        assert detail.status_code == 200, detail.text
        assert saved_alert(detail.content) == saved
        listed: Final = gateway.request("GET", "/alerts", params={"limit": "2"})
        assert listed.status_code == 200, listed.text
        assert saved_page(listed.content).data == (scoped, saved)
        update: Final = gateway.request("PATCH", f"/alerts/{saved.id}", {"status": "dismissed"})
        assert update.status_code == 200, update.text
        dismissed: Final = saved_alert(update.content)
        assert dismissed.status == "dismissed"
        assert await alerts.repository.record(detection) == dismissed
        assert await alerts.repository.get(int(scoped.id)) == scoped
    finally:
        write_rows(
            'DELETE FROM "LiteLLM_Alert" WHERE resource_id=%s AND type=%s AND window_start=%s AND window_end=%s',
            ("gateway", kind, detection.window_start, detection.window_end),
        )


@pytest.mark.asyncio
async def test_concurrent_detection_retries_preserve_dismissal_and_original_evidence(alerts: Alerts) -> None:
    detection: Final = alerts.detection()
    duplicates: Final = await asyncio.gather(*(alerts.repository.record(detection) for _ in range(10)))
    assert len({alert.id for alert in duplicates}) == 1
    dismissed: Final = await alerts.repository.set_status(int(duplicates[0].id), "dismissed", "reviewer")
    revised: Final = AlertDetection.model_validate(
        {**detection.model_dump(), "evidence": {"value": 100, "baseline": 2, "unit": "usd", "sample_size": 100}}
    )
    retried: Final = await alerts.repository.record(revised)
    assert retried == dismissed
    assert retried.evidence == detection.evidence
    assert read_rows('SELECT count(*) AS count FROM "LiteLLM_Alert" WHERE resource_id=%s', (alerts.resource_id,)) == [
        {"count": 1}
    ]


@pytest.mark.asyncio
async def test_keyset_pages_survive_status_changes_and_do_not_include_newer_alerts(
    alerts: Alerts, gateway: Gateway
) -> None:
    oldest: Final = await alerts.repository.record(alerts.detection(day=0))
    middle: Final = await alerts.repository.record(alerts.detection(day=1))
    newest: Final = await alerts.repository.record(alerts.detection(day=2))
    first: Final = gateway.request("GET", "/alerts", params={"limit": "2"})
    assert first.status_code == 200, first.text
    page: Final = saved_page(first.content)
    assert tuple(alert.id for alert in page.data) == (newest.id, middle.id)
    assert page.next_cursor == middle.id
    await alerts.repository.set_status(int(middle.id), "resolved", "reviewer")
    late: Final = await alerts.repository.record(alerts.detection(day=3))
    second: Final = gateway.request("GET", "/alerts", params={"limit": "2", "cursor": middle.id})
    assert second.status_code == 200, second.text
    remainder: Final = saved_page(second.content)
    assert oldest.id in tuple(alert.id for alert in remainder.data)
    assert not {newest.id, middle.id, late.id}.intersection(alert.id for alert in remainder.data)
    resolved: Final = gateway.request("GET", "/alerts", params={"status": "resolved", "limit": "1"})
    assert resolved.status_code == 200, resolved.text
    assert saved_page(resolved.content).data[0].id == middle.id


@pytest.mark.asyncio
async def test_repository_caps_page_size_and_follows_to_end(alerts: Alerts) -> None:
    for day in range(102):
        await alerts.repository.record(alerts.detection(day=day))
    page: Final = await alerts.repository.list("open", 100_000, None)
    assert len(page.data) == 100
    assert page.next_cursor == page.data[-1].id
    next_page: Final = await alerts.repository.list("open", 100, int(page.next_cursor))
    ours: Final = tuple(alert.id for alert in next_page.data if alert.resource_id == alerts.resource_id)
    assert len(ours) == 2
    assert not set(ours).intersection(alert.id for alert in page.data)
    empty: Final = await alerts.repository.list("dismissed", 25, None)
    assert all(alert.status == "dismissed" for alert in empty.data)


@pytest.mark.asyncio
async def test_live_auth_denies_nonadmins_and_viewers_cannot_write(alerts: Alerts, gateway: Gateway) -> None:
    saved: Final = await alerts.repository.record(alerts.detection())
    with gateway.scenario() as scenario:
        for role in ("internal_user", "proxy_admin_viewer"):
            user: Final = scenario.user(user_role=role)
            key: Final = scenario.key(user_id=user)
            for path in ("/alerts", f"/alerts/{saved.id}", "/alerts/monitoring"):
                response: Final = gateway.request("GET", path, key=key)
                assert response.status_code in ((200,) if role == "proxy_admin_viewer" else (401, 403)), response.text
            write: Final = gateway.request("PATCH", f"/alerts/{saved.id}", {"status": "dismissed"}, key=key)
            assert write.status_code in (401, 403), write.text
            monitoring: Final = gateway.request("PATCH", "/alerts/monitoring", {"enabled": False}, key=key)
            assert monitoring.status_code in (401, 403), monitoring.text
    assert await alerts.repository.get(int(saved.id)) == saved


@pytest.mark.asyncio
async def test_unknown_alert_returns_none_for_read_and_write(alerts: Alerts) -> None:
    assert await alerts.repository.get(2**63 - 1) is None
    assert await alerts.repository.set_status(2**63 - 1, "dismissed", "reviewer") is None


@pytest.mark.asyncio
async def test_windows_round_trip_at_database_precision_and_timezone(alerts: Alerts) -> None:
    start: Final = datetime(2026, 1, 1, microsecond=123456, tzinfo=timezone.utc)
    detection: Final = AlertDetection.model_validate(
        {**alerts.detection().model_dump(), "window_start": start, "window_end": start + timedelta(seconds=1)}
    )
    saved: Final = await alerts.repository.record(detection)
    assert saved.window_start == start.replace(microsecond=123000)
    assert saved.window_end == saved.window_start + timedelta(seconds=1)
    offset: Final = timezone(timedelta(hours=8))
    shifted: Final = AlertDetection.model_validate(
        {
            **detection.model_dump(),
            "window_start": start.astimezone(offset),
            "window_end": (start + timedelta(seconds=1)).astimezone(offset),
        }
    )
    assert await alerts.repository.record(shifted) == saved


@pytest.mark.parametrize("scope", ("model", "key", "user", "team", "gateway"))
@pytest.mark.asyncio
async def test_daily_history_matches_scope_and_weighted_metrics(
    alerts: Alerts, gateway: Gateway, scope: AlertResourceType
) -> None:
    resource: Final = {"key": uuid.uuid4().hex * 2, "gateway": "gateway"}.get(scope, alerts.resource_id)
    model_source: Final = scope in ("model", "gateway")
    table, column = {
        "model": ("LiteLLM_DailyModelUsage", "model_group"),
        "gateway": ("LiteLLM_DailyModelUsage", "model_group"),
        "team": ("LiteLLM_DailyTeamSpend", "team_id"),
        "user": ("LiteLLM_DailyUserSpend", "user_id"),
        "key": ("LiteLLM_DailyUserSpend", "api_key"),
    }[scope]
    marker: Final = alerts.resource_id
    request_column: Final = "request_count" if model_source else "api_requests"
    insert: Final = sql.SQL("""INSERT INTO {} ({}, date, model, {}, {}, spend, {}, failed_requests,
        total_response_time_ms, timed_requests, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())""").format(
        sql.Identifier(table),
        sql.Identifier(column),
        sql.Identifier("custom_llm_provider" if model_source else "id"),
        sql.Identifier("task_type" if model_source else ("user_id" if scope == "key" else "api_key")),
        sql.Identifier(request_column),
    )
    try:
        for day in ("2025-12-24", "2025-12-25", "2026-01-01", "2026-01-02"):
            for deployment, count in (("a", 2), ("b", 8)):
                write_rows(
                    insert,
                    (
                        resource,
                        day,
                        marker + deployment,
                        "example" if model_source else uuid.uuid4().hex,
                        "chat" if model_source else "",
                        float(count),
                        count,
                        1,
                        count * count * 100,
                        count,
                    ),
                )
        for kind, expected in (("spend", 10), ("traffic", 10), ("latency", 680), ("reliability", 0.2)):
            detection: Final = AlertDetection.model_validate(
                {
                    **alerts.detection(kind).model_dump(),
                    "resource_type": scope,
                    "resource_id": resource,
                }
            )
            saved: Final = await alerts.repository.record(detection)
            response: Final = gateway.request("GET", f"/alerts/{saved.id}")
            assert response.status_code == 200, response.text
            view: Final = AlertView.model_validate_json(response.content)
            assert view.history is not None
            assert tuple(point.date.isoformat() for point in view.history) == ("2025-12-25", "2026-01-01")
            assert tuple(point.value for point in view.history) == pytest.approx((expected, expected))
            assert view.evidence == saved.evidence
            assert view.resource_name == (
                resource if model_source and scope != "gateway" else "Gateway" if scope == "gateway" else None
            )
            listed: Final = gateway.request("GET", "/alerts", params={"type": kind, "limit": "1"})
            assert listed.status_code == 200, listed.text
            assert AlertViewPage.model_validate_json(listed.content).data[0] == view
    finally:
        write_rows(
            sql.SQL("DELETE FROM {} WHERE model IN (%s, %s)").format(sql.Identifier(table)),
            (marker + "a", marker + "b"),
        )
        write_rows('DELETE FROM "LiteLLM_Alert" WHERE resource_id=%s', (resource,))


@pytest.mark.asyncio
async def test_type_filter_applies_before_cursor_limit(alerts: Alerts, gateway: Gateway) -> None:
    oldest: Final = await alerts.repository.record(alerts.detection("traffic"))
    await alerts.repository.record(alerts.detection("spend"))
    newest: Final = await alerts.repository.record(alerts.detection("traffic", day=1))
    await alerts.repository.record(alerts.detection("model_deprecation"))
    page: Final = gateway.request("GET", "/alerts", params={"type": "traffic", "limit": "1"})
    assert page.status_code == 200, page.text
    first: Final = AlertViewPage.model_validate_json(page.content)
    assert first.data[0].id == newest.id
    assert first.next_cursor == newest.id
    following: Final = gateway.request("GET", "/alerts", params={"type": "traffic", "limit": "1", "cursor": newest.id})
    assert following.status_code == 200, following.text
    assert AlertViewPage.model_validate_json(following.content).data[0].id == oldest.id
    performance: Final = gateway.request("GET", "/alerts", params={"type": "performance"})
    assert performance.status_code == 200, performance.text
    assert all(
        alert.type != "model_deprecation" for alert in AlertViewPage.model_validate_json(performance.content).data
    )


@pytest.mark.asyncio
async def test_deleted_resource_preserves_history_and_history_timeout_preserves_alert(
    alerts: Alerts, gateway: Gateway
) -> None:
    import psycopg

    team: Final = alerts.resource_id
    write_rows('INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias) VALUES (%s, %s)', (team, "Example team"))
    write_rows(
        """INSERT INTO "LiteLLM_DailyTeamSpend" (id, team_id, api_key, date, spend, updated_at)
        VALUES (%s, %s, '', '2026-01-01', 20, NOW())""",
        (team, team),
    )
    try:
        saved: Final = await alerts.repository.record(
            AlertDetection.model_validate(
                {
                    **alerts.detection().model_dump(),
                    "resource_type": "team",
                }
            )
        )
        first: Final = gateway.request("GET", f"/alerts/{saved.id}")
        assert first.status_code == 200, first.text
        assert AlertView.model_validate_json(first.content).resource_name == "Example team"
        write_rows('DELETE FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team,))
        deleted: Final = gateway.request("GET", f"/alerts/{saved.id}")
        assert deleted.status_code == 200, deleted.text
        view: Final = AlertView.model_validate_json(deleted.content)
        assert view.resource_name is None
        assert view.history and view.history[0].value == 20
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute('LOCK TABLE "LiteLLM_DailyTeamSpend" IN ACCESS EXCLUSIVE MODE')
            unavailable: Final = gateway.request("GET", f"/alerts/{saved.id}")
            assert unavailable.status_code == 200, unavailable.text
            fallback: Final = AlertView.model_validate_json(unavailable.content)
            assert fallback.history is None
            assert fallback.evidence == saved.evidence
    finally:
        write_rows('DELETE FROM "LiteLLM_TeamTable" WHERE team_id=%s', (team,))
        write_rows('DELETE FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,))
