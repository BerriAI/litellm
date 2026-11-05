import json
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway

from litellm.repositories.issue_activity import IssueActivityRepository
from litellm.repositories.issues import IssuesRepository
from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.issues import (
    IssueCohort,
    IssueComparison,
    IssueInput,
    IssueList,
    IssueModelPair,
    IssueSignalInput,
    IssueSignalList,
)
from litellm.types.proxy.usage_alerts import UsageAlertObservation, UsageAlertRuleInput


def _issue(cohort: IssueCohort) -> IssueInput:
    identity: Final = str(uuid4())
    day: Final = date.today() - timedelta(days=3)
    return IssueInput(
        id=identity,
        cohort_key=identity,
        kind="performance",
        family="traffic",
        title="Requests increased",
        cohort=cohort,
        observed_on=day,
        observed_at=datetime.now(timezone.utc),
        range_start=day - timedelta(days=7),
        range_end=day,
        grouping_reason="same_resource",
        explanation="Requests increased for this resource.",
        comparisons=(
            IssueComparison(
                metric="requests",
                direction="increase",
                observed_value=420,
                baseline_value=120,
                threshold=360,
                baseline_start=day - timedelta(days=7),
                baseline_end=day - timedelta(days=1),
                baseline_sample_days=7,
            ),
        ),
    )


async def _remove_issues(db: ManagementDatabase, ids: tuple[str, ...]) -> None:
    await db.prisma.writer_db.query_raw(
        'DELETE FROM "LiteLLM_IssueDeliveryRule" WHERE delivery_id IN '
        '(SELECT id FROM "LiteLLM_IssueDelivery" WHERE issue_id=ANY($1::text[])) RETURNING delivery_id',
        list(ids),
    )
    await db.prisma.writer_db.query_raw(
        'DELETE FROM "LiteLLM_IssueDelivery" WHERE issue_id=ANY($1::text[]) RETURNING id', list(ids)
    )
    await db.prisma.writer_db.query_raw(
        'DELETE FROM "LiteLLM_IssueSignal" WHERE issue_id=ANY($1::text[]) RETURNING source_id', list(ids)
    )
    await db.prisma.writer_db.query_raw('DELETE FROM "LiteLLM_Issue" WHERE id=ANY($1::text[]) RETURNING id', list(ids))


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_http_scope_pagination_and_source_visibility(gateway: Gateway, db: ManagementDatabase) -> None:
    store: Final = IssuesRepository.from_prisma(db.prisma)
    alerts: Final = UsageAlertsRepository.from_prisma(db.prisma)
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        outsider: Final = scenario.user(user_role="internal_user")
        viewer: Final = scenario.user(user_role="proxy_admin_viewer")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        outsider_key: Final = scenario.key(user_id=outsider, key_type="management")
        viewer_key: Final = scenario.key(user_id=viewer, key_type="read_only")
        inference_key: Final = scenario.key(user_id=owner, key_type="llm_api", key_alias="Example checkout")
        reporting_id: Final = sha256(inference_key.encode()).hexdigest()
        first: Final = _issue(IssueCohort(scope_type="key", scope_id=reporting_id))
        second: Final = _issue(first.cohort)
        global_issue: Final = _issue(IssueCohort(scope_type="provider", scope_id="example-provider"))
        rule: Final = await alerts.save_rule(
            UsageAlertRuleInput(
                name="Visibility fixture",
                scope_type="key",
                scope_id=reporting_id,
                metric="requests",
                direction="increase",
            ),
            owner,
        )
        assert rule is not None
        try:
            await store.upsert_issues((first, second, global_issue))
            source: Final = (
                await alerts.record_observations(
                    rule,
                    (
                        UsageAlertObservation(
                            entity_id=reporting_id,
                            day=first.observed_on,
                            observed=420,
                            baseline=120,
                            threshold=360,
                            breached=True,
                        ),
                    ),
                )
            )[0]
            broader: Final = source.model_copy(
                update={
                    "id": str(uuid4()),
                    "scope_type": "deployment",
                    "scope_id": None,
                    "observed_value": 99999,
                    "baseline_value": 9999,
                }
            )
            await store.add_signals(
                first.id,
                (
                    IssueSignalInput(source="usage_alert", source_id=source.id, usage_alert=source),
                    IssueSignalInput(
                        source="usage_alert",
                        source_id=broader.id,
                        usage_alert=broader,
                        grouping_reason="dominant_key",
                        explanation="Parent contribution evidence.",
                    ),
                ),
            )
            path: Final = f"/issues/{first.id}"
            own: Final = gateway.request("GET", path, key=owner_key)
            assert own.status_code == 200 and own.json()["scope_label"] == "Example checkout"
            assert "99999" not in own.text and "9999" not in own.text
            for offset in (0, 1, 2):
                page: Final = gateway.request(
                    "GET",
                    "/issues",
                    key=owner_key,
                    params={"scope_type": "key", "scope_id": reporting_id, "limit": "1", "offset": str(offset)},
                )
                assert page.status_code == 200
                parsed: Final = IssueList.model_validate(page.json())
                assert parsed.total == 2 and len(parsed.data) == (1 if offset < 2 else 0)
            assert gateway.request("GET", "/issues", key=owner_key).status_code == 403
            for suffix in ("", "/signals", "/activity", "/keys", "/users", "/impact"):
                assert gateway.request("GET", f"/issues/{global_issue.id}{suffix}", key=owner_key).status_code == 403
                assert gateway.request("GET", path + suffix, key=outsider_key).status_code == 403
                assert gateway.request("GET", path + suffix, key=inference_key).status_code == 403
            public_signals: Final = gateway.request("GET", path + "/signals", key=owner_key, params={"limit": "1"})
            assert public_signals.status_code == 200
            visible: Final = IssueSignalList.model_validate(public_signals.json())
            assert visible.total == 1 and tuple(row.source_id for row in visible.data) == (source.id,)
            assert "99999" not in public_signals.text
            end: Final = gateway.request("GET", path + "/signals", key=owner_key, params={"limit": "1", "offset": "1"})
            assert end.json()["total"] == 1 and end.json()["data"] == []
            admin: Final = gateway.request("GET", path + "/signals", key=viewer_key)
            assert admin.status_code == 200 and admin.json()["total"] == 2 and "99999" in admin.text
            assert (
                gateway.request(
                    "PATCH", path, {"status": "dismissed", "expected_status": "open"}, key=viewer_key
                ).status_code
                == 403
            )
            assert (
                gateway.request(
                    "PATCH", path, {"status": "dismissed", "expected_status": "open"}, key=owner_key
                ).status_code
                == 200
            )
            assert (
                gateway.request(
                    "PATCH", path, {"status": "resolved", "expected_status": "open"}, key=owner_key
                ).status_code
                == 409
            )
            assert gateway.request("GET", f"/issues/{second.id}", key=owner_key).json()["status"] == "open"
        finally:
            await _remove_issues(db, (first.id, second.id, global_issue.id))
            await alerts.delete_rule(rule.id)
            await db.prisma.writer_db.query_raw(
                'DELETE FROM "LiteLLM_UsageAlertIncident" WHERE rule_id=$1 RETURNING id', rule.id
            )


@pytest.mark.asyncio(loop_scope="session")
async def test_issue_provider_pairs_filter_activity_pages_counts_and_exports(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    repository: Final = IssueActivityRepository.from_prisma(db.prisma)
    run_id: Final = str(uuid4())
    day: Final = date.today() - timedelta(days=3)
    provider: Final = f"example-provider-{run_id}"
    other: Final = f"other-provider-{run_id}"
    key: Final = sha256(run_id.encode()).hexdigest()
    entries: Final = tuple(
        {
            "id": f"{run_id}-{index}",
            "user_id": f"user-{run_id}",
            "date": day.isoformat(),
            "api_key": key,
            "model": model,
            "custom_llm_provider": name,
            "api_requests": requests,
            "spend": spend,
            "successful_requests": requests,
            "total_response_time_ms": requests * latency,
            "timed_requests": requests,
        }
        for index, (name, model, requests, spend, latency) in enumerate(
            (
                (provider, "example-model-a", 100, 10, 200),
                (other, "example-model-a", 1000, 100, 900),
                (provider, "example-model-b", 50, 5, 400),
            )
        )
    )
    await db.prisma.writer_db.query_raw(
        """INSERT INTO "LiteLLM_DailyUserSpend"
        (id,user_id,date,api_key,model,custom_llm_provider,api_requests,spend,successful_requests,total_response_time_ms,timed_requests,updated_at)
        SELECT id,user_id,date,api_key,model,custom_llm_provider,api_requests,spend,successful_requests,total_response_time_ms,timed_requests,CURRENT_TIMESTAMP
        FROM jsonb_to_recordset($1::jsonb) AS x(id text,user_id text,date text,api_key text,model text,
            custom_llm_provider text,api_requests int,spend float,successful_requests int,total_response_time_ms bigint,timed_requests int)
        RETURNING id""",
        json.dumps(entries),
    )
    cohort: Final = IssueCohort(
        scope_type="provider",
        scope_id=provider,
        model_pairs=(IssueModelPair(provider=provider, model="example-model-a"),),
    )
    store: Final = IssuesRepository.from_prisma(db.prisma)
    issue: Final = _issue(cohort).model_copy(update={"range_start": day, "range_end": day})
    try:
        await store.upsert_issue(issue)
        activity_http: Final = gateway.request("GET", f"/issues/{issue.id}/activity")
        assert activity_http.status_code == 200 and activity_http.json()["metadata"]["total_api_requests"] == 100
        keys_http: Final = gateway.request("GET", f"/issues/{issue.id}/keys", params={"limit": "1"})
        assert keys_http.status_code == 200 and keys_http.json()["api_keys"][0]["metrics"]["api_requests"] == 100
        users_http: Final = gateway.request("GET", f"/issues/{issue.id}/users")
        assert users_http.status_code == 200 and users_http.json()["total_users"] == 1
        impact_http: Final = gateway.request("GET", f"/issues/{issue.id}/impact")
        assert impact_http.status_code == 200 and impact_http.json()["requests"] == 100
        daily: Final = await repository.daily_activity(cohort, day, day)
        assert daily.metadata.total_api_requests == 100 and daily.metadata.total_spend == 10
        assert daily.metadata.total_response_time_ms == 20000 and daily.metadata.total_timed_requests == 100
        keys: Final = await repository.keys(cohort, day, day, 1, 0)
        assert keys.total_api_keys == 1 and keys.api_keys[0].metrics.api_requests == 100
        assert (
            keys.api_keys[0].metrics.total_response_time_ms == 20000 and keys.api_keys[0].metrics.timed_requests == 100
        )
        users: Final = await repository.users(cohort, day, day, 1, 0)
        assert users.total_users == 1 and users.users[0].api_requests == 100 and users.users[0].timed_requests == 100
        impact: Final = await repository.impact(cohort, day, day)
        assert (impact.keys, impact.users, impact.models, impact.requests) == (1, 1, 1, 100)
        pairs: Final = IssueCohort(
            scope_type="deployment",
            model_pairs=(
                IssueModelPair(provider=provider, model="example-model-a"),
                IssueModelPair(provider=other, model="example-model-b"),
            ),
        )
        assert (await repository.daily_activity(pairs, day, day)).metadata.total_api_requests == 100
        injected: Final = IssueCohort(scope_type="provider", scope_id=provider + "' OR true --")
        assert (await repository.impact(injected, day, day)).requests == 0
        from litellm.repositories.issue_activity import issue_activity_scope
        from litellm.types.repositories.daily_activity import ExportType

        exported: Final = tuple(
            [
                row
                async for row in repository.activity.export_rows(
                    issue_activity_scope(cohort, day, day), export_type=ExportType.DAILY
                )
            ]
        )
        assert len(exported) == 1 and exported[0].api_requests == 100
    finally:
        await _remove_issues(db, (issue.id,))
        await db.prisma.writer_db.query_raw(
            'DELETE FROM "LiteLLM_DailyUserSpend" WHERE id=ANY($1::text[]) RETURNING id',
            [entry["id"] for entry in entries],
        )
