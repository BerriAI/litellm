from datetime import date, timedelta
from hashlib import sha256
from typing import Final
from uuid import uuid4

import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway, string_value

from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alerts import (
    UsageAlertGroupList,
    UsageAlertMetric,
    UsageAlertModelList,
    UsageAlertObservation,
    UsageAlertRule,
    UsageAlertRuleInput,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_alert_resource_labels_bound_legacy_display_values(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    identity: Final = "label-bound-" + uuid4().hex
    scope: Final = UsageAlertRuleInput(
        name="Display label", scope_type="user", scope_id=identity, metric="spend", direction="increase"
    )
    await repository.database.query_raw(
        'INSERT INTO "LiteLLM_UserTable" (user_id,user_alias) VALUES ($1,repeat(chr(128640),65536))', identity
    )
    try:
        labels: Final = await repository.resource_labels((scope,))
        assert labels == {("user", identity): chr(128640) * 256}
        stored: Final = await repository.database.query_raw(
            'SELECT length(user_alias) AS characters FROM "LiteLLM_UserTable" WHERE user_id=$1', identity
        )
        assert stored[0]["characters"] == 65536
    finally:
        await repository.database.query_raw('DELETE FROM "LiteLLM_UserTable" WHERE user_id=$1', identity)


@pytest.mark.asyncio(loop_scope="session")
async def test_deleted_key_labels_use_the_latest_alias_and_prefer_live_keys(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    identity: Final = sha256(uuid4().bytes).hexdigest()
    scope: Final = UsageAlertRuleInput(
        name="Deleted key", scope_type="key", scope_id=identity, metric="spend", direction="increase"
    )
    try:
        await repository.database.query_raw(
            """INSERT INTO "LiteLLM_DeletedVerificationToken" (id,token,key_alias,deleted_at)
            VALUES ($1||'-old',$1,'Old alias',TIMESTAMP '2000-01-01'),
                   ($1||'-new',$1,repeat('x',512),TIMESTAMP '2000-01-02')""",
            identity,
        )
        assert await repository.resource_labels((scope,)) == {("key", identity): "x" * 256}
        await repository.database.query_raw(
            'INSERT INTO "LiteLLM_VerificationToken" (token,key_alias) VALUES ($1,$2)', identity, "Live alias"
        )
        assert await repository.resource_labels((scope,)) == {("key", identity): "Live alias"}
    finally:
        await repository.database.query_raw('DELETE FROM "LiteLLM_VerificationToken" WHERE token=$1', identity)
        await repository.database.query_raw('DELETE FROM "LiteLLM_DeletedVerificationToken" WHERE token=$1', identity)


@pytest.mark.parametrize("grant", ["management_routes", "/usage", "/usage/alert-rules"])
def test_scoped_rules_use_real_management_auth_and_preserve_read_only_roles(gateway: Gateway, grant: str) -> None:
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        outsider: Final = scenario.user(user_role="internal_user")
        viewer: Final = scenario.user(user_role="proxy_admin_viewer")
        owner_key: Final = scenario.key(user_id=owner, key_type="default", allowed_routes=[grant])
        outsider_key: Final = scenario.key(user_id=outsider, key_type="management")
        viewer_key: Final = scenario.key(user_id=viewer, key_type="read_only")
        inference_key: Final = scenario.key(user_id=owner, key_type="llm_api")
        config: Final = {
            "name": "Daily key spend",
            "scope_type": "key",
            "scope_id": sha256(inference_key.encode()).hexdigest(),
            "metric": "spend",
            "direction": "increase",
            "multiplier": 3,
        }
        created: Final = gateway.request("POST", "/usage/alert-rules", config, key=owner_key)
        assert created.status_code == 201, created.text
        rule_id: Final = string_value(created.json()["id"])
        path: Final = f"/usage/alert-rules/{rule_id}"
        try:
            assert gateway.request("GET", path, key=owner_key).status_code == 200
            assert gateway.request("GET", path, key=outsider_key).status_code == 403
            assert gateway.request("GET", path, key=inference_key).status_code == 403
            assert gateway.request("GET", path, key=viewer_key).status_code == 200
            assert gateway.request("PUT", path, config, key=viewer_key).status_code == 403
            assert gateway.request("PUT", path, config, key=outsider_key).status_code == 403
            assert (
                gateway.request("POST", "/usage/alert-rules", {**config, "scope_id": None}, key=owner_key).status_code
                == 403
            )
            disabled: Final = gateway.request("PUT", path, {**config, "enabled": False}, key=owner_key)
            assert disabled.status_code == 200 and disabled.json()["enabled"] is False
            page: Final = gateway.request(
                "GET",
                "/usage/alert-rules",
                key=owner_key,
                params={"scope_type": "key", "scope_id": str(config["scope_id"]), "limit": "1"},
            )
            assert page.status_code == 200 and page.json()["total"] == 1
            assert page.json()["data"][0]["id"] == rule_id
        finally:
            assert gateway.request("DELETE", path, key=owner_key).status_code == 204
        assert gateway.request("GET", path, key=owner_key).status_code == 404


def test_admin_edit_preserves_owner_rule_access(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        config: Final = {
            "name": "Owner policy",
            "scope_type": "user",
            "scope_id": owner,
            "metric": "spend",
            "direction": "increase",
        }
        created: Final = gateway.request("POST", "/usage/alert-rules", config, key=owner_key)
        assert created.status_code == 201, created.text
        path: Final = f"/usage/alert-rules/{created.json()['id']}"
        try:
            edited: Final = gateway.request("PUT", path, {**config, "name": "Renamed by admin"})
            assert edited.status_code == 200, edited.text
            owner_edit: Final = gateway.request("PUT", path, {**config, "multiplier": 5}, key=owner_key)
            assert owner_edit.status_code == 200, "Admin edit must preserve the owner's rule access"
        finally:
            assert gateway.request("DELETE", path).status_code == 204


def test_team_admin_can_manage_team_alerts_but_member_can_only_read(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        admin_id: Final = scenario.user(user_role="internal_user")
        member_id: Final = scenario.user(user_role="internal_user")
        team_id: Final = scenario.team(
            members_with_roles=[{"user_id": admin_id, "role": "admin"}, {"user_id": member_id, "role": "user"}]
        )
        admin_key: Final = scenario.key(user_id=admin_id, key_type="management")
        member_key: Final = scenario.key(user_id=member_id, key_type="management")
        config: Final = {
            "name": "Daily team requests",
            "scope_type": "team",
            "scope_id": team_id,
            "metric": "requests",
            "direction": "decrease",
            "multiplier": 2,
        }
        created: Final = gateway.request("POST", "/usage/alert-rules", config, key=admin_key)
        assert created.status_code == 201, created.text
        rule_id: Final = string_value(created.json()["id"])
        path: Final = f"/usage/alert-rules/{rule_id}"
        try:
            assert gateway.request("GET", path, key=member_key).status_code == 200
            assert gateway.request("PUT", path, config, key=member_key).status_code == 403
            assert gateway.request("DELETE", path, key=member_key).status_code == 403
        finally:
            assert gateway.request("DELETE", path, key=admin_key).status_code == 204


def test_info_only_credentials_cannot_mutate_alerts(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        admin: Final = scenario.user(user_role="proxy_admin")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        reader: Final = scenario.key(user_id=owner, key_type="default", allowed_routes=["info_routes"])
        admin_reader: Final = scenario.key(user_id=admin, key_type="default", allowed_routes=["info_routes"])
        config: Final = {
            "name": "Read-only route grant",
            "scope_type": "user",
            "scope_id": owner,
            "metric": "spend",
            "direction": "increase",
        }
        created: Final = gateway.request("POST", "/usage/alert-rules", config, key=owner_key)
        assert created.status_code == 201
        path: Final = f"/usage/alert-rules/{created.json()['id']}"
        try:
            assert gateway.request("GET", path, key=reader).status_code == 200
            assert gateway.request("POST", "/usage/alert-rules", config, key=reader).status_code == 403
            assert gateway.request("PUT", path, config, key=reader).status_code == 403
            assert gateway.request("DELETE", path, key=reader).status_code == 403
            assert (
                gateway.request("PATCH", "/issues/families/spend", {"enabled": False}, key=admin_reader).status_code
                == 403
            )
            assert (
                gateway.request(
                    "PATCH", "/issues/unavailable", {"status": "resolved", "expected_status": "open"}, key=admin_reader
                ).status_code
                == 403
            )
        finally:
            assert gateway.request("DELETE", path, key=owner_key).status_code == 204


def test_only_admins_can_configure_operator_notifications(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        config: Final = {
            "name": "Operator delivery",
            "scope_type": "user",
            "scope_id": owner,
            "metric": "spend",
            "direction": "increase",
        }
        for notification in (
            {"notification_channels": ["slack"]},
            {"notification_channels": ["email"], "email_recipients": ["ops@example.com"]},
        ):
            assert (
                gateway.request("POST", "/usage/alert-rules", {**config, **notification}, key=owner_key).status_code
                == 403
            )
        assert (
            gateway.request(
                "POST", "/usage/alert-rules", {**config, "overrides_defaults": True}, key=owner_key
            ).status_code
            == 422
        )
        assert (
            gateway.request("POST", "/usage/alert-rules", {**config, "name": "Spend\r\nInjected: header"}).status_code
            == 422
        )
        created: Final = gateway.request("POST", "/usage/alert-rules", {**config, "notification_channels": ["slack"]})
        assert created.status_code == 201
        assert "overrides_defaults" not in created.json()
        path: Final = f"/usage/alert-rules/{created.json()['id']}"
        try:
            assert gateway.request("PUT", path, config, key=owner_key).status_code == 403
            assert gateway.request("DELETE", path, key=owner_key).status_code == 403
        finally:
            assert gateway.request("DELETE", path).status_code == 204


@pytest.mark.asyncio(loop_scope="session")
async def test_incident_http_read_dismiss_and_scope_authorization(gateway: Gateway, db: ManagementDatabase) -> None:
    """Populate an explicit storage fixture, then exercise the real incident HTTP API."""
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        outsider: Final = scenario.user(user_role="internal_user")
        viewer: Final = scenario.user(user_role="proxy_admin_viewer")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        outsider_key: Final = scenario.key(user_id=outsider, key_type="management")
        viewer_key: Final = scenario.key(user_id=viewer, key_type="read_only")
        created: Final = gateway.request(
            "POST",
            "/usage/alert-rules",
            {
                "name": "Incident API fixture",
                "scope_type": "user",
                "scope_id": owner,
                "metric": "requests",
                "direction": "increase",
                "multiplier": 3,
            },
            key=owner_key,
        )
        assert created.status_code == 201
        rule: Final = UsageAlertRule.model_validate(created.json())
        try:
            incident: Final = (
                await repository.record_observations(
                    rule,
                    (
                        UsageAlertObservation(
                            entity_id=owner,
                            day=date(2026, 11, 4),
                            observed=420,
                            baseline=120,
                            baseline_sample_days=7,
                            threshold=360,
                            breached=True,
                        ),
                    ),
                )
            )[0]
            path: Final = f"/usage/alerts/{incident.id}"
            params: Final = {"scope_type": "user", "scope_id": owner, "limit": "1"}
            page: Final = gateway.request("GET", "/usage/alerts", key=owner_key, params=params)
            assert page.status_code == 200 and page.json()["data"][0]["id"] == incident.id
            detail: Final = gateway.request("GET", path, key=owner_key)
            assert detail.status_code == 200
            assert detail.json()["observed_value"] == 420
            assert detail.json()["baseline_value"] == 120
            assert detail.json()["threshold"] == 360
            assert detail.json()["baseline_sample_days"] == 7
            assert "detail_tab=activity" in detail.json()["activity_path"]
            assert gateway.request("GET", "/usage/alerts", key=outsider_key, params=params).status_code == 403
            assert gateway.request("GET", path, key=outsider_key).status_code == 403
            assert gateway.request("PATCH", path, {"status": "dismissed"}, key=outsider_key).status_code == 403
            assert gateway.request("GET", path, key=viewer_key).status_code == 200
            assert gateway.request("PATCH", path, {"status": "dismissed"}, key=viewer_key).status_code == 403
            dismissed: Final = gateway.request("PATCH", path, {"status": "dismissed"}, key=owner_key)
            assert dismissed.status_code == 200 and dismissed.json()["status"] == "dismissed"
            assert gateway.request("GET", path, key=owner_key).json()["status"] == "dismissed"
        finally:
            assert gateway.request("DELETE", f"/usage/alert-rules/{rule.id}", key=owner_key).status_code == 204


@pytest.mark.asyncio(loop_scope="session")
async def test_ongoing_episode_preserves_its_original_evidence_and_activity_link(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    definition: Final = UsageAlertRuleInput(
        name="Original episode", scope_type="user", scope_id="episode-example", metric="requests", direction="increase"
    )
    rule: Final = await repository.save_rule(definition, "qa-admin")
    assert rule is not None
    initial: Final = UsageAlertObservation(
        entity_id=rule.scope_id,
        day=date(2026, 11, 2),
        observed=420,
        baseline=120,
        baseline_sample_days=7,
        threshold=360,
        breached=True,
    )
    try:
        original: Final = (await repository.record_observations(rule, (initial,)))[0]
        edited: Final = await repository.save_rule(
            definition.model_copy(update={"name": "Edited rule", "multiplier": 4}), "qa-admin", rule
        )
        assert edited is not None
        later: Final = initial.model_copy(
            update={
                "day": date(2026, 11, 3),
                "observed": 600,
                "baseline": 100,
                "threshold": 400,
                "baseline_sample_days": 4,
            }
        )
        ongoing: Final = (await repository.record_observations(edited, (later,)))[0]
        assert ongoing.last_seen >= original.last_seen
        assert ongoing.model_copy(update={"last_seen": original.last_seen}) == original
        stored: Final = await repository.get_alert(original.id)
        assert stored == ongoing
        healthy: Final = later.model_copy(update={"day": date(2026, 11, 4), "breached": False})
        resolved: Final = (await repository.record_observations(edited, (healthy,)))[0]
        assert resolved.model_copy(update={"status": "open", "last_seen": original.last_seen}) == original
        repeated: Final = later.model_copy(update={"day": date(2026, 11, 5)})
        new_episode: Final = (await repository.record_observations(edited, (repeated,)))[0]
        assert new_episode.id != original.id
        assert new_episode.day == repeated.day and new_episode.observed_value == 600
        assert new_episode.threshold == 400 and new_episode.rule_name == "Edited rule"
        assert new_episode.activity_path != original.activity_path
    finally:
        await repository.delete_rule(rule.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_rule_scope_changes_reject_stale_updates_and_deletes(db: ManagementDatabase) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    definition: Final = UsageAlertRuleInput(
        name="Concurrent scope edit", scope_type="user", scope_id="original-owner", metric="spend", direction="increase"
    )
    original: Final = await repository.save_rule(definition, "qa-admin")
    assert original is not None
    try:
        moved: Final = await repository.save_rule(
            definition.model_copy(update={"scope_id": "different-owner"}), "qa-admin", existing=original
        )
        assert moved is not None
        assert await repository.save_rule(definition, "original-owner", existing=original) is None
        assert not await repository.delete_rule(original.id, existing=original)
        assert await repository.get_rule(original.id) == moved
        current: Final = await repository.save_rule(
            definition.model_copy(update={"scope_id": moved.scope_id, "name": "Current authorized edit"}),
            "qa-admin",
            existing=moved,
        )
        assert current is not None
        assert not await repository.delete_rule(original.id, existing=moved)
        assert await repository.delete_rule(original.id, existing=current)
        assert await repository.save_rule(definition, "qa-admin", existing=current) is None
        assert await repository.get_rule(original.id) is None
    finally:
        await repository.delete_rule(original.id)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "changed_field,new_value",
    [("metric", "latency"), ("direction", "decrease"), ("scope_type", "user"), ("scope_id", "semantic-other")],
)
async def test_rule_identity_edits_preserve_old_incident_and_filter_consistency(
    db: ManagementDatabase, changed_field: str, new_value: str
) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    definition: Final = UsageAlertRuleInput(
        name="Semantic identity",
        scope_type="team",
        scope_id="semantic-example",
        metric="spend",
        direction="increase",
        notification_channels=("slack",),
    )
    rule: Final = await repository.save_rule(definition, "qa-admin")
    assert rule is not None
    initial: Final = UsageAlertObservation(
        entity_id=rule.scope_id, day=date(2026, 11, 4), observed=40, baseline=5, threshold=15, breached=True
    )
    try:
        original: Final = (await repository.record_observations(rule, (initial,)))[0]
        edited: Final = await repository.save_rule(
            UsageAlertRuleInput.model_validate({**definition.model_dump(), changed_field: new_value}), "qa-admin", rule
        )
        assert edited is not None
        observation: Final = initial.model_copy(update={"entity_id": edited.scope_id})
        replacement: Final = (await repository.record_observations(edited, (observation,)))[0]
        assert replacement.id != original.id
        assert await repository.get_alert(original.id) == original
        before_page: Final = await repository.list_alerts(
            scope_type=original.scope_type, scope_id=original.scope_id, metric=original.metric
        )
        after_page: Final = await repository.list_alerts(
            scope_type=replacement.scope_type, scope_id=replacement.scope_id, metric=replacement.metric
        )
        assert original.id in {item.id for item in before_page.data}
        assert replacement.id in {item.id for item in after_page.data}
        assert all(
            item.scope_type == original.scope_type and item.metric == original.metric for item in before_page.data
        )
        healthy: Final = observation.model_copy(update={"day": date(2026, 11, 5), "breached": False})
        await repository.record_observations(edited, (healthy,))
        assert (await repository.get_alert(original.id)) == original
        resolved: Final = await repository.get_alert(replacement.id)
        assert resolved is not None and resolved.status == "resolved"
    finally:
        await repository.delete_rule(rule.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_grouped_http_pages_preserve_all_metrics_history_and_scoped_actions(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    day: Final = date(2026, 11, 4)
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user", user_alias="Checkout operator")
        outsider: Final = scenario.user(user_role="internal_user")
        viewer: Final = scenario.user(user_role="proxy_admin_viewer")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
        outsider_key: Final = scenario.key(user_id=outsider, key_type="management")
        viewer_key: Final = scenario.key(user_id=viewer, key_type="read_only")
        inference_key: Final = scenario.key(user_id=owner, key_type="llm_api")
        metrics: Final[tuple[UsageAlertMetric, ...]] = (
            *(("requests",) * 27),
            "spend",
            "errors",
            "latency",
            "error_rate",
        )
        rules: Final = tuple(
            [
                await repository.save_rule(
                    UsageAlertRuleInput(
                        name=f"Group rule {index}",
                        scope_type="user",
                        scope_id=owner,
                        metric=metric,
                        direction="increase",
                        baseline_days=7 if index % 2 else 14,
                    ),
                    owner,
                )
                for index, metric in enumerate(metrics)
            ]
        )
        assert all(rule is not None for rule in rules)
        incidents: Final = tuple(
            [
                (
                    await repository.record_observations(
                        rule,
                        (
                            UsageAlertObservation(
                                entity_id=owner,
                                day=day,
                                observed=14 if rule.metric == "error_rate" else 420,
                                baseline=4 if rule.metric == "error_rate" else 0 if index == 0 else 120,
                                threshold=12 if rule.metric == "error_rate" else 100 if index == 0 else 360,
                                breached=True,
                            ),
                        ),
                    )
                )[0]
                for index, rule in enumerate(rules)
                if rule is not None
            ]
        )
        old_rule: Final = await repository.save_rule(
            UsageAlertRuleInput(
                name="Earlier day", scope_type="user", scope_id=owner, metric="requests", direction="increase"
            ),
            owner,
        )
        assert old_rule is not None
        old: Final = (
            await repository.record_observations(
                old_rule,
                (
                    UsageAlertObservation(
                        entity_id=owner,
                        day=day - timedelta(days=1),
                        observed=420,
                        baseline=120,
                        threshold=360,
                        breached=True,
                    ),
                ),
            )
        )[0]
        other_rule: Final = await repository.save_rule(
            UsageAlertRuleInput(
                name="Other resource", scope_type="user", scope_id=outsider, metric="requests", direction="increase"
            ),
            outsider,
        )
        assert other_rule is not None
        other: Final = (
            await repository.record_observations(
                other_rule,
                (
                    UsageAlertObservation(
                        entity_id=outsider,
                        day=day,
                        observed=420,
                        baseline=120,
                        threshold=360,
                        breached=True,
                    ),
                ),
            )
        )[0]
        params: Final = {"scope_type": "user", "scope_id": owner, "status": "open", "limit": "1"}
        try:
            response: Final = gateway.request("GET", "/usage/alert-groups", key=owner_key, params=params)
            assert response.status_code == 200, response.text
            page: Final = UsageAlertGroupList.model_validate(response.json())
            assert page.total == 2 and len(page.data) == 1
            group: Final = page.data[0]
            assert group.day == day and group.scope_label == "Checkout operator"
            assert group.incident_count == 31 and len(group.metrics) == 5
            assert {metric.metric: metric.incident_count for metric in group.metrics} == {
                "requests": 27,
                "spend": 1,
                "errors": 1,
                "latency": 1,
                "error_rate": 1,
            }
            assert all(metric.representative.id in {incident.id for incident in incidents} for metric in group.metrics)
            assert group.activity_path in {incident.activity_path for incident in incidents}
            second: Final = UsageAlertGroupList.model_validate(
                gateway.request("GET", "/usage/alert-groups", key=owner_key, params={**params, "offset": "1"}).json()
            )
            assert second.total == 2 and second.data[0].day == old.day
            empty: Final = UsageAlertGroupList.model_validate(
                gateway.request("GET", "/usage/alert-groups", key=owner_key, params={**params, "offset": "2"}).json()
            )
            assert empty.total == 2 and empty.data == ()
            history: Final = gateway.request(
                "GET",
                "/usage/alerts",
                key=owner_key,
                params={**params, "start_date": day.isoformat(), "end_date": day.isoformat(), "metric": "requests"},
            )
            assert history.status_code == 200 and history.json()["total"] == 27
            await repository.set_status(incidents[0].id, "dismissed")
            body: Final = {
                "scope_type": "user",
                "scope_id": owner,
                "day": day.isoformat(),
                "expected_status": "open",
                "status": "resolved",
            }
            for forbidden in (outsider_key, inference_key):
                assert gateway.request("GET", "/usage/alert-groups", key=forbidden, params=params).status_code == 403
            for forbidden in (outsider_key, inference_key, viewer_key):
                assert gateway.request("PATCH", "/usage/alert-groups", body, key=forbidden).status_code == 403
            assert gateway.request("GET", "/usage/alert-groups", key=viewer_key, params=params).status_code == 200
            updated: Final = gateway.request("PATCH", "/usage/alert-groups", body, key=owner_key)
            assert updated.status_code == 200 and updated.json()["updated_count"] == 30
            assert gateway.request("PATCH", "/usage/alert-groups", body, key=owner_key).json()["updated_count"] == 0
            assert await repository.get_alert(old.id) == old
            assert await repository.get_alert(other.id) == other
            dismissed: Final = await repository.get_alert(incidents[0].id)
            assert dismissed is not None and dismissed.status == "dismissed" and dismissed.baseline_value == 0
            resolved: Final = await repository.get_alert(incidents[-1].id)
            assert resolved == incidents[-1].model_copy(update={"status": "resolved"})
            assert (
                gateway.request(
                    "GET", "/usage/alert-groups", params={"scope_type": "user", "scope_id": "x' OR 1=1 --"}
                ).json()["total"]
                == 0
            )
            assert (
                gateway.request(
                    "GET", "/usage/alert-groups", params={"start_date": "2026-11-05", "end_date": "2026-11-04"}
                ).status_code
                == 422
            )
        finally:
            for rule in (*rules, old_rule, other_rule):
                if rule is not None:
                    await repository.delete_rule(rule.id)
    deleted: Final = await repository.list_groups(scope_type="user", scope_id=owner)
    assert deleted.total == 2 and all(group.scope_label is None and group.scope_id == owner for group in deleted.data)


@pytest.mark.asyncio(loop_scope="session")
async def test_default_rule_summary_and_kind_filters_do_not_require_loading_every_rule(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    default: Final = UsageAlertRuleInput(
        name="Built-in daily model requests", scope_type="model", metric="requests", direction="increase"
    )
    await repository.ensure_default_rules((default,))
    custom: Final = await repository.save_rule(
        default.model_copy(update={"scope_id": f"model-{uuid4().hex}"}), "qa-admin"
    )
    assert custom is not None
    try:
        all_rules: Final = await repository.list_rules(scope_type="model", limit=1)
        builtins: Final = await repository.list_rules(scope_type="model", limit=1, kind="default")
        customs: Final = await repository.list_rules(scope_type="model", limit=1, kind="custom")
        assert builtins.total >= 1 and all(rule.is_default for rule in builtins.data)
        assert customs.total >= 1 and all(not rule.is_default for rule in customs.data)
        assert all_rules.total == builtins.total + customs.total
        assert all_rules.default_summary == builtins.default_summary == customs.default_summary
        assert builtins.default_summary.total == builtins.total
        assert (
            sum(builtins.default_summary.evaluation_status_counts.model_dump().values())
            == builtins.default_summary.enabled
        )
        assert (await repository.list_rules(scope_type="model", kind="default", offset=builtins.total)).data == ()
        stored: Final = await repository.get_rule(builtins.data[0].id)
        assert stored is not None and stored.is_default
        filtered: Final = gateway.request(
            "GET", "/usage/alert-rules", params={"scope_type": "model", "kind": "custom", "limit": "1"}
        )
        assert filtered.status_code == 200 and filtered.json()["total"] == customs.total
        assert all(not rule["is_default"] for rule in filtered.json()["data"])
        configuration: Final = stored.model_dump(mode="json", include=frozenset(UsageAlertRuleInput.model_fields))
        path: Final = f"/usage/alert-rules/{stored.id}"
        for change in (
            {"scope_type": "key"},
            {"scope_id": "model-example"},
            {"metric": "requests" if stored.metric != "requests" else "spend"},
            {"direction": "increase" if stored.direction == "decrease" else "decrease"},
        ):
            assert gateway.request("PUT", path, {**configuration, **change}).status_code == 422
        assert await repository.get_rule(stored.id) == stored
        try:
            updated: Final = gateway.request("PUT", path, {**configuration, "min_value": stored.min_value + 1})
            assert updated.status_code == 200 and updated.json()["is_default"] is True
            assert updated.json()["min_value"] == stored.min_value + 1
        finally:
            assert gateway.request("PUT", path, configuration).status_code == 200
        custom_input: Final = custom.model_dump(mode="json", include=frozenset(UsageAlertRuleInput.model_fields))
        assert (
            gateway.request("PUT", f"/usage/alert-rules/{custom.id}", {**custom_input, "metric": "latency"}).status_code
            == 200
        )
    finally:
        await repository.delete_rule(custom.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_reported_model_search_preserves_exact_identity_and_cursor(
    gateway: Gateway, db: ManagementDatabase
) -> None:
    repository: Final = UsageAlertsRepository.from_prisma(db.prisma)
    prefix: Final = f"reported-{uuid4().hex}"
    initial: Final = await repository.list_reported_models(prefix, None, 1)
    assert initial.data == ()
    models: Final = (f"{prefix}/first", f"{prefix}/literal%quote'", f"{prefix}/second")
    try:
        for index, model in enumerate((*models, models[0])):
            await db.prisma.writer_db.execute_raw(
                'INSERT INTO "LiteLLM_DailyUserSpend" (id,date,api_key,model,updated_at) VALUES ($1,$2,$3,$4,CURRENT_TIMESTAMP)',
                f"{prefix}-{index}",
                initial.end_date.isoformat(),
                prefix,
                model,
            )
        await db.prisma.writer_db.execute_raw(
            'INSERT INTO "LiteLLM_DailyUserSpend" (id,date,api_key,model,updated_at) VALUES ($1,$2,$3,$4,CURRENT_TIMESTAMP)',
            f"{prefix}-old",
            (initial.start_date - timedelta(days=1)).isoformat(),
            prefix,
            f"{prefix}/old",
        )
        response: Final = gateway.request("GET", "/usage/alert-models", params={"search": prefix, "limit": "1"})
        assert response.status_code == 200, response.text
        first: Final = UsageAlertModelList.model_validate(response.json())
        assert first.data[0].scope_id == models[0] and first.next_cursor == models[0]
        second: Final = await repository.list_reported_models(prefix, first.next_cursor, 1)
        third: Final = await repository.list_reported_models(prefix, second.next_cursor, 1)
        assert second.data[0].scope_id == models[1] and third.data[0].scope_id == models[2]
        assert third.next_cursor is None
        literal: Final = await repository.list_reported_models(f"{prefix}/literal%quote'", None, 20)
        assert tuple(option.scope_id for option in literal.data) == (models[1],)
        assert literal.data[0].label == models[1]
        with gateway.scenario() as scenario:
            user: Final = scenario.user(user_role="internal_user")
            key: Final = scenario.key(user_id=user, key_type="management")
            assert gateway.request("GET", "/usage/alert-models", key=key).status_code == 403
    finally:
        await db.prisma.writer_db.execute_raw('DELETE FROM "LiteLLM_DailyUserSpend" WHERE api_key=$1', prefix)
