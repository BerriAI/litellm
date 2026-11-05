from datetime import date
from hashlib import sha256
from typing import Final

import pytest
from integration._support.actors import ManagementDatabase
from integration._support.client import Gateway, string_value

from litellm.repositories.usage_alerts import UsageAlertsRepository
from litellm.types.proxy.usage_alerts import UsageAlertObservation, UsageAlertRule, UsageAlertRuleInput


def test_scoped_rules_use_real_management_auth_and_preserve_read_only_roles(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owner: Final = scenario.user(user_role="internal_user")
        outsider: Final = scenario.user(user_role="internal_user")
        viewer: Final = scenario.user(user_role="proxy_admin_viewer")
        owner_key: Final = scenario.key(user_id=owner, key_type="management")
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
