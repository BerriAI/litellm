from hashlib import sha256
from typing import Final

from integration._support.client import Gateway, string_value


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
