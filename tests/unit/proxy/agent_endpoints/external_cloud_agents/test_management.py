import json
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.external_cloud_agents import endpoints
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from tests.unit.proxy.agent_endpoints.external_cloud_agents.conftest import NOW, Harness


@pytest.mark.parametrize("scope", ["llm_api_routes", "agent_routes"])
def test_admin_owned_inference_key_cannot_reveal_connection(harness: Harness, scope: str) -> None:
    route: Final = f"/v1/external_cloud_agents/sources/{harness.source_id}/connection"
    restricted: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, allowed_routes=[scope])
    with pytest.raises(HTTPException) as denied:
        RouteChecks.is_virtual_key_allowed_to_call_route(route=route, valid_token=restricted)
    assert denied.value.status_code == 403
    assert RouteChecks.is_virtual_key_allowed_to_call_route(
        route=route, valid_token=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )
    shown: Final = harness.client.get(route)
    assert shown.status_code == 200
    assert shown.json()["ingest_key"]


def test_no_body_creation_encrypts_and_connection_reshows_same_key(harness: Harness) -> None:
    response: Final = harness.client.post("/v1/external_cloud_agents/sources")
    assert response.status_code == 200
    connection: Final = response.json()
    assert connection["hook_path"] == f"/grok_bot/inference_hook/{connection['source_id']}"
    assert harness.db.source is not None
    assert harness.db.source.ingest_key != connection["ingest_key"]
    assert decrypt_value_helper(harness.db.source.ingest_key, "ingest_key") == connection["ingest_key"]
    shown: Final = harness.client.get(f"/v1/external_cloud_agents/sources/{connection['source_id']}/connection")
    assert shown.status_code == 200
    assert shown.json() == connection


def test_sources_never_return_either_secret(harness: Harness) -> None:
    harness.db.rows = (
        {
            "source_id": harness.source_id,
            "vendor": "grok_bot",
            "display_name": "GrokBot",
            "tenant_id": None,
            "ingest_key": "private-ingest",
            "admin_api_key": "private-admin",
            "bot_count": 4,
            "connected": True,
            "has_admin_api_key": True,
            "last_event_at": NOW,
            "last_spend_sync_at": NOW,
            "last_spend_sync_error": None,
            "created_at": NOW,
            "created_by": "admin",
        },
    )
    response: Final = harness.client.get("/v1/external_cloud_agents/sources")
    assert response.status_code == 200
    assert response.json() == [
        {
            "source_id": harness.source_id,
            "vendor": "grok_bot",
            "display_name": "GrokBot",
            "tenant_id": None,
            "bot_count": 4,
            "connected": True,
            "has_admin_api_key": True,
            "last_event_at": NOW.isoformat().replace("+00:00", "Z"),
            "last_spend_sync_at": NOW.isoformat().replace("+00:00", "Z"),
            "last_spend_sync_error": None,
            "created_at": NOW.isoformat().replace("+00:00", "Z"),
            "created_by": "admin",
        }
    ]
    assert "private" not in json.dumps(response.json())


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/sources"),
        ("GET", "/sources"),
        ("GET", "/sources/source/connection"),
        ("DELETE", "/sources/source"),
        ("PUT", "/sources/source/admin_api_key"),
        ("DELETE", "/sources/source/admin_api_key"),
        ("POST", "/sources/source/sync_spend"),
        ("GET", "/agents?source_id=source"),
        ("GET", "/spend?source_id=source"),
    ],
)
def test_every_management_route_rejects_non_admin(harness: Harness, method: str, path: str) -> None:
    harness.app.dependency_overrides = {
        **harness.app.dependency_overrides,
        endpoints.user_api_key_auth: lambda: UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER),
    }
    response: Final = harness.client.request(method, "/v1/external_cloud_agents" + path, json={"api_key": "key"})
    assert response.status_code == 403
    assert harness.db.writes == ()
