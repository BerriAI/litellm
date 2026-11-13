from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.external_agents.endpoints import admin
from litellm.proxy.agent_endpoints.external_agents.models import Source


@pytest.mark.parametrize("role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY])
def test_management_rejects_non_admins_including_read_only_admin(role: LitellmUserRoles) -> None:
    with pytest.raises(HTTPException) as error:
        admin(UserAPIKeyAuth(user_role=role))
    assert error.value.status_code == 403


def test_proxy_admin_can_manage_external_agents() -> None:
    auth: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    assert admin(auth) is auth


def test_source_response_never_serializes_current_or_previous_secret() -> None:
    source: Final = Source(
        source_id="source",
        display_name="Bots",
        signing_secret="encrypted-current",
        previous_signing_secret="encrypted-previous",
        ingest_key="encrypted-ingest",
        admin_api_key="encrypted-admin",
    )
    assert source.model_dump(mode="json") == {
        "source_id": "source",
        "display_name": "Bots",
        "vendor": "grok_bot",
        "tenant_id": None,
        "mode": "shadow",
        "default_policy_id": None,
        "last_event_at": None,
    }
