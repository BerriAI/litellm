from datetime import date
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.alerting_endpoints.usage_alerts import require_scope
from litellm.repositories.usage_alerts import UsageAlertsRepository, activity_path
from litellm.types.proxy.auth.user_roles import LitellmUserRoles


class NoDatabase:
    async def query_raw(self, query: str, *args: object) -> object:
        raise AssertionError("Unauthorized callers must not reach the database")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,key_type,write",
    [
        (LitellmUserRoles.PROXY_ADMIN, "llm_api", False),
        (LitellmUserRoles.PROXY_ADMIN, "read_only", True),
        (LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, None, True),
        (LitellmUserRoles.INTERNAL_USER_VIEW_ONLY, None, True),
        (LitellmUserRoles.INTERNAL_USER, None, False),
        (LitellmUserRoles.CUSTOMER, None, False),
    ],
)
async def test_global_access_rejects_inference_keys_and_unprivileged_roles(
    role: LitellmUserRoles, key_type: str | None, write: bool
) -> None:
    user: Final = UserAPIKeyAuth(user_id="qa-user", user_role=role, key_type=key_type)
    with pytest.raises(HTTPException) as error:
        await require_scope(user, UsageAlertsRepository(NoDatabase()), "deployment", None, write=write)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_user_can_read_and_manage_only_their_user_scope() -> None:
    user: Final = UserAPIKeyAuth(user_id="qa-user", user_role=LitellmUserRoles.INTERNAL_USER)
    repository: Final = UsageAlertsRepository(NoDatabase())
    await require_scope(user, repository, "user", "qa-user", write=True)
    with pytest.raises(HTTPException) as error:
        await require_scope(user, repository, "user", "other-user", write=False)
    assert error.value.status_code == 403


def test_activity_link_escapes_entity_identity_and_includes_baseline() -> None:
    path: Final = activity_path("model", "provider/model?foo=bar&other=1", date(2026, 11, 4), "qa-alert", 7)
    parsed: Final = urlsplit(path)
    query: Final = parse_qs(parsed.query)
    assert parsed.path == "/models-and-endpoints"
    assert query["activity_model"] == ["provider/model?foo=bar&other=1"]
    assert query["activity_from"] == ["2026-10-28"]
    assert query["activity_to"] == ["2026-11-04"]
    assert query["alert"] == ["qa-alert"]
    assert query["detail_tab"] == ["activity"]


def test_global_activity_link_uses_the_usage_page_date_contract() -> None:
    query: Final = parse_qs(urlsplit(activity_path("deployment", None, date(2026, 11, 4), "qa-alert")).query)
    assert query["tab"] == ["keys"]
    assert "range" not in query
    assert query["from"] == ["2026-10-28"]
    assert query["to"] == ["2026-11-04"]
