"""`PATCH /management/v1/users/{user_id}`.

The point of the endpoint is that an explicit `null` clears a setting, so most of what follows is
about telling "the caller sent null" apart from "the caller sent nothing" all the way down to the
values handed to the query engine.
"""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import ValidationError

import litellm
from litellm.proxy._types import (
    LitellmUserRoles,
    UpdateUserRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    _update_internal_user_params,
)
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    validation_problem,
)
from litellm.proxy.management_endpoints.management_v1.users import (
    UserPatchRequest,
    resolve_user_patch_fields,
)

app = FastAPI()


@app.exception_handler(ManagementProblem)
async def _management_problem_handler(request: Request, exc: ManagementProblem):
    return problem_response(exc.problem)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    from_body = any(error["loc"][:1] == ("body",) for error in exc.errors())
    if not from_body:
        raise exc
    return problem_response(validation_problem("; ".join(error["msg"] for error in exc.errors())))


app.include_router(router)

ADMIN = UserAPIKeyAuth(user_id="admin-1", user_role=LitellmUserRoles.PROXY_ADMIN.value)
TARGET_ID = "user-1"
PATCH_PATH = f"{MANAGEMENT_V1_PREFIX}/users/{TARGET_ID}"


def _resolve(**body: Any) -> dict[str, Any]:
    """Run a patch body through the endpoint's field resolution, as the write path would."""
    request = UpdateUserRequest(
        user_id=TARGET_ID, **UserPatchRequest.model_validate(body).model_dump(exclude_unset=True)
    )
    return resolve_user_patch_fields(request.model_dump(exclude_unset=True), request)


def _user_row(**overrides: Any) -> MagicMock:
    row = MagicMock()
    row.model_dump.return_value = {
        "user_id": TARGET_ID,
        "user_email": "u1@example.com",
        "user_alias": None,
        "user_role": LitellmUserRoles.INTERNAL_USER.value,
        "models": [],
        "spend": 0.0,
        "max_budget": None,
        "budget_duration": None,
        "budget_reset_at": None,
        "tpm_limit": 500,
        "rpm_limit": None,
        "max_parallel_requests": None,
        "metadata": {},
        "model_max_budget": {},
        "object_permission_id": None,
        "teams": [],
        "created_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 8, 2, tzinfo=timezone.utc),
        **overrides,
    }
    for key, value in row.model_dump.return_value.items():
        setattr(row, key, value)
    return row


@pytest.fixture
def caller():
    """Whoever the route should authenticate as. Rebind `.value` inside a test to change it."""
    holder = MagicMock()
    holder.value = ADMIN
    app.dependency_overrides[user_api_key_auth] = lambda: holder.value
    yield holder
    app.dependency_overrides.clear()


@pytest.fixture
def prisma(mocker):
    """A prisma double whose `update_data` call is what the assertions inspect."""
    client = MagicMock()
    client.update_data = AsyncMock(return_value={"user_id": TARGET_ID, "data": _user_row()})
    client.get_data = AsyncMock(return_value=[])
    mocker.patch("litellm.proxy.proxy_server.prisma_client", client)
    mocker.patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "default_user_id")
    mocker.patch("litellm.proxy.proxy_server._invalidate_spend_counter", AsyncMock())
    table = MagicMock()
    table.find_first = AsyncMock(return_value=_user_row())
    mocker.patch(
        "litellm.repositories.user_repository.UserRepository.table",
        new_callable=mocker.PropertyMock,
        return_value=table,
    )
    client.table = table
    return client


def _written(prisma) -> dict[str, Any]:
    assert prisma.update_data.await_count == 1, "expected exactly one user write"
    return prisma.update_data.await_args.kwargs["data"]


# --- field resolution: null clears, absent is left alone -----------------------------------------


def test_explicit_null_clears_a_nullable_column():
    """The bug this endpoint exists for: `/user/update` drops this null and answers 200 unchanged."""
    assert _resolve(tpm_limit=None)["tpm_limit"] is None


def test_omitted_field_is_not_written():
    resolved = _resolve(rpm_limit=60)
    assert resolved["rpm_limit"] == 60
    assert "tpm_limit" not in resolved
    assert "max_budget" not in resolved


@pytest.mark.parametrize(
    ("field", "empty"),
    [("models", []), ("metadata", {}), ("model_max_budget", {})],
)
def test_null_on_a_not_null_column_clears_to_empty(field, empty):
    """These columns are NOT NULL in the schema, so the clear is the default, not SQL NULL."""
    assert _resolve(**{field: None})[field] == empty


def test_null_budget_duration_also_clears_the_reset_time():
    resolved = _resolve(budget_duration=None)
    assert resolved["budget_duration"] is None
    assert resolved["budget_reset_at"] is None


def test_setting_budget_duration_derives_a_reset_time():
    resolved = _resolve(budget_duration="30d")
    assert resolved["budget_duration"] == "30d"
    assert isinstance(resolved["budget_reset_at"], datetime)


def test_object_permission_null_is_left_for_the_entitlement_unlink():
    """Passing it through would make the shared path's upsert branch swallow the clear."""
    assert "object_permission" not in _resolve(object_permission=None)


def test_internal_user_default_budget_does_not_override_an_explicit_clear(monkeypatch):
    monkeypatch.setattr(litellm, "max_internal_user_budget", 25.0)
    assert _resolve(user_role=LitellmUserRoles.INTERNAL_USER, max_budget=None)["max_budget"] is None


def test_internal_user_default_budget_still_applies_when_unmentioned(monkeypatch):
    monkeypatch.setattr(litellm, "max_internal_user_budget", 25.0)
    assert _resolve(user_role=LitellmUserRoles.INTERNAL_USER)["max_budget"] == 25.0


def test_legacy_user_update_still_drops_nulls():
    """The old contract is unchanged: callers relying on null-as-no-op are not broken by this PR."""
    request = UpdateUserRequest(user_id=TARGET_ID, tpm_limit=None, rpm_limit=60)
    resolved = _update_internal_user_params(request.model_dump(exclude_unset=True), request)
    assert "tpm_limit" not in resolved
    assert resolved["rpm_limit"] == 60


# --- request validation --------------------------------------------------------------------------


def test_unknown_body_key_is_rejected():
    """Ignoring it would read as "omitted", i.e. a typo would silently do nothing."""
    with pytest.raises(ValidationError):
        UserPatchRequest.model_validate({"tpm_limitt": 5})


def test_unknown_body_key_answers_422_problem(caller, prisma):
    response = TestClient(app).patch(PATCH_PATH, json={"tpm_limitt": 5})
    assert response.status_code == 422
    assert response.json()["type"].endswith("invalid-request-body")
    assert prisma.update_data.await_count == 0


# --- route behaviour -----------------------------------------------------------------------------


def test_patch_writes_the_null_through_to_the_database(caller, prisma):
    response = TestClient(app).patch(PATCH_PATH, json={"tpm_limit": None, "rpm_limit": 60})
    assert response.status_code == 200
    written = _written(prisma)
    assert written["tpm_limit"] is None
    assert written["rpm_limit"] == 60


def test_patch_returns_the_row_read_back_after_the_write(caller, prisma):
    body = TestClient(app).patch(PATCH_PATH, json={"rpm_limit": 60}).json()
    assert body["data"]["user_id"] == TARGET_ID
    assert body["data"]["tpm_limit"] == 500


def test_missing_user_is_404_and_writes_nothing(caller, prisma):
    prisma.table.find_first = AsyncMock(return_value=None)
    response = TestClient(app).patch(PATCH_PATH, json={"rpm_limit": 60})
    assert response.status_code == 404
    assert response.json()["type"].endswith("user-not-found")
    # `update_data` upserts, so a create here would be a silent user creation.
    assert prisma.update_data.await_count == 0


def test_non_admin_cannot_clear_their_own_role(caller, prisma):
    caller.value = UserAPIKeyAuth(user_id=TARGET_ID, user_role=LitellmUserRoles.INTERNAL_USER.value)
    response = TestClient(app).patch(PATCH_PATH, json={"user_role": None})
    assert response.status_code == 403
    assert prisma.update_data.await_count == 0


def test_non_admin_cannot_clear_their_own_budget(caller, prisma):
    caller.value = UserAPIKeyAuth(user_id=TARGET_ID, user_role=LitellmUserRoles.INTERNAL_USER.value)
    response = TestClient(app).patch(PATCH_PATH, json={"max_budget": None})
    assert response.status_code == 403
    assert prisma.update_data.await_count == 0


def test_non_admin_cannot_patch_someone_else(caller, prisma):
    caller.value = UserAPIKeyAuth(user_id="other", user_role=LitellmUserRoles.INTERNAL_USER.value)
    response = TestClient(app).patch(PATCH_PATH, json={"rpm_limit": 60})
    assert response.status_code == 403
    assert prisma.update_data.await_count == 0
