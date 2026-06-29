import json
import os
import sys
from datetime import datetime, timezone
from litellm._uuid import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_UserTable,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.management_helpers.utils import add_new_member


@pytest.mark.asyncio
async def test_management_otel_span_redacts_mcp_global_env_var_secrets(monkeypatch):
    """A decrypted MCP global env var secret must never reach telemetry.

    MCP create/update endpoints return the server with decrypted
    ``scope="global"`` env var values so the admin UI can pre-fill the edit
    form. ``management_endpoint_wrapper`` serializes the response into an OTEL
    span, and that span is readable by observability users, so the secret value
    must be blanked there while names/scopes stay for usefulness. The endpoint's
    own return value must keep the decrypted value for the admin.
    """
    import datetime

    from litellm.proxy._types import (
        LiteLLM_MCPServerTable,
        MCPEnvVar,
        MCPEnvVarScope,
    )
    from litellm.proxy.management_helpers import utils as mgmt_utils

    captured = {}

    class _FakeOtelLogger:
        async def async_management_endpoint_success_hook(
            self, logging_payload, parent_otel_span
        ):
            captured["response"] = logging_payload.response

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "open_telemetry_logger", _FakeOtelLogger())
    monkeypatch.setattr(mgmt_utils, "is_otel_v2_enabled", lambda: False)

    secret = "s3cr3t-p@ss"
    result = LiteLLM_MCPServerTable(
        server_id="srv-1",
        alias="echo",
        url="http://localhost:8765/mcp",
        transport="http",
        env_vars=[
            MCPEnvVar(name="DB_PASSWORD", value=secret, scope=MCPEnvVarScope.global_),
            MCPEnvVar(
                name="CORP_USER",
                value="",
                scope=MCPEnvVarScope.user,
                description="Your DB username",
            ),
        ],
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
    )

    await mgmt_utils._emit_management_endpoint_otel_span(
        func=lambda: None,
        kwargs={},
        parent_otel_span=object(),
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now(),
        result=result,
    )

    serialized = captured["response"]["env_vars"]
    # The secret must not appear anywhere the span serializer would stringify.
    assert secret not in str(captured["response"])
    assert all(entry["value"] == "" for entry in serialized)
    # Names and scopes survive so the trace stays useful.
    assert {entry["name"] for entry in serialized} == {"DB_PASSWORD", "CORP_USER"}
    assert any(entry["scope"] == MCPEnvVarScope.global_ for entry in serialized)
    # The endpoint's own return value is untouched: the admin still gets the
    # decrypted value to pre-fill the edit form.
    assert result.env_vars[0].value == secret


@pytest.mark.asyncio
async def test_management_otel_span_redacts_nested_submission_env_var_secrets(
    monkeypatch,
):
    """Decrypted global env var secrets nested under ``items`` must also be blanked.

    ``GET /v1/mcp/server/submissions`` returns ``MCPSubmissionsSummary`` whose
    ``items[].env_vars`` carry decrypted ``scope="global"`` values for full admins.
    ``management_endpoint_wrapper`` stringifies that nested ``items`` value into the
    OTEL span, so redaction has to walk into ``items`` and not just the top level,
    while the endpoint's own return value keeps the value for the admin UI.
    """
    import datetime

    from litellm.proxy._types import (
        LiteLLM_MCPServerTable,
        MCPEnvVar,
        MCPEnvVarScope,
        MCPSubmissionsSummary,
    )
    from litellm.proxy.management_helpers import utils as mgmt_utils

    captured = {}

    class _FakeOtelLogger:
        async def async_management_endpoint_success_hook(
            self, logging_payload, parent_otel_span
        ):
            captured["response"] = logging_payload.response

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "open_telemetry_logger", _FakeOtelLogger())
    monkeypatch.setattr(mgmt_utils, "is_otel_v2_enabled", lambda: False)

    secret = "s3cr3t-submission"
    server = LiteLLM_MCPServerTable(
        server_id="srv-sub",
        alias="echo",
        url="http://localhost:8765/mcp",
        transport="http",
        env_vars=[
            MCPEnvVar(name="DB_PASSWORD", value=secret, scope=MCPEnvVarScope.global_),
        ],
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
    )
    result = MCPSubmissionsSummary(
        total=1, pending_review=1, active=0, rejected=0, items=[server]
    )

    await mgmt_utils._emit_management_endpoint_otel_span(
        func=lambda: None,
        kwargs={},
        parent_otel_span=object(),
        start_time=datetime.datetime.now(),
        end_time=datetime.datetime.now(),
        result=result,
    )

    # The nested secret must not appear anywhere the span serializer stringifies.
    assert secret not in str(captured["response"])

    redacted_item = captured["response"]["items"][0]
    redacted_env_vars = (
        redacted_item["env_vars"]
        if isinstance(redacted_item, dict)
        else redacted_item.env_vars
    )
    assert [entry["value"] for entry in redacted_env_vars] == [""]
    assert redacted_env_vars[0]["name"] == "DB_PASSWORD"
    # The endpoint's own return value is untouched for the admin UI.
    assert result.items[0].env_vars[0].value == secret


@pytest.mark.asyncio
async def test_add_new_member_clones_default_team_budget_id():
    """
    Test that add_new_member CLONES the team's default member budget when
    max_budget_in_team is None and a default_team_budget_id is provided.

    Cloning (rather than sharing the same budget row) is what lets admins later
    edit one member's budget without mutating every other member's budget.
    """
    from litellm.proxy._types import LitellmUserRoles

    # Setup test data
    test_user_id = "test_user_123"
    test_team_id = "test_team_456"
    test_default_budget_id = "default_budget_789"
    test_cloned_budget_id = "cloned_budget_xyz"
    test_admin_name = "test_admin"

    new_member = Member(user_id=test_user_id, role="user")

    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    mock_prisma_client = AsyncMock()

    # Mock the user table upsert operation
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": test_user_id,
        "user_email": None,
        "teams": [test_team_id],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )

    # Mock the default budget row fetched for cloning.
    mock_default_budget_row = MagicMock()
    mock_default_budget_row.model_dump.return_value = {
        "budget_id": test_default_budget_id,
        "max_budget": 100.0,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": 1000,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": "1d",
        "allowed_models": [],
    }
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=mock_default_budget_row
    )

    # Mock the cloned budget row that .create() returns.
    mock_cloned_budget_row = MagicMock()
    mock_cloned_budget_row.budget_id = test_cloned_budget_id
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_cloned_budget_row
    )

    # Mock the team membership creation
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": test_team_id,
        "user_id": test_user_id,
        "budget_id": test_cloned_budget_id,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    result_user, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=None,
        prisma_client=mock_prisma_client,
        team_id=test_team_id,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=test_admin_name,
        default_team_budget_id=test_default_budget_id,
    )

    assert result_user is not None
    assert result_user.user_id == test_user_id

    # Membership should be linked to the new cloned budget, not the shared default.
    assert result_team_membership is not None
    assert result_team_membership.budget_id == test_cloned_budget_id
    assert result_team_membership.budget_id != test_default_budget_id

    mock_prisma_client.db.litellm_usertable.upsert.assert_called_once()
    mock_prisma_client.db.litellm_teammembership.create.assert_called_once()

    # The clone must have happened: find_unique on the default, create for the clone.
    mock_prisma_client.db.litellm_budgettable.find_unique.assert_called_once_with(
        where={"budget_id": test_default_budget_id}
    )
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    cloned_create_data = (
        mock_prisma_client.db.litellm_budgettable.create.call_args.kwargs["data"]
    )
    # Cloned values from the default budget row
    assert cloned_create_data["max_budget"] == 100.0
    assert cloned_create_data["tpm_limit"] == 1000
    assert cloned_create_data["budget_duration"] == "1d"
    assert cloned_create_data["created_by"] == user_api_key_dict.user_id

    team_membership_call_args = (
        mock_prisma_client.db.litellm_teammembership.create.call_args
    )
    create_data = team_membership_call_args.kwargs["data"]
    assert create_data["budget_id"] == test_cloned_budget_id


@pytest.mark.asyncio
async def test_add_new_member_budget_duration_only_clones_default_max_budget():
    """When only a budget_duration is given and the team has a default member
    budget, the member must clone the default (keeping its max_budget) and just
    override the reset window. Creating a fresh duration-only row instead would
    silently drop the team default's cap, leaving the member uncapped."""
    from litellm.proxy._types import LitellmUserRoles

    new_member = Member(user_id="dur-clone-user", role="user")
    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    mock_prisma_client = AsyncMock()
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": "dur-clone-user",
        "user_email": None,
        "teams": ["team-dc"],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )
    mock_default_budget_row = MagicMock()
    mock_default_budget_row.model_dump.return_value = {
        "budget_id": "default-dc",
        "max_budget": 100.0,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": 1000,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": "1d",
        "allowed_models": [],
    }
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=mock_default_budget_row
    )
    mock_cloned_budget_row = MagicMock()
    mock_cloned_budget_row.budget_id = "cloned-dc"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_cloned_budget_row
    )
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": "team-dc",
        "user_id": "dur-clone-user",
        "budget_id": "cloned-dc",
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    await add_new_member(
        new_member=new_member,
        max_budget_in_team=None,
        prisma_client=mock_prisma_client,
        team_id="team-dc",
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name="test_admin",
        default_team_budget_id="default-dc",
        budget_duration="7d",
    )

    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    cloned_create_data = (
        mock_prisma_client.db.litellm_budgettable.create.call_args.kwargs["data"]
    )
    assert cloned_create_data["max_budget"] == 100.0  # kept from the team default
    assert cloned_create_data["budget_duration"] == "7d"  # overridden by the caller
    assert cloned_create_data["budget_reset_at"] > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_add_new_member_no_budget_when_no_default_and_no_max_budget():
    """
    Test that add_new_member links no budget to the team membership when
    neither max_budget_in_team nor default_team_budget_id is provided.

    When the team has no default member budget, new members get nothing.
    """
    from litellm.proxy._types import LitellmUserRoles

    test_user_id = "test_user_no_budget"
    test_team_id = "test_team_no_budget"
    test_admin_name = "test_admin"

    new_member = Member(user_id=test_user_id, role="user")

    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    mock_prisma_client = AsyncMock()

    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": test_user_id,
        "user_email": None,
        "teams": [test_team_id],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )

    # Even though we mock these, they must NOT be called on the no-budget path.
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock()
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock()
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock()

    result_user, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=None,
        prisma_client=mock_prisma_client,
        team_id=test_team_id,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=test_admin_name,
        default_team_budget_id=None,
    )

    assert result_user is not None
    assert result_user.user_id == test_user_id

    # No budget id, so no team membership row is created.
    assert result_team_membership is None
    mock_prisma_client.db.litellm_budgettable.find_unique.assert_not_called()
    mock_prisma_client.db.litellm_budgettable.create.assert_not_called()
    mock_prisma_client.db.litellm_teammembership.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_new_member_creates_new_budget_when_max_budget_provided():
    """
    Test that add_new_member creates a new budget when max_budget_in_team is provided.

    This test verifies that:
    1. When max_budget_in_team is provided
    2. A new budget is created in the litellm_budgettable
    3. The new budget_id is used for the team membership
    """
    from litellm.proxy._types import LitellmUserRoles

    # Setup test data
    test_user_id = "test_user_123"
    test_team_id = "test_team_456"
    test_max_budget = 100.0
    test_new_budget_id = "new_budget_789"
    test_admin_name = "test_admin"

    # Create a Member object with user_id
    new_member = Member(user_id=test_user_id, role="user")

    # Create UserAPIKeyAuth object
    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Mock the user table upsert operation
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": test_user_id,
        "user_email": None,
        "teams": [test_team_id],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )

    # Mock the budget table creation
    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = test_new_budget_id
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_budget_response
    )

    # Mock the team membership creation
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": test_team_id,
        "user_id": test_user_id,
        "budget_id": test_new_budget_id,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    # Call the function with max_budget_in_team provided
    result_user, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=test_max_budget,  # This should trigger budget creation
        prisma_client=mock_prisma_client,
        team_id=test_team_id,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=test_admin_name,
        default_team_budget_id=None,  # Should be ignored since max_budget_in_team is provided
    )

    # Verify that the budget was created
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    budget_call_args = mock_prisma_client.db.litellm_budgettable.create.call_args
    budget_data = budget_call_args.kwargs["data"]
    assert budget_data["max_budget"] == test_max_budget
    assert budget_data["created_by"] == user_api_key_dict.user_id
    assert budget_data["updated_by"] == user_api_key_dict.user_id

    # Verify that the team membership was created with the new budget_id
    assert result_team_membership is not None
    assert result_team_membership.budget_id == test_new_budget_id

    # Verify the team membership was created with the correct budget_id
    team_membership_call_args = (
        mock_prisma_client.db.litellm_teammembership.create.call_args
    )
    assert team_membership_call_args is not None
    create_data = team_membership_call_args.kwargs["data"]
    assert create_data["budget_id"] == test_new_budget_id


@pytest.mark.asyncio
async def test_add_new_member_persists_budget_duration():
    """Regression for the member_add half of the recurring-member-budget gap:
    a budget_duration passed to add_new_member must be written to the new
    member budget along with a future budget_reset_at, so the per-member budget
    recurs instead of acting as a lifetime cap."""
    from litellm.proxy._types import LitellmUserRoles

    new_member = Member(user_id="user-dur", role="user")
    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    mock_prisma_client = AsyncMock()
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": "user-dur",
        "user_email": None,
        "teams": ["team-dur"],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )
    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = "budget-dur"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_budget_response
    )
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": "team-dur",
        "user_id": "user-dur",
        "budget_id": "budget-dur",
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    await add_new_member(
        new_member=new_member,
        max_budget_in_team=10.0,
        prisma_client=mock_prisma_client,
        team_id="team-dur",
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name="test_admin",
        default_team_budget_id=None,
        allowed_models=["gpt-4o-mini"],
        budget_duration="30d",
    )

    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    budget_data = mock_prisma_client.db.litellm_budgettable.create.call_args.kwargs[
        "data"
    ]
    assert budget_data["max_budget"] == 10.0
    assert budget_data["allowed_models"] == ["gpt-4o-mini"]
    assert budget_data["budget_duration"] == "30d"
    reset_at = budget_data["budget_reset_at"]
    assert isinstance(reset_at, datetime)
    assert reset_at.tzinfo is not None
    assert reset_at > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_add_new_member_persists_budget_duration_without_max_budget():
    """budget_duration alone must still create a member budget; otherwise an
    explicit recurring window passed without a cap would be silently dropped."""
    from litellm.proxy._types import LitellmUserRoles

    new_member = Member(user_id="user-dur2", role="user")
    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    mock_prisma_client = AsyncMock()
    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": "user-dur2",
        "user_email": None,
        "teams": ["team-dur2"],
        "user_role": "internal_user",
    }
    mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
        return_value=mock_user_response
    )
    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = "budget-dur2"
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_budget_response
    )
    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": "team-dur2",
        "user_id": "user-dur2",
        "budget_id": "budget-dur2",
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    _, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=None,
        prisma_client=mock_prisma_client,
        team_id="team-dur2",
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name="test_admin",
        default_team_budget_id=None,
        budget_duration="7d",
    )

    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()
    budget_data = mock_prisma_client.db.litellm_budgettable.create.call_args.kwargs[
        "data"
    ]
    assert budget_data["budget_duration"] == "7d"
    assert budget_data["budget_reset_at"] > datetime.now(timezone.utc)
    assert result_team_membership is not None
    assert result_team_membership.budget_id == "budget-dur2"


@pytest.mark.asyncio
async def test_add_new_member_with_user_email_clones_default_budget():
    """
    Test add_new_member with user_email instead of user_id and a team default
    budget. The default budget should be CLONED into a new private row for
    this user, not shared with other members of the team.
    """
    from litellm.proxy._types import LitellmUserRoles

    test_user_email = "test@example.com"
    test_team_id = "test_team_456"
    test_default_budget_id = "default_budget_789"
    test_cloned_budget_id = "cloned_budget_for_email_user"
    test_admin_name = "test_admin"

    new_member = Member(user_email=test_user_email, role="user")

    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin_user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    mock_prisma_client = AsyncMock()

    mock_prisma_client.get_data = AsyncMock(return_value=[])

    mock_user_response = MagicMock()
    mock_user_response.model_dump.return_value = {
        "user_id": "generated_user_id",
        "user_email": test_user_email,
        "teams": [test_team_id],
        "user_role": "internal_user",
    }
    mock_prisma_client.insert_data = AsyncMock(return_value=mock_user_response)

    # Default budget that will be cloned
    mock_default_budget_row = MagicMock()
    mock_default_budget_row.model_dump.return_value = {
        "budget_id": test_default_budget_id,
        "max_budget": 25.0,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": None,
        "allowed_models": [],
    }
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=mock_default_budget_row
    )

    # Cloned budget result
    mock_cloned_budget_row = MagicMock()
    mock_cloned_budget_row.budget_id = test_cloned_budget_id
    mock_prisma_client.db.litellm_budgettable.create = AsyncMock(
        return_value=mock_cloned_budget_row
    )

    mock_team_membership_response = MagicMock()
    mock_team_membership_response.model_dump.return_value = {
        "team_id": test_team_id,
        "user_id": "generated_user_id",
        "budget_id": test_cloned_budget_id,
        "litellm_budget_table": None,
    }
    mock_prisma_client.db.litellm_teammembership.create = AsyncMock(
        return_value=mock_team_membership_response
    )

    result_user, result_team_membership = await add_new_member(
        new_member=new_member,
        max_budget_in_team=None,
        prisma_client=mock_prisma_client,
        team_id=test_team_id,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=test_admin_name,
        default_team_budget_id=test_default_budget_id,
    )

    assert result_user is not None
    assert result_user.user_email == test_user_email

    # Membership should point at the cloned (private) budget, not the shared default.
    assert result_team_membership is not None
    assert result_team_membership.budget_id == test_cloned_budget_id

    mock_prisma_client.get_data.assert_called_once_with(
        key_val={"user_email": test_user_email},
        table_name="user",
        query_type="find_all",
    )

    mock_prisma_client.insert_data.assert_called_once()
    insert_call_args = mock_prisma_client.insert_data.call_args
    insert_data = insert_call_args.kwargs["data"]
    assert insert_data["user_email"] == test_user_email
    assert insert_data["teams"] == [test_team_id]

    # Confirm the clone path ran
    mock_prisma_client.db.litellm_budgettable.find_unique.assert_called_once_with(
        where={"budget_id": test_default_budget_id}
    )
    mock_prisma_client.db.litellm_budgettable.create.assert_called_once()


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_object_permission_id():
    """
    Test that attach_object_permission_to_dict correctly attaches object_permission
    when object_permission_id is present and found in database.
    """
    from litellm.proxy.management_helpers.object_permission_utils import (
        attach_object_permission_to_dict,
    )

    # Setup test data
    test_object_permission_id = "test_perm_123"
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": test_object_permission_id,
        "other_field": "other_value",
    }

    expected_object_permission = {
        "object_permission_id": test_object_permission_id,
        "vector_stores": ["store1", "store2"],
        "assistants": ["assistant1"],
        "models": ["gpt-4", "claude-3"],
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Mock the object permission response
    mock_object_permission = MagicMock()
    mock_object_permission.model_dump.return_value = expected_object_permission

    # Mock the database query
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=mock_object_permission
    )

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict, prisma_client=mock_prisma_client
    )

    # Verify the result
    assert result is not None
    assert result["user_id"] == "test_user_456"
    assert result["object_permission_id"] == test_object_permission_id
    assert result["other_field"] == "other_value"
    assert result["object_permission"] == expected_object_permission

    # Verify the database query was called correctly
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": test_object_permission_id}
    )


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_without_object_permission_id():
    """
    Test that attach_object_permission_to_dict returns the original dict unchanged
    when object_permission_id is not present.
    """
    from litellm.proxy.management_helpers.object_permission_utils import (
        attach_object_permission_to_dict,
    )

    # Setup test data without object_permission_id
    test_data_dict = {"user_id": "test_user_456", "other_field": "other_value"}

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict, prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == test_data_dict
    assert "object_permission" not in result

    # Verify no database query was made
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_object_permission_not_found():
    """
    Test that attach_object_permission_to_dict returns the original dict unchanged
    when object_permission_id is present but not found in database.
    """
    from litellm.proxy.management_helpers.object_permission_utils import (
        attach_object_permission_to_dict,
    )

    # Setup test data
    test_object_permission_id = "test_perm_123"
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": test_object_permission_id,
        "other_field": "other_value",
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Mock the database query to return None (not found)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=None
    )

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict, prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == test_data_dict
    assert "object_permission" not in result

    # Verify the database query was called
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": test_object_permission_id}
    )


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_dict_method():
    """
    Test that attach_object_permission_to_dict handles object permissions that use .dict() method
    instead of .model_dump() method.
    """
    from litellm.proxy.management_helpers.object_permission_utils import (
        attach_object_permission_to_dict,
    )

    # Setup test data
    test_object_permission_id = "test_perm_123"
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": test_object_permission_id,
        "other_field": "other_value",
    }

    expected_object_permission = {
        "object_permission_id": test_object_permission_id,
        "vector_stores": ["store1"],
        "assistants": [],
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Mock the object permission response that uses .dict() method
    mock_object_permission = MagicMock()
    mock_object_permission.model_dump.side_effect = AttributeError(
        "No model_dump method"
    )
    mock_object_permission.dict.return_value = expected_object_permission

    # Mock the database query
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=mock_object_permission
    )

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict, prisma_client=mock_prisma_client
    )

    # Verify the result
    assert result is not None
    assert result["object_permission"] == expected_object_permission

    # Verify both methods were attempted
    mock_object_permission.model_dump.assert_called_once()
    mock_object_permission.dict.assert_called_once()


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_none_prisma_client():
    """
    Test that attach_object_permission_to_dict raises ValueError when prisma_client is None.
    """
    from litellm.proxy.management_helpers.object_permission_utils import (
        attach_object_permission_to_dict,
    )

    # Setup test data
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": "test_perm_123",
    }

    # Call the function with None prisma_client
    with pytest.raises(ValueError, match="Prisma client not found"):
        await attach_object_permission_to_dict(
            data_dict=test_data_dict, prisma_client=None  # type: ignore
        )


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_empty_dict():
    """
    Test that attach_object_permission_to_dict handles empty dictionaries correctly.
    """
    from litellm.proxy.management_helpers.object_permission_utils import (
        attach_object_permission_to_dict,
    )

    # Setup empty test data
    test_data_dict = {}

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict, prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == {}
    assert "object_permission" not in result

    # Verify no database query was made
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_attach_object_permission_to_dict_with_none_object_permission_id():
    """
    Test that attach_object_permission_to_dict handles None object_permission_id correctly.
    """
    from litellm.proxy.management_helpers.object_permission_utils import (
        attach_object_permission_to_dict,
    )

    # Setup test data with None object_permission_id
    test_data_dict = {
        "user_id": "test_user_456",
        "object_permission_id": None,
        "other_field": "other_value",
    }

    # Mock the prisma client
    mock_prisma_client = AsyncMock()

    # Call the function
    result = await attach_object_permission_to_dict(
        data_dict=test_data_dict, prisma_client=mock_prisma_client
    )

    # Verify the result is unchanged
    assert result == test_data_dict
    assert "object_permission" not in result

    # Verify no database query was made
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_not_called()
