import asyncio
import json
import os
import sys
from litellm._uuid import uuid
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../"))  # Adds the parent directory to the system path


@pytest.mark.asyncio
async def test_organization_update_object_permissions_existing_permission(monkeypatch):
    """
    Test updating object permissions when an organization already has an existing object_permission_id.

    This test verifies that when updating vector stores for an organization that already has an
    object_permission_id, the existing LiteLLM_ObjectPermissionTable record is updated
    with the new permissions and the object_permission_id remains the same.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock existing organization with object_permission_id
    existing_organization_row = LiteLLM_OrganizationTable(
        organization_id="test_org_id",
        object_permission_id="existing_perm_id_123",
        organization_alias="test_org",
        budget_id="test_budget_id",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by",
        updated_by="test_updated_by",
    )

    # Mock existing object permission record
    existing_object_permission = MagicMock()
    existing_object_permission.model_dump.return_value = {
        "object_permission_id": "existing_perm_id_123",
        "vector_stores": ["old_store_1", "old_store_2"],
    }

    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=existing_object_permission)

    # Mock upsert operation
    updated_permission = MagicMock()
    updated_permission.object_permission_id = "existing_perm_id_123"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=updated_permission)

    # Test data with new object permission
    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["new_store_1", "new_store_2", "new_store_3"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "organization_alias": "updated_org",
    }

    # Call the function
    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row,
    )

    # Verify the object_permission was removed from data_json and object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "existing_perm_id_123"

    # Verify database operations were called correctly
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "existing_perm_id_123"}
    )
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_get_organization_daily_activity_admin_param_passing(monkeypatch):
    """
    As admin, ensure parsed params are forwarded to get_daily_activity with correct values.
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import (
        get_organization_daily_activity,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Admin view -> skip membership restriction
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints._user_has_admin_view",
        lambda _: True,
    )

    # Patch downstream common function and verify call args
    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(organization_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    result = await get_organization_daily_activity(
        organization_ids="org1,org2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model="gpt-4",
        api_key="test-key",
        page=2,
        page_size=5,
        exclude_organization_ids="org3",
        user_api_key_dict=auth,
    )

    # Ensure passthrough to common method with correct args
    get_daily_activity_mock.assert_awaited_once()
    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["table_name"] == "litellm_dailyorganizationspend"
    assert kwargs["entity_id_field"] == "organization_id"
    assert kwargs["entity_id"] == ["org1", "org2"]
    assert kwargs["exclude_entity_ids"] == ["org3"]
    assert kwargs["start_date"] == "2024-01-01"
    assert kwargs["end_date"] == "2024-01-31"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["page"] == 2
    assert kwargs["page_size"] == 5

    assert result is mocked_response


@pytest.mark.asyncio
async def test_get_organization_daily_activity_non_admin_defaults_to_admin_orgs(
    monkeypatch,
):
    """
    Non-admin with no explicit organization_ids should default to orgs they are ORG_ADMIN of.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import (
        get_organization_daily_activity,
    )

    # Mock prisma client and memberships
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_organizationmembership.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(organization_id="orgA", user_role=LitellmUserRoles.ORG_ADMIN.value),
            SimpleNamespace(organization_id="orgB", user_role=LitellmUserRoles.ORG_ADMIN.value),
        ]
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Non-admin view
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints._user_has_admin_view",
        lambda _: False,
    )

    # Patch downstream aggregator
    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(organization_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="regular-user")
    await get_organization_daily_activity(
        organization_ids=None,
        start_date="2024-02-01",
        end_date="2024-02-28",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
        exclude_organization_ids=None,
        user_api_key_dict=auth,
    )

    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["entity_id"] == ["orgA", "orgB"]
    assert kwargs["start_date"] == "2024-02-01"
    assert kwargs["end_date"] == "2024-02-28"


@pytest.mark.asyncio
async def test_get_organization_daily_activity_non_admin_unauthorized_org_raises(
    monkeypatch,
):
    """
    Non-admin requesting an org they aren't ORG_ADMIN for should raise 403.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import (
        get_organization_daily_activity,
    )

    # Mock prisma client and memberships (only orgA is admin)
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_organizationmembership.find_many = AsyncMock(
        return_value=[SimpleNamespace(organization_id="orgA", user_role=LitellmUserRoles.ORG_ADMIN.value)]
    )
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Non-admin view
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.organization_endpoints._user_has_admin_view",
        lambda _: False,
    )

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="regular-user")

    with pytest.raises(HTTPException) as exc:
        await get_organization_daily_activity(
            organization_ids="orgA,orgX",  # orgX is unauthorized
            start_date="2024-03-01",
            end_date="2024-03-31",
            model=None,
            api_key=None,
            page=1,
            page_size=10,
            exclude_organization_ids=None,
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_organization_update_object_permissions_no_existing_permission(
    monkeypatch,
):
    """
    Test creating object permissions when an organization has no existing object_permission_id.

    This test verifies that when updating object permissions for an organization that has
    object_permission_id set to None, a new entry is created in the
    LiteLLM_ObjectPermissionTable and the organization is updated with the new object_permission_id.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_organization_row_no_perm = LiteLLM_OrganizationTable(
        organization_id="test_org_id_2",
        object_permission_id=None,
        organization_alias="test_org_2",
        budget_id="test_budget_id_2",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by_2",
        updated_by="test_updated_by_2",
    )

    # Mock find_unique to return None (no existing permission)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=None)

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "new_perm_id_456"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=new_permission)

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(vector_stores=["brand_new_store"]).model_dump(
            exclude_unset=True, exclude_none=True
        ),
        "organization_alias": "updated_org_2",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row_no_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "new_perm_id_456"

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_organization_update_object_permissions_missing_permission_record(
    monkeypatch,
):
    """
    Test creating object permissions when existing object_permission_id record is not found.

    This test verifies that when updating object permissions for an organization that has an
    object_permission_id but the corresponding record cannot be found in the database,
    a new entry is created in the LiteLLM_ObjectPermissionTable with the new permissions.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_OrganizationTable,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_organization_row_missing_perm = LiteLLM_OrganizationTable(
        organization_id="test_org_id_3",
        object_permission_id="missing_perm_id_789",
        organization_alias="test_org_3",
        budget_id="test_budget_id_3",
        models=["test_model_1", "test_model_2"],
        created_by="test_created_by_3",
        updated_by="test_updated_by_3",
    )

    # Mock find_unique to return None (permission record not found)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=None)

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "recreated_perm_id_789"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(return_value=new_permission)

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(vector_stores=["recreated_store"]).model_dump(
            exclude_unset=True, exclude_none=True
        ),
        "organization_alias": "updated_org_3",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_organization_row=existing_organization_row_missing_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "recreated_perm_id_789"

    # Verify find_unique was called with the missing permission ID
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "missing_perm_id_789"}
    )

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_list_organization_filter_by_org_id(monkeypatch):
    """
    Test filtering organizations by org_id query parameter.

    This test verifies that when org_id is provided, only the organization
    with that exact organization_id is returned.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import (
        list_organization,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()

    # Mock organization data
    mock_org1 = SimpleNamespace(
        organization_id="org-123",
        organization_alias="Test Org 1",
        model_dump=lambda: {
            "organization_id": "org-123",
            "organization_alias": "Test Org 1",
        },
    )

    # Mock find_many to return filtered results
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[mock_org1])

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test as proxy admin
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-user")

    result = await list_organization(org_id="org-123", org_alias=None, user_api_key_dict=auth)

    # Verify the correct organization was returned
    assert len(result) == 1
    assert result[0].organization_id == "org-123"
    assert result[0].organization_alias == "Test Org 1"

    # Verify find_many was called with correct where conditions
    mock_prisma_client.db.litellm_organizationtable.find_many.assert_called_once()
    call_args = mock_prisma_client.db.litellm_organizationtable.find_many.call_args
    assert call_args.kwargs["where"] == {"organization_id": "org-123"}
    assert call_args.kwargs["include"] == {
        "litellm_budget_table": True,
        "members": True,
        "teams": True,
    }


@pytest.mark.asyncio
async def test_list_organization_filter_by_org_alias(monkeypatch):
    """
    Test filtering organizations by org_alias query parameter with case-insensitive partial matching.

    This test verifies that when org_alias is provided, organizations with matching
    organization_alias (case-insensitive partial match) are returned.
    """
    from types import SimpleNamespace

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import (
        list_organization,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()

    # Mock organization data
    mock_org1 = SimpleNamespace(
        organization_id="org-123",
        organization_alias="My Test Organization",
        model_dump=lambda: {
            "organization_id": "org-123",
            "organization_alias": "My Test Organization",
        },
    )
    mock_org2 = SimpleNamespace(
        organization_id="org-456",
        organization_alias="Another Test Org",
        model_dump=lambda: {
            "organization_id": "org-456",
            "organization_alias": "Another Test Org",
        },
    )

    # Mock find_many to return filtered results
    mock_prisma_client.db.litellm_organizationtable.find_many = AsyncMock(return_value=[mock_org1, mock_org2])

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test as proxy admin with org_alias filter
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-user")

    result = await list_organization(org_id=None, org_alias="test", user_api_key_dict=auth)

    # Verify organizations with "test" in alias were returned
    assert len(result) == 2
    assert all("test" in org.organization_alias.lower() for org in result)

    # Verify find_many was called with correct where conditions (case-insensitive contains)
    mock_prisma_client.db.litellm_organizationtable.find_many.assert_called_once()
    call_args = mock_prisma_client.db.litellm_organizationtable.find_many.call_args
    assert call_args.kwargs["where"] == {"organization_alias": {"contains": "test", "mode": "insensitive"}}
    assert call_args.kwargs["include"] == {
        "litellm_budget_table": True,
        "members": True,
        "teams": True,
    }


@pytest.mark.asyncio
async def test_organization_info_includes_user_email(monkeypatch):
    """
    Test that GET /organization/info returns user_email in members list.
    """
    from litellm.proxy._types import LiteLLM_OrganizationMembershipTable
    from datetime import datetime

    # Simulate a membership row with a nested user object that has user_email
    raw_membership = {
        "user_id": "user_abc",
        "organization_id": "org_xyz",
        "user_role": "org_admin",
        "spend": 0.0,
        "budget_id": None,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "user": {"user_email": "alice@example.com"},
        "litellm_budget_table": None,
    }

    membership = LiteLLM_OrganizationMembershipTable(**raw_membership)
    assert membership.user_email == "alice@example.com"


# Regression tests for IDOR fixes on org-scoped endpoints. Sibling cluster
# to GHSA-xxv2-fprq-9x93 (team callback IDOR): the same shape of "any
# authenticated key holder reaches an endpoint that takes an
# organization_id from the request body without an access guard." The
# fix routes ``update_organization``, ``organization_member_add``,
# ``organization_member_update``, and ``organization_member_delete``
# through the existing ``_verify_org_access`` helper.


@pytest.fixture
def unauthorized_caller():
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="random_authenticated_user",
        api_key="sk-random",
    )


@pytest.fixture
def patched_org_prisma():
    """Mock prisma so that find_unique returns a victim org and
    get_user_object reports the caller has no org membership — so
    _verify_org_access raises 403."""
    victim_row = MagicMock()
    victim_row.organization_id = "org-victim"
    victim_row.metadata = {}
    victim_row.model_dump.return_value = {"organization_id": "org-victim"}

    caller_user = MagicMock()
    caller_user.organization_memberships = []  # no admin role anywhere

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.organization_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=caller_user,
        ),
        patch(
            "litellm.proxy.proxy_server.user_api_key_cache",
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
    ):
        mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(return_value=victim_row)
        yield mock_prisma


@pytest.mark.asyncio
async def test_organization_member_add_rejects_unauthorized_caller(patched_org_prisma, unauthorized_caller):
    # ``organization_member_add`` catches HTTPException in its
    # catch-all and re-wraps as ProxyException with the original status
    # code preserved.
    from litellm.proxy._types import (
        OrganizationMemberAddRequest,
        OrgMember,
        ProxyException,
    )
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_add,
    )
    from unittest.mock import Mock

    from fastapi import Request

    data = OrganizationMemberAddRequest(
        organization_id="org-victim",
        member=OrgMember(role="internal_user", user_id="attacker-user"),
    )

    with pytest.raises((HTTPException, ProxyException)) as exc:
        await organization_member_add(
            data=data,
            http_request=Mock(spec=Request),
            user_api_key_dict=unauthorized_caller,
        )
    code = getattr(exc.value, "status_code", None) or getattr(exc.value, "code", None)
    assert int(code) == 403


@pytest.mark.asyncio
async def test_organization_member_update_rejects_unauthorized_caller(patched_org_prisma, unauthorized_caller):
    from litellm.proxy._types import OrganizationMemberUpdateRequest
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_update,
    )

    data = OrganizationMemberUpdateRequest(
        organization_id="org-victim",
        user_id="some-other-user",
        role="org_admin",
    )

    with pytest.raises(HTTPException) as exc:
        await organization_member_update(
            data=data,
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_organization_member_delete_rejects_unauthorized_caller(patched_org_prisma, unauthorized_caller):
    from litellm.proxy._types import OrganizationMemberDeleteRequest
    from litellm.proxy.management_endpoints.organization_endpoints import (
        organization_member_delete,
    )

    data = OrganizationMemberDeleteRequest(
        organization_id="org-victim",
        user_id="some-other-user",
    )

    with pytest.raises(HTTPException) as exc:
        await organization_member_delete(
            data=data,
            user_api_key_dict=unauthorized_caller,
        )
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "body",
    [{"tpm_limit": ""}, {"tmp_limit": None}],
    ids=["non-numeric-limit", "unknown-key"],
)
def test_v2_model_rejects_invalid_body(body):
    """A non-numeric limit and an unknown/misspelled key are both rejected at model validation (422 at the route)."""
    from pydantic import ValidationError

    from litellm.proxy._types import OrganizationUpdateRequestV2

    with pytest.raises(ValidationError):
        OrganizationUpdateRequestV2.model_validate(body)


class _FakeTxContext:
    def __init__(self, tx):
        self._tx = tx

    async def __aenter__(self):
        return self._tx

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _run_update_organization_v2(
    monkeypatch,
    *,
    body: dict,
    existing_budget_id,
    existing_metadata,
    existing_object_permission_id=None,
    existing_object_permission_row=None,
):
    from litellm.proxy._types import (
        LitellmUserRoles,
        OrganizationUpdateRequestV2,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import (
        update_organization_v2,
    )
    from litellm.proxy.utils import jsonify_object

    mock_prisma_client = AsyncMock()
    mock_prisma_client.jsonify_object = jsonify_object

    existing_org = MagicMock()
    existing_org.budget_id = existing_budget_id
    existing_org.object_permission_id = existing_object_permission_id
    existing_org.metadata = existing_metadata

    mock_prisma_client.db.litellm_organizationtable.find_unique = AsyncMock(return_value=existing_org)
    mock_prisma_client.db.litellm_organizationtable.update = AsyncMock(return_value=MagicMock())
    mock_prisma_client.db.litellm_budgettable.update = AsyncMock()
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=existing_object_permission_row
    )
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock()

    tx = MagicMock()
    tx.litellm_organizationtable = mock_prisma_client.db.litellm_organizationtable
    tx.litellm_budgettable = mock_prisma_client.db.litellm_budgettable
    tx.litellm_objectpermissiontable.upsert = AsyncMock()
    mock_prisma_client.db.tx = MagicMock(return_value=_FakeTxContext(tx))
    mock_prisma_client.tx = tx

    call_order = MagicMock()
    call_order.attach_mock(tx.litellm_objectpermissiontable.upsert, "permission_upsert")
    call_order.attach_mock(mock_prisma_client.db.litellm_organizationtable.update, "org_update")
    mock_prisma_client.call_order = call_order

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(organization_endpoints, "_verify_org_access", AsyncMock())

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    await update_organization_v2(
        organization_id="org-1",
        data=OrganizationUpdateRequestV2.model_validate(body),
        user_api_key_dict=auth,
    )
    return mock_prisma_client


@pytest.mark.asyncio
async def test_v2_update_clears_tpm_limit_and_metadata(monkeypatch):
    """A cleared tpm_limit is written to the budget row as None; a cleared metadata is written as {}."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"tpm_limit": None, "metadata": None},
        existing_budget_id="budget-1",
        existing_metadata={"stale": "value"},
    )

    budget_write = prisma.db.litellm_budgettable.update.await_args
    assert budget_write.kwargs["where"] == {"budget_id": "budget-1"}
    assert budget_write.kwargs["data"]["tpm_limit"] is None
    assert "soft_budget" not in budget_write.kwargs["data"]

    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert json.loads(write_data["metadata"]) == {}
    assert "budget_id" not in write_data


@pytest.mark.asyncio
async def test_v2_update_untouched_fields_not_written(monkeypatch):
    """Omitted fields are left untouched: only organization_alias is written, no budget-row write."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"organization_alias": "renamed"},
        existing_budget_id="budget-1",
        existing_metadata={"keep": "me"},
    )

    prisma.db.litellm_budgettable.update.assert_not_awaited()
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert write_data["organization_alias"] == "renamed"
    assert "metadata" not in write_data
    assert "tpm_limit" not in write_data


@pytest.mark.asyncio
async def test_v2_update_metadata_replaces_not_merges(monkeypatch):
    """Sending metadata replaces the stored blob wholesale; a previously-present key is gone."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"metadata": {"a": 1}},
        existing_budget_id="budget-1",
        existing_metadata={"stale": "value"},
    )
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert json.loads(write_data["metadata"]) == {"a": 1}


@pytest.mark.asyncio
async def test_v2_rejects_null_clear_of_non_nullable_fields(monkeypatch):
    """organization_alias and models are non-nullable columns, so a null clear is a 422, not a 500."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", AsyncMock())
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")

    for body in ({"organization_alias": None}, {"models": None}):
        with pytest.raises(HTTPException) as exc:
            await update_organization_v2(
                organization_id="org-1",
                data=OrganizationUpdateRequestV2.model_validate(body),
                user_api_key_dict=auth,
            )
        assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_v2_rejects_negative_max_budget(monkeypatch):
    """v2 rejects a negative max_budget with a 422 before touching the DB."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", AsyncMock())

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({"max_budget": -5}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 422
    assert "max_budget" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_v2_rejects_caller_without_org_access(monkeypatch):
    """v2 runs the real _verify_org_access guard: a non-admin without ORG_ADMIN on the org gets 403 and no write."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(organization_endpoints, "_user_has_admin_view", lambda _: False)

    caller = MagicMock()
    caller.organization_memberships = []
    monkeypatch.setattr(organization_endpoints, "get_user_object", AsyncMock(return_value=caller))

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="user-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({"tpm_limit": 5}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 403
    mock_prisma_client.db.litellm_organizationtable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_v2_wires_object_permission_onto_org_write(monkeypatch):
    """A sent object_permission merges over the existing permission row and its id is linked onto the org write."""
    existing_row = MagicMock()
    existing_row.model_dump.return_value = {
        "object_permission_id": "op-123",
        "mcp_servers": ["server-1"],
    }

    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"object_permission": {"vector_stores": ["vs-1"]}},
        existing_budget_id="budget-1",
        existing_metadata={},
        existing_object_permission_id="op-123",
        existing_object_permission_row=existing_row,
    )

    upsert = prisma.tx.litellm_objectpermissiontable.upsert.await_args.kwargs
    assert upsert["where"] == {"object_permission_id": "op-123"}
    assert upsert["data"]["update"]["mcp_servers"] == ["server-1"]
    assert upsert["data"]["update"]["vector_stores"] == ["vs-1"]
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert write_data["object_permission_id"] == "op-123"


@pytest.mark.asyncio
async def test_v2_object_permission_upsert_runs_inside_transaction(monkeypatch):
    """The permission upsert runs on the tx client, before the org write that links it, so a rollback cannot
    leave merged grants live on a row the org still points at."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"object_permission": {"vector_stores": ["vs-1"]}},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    prisma.tx.litellm_objectpermissiontable.upsert.assert_awaited_once()
    prisma.db.litellm_objectpermissiontable.upsert.assert_not_awaited()

    upsert = prisma.tx.litellm_objectpermissiontable.upsert.await_args.kwargs
    linked_id = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]["object_permission_id"]
    assert upsert["where"] == {"object_permission_id": linked_id}
    assert upsert["data"]["create"]["object_permission_id"] == linked_id

    ordered = [name for name, _, _ in prisma.call_order.mock_calls if name in ("permission_upsert", "org_update")]
    assert ordered == ["permission_upsert", "org_update"]


@pytest.mark.asyncio
async def test_v2_clears_object_permission_when_sent_null(monkeypatch):
    """object_permission: null detaches the org's permission row (object_permission_id -> None), no merge."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"object_permission": None},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    prisma.tx.litellm_objectpermissiontable.upsert.assert_not_awaited()
    prisma.db.litellm_objectpermissiontable.find_unique.assert_not_awaited()
    write_data = prisma.db.litellm_organizationtable.update.await_args.kwargs["data"]
    assert write_data["object_permission_id"] is None


@pytest.mark.asyncio
async def test_v2_rejects_empty_object_permission(monkeypatch):
    """object_permission: {} merges nothing, so it is rejected (send null to clear) rather than silently leaving grants."""
    from litellm.proxy._types import LitellmUserRoles, OrganizationUpdateRequestV2, UserAPIKeyAuth
    from litellm.proxy.management_endpoints import organization_endpoints
    from litellm.proxy.management_endpoints.organization_endpoints import update_organization_v2

    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(organization_endpoints, "_verify_org_access", AsyncMock())

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1")
    with pytest.raises(HTTPException) as exc:
        await update_organization_v2(
            organization_id="org-1",
            data=OrganizationUpdateRequestV2.model_validate({"object_permission": {}}),
            user_api_key_dict=auth,
        )
    assert exc.value.status_code == 422
    assert "object_permission" in str(exc.value.detail)
    mock_prisma_client.db.litellm_organizationtable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_v2_writes_budget_and_org_in_one_transaction(monkeypatch):
    """A change touching both the budget row and the org row runs both writes inside one prisma transaction."""
    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"tpm_limit": 500, "metadata": {"a": 1}},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    prisma.db.tx.assert_called_once()
    prisma.db.litellm_budgettable.update.assert_awaited_once()
    prisma.db.litellm_organizationtable.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_v2_serializes_model_max_budget_on_budget_write(monkeypatch):
    """model_max_budget is a Json column, so it is JSON-serialized on the budget-row write like new_budget/metadata."""
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.validate_model_max_budget",
        lambda _: None,
    )

    prisma = await _run_update_organization_v2(
        monkeypatch,
        body={"model_max_budget": {"gpt-4o": {"max_budget": 10}}},
        existing_budget_id="budget-1",
        existing_metadata={},
    )

    written = prisma.db.litellm_budgettable.update.await_args.kwargs["data"]["model_max_budget"]
    assert isinstance(written, str)
    assert json.loads(written) == {"gpt-4o": {"max_budget": 10}}


def test_build_budget_write_data_recomputes_reset_at_on_duration():
    """A sent budget_duration recomputes budget_reset_at so the reset window follows the new duration."""
    from litellm.proxy.management_endpoints.organization_endpoints import build_budget_write_data

    data = build_budget_write_data({"budget_duration": "30d"}, "admin-1")
    assert data["budget_duration"] == "30d"
    assert "budget_reset_at" in data
    assert data["updated_by"] == "admin-1"


def test_build_budget_write_data_no_reset_at_without_duration():
    """Clearing a limit writes it through untouched and does not recompute budget_reset_at."""
    from litellm.proxy.management_endpoints.organization_endpoints import build_budget_write_data

    data = build_budget_write_data({"tpm_limit": None}, "admin-1")
    assert data["tpm_limit"] is None
    assert "budget_reset_at" not in data
