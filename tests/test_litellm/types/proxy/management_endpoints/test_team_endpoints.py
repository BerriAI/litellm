import pytest
from pydantic import ValidationError

from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkUpdateTeamMemberPermissionsRequest,
    UpdateTeamMemberPermissionsRequest,
)


def test_permissions_update_accepts_enforced_grants():
    request = UpdateTeamMemberPermissionsRequest(
        team_id="team-1",
        team_member_permissions=["/key/generate", "/key/delete"],
    )
    assert [permission.value for permission in request.team_member_permissions] == ["/key/generate", "/key/delete"]


@pytest.mark.parametrize("dead_route", ["/key/block", "/key/bulk_update", "/made/up/route"])
def test_permissions_update_rejects_unenforced_grants(dead_route):
    with pytest.raises(ValidationError):
        UpdateTeamMemberPermissionsRequest(team_id="team-1", team_member_permissions=[dead_route])


@pytest.mark.parametrize("dead_route", ["/key/unblock", "/spend/logs/v2"])
def test_bulk_permissions_update_rejects_unenforced_grants(dead_route):
    with pytest.raises(ValidationError):
        BulkUpdateTeamMemberPermissionsRequest(permissions=[dead_route], apply_to_all_teams=True)
