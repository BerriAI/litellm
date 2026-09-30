from __future__ import annotations

from types import SimpleNamespace
from typing import Final

import pytest

from litellm.repositories.organization_membership_repository import OrganizationMembershipRepository


class _MembershipTable:
    def __init__(self, rows: dict[tuple[str, str], object]) -> None:
        self.rows: Final = rows
        self.queries: Final[list[dict[str, dict[str, str]]]] = []

    async def find_unique(self, where: dict[str, dict[str, str]]) -> object | None:
        self.queries.append(where)
        key: Final = where["user_id_organization_id"]
        return self.rows.get((key["user_id"], key["organization_id"]))


def _repository(rows: dict[tuple[str, str], object]) -> tuple[OrganizationMembershipRepository, _MembershipTable]:
    table: Final = _MembershipTable(rows)
    client: Final = SimpleNamespace(db=SimpleNamespace(litellm_organizationmembership=table))
    return OrganizationMembershipRepository(client), table


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param({("u1", "org-1"): SimpleNamespace(user_role="org_admin")}, True, id="org-admin-row"),
        pytest.param({("u1", "org-1"): SimpleNamespace(user_role="internal_user")}, False, id="member-row"),
        pytest.param({("u1", "org-1"): SimpleNamespace(user_role=None)}, False, id="row-without-role"),
        pytest.param({("u1", "org-2"): SimpleNamespace(user_role="org_admin")}, False, id="admin-of-another-org"),
        pytest.param({("u2", "org-1"): SimpleNamespace(user_role="org_admin")}, False, id="another-users-row"),
        pytest.param({}, False, id="no-row"),
    ],
)
async def test_is_org_admin_reads_the_callers_membership_row(rows: dict[tuple[str, str], object], expected: bool) -> None:
    repository, table = _repository(rows)
    assert await repository.is_org_admin("u1", "org-1") is expected
    assert table.queries == [{"user_id_organization_id": {"user_id": "u1", "organization_id": "org-1"}}]
