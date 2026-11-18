from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.models.mcp_policy import PolicyFingerprint, PolicySubject

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

# Spend, activity and credential columns change on every request or are secrets, and never describe access.
_FINGERPRINT_SQL: Final = """
WITH k AS (SELECT * FROM "LiteLLM_VerificationToken" WHERE token = $1::text),
u AS (SELECT * FROM "LiteLLM_UserTable" WHERE user_id IN (SELECT user_id FROM k) OR user_id = $2::text),
t AS (SELECT * FROM "LiteLLM_TeamTable" WHERE team_id IN (SELECT team_id FROM k WHERE team_id IS NOT NULL)
         OR team_id IN (SELECT unnest(teams) FROM u) OR team_id = $3::text),
o AS (SELECT * FROM "LiteLLM_OrganizationTable"
      WHERE organization_id IN (SELECT organization_id FROM k UNION SELECT organization_id FROM u
                                UNION SELECT organization_id FROM t)
         OR organization_id = $4::text),
tm AS (SELECT * FROM "LiteLLM_TeamMembership" WHERE user_id IN (SELECT user_id FROM u) OR user_id IN (SELECT user_id FROM k)),
om AS (SELECT * FROM "LiteLLM_OrganizationMembership" WHERE user_id IN (SELECT user_id FROM u) OR user_id IN (SELECT user_id FROM k)),
p AS (SELECT * FROM "LiteLLM_ObjectPermissionTable" WHERE object_permission_id IN (SELECT object_permission_id FROM k UNION SELECT object_permission_id FROM u
                                     UNION SELECT object_permission_id FROM t UNION SELECT object_permission_id FROM o)),
ag AS (SELECT * FROM "LiteLLM_AccessGroupTable" WHERE access_group_id IN (SELECT unnest(access_group_ids) FROM k UNION SELECT unnest(access_group_ids) FROM t)),
b AS (SELECT * FROM "LiteLLM_BudgetTable" WHERE budget_id IN (SELECT budget_id FROM k UNION SELECT budget_id FROM o
                          UNION SELECT budget_id FROM tm UNION SELECT budget_id FROM om))
SELECT
  md5(concat_ws('#',
    (SELECT COALESCE(string_agg((to_jsonb(k) - 'spend' - 'total_spend' - 'model_spend' - 'budget_reset_at'
        - 'updated_at' - 'last_active' - 'last_rotation_at')::text, ',' ORDER BY token), '') FROM k),
    (SELECT COALESCE(string_agg((to_jsonb(u) - 'spend' - 'model_spend' - 'budget_reset_at' - 'updated_at'
        - 'last_breach_check_at' - 'password' - 'password_reset_required')::text, ',' ORDER BY user_id), '') FROM u),
    (SELECT COALESCE(string_agg((to_jsonb(t) - 'spend' - 'model_spend' - 'budget_reset_at' - 'updated_at')::text, ',' ORDER BY team_id), '') FROM t),
    (SELECT COALESCE(string_agg((to_jsonb(o) - 'spend' - 'model_spend' - 'budget_reset_at' - 'updated_at')::text, ',' ORDER BY organization_id), '') FROM o),
    (SELECT COALESCE(string_agg((to_jsonb(tm) - 'spend' - 'total_spend' - 'updated_at')::text, ',' ORDER BY user_id, team_id), '') FROM tm),
    (SELECT COALESCE(string_agg((to_jsonb(om) - 'spend' - 'updated_at')::text, ',' ORDER BY user_id, organization_id), '') FROM om),
    (SELECT COALESCE(string_agg(to_jsonb(p)::text, ',' ORDER BY object_permission_id), '') FROM p),
    (SELECT COALESCE(string_agg(to_jsonb(ag)::text, ',' ORDER BY access_group_id), '') FROM ag),
    (SELECT COALESCE(string_agg((to_jsonb(b) - 'budget_reset_at' - 'updated_at')::text, ',' ORDER BY budget_id), '') FROM b)
  )) AS digest,
  EXISTS (SELECT 1 FROM p WHERE cardinality(mcp_toolsets) > 0) AS uses_toolsets,
  jsonb_build_object(
    'user_id', COALESCE((SELECT user_id FROM k), $2::text),
    'project_id', (SELECT project_id FROM k),
    'team_ids', (SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
        SELECT team_id AS x FROM k WHERE team_id IS NOT NULL UNION SELECT unnest(teams) FROM u
        UNION SELECT $3::text) q WHERE x IS NOT NULL),
    'organization_ids', (SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
        SELECT organization_id AS x FROM k UNION SELECT organization_id FROM u UNION SELECT organization_id FROM t
        UNION SELECT $4::text) q WHERE x IS NOT NULL),
    'permission_ids', (SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
        SELECT object_permission_id AS x FROM k UNION SELECT object_permission_id FROM u
        UNION SELECT object_permission_id FROM t UNION SELECT object_permission_id FROM o) q WHERE x IS NOT NULL),
    'access_group_ids', (SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
        SELECT unnest(access_group_ids) AS x FROM k UNION SELECT unnest(access_group_ids) FROM t) q WHERE x IS NOT NULL)
  ) AS reachable
"""


class MCPPolicyRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._prisma_client: Final = prisma_client

    async def get_fingerprint(self, subject: PolicySubject) -> PolicyFingerprint:
        rows: Final[Sequence[Mapping[str, object]]] = await self._prisma_client.writer_db.query_raw(  # pyright: ignore[reportAny]  # Prisma's raw client is untyped
            _FINGERPRINT_SQL, subject.key_hash, subject.user_id, subject.pinned_team_id, subject.pinned_org_id
        )
        return PolicyFingerprint.model_validate(rows[0])
