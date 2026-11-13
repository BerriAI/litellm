import secrets
import uuid
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.agent_endpoints.external_agents.cache import GovernanceMetadata
from litellm.proxy.agent_endpoints.external_agents.database import Database
from litellm.proxy.agent_endpoints.external_agents.members import MemberHTTPClient, sync_members
from litellm.proxy.agent_endpoints.external_agents.models import (
    ActorsPut,
    Agent,
    AgentPatch,
    Decision,
    ExternalAgentSummary,
    HookFailure,
    MemberSyncResult,
    Policy,
    PolicyPut,
    Source,
    SourceConnection,
    SourceCreate,
    SourcePatch,
    SourceSaved,
)
from litellm.proxy.agent_endpoints.external_agents.source import signing_key
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper


class ExternalAgentManagement:
    def __init__(
        self, db: Database, cache: GovernanceMetadata | None = None, client: MemberHTTPClient | None = None
    ) -> None:
        self.db = db
        self.cache = cache
        self.client = client

    def invalidate(self) -> None:
        if self.cache is not None:
            self.cache.invalidate()

    async def create_source(self, data: SourceCreate, user_id: str) -> SourceConnection | HookFailure:
        secret: Final = data.signing_secret.get_secret_value() if data.signing_secret else None
        valid: Final = signing_key(secret) if secret is not None else None
        if isinstance(valid, HookFailure):
            return valid
        ingest: Final = "litellm_ingest_" + secrets.token_urlsafe(32)
        policy_id: Final = str(uuid.uuid4())
        rows: Final = TypeAdapter(list[Source]).validate_python(
            await self.db.query_raw(
                'WITH source AS (INSERT INTO "LiteLLM_ExternalAgentSource" '
                "(source_id,display_name,signing_secret,mode,created_by,updated_by,ingest_key,admin_api_key,default_policy_id) "
                "VALUES ($1,$2,$3,$4,$5,$5,$6,$7,$8) RETURNING *), "
                'policy AS (INSERT INTO "LiteLLM_ExternalAgentPolicy" '
                "(policy_id,source_id,blocked_tools,allow_unmapped_actors,require_owner) "
                "SELECT $8,source_id,ARRAY[]::text[],true,false FROM source) SELECT * FROM source",
                str(uuid.uuid4()),
                data.display_name,
                encrypt_value_helper(secret) if secret else None,
                data.mode,
                user_id,
                encrypt_value_helper(ingest),
                encrypt_value_helper(data.admin_api_key.get_secret_value()) if data.admin_api_key else None,
                policy_id,
            )
        )
        self.invalidate()
        synced: Final = await self.sync_members(rows[0].source_id) if data.admin_api_key else None
        if isinstance(synced, HookFailure):
            return synced
        return SourceConnection.model_validate(
            rows[0].model_dump()
            | {
                "ingest_key": ingest,
                "hook_path": f"/grok_bot/inference_hook/{rows[0].source_id}",
                "sync_result": synced,
            }
        )

    async def connection(self, source_id: str) -> SourceConnection | HookFailure:
        rows: Final = TypeAdapter(list[Source]).validate_python(
            await self.db.query_raw(
                'UPDATE "LiteLLM_ExternalAgentSource" SET ingest_key=COALESCE(ingest_key,$2) '
                "WHERE source_id=$1 RETURNING *",
                source_id,
                encrypt_value_helper("litellm_ingest_" + secrets.token_urlsafe(32)),
            )
        )
        if not rows:
            return HookFailure(status=404, reason="Unknown source")
        self.invalidate()
        key: Final = decrypt_value_helper(rows[0].ingest_key, "ingest_key") if rows[0].ingest_key else None
        if not key:
            return HookFailure(status=503, reason="Connection credentials unavailable")
        return SourceConnection.model_validate(
            rows[0].model_dump() | {"ingest_key": key, "hook_path": f"/grok_bot/inference_hook/{source_id}"}
        )

    async def sync_members(self, source_id: str) -> MemberSyncResult | HookFailure:
        rows: Final = TypeAdapter(list[Source]).validate_python(
            await self.db.query_raw('SELECT * FROM "LiteLLM_ExternalAgentSource" WHERE source_id=$1', source_id)
        )
        if not rows:
            return HookFailure(status=404, reason="Unknown source")
        key: Final = decrypt_value_helper(rows[0].admin_api_key, "admin_api_key") if rows[0].admin_api_key else None
        if not key or self.client is None:
            return HookFailure(status=400, reason="Save a GrokBot Admin API key before syncing members")
        result: Final = await sync_members(self.db, self.client, source_id, key)
        self.invalidate()
        return result

    async def sources(self) -> tuple[Source, ...]:
        return TypeAdapter(tuple[Source, ...]).validate_python(
            await self.db.query_raw(
                'SELECT * FROM "LiteLLM_ExternalAgentSource" ORDER BY created_at,source_id',
            )
        )

    async def patch_source(self, source_id: str, data: SourcePatch, user_id: str) -> SourceSaved | HookFailure:
        secret: Final = data.signing_secret.get_secret_value() if data.signing_secret else None
        valid: Final = signing_key(secret) if secret is not None else None
        if isinstance(valid, HookFailure):
            return valid
        rows: Final = TypeAdapter(list[Source]).validate_python(
            await self.db.query_raw(
                'UPDATE "LiteLLM_ExternalAgentSource" SET display_name=COALESCE($2,display_name),mode=COALESCE($3,mode),'
                "previous_signing_secret=CASE WHEN $4::text IS NULL THEN previous_signing_secret ELSE signing_secret END,"
                "signing_secret=COALESCE($4,signing_secret),updated_by=$5,updated_at=NOW(),"
                "admin_api_key=COALESCE($6,admin_api_key) WHERE source_id=$1 RETURNING *",
                source_id,
                data.display_name,
                data.mode,
                encrypt_value_helper(secret) if secret else None,
                user_id,
                encrypt_value_helper(data.admin_api_key.get_secret_value()) if data.admin_api_key else None,
            )
        )
        self.invalidate()
        if not rows:
            return HookFailure(status=404, reason="Unknown source")
        synced: Final = await self.sync_members(source_id) if data.admin_api_key else None
        if isinstance(synced, HookFailure):
            return synced
        return SourceSaved.model_validate(rows[0].model_dump() | {"sync_result": synced})

    async def agents(
        self,
        source_id: str | None,
        team_id: str | None,
        exceptions_only: bool,
        search: str | None,
        offset: int,
        limit: int,
    ) -> tuple[Agent, ...]:
        return TypeAdapter(tuple[Agent, ...]).validate_python(
            await self.db.query_raw(
                "WITH bots AS (SELECT a.*,COALESCE(NULLIF(u.user_email,''),NULLIF(u.user_alias,'')) "
                "AS owner_display_name,array_remove(ARRAY["
                "CASE WHEN a.team_id IS NULL THEN 'no team' END, "
                "CASE WHEN a.owner_user_id IS NULL THEN 'no owner' END, "
                "CASE WHEN a.owner_inferred THEN 'owner inferred' END, "
                'CASE WHEN NOT EXISTS (SELECT 1 FROM "LiteLLM_ExternalAgentActor" m '
                'JOIN "LiteLLM_UserTable" u ON u.user_id=m.user_id '
                "WHERE m.source_id=a.source_id AND m.external_actor_id=a.last_actor_id) THEN 'unmapped member' END, "
                'CASE WHEN EXISTS (SELECT 1 FROM "LiteLLM_ExternalAgentDecision" d WHERE d.source_id=a.source_id '
                "AND d.external_agent_id=a.external_agent_id AND d.would_verdict='deny' "
                "AND d.created_at >= NOW()-INTERVAL '24 hours') THEN 'denied recently' END],NULL) AS issues "
                'FROM "LiteLLM_ExternalAgent" a LEFT JOIN "LiteLLM_UserTable" u ON u.user_id=a.owner_user_id '
                "WHERE ($1::text IS NULL OR a.source_id=$1) "
                "AND ($2::text IS NULL OR a.team_id=$2) "
                "AND ($3::text IS NULL OR a.display_name ILIKE $3 OR a.external_agent_id ILIKE $3)) "
                "SELECT * FROM bots WHERE (NOT $4 OR cardinality(issues)>0) "
                "ORDER BY last_seen_at DESC,id LIMIT $5 OFFSET $6",
                source_id,
                team_id,
                f"%{search}%" if search else None,
                exceptions_only,
                limit,
                offset,
            )
        )

    async def summary(self, source_id: str) -> ExternalAgentSummary:
        rows: Final = TypeAdapter(list[ExternalAgentSummary]).validate_python(
            await self.db.query_raw(
                "SELECT COUNT(*)::int AS discovered, "
                'COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM "LiteLLM_ExternalAgentPolicy" p '
                "WHERE p.source_id=a.source_id AND (p.team_id=a.team_id OR p.team_id IS NULL)))::int "
                "AS covered_by_team_policy, "
                "COUNT(*) FILTER (WHERE a.team_id IS NULL OR a.owner_user_id IS NULL OR a.owner_inferred "
                'OR NOT EXISTS (SELECT 1 FROM "LiteLLM_ExternalAgentActor" m JOIN "LiteLLM_UserTable" u '
                "ON u.user_id=m.user_id WHERE m.source_id=a.source_id AND m.external_actor_id=a.last_actor_id) "
                'OR EXISTS (SELECT 1 FROM "LiteLLM_ExternalAgentDecision" d WHERE d.source_id=a.source_id '
                "AND d.external_agent_id=a.external_agent_id AND d.would_verdict='deny' "
                "AND d.created_at >= NOW()-INTERVAL '24 hours'))::int AS needs_attention, "
                "COUNT(*) FILTER (WHERE owner_user_id IS NULL OR owner_inferred)::int AS needs_owner, "
                '(SELECT COUNT(DISTINCT d.actor_id)::int FROM "LiteLLM_ExternalAgentDecision" d '
                'WHERE d.source_id=$1 AND NOT EXISTS (SELECT 1 FROM "LiteLLM_ExternalAgentActor" m '
                'JOIN "LiteLLM_UserTable" u ON u.user_id=m.user_id '
                "WHERE m.source_id=d.source_id AND m.external_actor_id=d.actor_id)) AS unmapped_actors, "
                '(SELECT COUNT(*)::int FROM "LiteLLM_ExternalAgentDecision" d WHERE d.source_id=$1 '
                "AND d.enforced AND d.verdict='deny' "
                "AND d.created_at >= NOW()-INTERVAL '24 hours') AS denied_24h, "
                'MAX(a.last_seen_at) AS last_event_at FROM "LiteLLM_ExternalAgent" a WHERE a.source_id=$1',
                source_id,
            )
        )
        return rows[0]

    async def bulk(self, data: AgentPatch) -> int:
        return await self.db.execute_raw(
            'UPDATE "LiteLLM_ExternalAgent" SET team_id=CASE WHEN $2 THEN $3 ELSE team_id END, '
            "owner_user_id=CASE WHEN $4 THEN $5 ELSE owner_user_id END, "
            "owner_inferred=CASE WHEN $4 THEN false ELSE owner_inferred END, "
            "status=COALESCE($6,status),display_name=CASE WHEN $7 THEN $8 ELSE display_name END "
            "WHERE id=ANY($1::text[])",
            list(data.agent_ids),
            "team_id" in data.model_fields_set,
            data.team_id,
            "owner_user_id" in data.model_fields_set,
            data.owner_user_id,
            data.status,
            "display_name" in data.model_fields_set,
            data.display_name,
        )

    async def policies(self, source_id: str) -> tuple[Policy, ...]:
        return TypeAdapter(tuple[Policy, ...]).validate_python(
            await self.db.query_raw(
                'SELECT * FROM "LiteLLM_ExternalAgentPolicy" WHERE source_id=$1 ORDER BY policy_id',
                source_id,
            )
        )

    async def put_policy(self, data: PolicyPut) -> Policy | HookFailure:
        async with self.db.tx() as tx:
            source: Final = await tx.query_raw(
                'SELECT source_id FROM "LiteLLM_ExternalAgentSource" WHERE source_id=$1 FOR UPDATE',
                data.source_id,
            )
            if not source:
                return HookFailure(status=404, reason="Unknown source")
            existing: Final = TypeAdapter(list[Policy]).validate_python(
                await tx.query_raw(
                    'SELECT * FROM "LiteLLM_ExternalAgentPolicy" WHERE source_id=$1 AND team_id IS NOT DISTINCT FROM $2',
                    data.source_id,
                    data.team_id,
                )
            )
            policy_id: Final = existing[0].policy_id if existing else str(uuid.uuid4())
            rows: Final = TypeAdapter(list[Policy]).validate_python(
                await tx.query_raw(
                    'INSERT INTO "LiteLLM_ExternalAgentPolicy" '
                    "(policy_id,source_id,team_id,blocked_tools,allow_unmapped_actors,require_owner) "
                    "VALUES ($1,$2,$3,$4::text[],$5,$6) ON CONFLICT (policy_id) DO UPDATE SET "
                    "blocked_tools=EXCLUDED.blocked_tools,allow_unmapped_actors=EXCLUDED.allow_unmapped_actors,"
                    "require_owner=EXCLUDED.require_owner RETURNING *",
                    policy_id,
                    data.source_id,
                    data.team_id,
                    list(data.blocked_tools),
                    data.allow_unmapped_actors,
                    data.require_owner,
                )
            )
            if data.team_id is None:
                await tx.execute_raw(
                    'UPDATE "LiteLLM_ExternalAgentSource" SET default_policy_id=$1 WHERE source_id=$2',
                    policy_id,
                    data.source_id,
                )
        self.invalidate()
        return rows[0]

    async def actors(self, data: ActorsPut) -> int | HookFailure:
        async with self.db.tx() as tx:
            source: Final = await tx.query_raw(
                'SELECT source_id FROM "LiteLLM_ExternalAgentSource" WHERE source_id=$1 FOR UPDATE', data.source_id
            )
            if not source:
                return HookFailure(status=404, reason="Unknown source")
            for mapping in data.mappings:
                user = await tx.query_raw('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=$1', mapping.user_id)
                if not user:
                    return HookFailure(status=400, reason="Unknown mapped user")
            for mapping in data.mappings:
                await tx.execute_raw(
                    'INSERT INTO "LiteLLM_ExternalAgentActor" (id,source_id,external_actor_id,user_id) VALUES ($1,$2,$3,$4) '
                    "ON CONFLICT (source_id,external_actor_id) DO UPDATE SET user_id=EXCLUDED.user_id",
                    str(uuid.uuid4()),
                    data.source_id,
                    mapping.external_actor_id,
                    mapping.user_id,
                )
        self.invalidate()
        return len(data.mappings)

    async def decisions(
        self,
        source_id: str | None,
        external_agent_id: str | None,
        verdict: str | None,
        offset: int,
        limit: int,
    ) -> tuple[Decision, ...]:
        return TypeAdapter(tuple[Decision, ...]).validate_python(
            await self.db.query_raw(
                "SELECT d.*,COALESCE(NULLIF(u.user_email,''),NULLIF(u.user_alias,'')) AS user_display_name, "
                'a.display_name AS agent_display_name FROM "LiteLLM_ExternalAgentDecision" d '
                'LEFT JOIN "LiteLLM_UserTable" u ON u.user_id=d.user_id '
                'LEFT JOIN "LiteLLM_ExternalAgent" a ON a.source_id=d.source_id '
                "AND a.external_agent_id=d.external_agent_id WHERE ($1::text IS NULL OR d.source_id=$1) "
                "AND ($2::text IS NULL OR d.external_agent_id=$2) AND ($3::text IS NULL OR d.verdict=$3) "
                "ORDER BY d.created_at DESC,d.request_id LIMIT $4 OFFSET $5",
                source_id,
                external_agent_id,
                verdict,
                limit,
                offset,
            )
        )
