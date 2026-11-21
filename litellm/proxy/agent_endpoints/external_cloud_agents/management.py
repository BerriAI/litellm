import secrets
import uuid
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy.agent_endpoints.external_cloud_agents.database import Database
from litellm.proxy.agent_endpoints.external_cloud_agents.inventory import SourceCache, get_source
from litellm.proxy.agent_endpoints.external_cloud_agents.models import (
    Agent,
    AgentPage,
    Connection,
    Count,
    Failure,
    Source,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper, encrypt_value_helper


class Management:
    def __init__(self, db: Database, cache: SourceCache) -> None:
        self.db = db
        self.cache = cache

    async def create(self, user_id: str) -> Connection:
        source_id: Final = str(uuid.uuid4())
        ingest_key: Final = "litellm_ingest_" + secrets.token_urlsafe(32)
        await self.db.execute_raw(
            'INSERT INTO "LiteLLM_ExternalCloudAgentSource" (id,ingest_key,created_by) VALUES ($1,$2,$3)',
            source_id,
            encrypt_value_helper(ingest_key),
            user_id,
        )
        self.cache.discard(source_id)
        return Connection(source_id=source_id, hook_path=f"/grok_bot/inference_hook/{source_id}", ingest_key=ingest_key)

    async def sources(self) -> tuple[Source, ...]:
        return TypeAdapter(tuple[Source, ...]).validate_python(
            await self.db.query_raw(
                "SELECT s.id AS source_id,s.vendor,s.display_name,s.tenant_id,s.last_event_at,"
                "s.last_spend_sync_at,s.last_spend_sync_error,s.created_at,s.created_by,"
                "(s.admin_api_key IS NOT NULL) AS has_admin_api_key,(s.last_event_at IS NOT NULL) AS connected,"
                '(SELECT COUNT(*) FROM "LiteLLM_ExternalCloudAgent" a WHERE a.source_id=s.id)::int AS bot_count '
                'FROM "LiteLLM_ExternalCloudAgentSource" s ORDER BY s.created_at,s.id'
            )
        )

    async def connection(self, source_id: str) -> Connection | Failure:
        source: Final = await get_source(self.db, source_id)
        if source is None:
            return Failure(status=404, message="Unknown source")
        key: Final = decrypt_value_helper(source.ingest_key, "ingest_key")
        if not key:
            return Failure(status=503, message="Connection credentials are unavailable")
        return Connection(source_id=source_id, hook_path=f"/grok_bot/inference_hook/{source_id}", ingest_key=key)

    async def delete(self, source_id: str) -> None:
        await self.db.execute_raw('DELETE FROM "LiteLLM_ExternalCloudAgentSource" WHERE id=$1', source_id)
        self.cache.discard(source_id)

    async def remove_key(self, source_id: str) -> Failure | None:
        rows: Final = await self.db.execute_raw(
            'UPDATE "LiteLLM_ExternalCloudAgentSource" SET admin_api_key=NULL,last_spend_sync_error=NULL WHERE id=$1',
            source_id,
        )
        return Failure(status=404, message="Unknown source") if not rows else None

    async def agents(self, source_id: str, search: str, page: int, page_size: int) -> AgentPage:
        rows: Final = TypeAdapter(tuple[Agent, ...]).validate_python(
            await self.db.query_raw(
                'SELECT * FROM "LiteLLM_ExternalCloudAgent" WHERE source_id=$1 AND '
                "($2='' OR STRPOS(LOWER(external_id),LOWER($2))>0) "
                "ORDER BY last_seen_at DESC,external_id LIMIT $3 OFFSET $4",
                source_id,
                search,
                page_size,
                (page - 1) * page_size,
            )
        )
        counts: Final = TypeAdapter(tuple[Count, ...]).validate_python(
            await self.db.query_raw(
                'SELECT COUNT(*)::int AS total FROM "LiteLLM_ExternalCloudAgent" WHERE source_id=$1 AND '
                "($2='' OR STRPOS(LOWER(external_id),LOWER($2))>0)",
                source_id,
                search,
            )
        )
        return AgentPage(data=rows, page=page, page_size=page_size, total=counts[0].total)
