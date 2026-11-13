import asyncio
import base64
import uuid
from collections.abc import Mapping
from typing import Final, Protocol

import httpx
from pydantic import ValidationError

from litellm.proxy.agent_endpoints.external_agents.database import Database
from litellm.proxy.agent_endpoints.external_agents.models import FrozenModel, HookFailure, MemberSyncResult


class MemberHTTPClient(Protocol):
    async def get(self, url: str, *, headers: Mapping[str, str], timeout: float) -> httpx.Response: ...


class VendorMember(FrozenModel):
    id: str
    email: str
    isRemoved: bool = False


class VendorMembers(FrozenModel):
    teamMembers: tuple[VendorMember, ...]


async def sync_members(
    db: Database, client: MemberHTTPClient, source_id: str, key: str
) -> MemberSyncResult | HookFailure:
    credentials: Final = base64.b64encode(f"{key}:".encode()).decode()
    try:
        response: Final = await client.get(
            "https://api.cursor.com/teams/members",
            headers={"Authorization": f"Basic {credentials}"},
            timeout=5.0,
        )
        response.raise_for_status()
        members: Final = VendorMembers.model_validate_json(response.content)
    except (httpx.HTTPError, ValidationError):
        return HookFailure(status=502, reason="Member sync failed. Check the GrokBot Admin API key and retry.")
    active: Final = tuple(member for member in members.teamMembers if not member.isRemoved)
    results: Final = tuple(await asyncio.gather(*(match_member(db, source_id, member) for member in active)))
    return MemberSyncResult(
        matched=results.count("matched"),
        unmatched=results.count("unmatched"),
        already_mapped=results.count("already_mapped"),
    )


async def match_member(db: Database, source_id: str, member: VendorMember) -> str:
    existing: Final = await db.query_raw(
        'SELECT id FROM "LiteLLM_ExternalAgentActor" WHERE source_id=$1 AND external_actor_id=$2',
        source_id,
        member.id,
    )
    if existing:
        return "already_mapped"
    inserted: Final = await db.execute_raw(
        'INSERT INTO "LiteLLM_ExternalAgentActor" (id,source_id,external_actor_id,user_id) '
        'SELECT $1,$2,$3,u.user_id FROM "LiteLLM_UserTable" u WHERE lower(u.user_email)=lower($4) '
        'AND (SELECT COUNT(*) FROM "LiteLLM_UserTable" v WHERE lower(v.user_email)=lower($4))=1 '
        "ON CONFLICT (source_id,external_actor_id) DO NOTHING",
        str(uuid.uuid4()),
        source_id,
        member.id,
        member.email,
    )
    return "matched" if inserted else "unmatched"
