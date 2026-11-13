from collections.abc import Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma import Prisma

from litellm.proxy.agent_endpoints.external_agents.cache import GovernanceMetadata
from litellm.proxy.agent_endpoints.external_agents.governance import ExternalAgentStore
from litellm.proxy.agent_endpoints.external_agents.management import ExternalAgentManagement
from litellm.proxy.agent_endpoints.external_agents.models import (
    ActorMapping,
    ActorsPut,
    MappedUser,
    Policy,
    PolicyPut,
    Source,
    SourcePatch,
)

SOURCE: Final = Source(source_id="source", display_name="Bots", signing_secret="encrypted", tenant_id="tenant")


def database_with_rows(rows: Sequence[object]) -> MagicMock:
    db: Final = MagicMock(spec=Prisma)
    db.query_raw = AsyncMock(side_effect=rows)
    db.execute_raw = AsyncMock(return_value=1)
    db.tx.return_value.__aenter__ = AsyncMock(return_value=db)
    db.tx.return_value.__aexit__ = AsyncMock(return_value=False)
    return db


@pytest.mark.asyncio
async def test_source_patch_refreshes_cached_authentication_and_mode() -> None:
    cache: Final = GovernanceMetadata()
    cache.sources.put("source", SOURCE)
    updated: Final = SOURCE.model_copy(update={"mode": "enforce", "signing_secret": "rotated"})
    row: Final = updated.model_dump() | {"signing_secret": "rotated"}
    db: Final = database_with_rows([[row], [row]])
    assert (
        await ExternalAgentManagement(db, cache).patch_source("source", SourcePatch(mode="enforce"), "admin") == updated
    )
    assert await ExternalAgentStore(db, cache).source("source") == updated


@pytest.mark.asyncio
async def test_policy_write_invalidates_cached_allow_all_policy() -> None:
    cache: Final = GovernanceMetadata()
    cache.policies.put(("source", None, None), None)
    policy: Final = Policy(policy_id="policy", source_id="source", blocked_tools=("Shell",))
    db: Final = database_with_rows([[{"source_id": "source"}], [], [policy.model_dump()], [policy.model_dump()]])
    assert (
        await ExternalAgentManagement(db, cache).put_policy(PolicyPut(source_id="source", blocked_tools=("Shell",)))
        == policy
    )
    assert await ExternalAgentStore(db, cache).policy(SOURCE, None) == policy


@pytest.mark.asyncio
async def test_actor_write_invalidates_cached_unmapped_actor() -> None:
    cache: Final = GovernanceMetadata()
    cache.actors.put(("source", "actor"), None)
    user: Final = MappedUser(user_id="user", teams=("team",))
    db: Final = database_with_rows([[{"source_id": "source"}], [{"user_id": "user"}], [user.model_dump()]])
    assert (
        await ExternalAgentManagement(db, cache).actors(
            ActorsPut(source_id="source", mappings=(ActorMapping(external_actor_id="actor", user_id="user"),))
        )
        == 1
    )
    assert await ExternalAgentStore(db, cache).actor("source", "actor") == user
