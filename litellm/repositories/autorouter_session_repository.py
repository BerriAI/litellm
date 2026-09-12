"""
Repository for the auto-router per-session rollup (LiteLLM_AutoRouterSession).
"""

from typing import TYPE_CHECKING, Final

from litellm.models.autorouter_session import LiteLLM_AutoRouterSession
from litellm.repositories.base_repository import BaseRepository
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models


class AutoRouterSessionRepository(BaseRepository[LiteLLM_AutoRouterSession]):
    @property
    def table(self) -> TableActions["prisma_models.LiteLLM_AutoRouterSession"]:
        return self.prisma_client.db.litellm_autoroutersession

    @property
    def model_class(self) -> type[LiteLLM_AutoRouterSession]:
        return LiteLLM_AutoRouterSession

    async def find_latest_for_key(self, api_key: str, session_id: str) -> LiteLLM_AutoRouterSession | None:
        """The session's most recently active router row under exactly this key hash, or None.

        The key is the row's own partition, not a filter over a wider read: the spend writer keyed the
        row under the caller's api_key, so a key can only ever see what it wrote itself.
        """
        record: Final = await self.table.find_first(
            where={"api_key": api_key, "session_id": session_id},  # mutable-ok: Prisma where filter must be a dict
            order={"last_turn_at": "desc"},  # mutable-ok: Prisma order clause must be a dict
        )
        return self._to_model(record)
