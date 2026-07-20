"""
PTU Reservation repository for database operations on LiteLLM_PTUReservation.
"""

from datetime import datetime
from typing import Any

from litellm.models.ptu_reservation import LiteLLM_PTUReservationFull
from litellm.repositories.base_repository import BaseRepository


class PTUReservationRepository(BaseRepository[LiteLLM_PTUReservationFull]):
    """Repository for PTU reservation database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_ptureservation

    @property
    def model_class(self) -> type[LiteLLM_PTUReservationFull]:
        return LiteLLM_PTUReservationFull

    async def find_active(
        self,
        as_of: datetime,
        team_id: str | None = None,
        model: str | None = None,
    ) -> list[Any]:
        """Return reservations live at ``as_of``."""
        where: dict = {
            "effective_from": {"lte": as_of},
            "OR": [
                {"effective_to": None},
                {"effective_to": {"gt": as_of}},
            ],
        }
        if team_id is not None:
            where["team_id"] = team_id
        if model is not None:
            where["model"] = model
        return await self.table.find_many(where=where)

    async def find_overlapping(
        self,
        team_id: str,
        model: str,
        effective_from: datetime,
        effective_to: datetime | None,
    ) -> list[Any]:
        """Return existing reservations overlapping the proposed window for the same (team, model)."""
        where: dict = {
            "team_id": team_id,
            "model": model,
        }
        if effective_to is None:
            where["OR"] = [
                {"effective_to": None},
                {"effective_to": {"gt": effective_from}},
            ]
        else:
            where["effective_from"] = {"lt": effective_to}
            where["OR"] = [
                {"effective_to": None},
                {"effective_to": {"gt": effective_from}},
            ]
        return await self.table.find_many(where=where)
