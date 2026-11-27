"""The rows a key's auth checks read about its owners, read in one round trip for the auth prefetch."""

from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.repositories import rust_database
from litellm.rust_bridge._native import RustKeyOwnersRepository

_Rows: Final = TypeAdapter(dict[str, object])


class KeyOwnersRepository:
    async def find(
        self,
        *,
        user_id: str | None,
        team_id: str | None,
        membership_user_id: str | None,
        organization_id: str | None,
        project_id: str | None,
    ) -> Mapping[str, object]:
        """``user_row``, ``team_row``, ``membership_row``, ``organization_row`` and ``project_row`` as JSON objects
        with the budget, object permission and model alias rows auth reads joined in, each None when its id is None
        or no row matches. The membership is ``membership_user_id``'s in ``team_id``."""
        return _Rows.validate_json(
            await rust_database.required_repository(RustKeyOwnersRepository).find(
                user_id=user_id,
                team_id=team_id,
                membership_user_id=membership_user_id,
                organization_id=organization_id,
                project_id=project_id,
            )
        )
