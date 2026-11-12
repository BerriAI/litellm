from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest

from litellm.integrations.email_alerting import get_all_team_member_emails
from litellm.proxy import proxy_server


@pytest.mark.asyncio
async def test_team_member_emails_are_the_set_emails_of_the_members_with_a_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team: Final = SimpleNamespace(
        members_with_roles=[{"user_id": "u-1", "role": "admin"}, {"role": "user"}, {"user_id": "u-2", "role": "user"}]
    )
    query_raw: Final = AsyncMock(
        return_value=[{"user_email": "a@example.com"}, {"user_email": None}, {}, {"user_email": "b@example.com"}]
    )
    prisma_client: Final = SimpleNamespace(
        db=SimpleNamespace(
            litellm_teamtable=SimpleNamespace(find_unique=AsyncMock(return_value=team)),
            query_raw=query_raw,
        )
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_client)

    emails: Final = await get_all_team_member_emails("team-1")

    assert emails == ["a@example.com", "b@example.com"]
    assert query_raw.await_args is not None
    assert query_raw.await_args.args[1] == ["u-1", "u-2"]
