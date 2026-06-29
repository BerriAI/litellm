"""
Test that user max_budget is enforced for team-associated API keys.

Regression test for https://github.com/BerriAI/litellm/issues/27394
The user budget check in common_checks() was gated behind a condition that
skipped it when the key belonged to a team.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import common_checks


@pytest.fixture(autouse=True)
def set_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-1234")


def _make_user(user_id: str, max_budget: float, spend: float) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=user_id,
        max_budget=max_budget,
        spend=spend,
    )


def _make_team(team_id: str) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        models=[],
        max_budget=None,
    )


def _make_request():
    mock_request = MagicMock()
    mock_request.url = MagicMock()
    mock_request.url.path = "/chat/completions"
    return mock_request


def _common_patches():
    """Shared mocks for common_checks dependencies."""
    return [
        patch(
            "litellm.proxy.proxy_server.get_current_spend",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.proxy_server.prisma_client",
            MagicMock(),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "litellm.proxy.auth.auth_checks._tag_max_budget_check",
            new_callable=AsyncMock,
        ),
    ]


@pytest.mark.asyncio
async def test_user_budget_enforced_with_team_key():
    """User over budget should be rejected even when the key has a team_id."""
    user = _make_user(user_id="user-1", max_budget=30.0, spend=35.0)
    team = _make_team(team_id="team-1")
    token = UserAPIKeyAuth(token="sk-test", user_id="user-1", team_id="team-1")

    patches = _common_patches()
    for p in patches:
        p.start()

    # Set the spend return value
    from litellm.proxy.proxy_server import get_current_spend

    get_current_spend.return_value = 35.0

    try:
        with pytest.raises(litellm.BudgetExceededError):
            await common_checks(
                request_body={"model": "gpt-4o"},
                team_object=team,
                user_object=user,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=_make_request(),
            )
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_user_under_budget_with_team_key_passes():
    """User under budget with a team key should pass the check."""
    user = _make_user(user_id="user-2", max_budget=100.0, spend=50.0)
    team = _make_team(team_id="team-2")
    token = UserAPIKeyAuth(token="sk-test", user_id="user-2", team_id="team-2")

    patches = _common_patches()
    for p in patches:
        p.start()

    from litellm.proxy.proxy_server import get_current_spend

    get_current_spend.return_value = 50.0

    try:
        result = await common_checks(
            request_body={"model": "gpt-4o"},
            team_object=team,
            user_object=user,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/chat/completions",
            llm_router=None,
            proxy_logging_obj=MagicMock(),
            valid_token=token,
            request=_make_request(),
        )
        assert result is True
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_user_budget_still_enforced_without_team():
    """User over budget without a team should still be rejected (no regression)."""
    user = _make_user(user_id="user-3", max_budget=10.0, spend=15.0)
    token = UserAPIKeyAuth(token="sk-test", user_id="user-3")

    patches = _common_patches()
    for p in patches:
        p.start()

    from litellm.proxy.proxy_server import get_current_spend

    get_current_spend.return_value = 15.0

    try:
        with pytest.raises(litellm.BudgetExceededError):
            await common_checks(
                request_body={"model": "gpt-4o"},
                team_object=None,
                user_object=user,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/chat/completions",
                llm_router=None,
                proxy_logging_obj=MagicMock(),
                valid_token=token,
                request=_make_request(),
            )
    finally:
        for p in patches:
            p.stop()
