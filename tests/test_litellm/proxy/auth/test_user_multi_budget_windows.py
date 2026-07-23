"""
Unit tests for multi-budget-window enforcement on users.
"""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.models.team import BudgetLimitEntry
from litellm.models.user import LiteLLM_UserTable


def _make_user(**kwargs) -> LiteLLM_UserTable:
    defaults = dict(
        user_id="test-user-123",
        spend=0.0,
        max_budget=None,
        budget_limits=[],
    )
    defaults.update(kwargs)
    return LiteLLM_UserTable(**defaults)


def test_user_model_accepts_budget_limits():
    """LiteLLM_UserTable must accept a budget_limits field."""
    user = _make_user(
        budget_limits=[
            {"budget_duration": "1hr", "max_budget": 5.0, "reset_at": None},
            {"budget_duration": "1d", "max_budget": 50.0, "reset_at": None},
        ]
    )
    assert user.budget_limits is not None
    assert len(user.budget_limits) == 2
