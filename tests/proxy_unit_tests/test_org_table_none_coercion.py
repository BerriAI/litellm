"""
Tests for _coerce_none_to_empty_list field validator on LiteLLM_OrganizationTableWithMembers.

Validates that None values for members and teams are coerced to [] (Prisma returns None
for empty relations), while populated and empty lists are preserved.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import LiteLLM_OrganizationTableWithMembers


def _base_org_data():
    """Minimal required fields for LiteLLM_OrganizationTableWithMembers."""
    now = datetime(2025, 1, 1, 12, 0, 0)
    return {
        "budget_id": "budget-1",
        "models": ["gpt-4"],
        "created_by": "user-1",
        "updated_by": "user-1",
        "created_at": now,
        "updated_at": now,
    }


def test_teams_none_becomes_empty_list():
    """teams=None -> becomes []"""
    data = {**_base_org_data(), "members": [], "teams": None}
    obj = LiteLLM_OrganizationTableWithMembers.model_validate(data)
    assert obj.teams == []
    assert obj.members == []


def test_members_none_becomes_empty_list():
    """members=None -> becomes []"""
    data = {**_base_org_data(), "members": None, "teams": []}
    obj = LiteLLM_OrganizationTableWithMembers.model_validate(data)
    assert obj.members == []
    assert obj.teams == []


def test_both_none_become_empty_lists():
    """Both None -> both become []"""
    data = {**_base_org_data(), "members": None, "teams": None}
    obj = LiteLLM_OrganizationTableWithMembers.model_validate(data)
    assert obj.members == []
    assert obj.teams == []


def test_populated_lists_preserved():
    """Populated lists -> preserved"""
    now = datetime(2025, 1, 1, 12, 0, 0)
    member_data = {
        "user_id": "user-1",
        "organization_id": "org-1",
        "created_at": now,
        "updated_at": now,
    }
    team_data = {"team_id": "team-1"}
    data = {
        **_base_org_data(),
        "members": [member_data],
        "teams": [team_data],
    }
    obj = LiteLLM_OrganizationTableWithMembers.model_validate(data)
    assert len(obj.members) == 1
    assert obj.members[0].user_id == "user-1"
    assert len(obj.teams) == 1
    assert obj.teams[0].team_id == "team-1"


def test_empty_lists_preserved():
    """Empty lists -> preserved"""
    data = {**_base_org_data(), "members": [], "teams": []}
    obj = LiteLLM_OrganizationTableWithMembers.model_validate(data)
    assert obj.members == []
    assert obj.teams == []
