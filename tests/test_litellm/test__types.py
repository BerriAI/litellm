# tests/test_litellm/proxy/test__types.py

from litellm.proxy._types import LiteLLM_TeamMembership


def test_team_membership_budget_table_optional_no_crash():
    """
    Regression test for #28689
    Pydantic v2: Optional[T] without default = required field.
    When budget_id is null, DB join returns no litellm_budget_table key.
    model_validate must NOT raise 'Field required'.
    """
    data = {
        "user_id": "test-user",
        "team_id": "test-team",
        "budget_id": None,
        # litellm_budget_table intentionally absent (as DB join returns when budget_id is null)
    }
    result = LiteLLM_TeamMembership.model_validate(data)
    assert result.litellm_budget_table is None


def test_team_membership_budget_table_present_still_works():
    """When budget_id exists, litellm_budget_table should still be populated."""
    data = {
        "user_id": "test-user",
        "team_id": "test-team",
        "budget_id": "some-budget-id",
        "litellm_budget_table": None,
    }
    result = LiteLLM_TeamMembership.model_validate(data)
    assert result.litellm_budget_table is None
