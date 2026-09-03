import pytest

from litellm.budget_manager import BudgetManager


@pytest.fixture()
def manager() -> BudgetManager:
    bm = BudgetManager(project_name="test", client_type="local")
    bm.create_budget(total_budget=10, user="u", duration="daily")
    return bm


def test_projected_cost_string_content(manager: BudgetManager):
    cost = manager.projected_cost(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        user="u",
    )
    assert cost >= 0


def test_projected_cost_vision_content(manager: BudgetManager):
    cost = manager.projected_cost(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            }
        ],
        user="u",
    )
    assert cost >= 0


def test_projected_cost_none_and_missing_content(manager: BudgetManager):
    assert (
        manager.projected_cost(
            model="gpt-4o-mini",
            messages=[{"role": "assistant", "content": None}],
            user="u",
        )
        >= 0
    )
    assert manager.projected_cost(model="gpt-4o-mini", messages=[{"role": "user"}], user="u") >= 0
