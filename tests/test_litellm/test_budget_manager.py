import pytest

from litellm.budget_manager import BudgetManager


@pytest.fixture()
def manager(tmp_path, monkeypatch) -> BudgetManager:
    # BudgetManager persists to ./user_cost.json via a background thread;
    # isolate cwd so the suite never litters the repo or races parallel workers.
    monkeypatch.chdir(tmp_path)
    bm = BudgetManager(project_name="test", client_type="local")
    bm.create_budget(total_budget=10, user="u", duration="daily")
    return bm


def test_projected_cost_string_content(manager: BudgetManager):
    cost = manager.projected_cost(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        user="u",
    )
    assert cost > 0


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
    assert cost > 0


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


def test_projected_cost_tool_calls_with_null_content(manager: BudgetManager):
    # The expensive shape: assistant turn carrying tool calls and no text.
    cost = manager.projected_cost(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "what is the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
        ],
        user="u",
    )
    assert cost > 0


def test_projected_cost_provider_specific_fields(manager: BudgetManager):
    # Provider extras (name, cache_control, reasoning_content) must not break counting.
    cost = manager.projected_cost(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "hi", "name": "soroush"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
                ],
                "reasoning_content": "thinking...",
            },
        ],
        user="u",
    )
    assert cost > 0
