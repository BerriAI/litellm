from litellm import BudgetManager
import litellm.budget_manager as budget_manager_module


def test_create_budget_uses_current_time(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manager = BudgetManager(project_name="test", client_type="local")
    monkeypatch.setattr(manager, "_save_data_thread", lambda: None)
    times = iter([1000.0, 1001.0])
    monkeypatch.setattr(budget_manager_module.time, "time", lambda: next(times))

    budget1 = manager.create_budget(100.0, "user1", duration="daily")
    budget2 = manager.create_budget(100.0, "user2", duration="daily")

    assert budget1["created_at"] == 1000.0
    assert budget2["created_at"] == 1001.0
