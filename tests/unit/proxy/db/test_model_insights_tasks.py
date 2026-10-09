from litellm.proxy.db.model_insights_tasks import load_model_insight_tasks
from litellm.proxy.db.model_usage_rollup import model_usage_task_type


def test_every_task_has_a_label_and_a_category() -> None:
    tasks = load_model_insight_tasks()

    assert tasks
    for name, task in tasks.items():
        assert task.task_type == name
        assert task.label
        assert task.category in {"General", "Agent", "Code", "Data"}


def test_tasks_in_the_json_file_are_the_ones_the_rollup_accepts() -> None:
    for name in load_model_insight_tasks():
        assert model_usage_task_type(f'["task:{name}"]') == name

    assert model_usage_task_type('["task:not_in_the_file"]') == "uncategorized"
