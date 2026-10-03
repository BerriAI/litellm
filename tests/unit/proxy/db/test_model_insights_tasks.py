from litellm.proxy.db.model_insights_tasks import load_model_insight_tasks


def test_every_task_has_a_label_category_and_description() -> None:
    tasks = load_model_insight_tasks()

    assert tasks
    for name, task in tasks.items():
        assert task.task_type == name
        assert task.label
        assert task.category in {"General", "Agent", "Code", "Data"}
        assert task.description.endswith(".")
