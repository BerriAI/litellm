import json
from functools import lru_cache
from pathlib import Path
from typing import Final

from litellm.types.model_insights import ModelInsightTask

_TASKS_FILE: Final = Path(__file__).resolve().parent.parent / "model_insights_tasks.json"


@lru_cache(maxsize=1)
def load_model_insight_tasks() -> dict[str, ModelInsightTask]:
    raw: Final = json.loads(_TASKS_FILE.read_text())
    return {name: ModelInsightTask(task_type=name, **entry) for name, entry in raw.items()}
