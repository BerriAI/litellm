from typing import Any
from typing import Dict


def upgrade(baseline: Dict[str, Any]) -> None:
    if "exclude_regex" in baseline:
        baseline["exclude"] = {
            "files": baseline.pop("exclude_regex"),
            "lines": None,
        }

    baseline["word_list"] = {
        "file": None,
        "hash": None,
    }
