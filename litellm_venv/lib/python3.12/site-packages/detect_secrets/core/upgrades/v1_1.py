from typing import Any
from typing import Dict


def upgrade(baseline: Dict[str, Any]) -> None:
    for function in [
        _add_new_default_filters,
    ]:
        function(baseline)


def _add_new_default_filters(baseline: Dict[str, Any]) -> None:
    baseline["filters_used"].extend(
        [
            {
                "path": "detect_secrets.filters.heuristic.is_lock_file",
            },
            {
                "path": "detect_secrets.filters.heuristic.is_not_alphanumeric_string",
            },
            {
                "path": "detect_secrets.filters.heuristic.is_swagger_file",
            },
        ]
    )
