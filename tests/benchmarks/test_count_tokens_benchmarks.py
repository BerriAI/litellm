from typing import Final

import pytest
from pydantic import JsonValue

from litellm.llms.anthropic.count_tokens.transformation import without_dropped_paths

_LARGE_SCHEMA_OPTIONS: Final[dict[str, JsonValue]] = {
    "output_config": {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {f"field_{index}": {"type": "string", "description": "d" * 8} for index in range(3000)},
            },
        }
    },
    "thinking": {"type": "enabled", "budget_tokens": 2048},
}
_MANY_DROP_PATHS: Final = (*(f"output_config.missing_{index}" for index in range(1000)), "thinking")


@pytest.mark.benchmark
def test_without_dropped_paths_many_client_paths_over_a_large_schema() -> None:
    result: Final = without_dropped_paths(_LARGE_SCHEMA_OPTIONS, _MANY_DROP_PATHS)

    assert result == {"output_config": _LARGE_SCHEMA_OPTIONS["output_config"]}


@pytest.mark.benchmark
def test_without_dropped_paths_one_client_path_of_unclosed_brackets() -> None:
    options: Final[dict[str, JsonValue]] = {"thinking": {"type": "enabled", "budget_tokens": 2048}}

    assert without_dropped_paths(options, ("[" * 16_000,)) == options
