from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Final

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.proxy.common_utils.db_model_utils import strip_env_refs_for_team_model
from litellm.types.router import ModelInfo


@pytest.mark.parametrize("metadata", ({"team_id": "team-1"}, ModelInfo(team_id="team-1")))
def test_scoped_parameters_copy_nested_settings_without_changing_input(
    metadata: Mapping[str, object] | ModelInfo,
) -> None:
    params: Final = {
        "model": "openai/gpt-4o-mini",
        "options": [{"setting": "os.environ/SETTING"}, ["os.environ/SETTING", None, 3, False]],
    }
    result: Final = strip_env_refs_for_team_model(params, metadata)

    assert result == {"model": "openai/gpt-4o-mini", "options": [{"setting": ""}, ["", None, 3, False]]}
    assert params == {
        "model": "openai/gpt-4o-mini",
        "options": [{"setting": "os.environ/SETTING"}, ["os.environ/SETTING", None, 3, False]],
    }


@pytest.mark.parametrize("metadata", (None, {}, {"team_id": None}, ModelInfo()))
def test_unscoped_parameters_keep_their_original_values(metadata: Mapping[str, object] | ModelInfo | None) -> None:
    params: Final = {"setting": "os.environ/SETTING", "options": ["os.environ/SETTING"]}
    assert strip_env_refs_for_team_model(params, metadata) is params


@pytest.mark.parametrize("params", ({}, {"setting": "literal", "options": [False, None, 2, {"name": "value"}]}))
def test_empty_and_literal_scoped_parameters_keep_their_values(params: Mapping[str, object]) -> None:
    assert strip_env_refs_for_team_model(params, {"team_id": "team-1"}) == params


def test_nested_lists_do_not_depend_on_python_recursion_depth() -> None:
    def wrap(value: object, _: int) -> object:
        return [value]

    nested: Final = reduce(wrap, range(1100), "os.environ/SETTING")
    result: Final = strip_env_refs_for_team_model({"options": nested}, {"team_id": "team-1"})
    sequence: Final = TypeAdapter(Sequence[object])

    def first(value: object, _: int) -> object:
        return sequence.validate_python(value)[0]

    assert reduce(first, range(1100), result["options"]) == ""


def test_invalid_mapping_keys_are_not_silently_coerced() -> None:
    with pytest.raises(ValidationError):
        strip_env_refs_for_team_model({"options": {1: "value"}}, {"team_id": "team-1"})
