from collections.abc import Sequence
from typing import Final

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.proxy.common_utils.db_model_utils import strip_env_refs_for_team_model
from litellm.types.router import ModelInfo


@pytest.mark.parametrize("metadata", ({"team_id": "team-1"}, ModelInfo(team_id="team-1")))
def test_scoped_parameters_copy_nested_settings_without_changing_input(metadata):
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
def test_unscoped_parameters_keep_their_original_values(metadata):
    params: Final = {"setting": "os.environ/SETTING", "options": ["os.environ/SETTING"]}
    assert strip_env_refs_for_team_model(params, metadata) is params


@pytest.mark.parametrize("params", ({}, {"setting": "literal", "options": [False, None, 2, {"name": "value"}]}))
def test_empty_and_literal_scoped_parameters_keep_their_values(params):
    assert strip_env_refs_for_team_model(params, {"team_id": "team-1"}) == params


def test_nested_lists_do_not_depend_on_python_recursion_depth():
    nested: Final = list[object](("os.environ/SETTING",))
    for _ in range(1100):
        nested[0] = [nested[0]]
    result: Final = strip_env_refs_for_team_model({"options": nested[0]}, {"team_id": "team-1"})
    sequence: Final = TypeAdapter(Sequence[object])
    current: Final = [result["options"]]
    for _ in range(1100):
        current[0] = sequence.validate_python(current[0])[0]
    assert current[0] == ""


def test_invalid_mapping_keys_are_not_silently_coerced():
    with pytest.raises(ValidationError):
        strip_env_refs_for_team_model({"options": {1: "value"}}, {"team_id": "team-1"})
