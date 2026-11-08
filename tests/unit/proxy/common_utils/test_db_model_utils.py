from collections.abc import Mapping
from functools import reduce
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.common_utils.db_model_utils import (
    decrypt_db_model_params,
    reencrypt_db_model_params,
    team_model_params_are_allowed,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with, encrypt_value_helper
from litellm.types.router import ModelInfo


@pytest.mark.parametrize("metadata", ({"team_id": "team-1"}, ModelInfo(team_id="team-1")))
def test_scoped_parameters_reject_nested_settings_without_changing_input(
    metadata: Mapping[str, object] | ModelInfo,
) -> None:
    params: Final = {
        "model": "openai/gpt-4o-mini",
        "options": [{"setting": "os.environ/SETTING"}, ["os.environ/SETTING", None, 3, False]],
    }
    result: Final = team_model_params_are_allowed(params, metadata)

    assert result is False
    assert params == {
        "model": "openai/gpt-4o-mini",
        "options": [{"setting": "os.environ/SETTING"}, ["os.environ/SETTING", None, 3, False]],
    }


@pytest.mark.parametrize("metadata", (None, {}, {"team_id": None}, ModelInfo()))
def test_unscoped_parameters_keep_their_original_values(metadata: Mapping[str, object] | ModelInfo | None) -> None:
    params: Final = {"setting": "os.environ/SETTING", "options": ["os.environ/SETTING"]}
    assert team_model_params_are_allowed(params, metadata) is True


@pytest.mark.parametrize("params", ({}, {"setting": "literal", "options": [False, None, 2, {"name": "value"}]}))
def test_empty_and_literal_scoped_parameters_keep_their_values(params: Mapping[str, object]) -> None:
    assert team_model_params_are_allowed(params, {"team_id": "team-1"}) is True


def test_nested_lists_do_not_depend_on_python_recursion_depth() -> None:
    def wrap(value: object, _: int) -> object:
        return [value]

    nested: Final = reduce(wrap, range(1100), "os.environ/SETTING")
    assert team_model_params_are_allowed({"options": nested}, {"team_id": "team-1"}) is False


def test_invalid_mapping_keys_are_not_silently_coerced() -> None:
    with pytest.raises(ValidationError):
        team_model_params_are_allowed({"options": {1: "value"}}, {"team_id": "team-1"})


@pytest.mark.parametrize(
    "params",
    (
        {},
        {
            "model": "openai/gpt-4o-mini",
            "api_key": "os.environ/ROTATION_SETTING",
            "api_base": "literal",
            "options": [{"setting": "os.environ/ROTATION_SETTING", "empty": None}],
            "custom_null": None,
            "enabled": False,
        },
    ),
)
def test_rotation_preserves_complete_stored_parameters(
    monkeypatch: pytest.MonkeyPatch, params: Mapping[str, object]
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "old-model-settings-key")
    monkeypatch.setenv("ROTATION_SETTING", "configured-value")
    stored: Final = {
        key: encrypt_value_helper(value=value) if isinstance(value, str) else value for key, value in params.items()
    }
    assert decrypt_db_model_params(stored) == params
    rotated: Final = reencrypt_db_model_params(stored, "new-model-settings-key")
    decoded: Final = {
        key: decrypt_if_encrypted_with(value, "new-model-settings-key") if isinstance(value, str) else value
        for key, value in rotated.items()
    }
    assert decoded == params
    assert decrypt_db_model_params(stored) == params
