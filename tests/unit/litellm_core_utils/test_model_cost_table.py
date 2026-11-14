from typing import Final

import pytest

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_model_cost_map import model_cost_inputs
from litellm.litellm_core_utils.model_cost_table import changed_in_place, publish, write

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


_MODEL: Final = "lit-table-model"


def _row(price: float) -> dict[str, object]:
    return {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": price, "output_cost_per_token": price}


def _billed() -> float:
    prompt, completion = cost_per_token(
        model=_MODEL, custom_llm_provider="openai", prompt_tokens=1, completion_tokens=0
    )
    return prompt + completion


def test_a_published_table_is_read_in_place_of_what_was_cached_for_the_one_before() -> None:
    publish({**litellm.model_cost, _MODEL: _row(1.0)})
    assert litellm.get_model_info(_MODEL)["input_cost_per_token"] == 1.0
    assert _billed() == 1.0

    publish({**litellm.model_cost, _MODEL: _row(2.0)})

    assert litellm.get_model_info(_MODEL)["input_cost_per_token"] == 2.0
    assert _billed() == 2.0


def test_rows_written_into_the_live_table_replace_a_cached_value_of_the_same_size() -> None:
    publish({**litellm.model_cost, _MODEL: _row(1.0)})
    assert _billed() == 1.0

    write(litellm.model_cost, {_MODEL: _row(3.0)})

    assert litellm.get_model_info(_MODEL)["input_cost_per_token"] == 3.0
    assert _billed() == 3.0


def test_a_provider_changed_in_place_regroups_the_key_once_the_change_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "gpt-4o" not in model_cost_inputs.keys_in(litellm.model_cost, "lit-moved")

    monkeypatch.setitem(litellm.model_cost["gpt-4o"], "litellm_provider", "lit-moved")
    changed_in_place()

    assert model_cost_inputs.keys_in(litellm.model_cost, "lit-moved") == ["gpt-4o"]
    assert "gpt-4o" not in model_cost_inputs.keys_in(litellm.model_cost, "openai")
