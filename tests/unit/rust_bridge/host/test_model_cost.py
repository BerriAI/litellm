from typing import Final

import pytest

import litellm
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_model_cost_map import model_cost_inputs
from litellm.rust_bridge.host.model_cost import (
    changed_in_place,
    entry,
    merged,
    publish,
    resolve,
    write,
)

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def test_resolve_lands_on_the_row_a_lookup_in_the_published_table_resolves() -> None:
    landing: Final = resolve(litellm.model_cost, "openai/gpt-4o", {"input_cost_per_token": 1.0}, None)

    assert landing is not None
    assert (landing.key, landing.matched_builtin) == ("gpt-4o", True)


def test_merged_keeps_the_alias_names_and_adds_only_the_fields_a_registration_set() -> None:
    current: Final = entry({"mode": "chat", "x": 1}, ("owner", "alias"), frozenset({"mode"}))

    row: Final = merged(current, {"x": 2}, {"litellm_provider": "openai", "output_cost_per_token": None})

    assert row == {"mode": "chat", "x": 2, "litellm_provider": "openai"}
    assert (row.names, row.registered_fields) == (("owner", "alias"), frozenset({"mode", "litellm_provider"}))


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
