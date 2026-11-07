from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
from litellm.rust_bridge.host.model_cost import entry, write

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def test_entry_is_the_row_type_alias_names_share() -> None:
    row: Final = entry({"mode": "chat"}, ("owner", "alias"), frozenset({"mode"}))

    assert isinstance(row, AliasedCostEntry)
    assert row == {"mode": "chat"}
    assert (row.names, row.registered_fields) == (("owner", "alias"), frozenset({"mode"}))


def test_write_lands_on_the_row_a_lookup_resolves_and_writes_only_the_dict_it_is_handed() -> None:
    live: Final = dict(litellm.model_cost)
    rows: Final = dict(litellm.model_cost)

    landed: Final = write(rows, "openai/gpt-4o", {"input_cost_per_token": 1.0}, None, None)

    assert landed == "gpt-4o"
    assert rows["gpt-4o"]["input_cost_per_token"] == 1.0
    assert isinstance(rows["gpt-4o"], AliasedCostEntry)
    assert rows["gpt-4o"].registered_fields == frozenset({"input_cost_per_token"})
    assert litellm.model_cost == live


def test_write_skips_none_values_in_the_registered_fields() -> None:
    rows: Final = {}

    landed: Final = write(
        rows, "lit-host-model", {"litellm_provider": "openai", "output_cost_per_token": None}, "openai", None
    )

    assert rows[landed].registered_fields == frozenset({"litellm_provider"})
