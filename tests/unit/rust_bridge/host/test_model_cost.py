from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.model_cost_table import AliasedCostEntry
from litellm.rust_bridge.host.model_cost import kept, merged, resolve

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def test_resolve_lands_on_the_row_a_lookup_in_the_published_table_resolves() -> None:
    landing: Final = resolve(litellm.model_cost, "openai/gpt-4o", {"input_cost_per_token": 1.0}, None)

    assert landing is not None
    assert (landing.key, landing.matched_builtin) == ("gpt-4o", True)


def test_merged_keeps_the_alias_names_and_adds_only_the_fields_a_registration_set() -> None:
    current: Final = AliasedCostEntry({"mode": "chat", "x": 1}, ("owner", "alias"), frozenset({"mode"}))

    row: Final = merged(current, {"x": 2}, {"litellm_provider": "openai", "output_cost_per_token": None})

    assert row == {"mode": "chat", "x": 2, "litellm_provider": "openai"}
    assert (row.names, row.registered_fields) == (("owner", "alias"), frozenset({"mode", "litellm_provider"}))


def test_kept_registration_drops_the_flat_video_price_a_later_per_second_price_replaced() -> None:
    replayed: Final = kept({"mode": "video_generation", "output_cost_per_video": 0.37}, {"output_cost_per_second": 0.1})

    assert replayed == {"mode": "video_generation", "output_cost_per_second": 0.1}


def test_kept_registration_keeps_a_flat_video_price_registered_beside_a_per_second_price() -> None:
    replayed: Final = kept(
        {"output_cost_per_video": 0.37}, {"output_cost_per_video": 0.4, "output_cost_per_second": 0.1}
    )

    assert replayed == {"output_cost_per_video": 0.4, "output_cost_per_second": 0.1}
