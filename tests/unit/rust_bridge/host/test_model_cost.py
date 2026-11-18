from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.model_cost_table import AliasedCostEntry, write
from litellm.rust_bridge.host.model_cost import generation, kept, merged, published, resolve

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


def test_generation_moves_with_a_table_write_and_keeps_the_published_table() -> None:
    before: Final = generation()

    write(published(), {"lit-generation-probe": {"input_cost_per_token": 1.0}})
    after: Final = generation()

    assert after[0] is published()
    assert (after[3], after[2]) == (before[3] + 1, before[2] + 1)


@pytest.mark.parametrize("config", ["cost_discount_config", "cost_margin_config"])
def test_generation_moves_when_a_cost_config_is_replaced(monkeypatch: pytest.MonkeyPatch, config: str) -> None:
    before: Final = generation()

    monkeypatch.setattr(litellm, config, {"anthropic": 0.1})
    replaced: Final = generation()

    assert replaced[1:4] == before[1:4]
    assert replaced[4:] != before[4:]


def test_generation_stays_when_a_cost_config_is_edited_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    before: Final = generation()

    monkeypatch.setitem(litellm.cost_discount_config, "anthropic", 0.1)

    assert generation() == before


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
