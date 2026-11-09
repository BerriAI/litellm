from collections.abc import Mapping
from typing import Final
from unittest.mock import Mock, call

import pytest

import litellm
from litellm._auth import model_budget_aliases as native
from litellm._auth.model_budget_data import ModelBudgetSnapshot
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
from litellm.proxy.auth.model_budget_adapter import ModelBudgetIO, read_alias_models
from litellm.proxy.hooks.model_budget_shared_windows import SharedWindows

from .model_alias_reference import expand


@pytest.mark.parametrize("model", ("latest", "openai/latest", "unknown/latest", "bedrock/latest"))
@pytest.mark.parametrize("names", ((), ("latest",), ("openai/latest",), ("target",), ("",), ("openai/target",)))
@pytest.mark.parametrize("provider", (None, "openai", "unknown", "bedrock_converse", 42))
def test_alias_lookup_matches_frozen_main(model: str, names: tuple[str, ...], provider: object) -> None:
    row: Final = AliasedCostEntry({"litellm_provider": provider}, names)
    catalog: Final[Mapping[str, object]] = {model: row, "latest": row}
    expected: Final = expand(model, catalog, litellm.provider_list)
    lookup: Final = Mock(side_effect=catalog.get)
    io: Final = ModelBudgetIO(
        DualCache(), lambda _model, _team: frozenset(), Mock(spec=SharedWindows), 0, price_entry=lookup
    )
    assert read_alias_models(model, io) == expected
    assert lookup.call_args_list[0].args == (model,)


@pytest.mark.parametrize("raw", (None, {}, {"alias_of": "target"}, {"names": ["target"]}))
def test_plain_catalog_rows_do_not_become_aliases(raw: object) -> None:
    io: Final = ModelBudgetIO(
        DualCache(), lambda _model, _team: frozenset(), Mock(spec=SharedWindows), 0, price_entry=lambda _: raw
    )
    assert read_alias_models("latest", io) == ("latest",)


@pytest.mark.parametrize("direct", (True, False))
def test_catalog_reads_bare_name_only_when_qualified_alias_misses(direct: bool) -> None:
    row: Final = AliasedCostEntry({"litellm_provider": "openai"}, ("target", "latest"))
    catalog: Final = {"openai/latest": row if direct else None, "latest": row}
    lookup: Final = Mock(side_effect=catalog.get)
    io: Final = ModelBudgetIO(
        DualCache(), lambda _model, _team: frozenset(), Mock(spec=SharedWindows), 0, price_entry=lookup
    )
    assert read_alias_models("openai/latest", io) == ("openai/latest", "openai/target")
    assert lookup.call_args_list == ([call("openai/latest")] if direct else [call("openai/latest"), call("latest")])


@pytest.mark.parametrize(
    ("snapshots", "expected"),
    (
        ((), native.ScopeDecision(0, None)),
        ((ModelBudgetSnapshot(1, 0),), native.ScopeDecision(1, None)),
        ((ModelBudgetSnapshot(1, 1),), native.ScopeDecision(None, 0)),
        ((ModelBudgetSnapshot(1, 0), ModelBudgetSnapshot(2, 2)), native.ScopeDecision(None, 1)),
        ((ModelBudgetSnapshot(1, 0), ModelBudgetSnapshot(2, 0)), native.ScopeDecision(None, 1)),
    ),
)
def test_scope_reads_only_until_first_denial(
    snapshots: tuple[ModelBudgetSnapshot, ...], expected: native.ScopeDecision
) -> None:
    assert native.scope(native.ScopeFacts(("alias", "target"), snapshots)) == expected
