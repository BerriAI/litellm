"""
Tests for the ``aliases`` feature in the model cost map.

An entry's ``aliases`` list serves the entry under every alias name, all of them
one shared ``AliasedCostEntry`` object, when a document is installed.
"""

import json
from typing import Final

from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
from litellm.rust_bridge._native import NativeInstalled, NativeModelCostMap, NativeRejected


def _installed(document: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    installed: Final = NativeModelCostMap.bundled(0, 0.0).install(json.dumps(document).encode())
    assert isinstance(installed, NativeInstalled)
    return installed.rows


def _row(**fields: object) -> dict[str, object]:
    return {"litellm_provider": "test", "mode": "chat", **fields}


# ---------------------------------------------------------------------------
# Core expansion behaviour
# ---------------------------------------------------------------------------


class TestExpandModelAliases:
    def test_basic_expansion(self):
        """Aliases are added as top-level entries in model_cost."""
        result = _installed({"my-model-latest": _row(aliases=["my-model-20250101"], input_cost_per_token=1e-06)})

        assert result["my-model-20250101"]["input_cost_per_token"] == 1e-06
        assert result["my-model-20250101"]["litellm_provider"] == "test"

    def test_multiple_aliases(self):
        """A single entry can declare multiple aliases."""
        result = _installed({"provider/model-latest": _row(aliases=["provider/model-v1", "provider/model-v2"])})

        assert list(result) == ["provider/model-latest", "provider/model-v1", "provider/model-v2"]

    def test_shared_entry_object(self):
        """Alias entries are the same object as the canonical entry, which knows every name it is served under."""
        result = _installed({"canonical-model": _row(aliases=["alias-model"], input_cost_per_token=2e-06)})

        entry: Final = result["canonical-model"]
        assert result["alias-model"] is entry
        assert isinstance(entry, AliasedCostEntry)
        assert entry.names == ("canonical-model", "alias-model")
        assert entry.registered_fields == frozenset()

    def test_aliases_key_removed(self):
        """The ``aliases`` key is removed from the entry after expansion."""
        result = _installed({"my-model": _row(aliases=["my-model-alias"])})

        assert "aliases" not in result["my-model"]
        assert "aliases" not in result["my-model-alias"]

    def test_entries_without_aliases_stay_plain_dicts(self):
        """Entries with no ``aliases`` key are served as written, as plain dicts."""
        result = _installed({"plain-model": _row(input_cost_per_token=3e-06)})

        assert result == {"plain-model": _row(input_cost_per_token=3e-06)}
        assert type(result["plain-model"]) is dict

    def test_empty_aliases_list(self):
        """An empty ``aliases`` list is treated the same as no aliases."""
        result = _installed({"model-a": _row(aliases=[])})

        assert list(result) == ["model-a"]
        assert "aliases" not in result["model-a"]


# ---------------------------------------------------------------------------
# Conflict handling
# ---------------------------------------------------------------------------


class TestAliasConflicts:
    def test_alias_conflicts_with_canonical_entry(self):
        """Alias that matches an existing canonical entry is skipped."""
        result = _installed(
            {
                "model-latest": _row(aliases=["model-dated"], input_cost_per_token=1e-06),
                "model-dated": _row(input_cost_per_token=5e-06),
            }
        )

        assert result["model-dated"]["input_cost_per_token"] == 5e-06
        assert result["model-dated"] is not result["model-latest"]

    def test_an_alias_naming_an_earlier_entry_leaves_that_entry_alone(self):
        """An alias declared after the entry it names never overwrites that entry."""
        result = _installed(
            {
                "existing-model": _row(input_cost_per_token=9.99e-06),
                "other-model": _row(aliases=["existing-model"], input_cost_per_token=1e-06),
            }
        )

        assert result["existing-model"] == _row(input_cost_per_token=9.99e-06)
        assert result["existing-model"] is not result["other-model"]

    def test_duplicate_alias_across_entries(self):
        """Same alias claimed by two different entries: the first one keeps it."""
        result = _installed(
            {
                "model-a": _row(aliases=["shared-alias"], input_cost_per_token=1e-06),
                "model-b": _row(aliases=["shared-alias"], input_cost_per_token=2e-06),
            }
        )

        assert result["shared-alias"] is result["model-a"]
        assert result["shared-alias"]["input_cost_per_token"] == 1e-06


# ---------------------------------------------------------------------------
# Integration with model_cost dict mutation
# ---------------------------------------------------------------------------


class TestAliasIntegration:
    def test_mutation_through_alias_visible_on_canonical(self):
        """Since alias is a shared reference, mutations are visible on both."""
        result = _installed({"canonical": _row(aliases=["alias"], input_cost_per_token=1e-06)})

        result["alias"]["input_cost_per_token"] = 999
        assert result["canonical"]["input_cost_per_token"] == 999

    def test_mixed_entries_with_and_without_aliases(self):
        """Aliases follow every document key, grouped by owner."""
        result = _installed(
            {
                "model-with-alias": _row(aliases=["alias-1", "alias-2"]),
                "plain-model": _row(),
            }
        )

        assert list(result) == ["model-with-alias", "plain-model", "alias-1", "alias-2"]

    def test_an_empty_document_is_rejected(self):
        """An empty document is never installed."""
        assert isinstance(NativeModelCostMap.bundled(0, 0.0).install(b"{}"), NativeRejected)
