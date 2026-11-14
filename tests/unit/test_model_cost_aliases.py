"""
Tests for the ``aliases`` feature in the model cost map.

An entry's ``aliases`` list serves the entry under every alias name, all of them
one shared ``AliasedCostEntry`` object, when a document is installed.
"""

import json
import logging
from typing import Final

import pytest

from litellm.rust_bridge._native import NativeInstalled, NativeModelCostMap, NativeRejected
from litellm.litellm_core_utils.model_cost_table import AliasedCostEntry


def _install(document: dict[str, dict[str, object]]) -> NativeInstalled:
    installed: Final = NativeModelCostMap.bundled(0, 0.0).install(json.dumps(document).encode())
    assert isinstance(installed, NativeInstalled)
    return installed


def _installed(document: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return _install(document).rows


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


@pytest.mark.parametrize(
    ("document", "name", "row", "other", "shared"),
    [
        pytest.param(
            {
                "model-latest": _row(aliases=["model-dated"], input_cost_per_token=1e-06),
                "model-dated": _row(input_cost_per_token=5e-06),
            },
            "model-dated",
            _row(input_cost_per_token=5e-06),
            "model-latest",
            False,
            id="alias-names-a-later-entry",
        ),
        pytest.param(
            {
                "existing-model": _row(input_cost_per_token=9.99e-06),
                "other-model": _row(aliases=["existing-model"], input_cost_per_token=1e-06),
            },
            "existing-model",
            _row(input_cost_per_token=9.99e-06),
            "other-model",
            False,
            id="alias-names-an-earlier-entry",
        ),
        pytest.param(
            {
                "model-a": _row(aliases=["shared-alias"], input_cost_per_token=1e-06),
                "model-b": _row(aliases=["shared-alias"], input_cost_per_token=2e-06),
            },
            "shared-alias",
            _row(input_cost_per_token=1e-06),
            "model-a",
            True,
            id="two-entries-claim-one-alias",
        ),
    ],
)
def test_an_alias_never_replaces_a_model_name_and_the_first_claim_wins(
    document: dict[str, dict[str, object]], name: str, row: dict[str, object], other: str, shared: bool
):
    result: Final = _installed(document)

    assert result[name] == row
    assert (result[name] is result[other]) is shared


class TestAliasOfRows:
    def test_an_alias_of_row_is_served_as_its_target_where_the_document_put_it(self):
        """An ``alias_of`` row is its target's object, even when written before the target."""
        result = _installed(
            {
                "model-latest": {"alias_of": "model"},
                "model": _row(input_cost_per_token=1e-06),
            }
        )

        assert list(result) == ["model-latest", "model"]
        assert result["model-latest"] is result["model"]
        assert result["model"] == _row(input_cost_per_token=1e-06)

    def test_an_alias_of_row_whose_target_is_not_a_model_is_dropped(self):
        result = _installed({"model": _row(), "model-latest": {"alias_of": "ghost"}})

        assert list(result) == ["model"]

    def test_an_alias_of_row_that_carries_other_fields_is_served_as_its_target(self):
        result = _installed(
            {
                "model": _row(input_cost_per_token=1e-06),
                "model-latest": {"alias_of": "model", "input_cost_per_token": 9e-06},
            }
        )

        assert result["model-latest"] is result["model"]
        assert result["model-latest"]["input_cost_per_token"] == 1e-06

    def test_the_model_count_is_every_key_the_document_serves(self):
        installed: Final = _install(
            {
                "model": _row(aliases=["model-dated"]),
                "model-latest": {"alias_of": "model"},
                "ghost-latest": {"alias_of": "ghost"},
                "sample_spec": {"mode": "chat"},
            }
        )

        assert list(installed.rows) == ["model", "model-latest", "sample_spec", "model-dated"]
        assert installed.model_count == 4


class TestAliasWarnings:
    @pytest.mark.parametrize(
        ("document", "warned"),
        [
            ({"model": _row(), "other": _row(aliases=["model"])}, 'alias "model" is already a model name'),
            (
                {"a": _row(aliases=["shared"]), "b": _row(aliases=["shared"])},
                'alias "shared" already belongs to a',
            ),
            ({"model": _row(), "ghost-latest": {"alias_of": "ghost"}}, 'alias_of "ghost" is not a model name'),
            (
                {"model": _row(), "model-latest": {"alias_of": "model", "mode": "chat"}},
                "model-latest.mode: an alias_of row carries no other field",
            ),
            ({"model": _row(aliases="model-dated")}, "model.aliases: invalid value"),
        ],
    )
    def test_an_alias_the_document_cannot_serve_is_logged(
        self, caplog: pytest.LogCaptureFixture, document: dict[str, dict[str, object]], warned: str
    ):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            _installed(document)

        assert any(warned in record.getMessage() for record in caplog.records), caplog.text

    def test_a_document_with_servable_aliases_logs_nothing(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            _installed({"model": _row(aliases=["model-dated"]), "model-latest": {"alias_of": "model"}})

        assert caplog.records == []


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
