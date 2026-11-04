"""
Tests for the ``aliases`` feature in the model cost map.

The ``_expand_model_aliases`` function processes ``aliases`` lists from model
entries, creating shared dict references for alias entries at load time.
"""

from collections.abc import Mapping
from unittest.mock import call, patch

import pytest

from litellm import verbose_logger
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry, _expand_model_aliases

_NOT_A_LIST = "LiteLLM model alias field for '%s' is not a list (got %s) — skipping."
_SHADOWS = "LiteLLM model alias conflict: alias '%s' (from '%s') already exists as a canonical entry — skipping."
_TAKEN = "LiteLLM model alias conflict: alias '%s' (from '%s') was already claimed by another entry — skipping."


# ---------------------------------------------------------------------------
# Core expansion behaviour
# ---------------------------------------------------------------------------


class TestExpandModelAliases:
    """Unit tests for _expand_model_aliases."""

    def test_basic_expansion(self):
        """Aliases are added as top-level entries in model_cost."""
        model_cost = {
            "my-model-latest": {
                "aliases": ["my-model-20250101"],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        assert "my-model-20250101" in result
        assert result["my-model-20250101"]["input_cost_per_token"] == 1e-06
        assert result["my-model-20250101"]["litellm_provider"] == "test"

    def test_multiple_aliases(self):
        """A single entry can declare multiple aliases."""
        model_cost = {
            "provider/model-latest": {
                "aliases": ["provider/model-v1", "provider/model-v2"],
                "input_cost_per_token": 5e-06,
                "litellm_provider": "provider",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        assert "provider/model-v1" in result
        assert "provider/model-v2" in result

    def test_shared_dict_reference(self):
        """Alias entries share the same dict object as the canonical entry (no copy)."""
        model_cost = {
            "canonical-model": {
                "aliases": ["alias-model"],
                "input_cost_per_token": 2e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        assert result["alias-model"] is result["canonical-model"]

    def test_aliases_key_removed(self):
        """The ``aliases`` key is removed from the entry after expansion."""
        model_cost = {
            "my-model": {
                "aliases": ["my-model-alias"],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        assert "aliases" not in result["my-model"]
        assert "aliases" not in result["my-model-alias"]

    def test_entries_without_aliases_unchanged(self):
        """Entries with no ``aliases`` key are left untouched."""
        model_cost = {
            "plain-model": {
                "input_cost_per_token": 3e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        assert "plain-model" in result
        assert result["plain-model"]["input_cost_per_token"] == 3e-06
        assert len(result) == 1

    def test_empty_aliases_list(self):
        """An empty ``aliases`` list is treated the same as no aliases."""
        model_cost = {
            "model-a": {
                "aliases": [],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        assert len(result) == 1
        assert "model-a" in result
        assert "aliases" not in result["model-a"]


# ---------------------------------------------------------------------------
# Conflict handling
# ---------------------------------------------------------------------------


class TestAliasConflicts:
    """Tests for alias conflict detection and handling."""

    def test_alias_conflicts_with_canonical_entry(self):
        """Alias that matches an existing canonical entry is skipped with a warning."""
        model_cost = {
            "model-latest": {
                "aliases": ["model-dated"],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
            "model-dated": {
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        with patch.object(verbose_logger, "warning") as mock_warn:
            result = _expand_model_aliases(model_cost)

        # The canonical "model-dated" entry is preserved, not overwritten
        assert "model-dated" in result
        # Verify a warning about the alias conflict was logged
        mock_warn.assert_called()
        warning_messages = " ".join(str(c) for c in mock_warn.call_args_list)
        assert "alias conflict" in warning_messages.lower()

    def test_duplicate_alias_across_entries(self):
        """Same alias claimed by two different entries: second one is skipped."""
        model_cost = {
            "model-a": {
                "aliases": ["shared-alias"],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
            "model-b": {
                "aliases": ["shared-alias"],
                "input_cost_per_token": 2e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        with patch.object(verbose_logger, "warning") as mock_warn:
            result = _expand_model_aliases(model_cost)

        # "shared-alias" should point to model-a (first one wins)
        assert "shared-alias" in result
        assert result["shared-alias"]["input_cost_per_token"] == 1e-06
        # Verify a warning about the alias conflict was logged
        mock_warn.assert_called()
        warning_messages = " ".join(str(c) for c in mock_warn.call_args_list)
        assert "alias conflict" in warning_messages.lower()

    def test_canonical_entry_not_overwritten_by_alias(self):
        """An alias must never overwrite an existing canonical entry's data."""
        original_cost = 9.99e-06
        model_cost = {
            "existing-model": {
                "input_cost_per_token": original_cost,
                "litellm_provider": "test",
                "mode": "chat",
            },
            "other-model": {
                "aliases": ["existing-model"],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        # Original entry must be preserved
        assert result["existing-model"]["input_cost_per_token"] == original_cost


# ---------------------------------------------------------------------------
# Integration with model_cost dict mutation
# ---------------------------------------------------------------------------


class TestAliasIntegration:
    """Higher-level tests verifying aliases work with the model_cost dict."""

    def test_mutation_through_alias_visible_on_canonical(self):
        """Since alias is a shared reference, mutations are visible on both."""
        model_cost = {
            "canonical": {
                "aliases": ["alias"],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        # Mutate via alias
        result["alias"]["input_cost_per_token"] = 999
        assert result["canonical"]["input_cost_per_token"] == 999

    def test_mixed_entries_with_and_without_aliases(self):
        """A model_cost dict with a mix of aliased and plain entries."""
        model_cost = {
            "model-with-alias": {
                "aliases": ["alias-1", "alias-2"],
                "input_cost_per_token": 1e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
            "plain-model": {
                "input_cost_per_token": 2e-06,
                "litellm_provider": "test",
                "mode": "chat",
            },
        }
        result = _expand_model_aliases(model_cost)

        assert len(result) == 4  # 2 canonical + 2 aliases
        assert "alias-1" in result
        assert "alias-2" in result
        assert "plain-model" in result
        assert "model-with-alias" in result

    def test_expand_on_empty_dict(self):
        """Expanding an empty dict returns an empty dict."""
        assert _expand_model_aliases({}) == {}


def _served(model_cost: Mapping[str, object]) -> list[tuple[object, object, tuple[object, ...] | None]]:
    return [
        (key, dict(row) if isinstance(row, dict) else row, row.names if isinstance(row, AliasedCostEntry) else None)
        for key, row in model_cost.items()
    ]


class TestExpansionMatchesMainExactly:
    """Every case pins the map, the row objects and the warnings main's Python expansion produced"""

    @pytest.mark.parametrize(
        ("model_cost", "served", "warnings"),
        [
            pytest.param(
                {"b": {"aliases": ["b1"]}, "a": {"aliases": ["a1", "a2"]}, "c": {}},
                [
                    ("b", {}, ("b", "b1")),
                    ("a", {}, ("a", "a1", "a2")),
                    ("c", {}, None),
                    ("b1", {}, ("b", "b1")),
                    ("a1", {}, ("a", "a1", "a2")),
                    ("a2", {}, ("a", "a1", "a2")),
                ],
                [],
                id="aliases_follow_rows_in_first_claim_order",
            ),
            pytest.param(
                {"m": {"aliases": None, "mode": "chat"}},
                [("m", {"aliases": None, "mode": "chat"}, None)],
                [],
                id="null_aliases_stay_on_the_row",
            ),
            pytest.param(
                {"m": {"aliases": "m2", "mode": "chat"}},
                [("m", {"mode": "chat"}, None)],
                [call(_NOT_A_LIST, "m", "str")],
                id="non_list_aliases_are_dropped_with_a_warning",
            ),
            pytest.param(
                {"m": {"aliases": []}},
                [("m", {}, None)],
                [],
                id="empty_aliases_are_dropped",
            ),
            pytest.param(
                {"m": {"aliases": ["sample_spec", "n", "m"]}, "n": {"aliases": ["n"]}, "sample_spec": {}},
                [("m", {}, None), ("n", {}, None), ("sample_spec", {}, None)],
                [
                    call(_SHADOWS, "sample_spec", "m"),
                    call(_SHADOWS, "n", "m"),
                    call(_SHADOWS, "m", "m"),
                    call(_SHADOWS, "n", "n"),
                ],
                id="an_alias_equal_to_any_key_is_skipped",
            ),
            pytest.param(
                {"m": {"aliases": ["x", "x"]}, "n": {"aliases": ["x"]}},
                [("m", {}, ("m", "x")), ("n", {}, None), ("x", {}, ("m", "x"))],
                [call(_TAKEN, "x", "m"), call(_TAKEN, "x", "n")],
                id="the_first_claim_wins",
            ),
            pytest.param(
                {"m": {"aliases": [1]}, "n": {"aliases": [True, 1.0]}},
                [("m", {}, ("m", 1)), ("n", {}, None), (1, {}, ("m", 1))],
                [call(_TAKEN, True, "n"), call(_TAKEN, 1.0, "n")],
                id="aliases_collide_under_python_equality",
            ),
        ],
    )
    def test_expansion(
        self,
        model_cost: dict[str, dict[str, object]],
        served: list[tuple[object, object, tuple[object, ...] | None]],
        warnings: list[object],
    ):
        with patch.object(verbose_logger, "warning") as warn:
            result = _expand_model_aliases(model_cost)

        assert result is model_cost
        assert _served(result) == served
        assert warn.call_args_list == warnings

    def test_each_alias_group_shares_one_copied_row(self):
        row: dict[str, object] = {"aliases": ["m1", "m2"], "nested": [1]}
        result = _expand_model_aliases({"m": row})

        assert result["m"] is result["m1"] is result["m2"]
        assert result["m"] is not row
        assert result["m"]["nested"] is row["nested"]

    @pytest.mark.parametrize(
        ("model_cost", "error", "message", "warnings"),
        [
            pytest.param(
                {"m": {"aliases": "x"}, "bad": None, "n": {"aliases": "y"}},
                AttributeError,
                "'NoneType' object has no attribute 'get'",
                [call(_NOT_A_LIST, "m", "str")],
                id="a_row_that_is_not_a_dict",
            ),
            pytest.param(
                {"m": {"aliases": "x"}, "n": {"aliases": ["ok", ["unhashable"]]}},
                TypeError,
                "unhashable type: 'list'",
                [call(_NOT_A_LIST, "m", "str")],
                id="an_unhashable_alias",
            ),
        ],
    )
    def test_main_raises_mid_scan_after_its_earlier_warnings_and_leaves_the_map(
        self,
        model_cost: dict[str, dict[str, object]],
        error: type[Exception],
        message: str,
        warnings: list[object],
    ):
        before = repr(model_cost)
        with patch.object(verbose_logger, "warning") as warn, pytest.raises(error, match=f"^{message}$"):
            _expand_model_aliases(model_cost)

        assert warn.call_args_list == warnings
        assert repr(model_cost) == before
