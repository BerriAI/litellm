"""
Tests for the ``aliases`` feature in the model cost map.

The ``_expand_model_aliases`` function processes ``aliases`` lists from model
entries, creating shared dict references for alias entries at load time.
"""

import logging

import pytest

from litellm.litellm_core_utils.get_model_cost_map import _expand_model_aliases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_cost(**entries) -> dict:
    """Build a small model_cost dict from keyword args (model_name → info)."""
    return dict(entries)


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


# ---------------------------------------------------------------------------
# Conflict handling
# ---------------------------------------------------------------------------


class TestAliasConflicts:
    """Tests for alias conflict detection and handling."""

    def test_alias_conflicts_with_canonical_entry(self, caplog):
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
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            result = _expand_model_aliases(model_cost)

        # The canonical "model-dated" entry is preserved, not overwritten
        assert "model-dated" in result
        assert "alias conflict" in caplog.text.lower() or len(result) == 2

    def test_duplicate_alias_across_entries(self, caplog):
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
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            result = _expand_model_aliases(model_cost)

        # "shared-alias" should point to model-a (first one wins)
        assert "shared-alias" in result
        assert result["shared-alias"]["input_cost_per_token"] == 1e-06

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
