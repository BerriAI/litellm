"""
Regression tests for #20885 – ``supports_response_schema`` (and related
capability flags) must be consistent between the bare model-name entry
(e.g. ``deepseek-chat``) and the provider-prefixed entry
(e.g. ``deepseek/deepseek-chat``) in the model-cost map.

The bug caused ``supports_response_schema("deepseek/deepseek-chat")`` to
return ``False`` even though the canonical ``deepseek-chat`` entry has the
field set to ``True``.
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.utils import (
    _supports_factory,
    supports_response_schema,
)


# ---------------------------------------------------------------------------
# Data-level tests – verify the JSON files are in sync
# ---------------------------------------------------------------------------


def _load_backup_json() -> dict:
    """Load the backup JSON directly from disk."""
    backup_path = os.path.join(
        os.path.dirname(litellm.__file__),
        "model_prices_and_context_window_backup.json",
    )
    with open(backup_path, encoding="utf-8") as f:
        return json.load(f)


class TestDeepSeekModelCostEntries:
    """Verify that provider-prefixed DeepSeek entries contain the same
    capability flags as their bare-name counterparts in the JSON files."""

    def test_deepseek_chat_supports_response_schema_in_backup(self):
        data = _load_backup_json()
        entry = data.get("deepseek/deepseek-chat", {})
        assert entry.get("supports_response_schema") is True

    def test_deepseek_reasoner_supports_response_schema_in_backup(self):
        data = _load_backup_json()
        entry = data.get("deepseek/deepseek-reasoner", {})
        assert entry.get("supports_response_schema") is True

    def test_deepseek_chat_supports_system_messages_in_backup(self):
        data = _load_backup_json()
        entry = data.get("deepseek/deepseek-chat", {})
        assert entry.get("supports_system_messages") is True

    def test_deepseek_reasoner_supports_system_messages_in_backup(self):
        data = _load_backup_json()
        entry = data.get("deepseek/deepseek-reasoner", {})
        assert entry.get("supports_system_messages") is True

    def test_deepseek_chat_max_input_tokens_matches_bare_in_backup(self):
        data = _load_backup_json()
        bare = data.get("deepseek-chat", {})
        prefixed = data.get("deepseek/deepseek-chat", {})
        assert prefixed.get("max_input_tokens") == bare.get("max_input_tokens")

    def test_deepseek_reasoner_max_output_tokens_matches_bare_in_backup(self):
        data = _load_backup_json()
        bare = data.get("deepseek-reasoner", {})
        prefixed = data.get("deepseek/deepseek-reasoner", {})
        assert prefixed.get("max_output_tokens") == bare.get("max_output_tokens")

    def test_main_json_deepseek_chat_supports_response_schema(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(litellm.__file__)),
            "model_prices_and_context_window.json",
        )
        with open(main_path, encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get("deepseek/deepseek-chat", {})
        assert entry.get("supports_response_schema") is True

    def test_main_json_deepseek_reasoner_supports_response_schema(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(litellm.__file__)),
            "model_prices_and_context_window.json",
        )
        with open(main_path, encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get("deepseek/deepseek-reasoner", {})
        assert entry.get("supports_response_schema") is True


# ---------------------------------------------------------------------------
# API-level tests – verify supports_response_schema returns True
# ---------------------------------------------------------------------------


class TestSupportsResponseSchemaDeepSeek:
    """All calling conventions for DeepSeek should return True for
    ``supports_response_schema``."""

    def test_provider_slash_model(self):
        assert supports_response_schema(model="deepseek/deepseek-chat") is True

    def test_explicit_provider(self):
        assert (
            supports_response_schema(
                model="deepseek-chat", custom_llm_provider="deepseek"
            )
            is True
        )

    def test_reasoner_provider_slash_model(self):
        assert supports_response_schema(model="deepseek/deepseek-reasoner") is True

    def test_reasoner_explicit_provider(self):
        assert (
            supports_response_schema(
                model="deepseek-reasoner", custom_llm_provider="deepseek"
            )
            is True
        )


# ---------------------------------------------------------------------------
# Fallback-logic test – bare model entry used when prefixed is incomplete
# ---------------------------------------------------------------------------


class TestBareModelFallback:
    """When a provider-prefixed entry is missing a capability flag, the
    ``_supports_factory`` fallback should consult the bare model-name
    entry in ``litellm.model_cost``."""

    def test_fallback_uses_bare_entry(self):
        """Temporarily remove ``supports_response_schema`` from the prefixed
        entry and verify the fallback still returns True."""
        key = "deepseek/deepseek-chat"
        original = litellm.model_cost.get(key, {}).get("supports_response_schema")
        try:
            # Simulate the pre-fix state: field missing from prefixed entry
            if key in litellm.model_cost:
                litellm.model_cost[key].pop("supports_response_schema", None)
            result = _supports_factory(
                model="deepseek-chat",
                custom_llm_provider="deepseek",
                key="supports_response_schema",
            )
            assert result is True
        finally:
            # Restore
            if key in litellm.model_cost and original is not None:
                litellm.model_cost[key]["supports_response_schema"] = original

    def test_no_fallback_when_explicitly_false(self):
        """If the prefixed entry explicitly sets a capability to ``False``,
        the fallback must NOT override it."""
        key = "deepseek/deepseek-reasoner"
        # After the data fix, deepseek/deepseek-reasoner has
        # supports_function_calling=false (matching the bare entry).
        # Explicitly set it to False to test the guard.
        original = litellm.model_cost.get(key, {}).get("supports_function_calling")
        try:
            if key in litellm.model_cost:
                litellm.model_cost[key]["supports_function_calling"] = False
            result = _supports_factory(
                model="deepseek-reasoner",
                custom_llm_provider="deepseek",
                key="supports_function_calling",
            )
            assert result is False
        finally:
            if key in litellm.model_cost and original is not None:
                litellm.model_cost[key]["supports_function_calling"] = original
