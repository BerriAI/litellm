"""
Regression tests for #29656 – the Bedrock ``amazon.titan-embed-text-v2:0``
input price was listed 10x too high (``2e-07`` = $0.20 / 1M tokens) in the
model-cost map.

Per the AWS Bedrock pricing API (us-west-2, 2026-06-04) Titan Text Embeddings
V2 costs $0.00002 / 1K tokens = $0.02 / 1M tokens = ``2e-08`` per token. The
same value is already used by the ``vercel_ai_gateway/amazon/titan-embed-text-v2``
entry, and Titan Embed V2 is cheaper than V1 (``1e-07``), so ``2e-07`` was
clearly the error.

The fix corrects the bare entry and both GovCloud region-prefixed entries
(``bedrock/us-gov-east-1/...`` and ``bedrock/us-gov-west-1/...``) in both the
canonical ``model_prices_and_context_window.json`` and the bundled
``model_prices_and_context_window_backup.json``.
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm

EXPECTED_INPUT_COST_PER_TOKEN = 2e-08  # $0.02 / 1M tokens

TITAN_EMBED_V2_KEYS = [
    "amazon.titan-embed-text-v2:0",
    "bedrock/us-gov-east-1/amazon.titan-embed-text-v2:0",
    "bedrock/us-gov-west-1/amazon.titan-embed-text-v2:0",
]


def _load_backup_json() -> dict:
    """Load the bundled backup JSON directly from disk."""
    backup_path = os.path.join(
        os.path.dirname(litellm.__file__),
        "model_prices_and_context_window_backup.json",
    )
    with open(backup_path, encoding="utf-8") as f:
        return json.load(f)


def _load_main_json() -> dict:
    """Load the canonical (repo-root) JSON directly from disk."""
    main_path = os.path.join(
        os.path.dirname(os.path.dirname(litellm.__file__)),
        "model_prices_and_context_window.json",
    )
    with open(main_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data-level tests – verify both JSON files carry the corrected price
# ---------------------------------------------------------------------------


class TestTitanEmbedV2PricingData:
    def test_main_json_titan_embed_v2_input_cost(self):
        data = _load_main_json()
        for key in TITAN_EMBED_V2_KEYS:
            entry = data.get(key, {})
            assert entry, f"missing entry: {key}"
            assert (
                entry.get("input_cost_per_token") == EXPECTED_INPUT_COST_PER_TOKEN
            ), f"{key} input_cost_per_token in main JSON should be {EXPECTED_INPUT_COST_PER_TOKEN}"

    def test_backup_json_titan_embed_v2_input_cost(self):
        data = _load_backup_json()
        for key in TITAN_EMBED_V2_KEYS:
            entry = data.get(key, {})
            assert entry, f"missing entry: {key}"
            assert (
                entry.get("input_cost_per_token") == EXPECTED_INPUT_COST_PER_TOKEN
            ), f"{key} input_cost_per_token in backup JSON should be {EXPECTED_INPUT_COST_PER_TOKEN}"

    def test_main_and_backup_titan_embed_v2_in_sync(self):
        main = _load_main_json()
        backup = _load_backup_json()
        for key in TITAN_EMBED_V2_KEYS:
            assert main.get(key, {}).get("input_cost_per_token") == backup.get(
                key, {}
            ).get(
                "input_cost_per_token"
            ), f"{key} input_cost_per_token out of sync between main and backup JSON"

    def test_titan_embed_v2_cheaper_than_v1(self):
        """Titan Embed V2 is cheaper than V1 on AWS; guards against a
        regression that re-inflates the V2 price above V1."""
        data = _load_main_json()
        v2 = data["amazon.titan-embed-text-v2:0"]["input_cost_per_token"]
        v1 = data["amazon.titan-embed-text-v1"]["input_cost_per_token"]
        assert v2 < v1


# ---------------------------------------------------------------------------
# API-level test – verify the loaded model-cost map reflects the fix
# ---------------------------------------------------------------------------


class TestTitanEmbedV2PricingModelCost:
    def test_model_cost_map_input_cost(self):
        entry = litellm.model_cost.get("amazon.titan-embed-text-v2:0", {})
        assert entry, "amazon.titan-embed-text-v2:0 missing from litellm.model_cost"
        assert entry.get("input_cost_per_token") == EXPECTED_INPUT_COST_PER_TOKEN
