"""
Validate that the four Bedrock cross-region Claude 3 Haiku CRIS entries
carry the correct Anthropic-published cache pricing ($0.30 write / $0.03 read
per Mtok), matching the direct `claude-3-haiku-20240307` entry.

Previous values were incorrectly derived from the generic 1.25x write / 0.1x
read ratios (write $0.3125, read $0.025), which understated cache-read costs
by 16.7% and overstated cache-write costs by 4.2%.

This test guards against regression of the fix in PR #40368.
"""

import json
import os

import pytest

JSON_PATH = os.path.join(
    os.path.dirname(__file__), "../../model_prices_and_context_window.json"
)

BACKUP_PATH = os.path.join(
    os.path.dirname(__file__), "../../litellm/model_prices_and_context_window_backup.json"
)

# The four CRIS (cross-region inference) Bedrock variants that were fixed
CRIS_KEYS = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "apac.anthropic.claude-3-haiku-20240307-v1:0",
    "eu.anthropic.claude-3-haiku-20240307-v1:0",
    "us.anthropic.claude-3-haiku-20240307-v1:0",
]

# The direct (non-CRIS) entry that has the correct values — used for comparison
DIRECT_KEY = "claude-3-haiku-20240307"

EXPECTED_CACHE_CREATION = 3e-07  # $0.30 per M tokens
EXPECTED_CACHE_READ = 3e-08  # $0.03 per M tokens


@pytest.fixture(scope="module")
def model_data():
    with open(JSON_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def backup_data():
    with open(BACKUP_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("key", CRIS_KEYS)
def test_cris_cache_creation_price(model_data, key):
    """Each CRIS variant should have the correct cache creation price."""
    assert key in model_data, f"Missing model entry: {key}"
    entry = model_data[key]
    actual = entry.get("cache_creation_input_token_cost")
    assert actual == EXPECTED_CACHE_CREATION, (
        f"{key}: expected cache_creation_input_token_cost={EXPECTED_CACHE_CREATION} "
        f"($0.30/M), got {actual}"
    )


@pytest.mark.parametrize("key", CRIS_KEYS)
def test_cris_cache_read_price(model_data, key):
    """Each CRIS variant should have the correct cache read price."""
    assert key in model_data, f"Missing model entry: {key}"
    entry = model_data[key]
    actual = entry.get("cache_read_input_token_cost")
    assert actual == EXPECTED_CACHE_READ, (
        f"{key}: expected cache_read_input_token_cost={EXPECTED_CACHE_READ} "
        f"($0.03/M), got {actual}"
    )


@pytest.mark.parametrize("key", CRIS_KEYS)
def test_cris_prices_match_direct_entry(model_data, key):
    """CRIS variants should match the direct Claude 3 Haiku entry."""
    assert key in model_data, f"Missing CRIS entry: {key}"
    assert DIRECT_KEY in model_data, f"Missing direct entry: {DIRECT_KEY}"
    cris = model_data[key]
    direct = model_data[DIRECT_KEY]
    assert cris.get("cache_creation_input_token_cost") == direct.get("cache_creation_input_token_cost"), (
        f"{key} cache_creation mismatch with {DIRECT_KEY}"
    )
    assert cris.get("cache_read_input_token_cost") == direct.get("cache_read_input_token_cost"), (
        f"{key} cache_read mismatch with {DIRECT_KEY}"
    )


@pytest.mark.parametrize("key", CRIS_KEYS)
def test_backup_matches_main(backup_data, model_data, key):
    """Backup cost map should match the main cost map for all CRIS entries."""
    assert key in backup_data, f"Missing CRIS entry in backup: {key}"
    assert backup_data[key].get("cache_creation_input_token_cost") == EXPECTED_CACHE_CREATION
    assert backup_data[key].get("cache_read_input_token_cost") == EXPECTED_CACHE_READ
    assert backup_data[key] == model_data[key], (
        f"Backup and main differ for {key}"
    )
