"""
Validate AWS Bedrock APAC (apac.*) Anthropic cross-region inference pricing entries.

AWS exposes geographic cross-region inference profiles per geography (US, EU, APAC, plus the
narrower Australia `au.` and Japan `jp.` profiles). A geographic profile prices identically
across its geography, so the `apac.` profile carries the same rates as `us.`/`eu.`/`au.`/`jp.`
for the same model. These models already had `us.`/`eu.`/`au.`/`jp.` entries but were missing
the broader `apac.` profile, so cross-region requests pinned to APAC fell back to the base
(non-regional) price.

Refs: https://github.com/BerriAI/litellm/issues/30768
"""

import json
import os

import pytest


@pytest.fixture(scope="module")
def model_data():
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        return json.load(f)


APAC_ANTHROPIC_MODELS = [
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-8",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-sonnet-4-6",
]


@pytest.mark.parametrize("base", APAC_ANTHROPIC_MODELS)
def test_apac_anthropic_entry_matches_us(model_data, base):
    """The apac. cross-region profile must exist and price identically to the us. profile,
    matching AWS geographic cross-region inference where us/eu/apac share the same rate.
    """
    apac_key = "apac." + base
    us_key = "us." + base
    assert apac_key in model_data, f"Missing model entry: {apac_key}"
    assert (
        model_data[apac_key] == model_data[us_key]
    ), f"{apac_key} must price identically to {us_key}"
