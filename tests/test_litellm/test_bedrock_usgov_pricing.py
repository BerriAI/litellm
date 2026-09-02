"""
Validate AWS GovCloud (Bedrock us-gov-*) Anthropic pricing entries.

AWS Bedrock pricing in GovCloud carries a +20% premium over the global
Anthropic prices (not the +10% commercial-US premium). Until 2026-05-22
these entries silently mirrored commercial US, undercharging customers
by ~9%.

Source: https://aws.amazon.com/bedrock/pricing/

  Sonnet 4.5 in us-gov-* (per million tokens):
    input          = $3.60
    output         = $18.00
    cache write 5m = $4.50
    cache write 1h = $7.20
    cache read     = $0.36

Reference: https://github.com/BerriAI/litellm/issues/27120
"""

import json
import os

import pytest


@pytest.fixture(scope="module")
def model_data():
    json_path = os.path.join(os.path.dirname(__file__), "../../model_prices_and_context_window.json")
    with open(json_path) as f:
        return json.load(f)


SONNET_4_5_USGOV_KEYS = [
    "bedrock/us-gov-east-1/anthropic.claude-sonnet-4-5-20250929-v1:0",
    "bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0",
    "bedrock/us-gov-east-1/claude-sonnet-4-5-20250929-v1:0",
    "bedrock/us-gov-west-1/claude-sonnet-4-5-20250929-v1:0",
    "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
]


@pytest.mark.parametrize("model_key", SONNET_4_5_USGOV_KEYS)
def test_usgov_sonnet_4_5_pricing(model_data, model_key):
    """Each us-gov sonnet-4-5 entry must carry the +20%-over-global rates
    that AWS publishes on the GovCloud pricing page.
    """
    assert model_key in model_data, f"Missing model entry: {model_key}"
    info = model_data[model_key]

    assert info["input_cost_per_token"] == 3.6e-06, (
        f"{model_key}: input_cost_per_token should be $3.60/MTok (got {info['input_cost_per_token']})"
    )
    assert info["output_cost_per_token"] == 1.8e-05, f"{model_key}: output_cost_per_token should be $18.00/MTok"
    assert info["cache_creation_input_token_cost"] == 4.5e-06, f"{model_key}: 5m cache write should be $4.50/MTok"
    assert info["cache_creation_input_token_cost_above_1hr"] == 7.2e-06, (
        f"{model_key}: 1h cache write should be $7.20/MTok"
    )
    assert info["cache_read_input_token_cost"] == 3.6e-07, f"{model_key}: cache read should be $0.36/MTok"


def test_usgov_carries_20_percent_premium_over_global(model_data):
    """The us-gov rates must equal 1.2x the global anthropic.* rates,
    matching AWS's documented GovCloud uplift.
    """
    global_key = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    usgov_key = "bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0"
    global_info = model_data[global_key]
    usgov_info = model_data[usgov_key]
    for field in (
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "cache_read_input_token_cost",
    ):
        ratio = usgov_info[field] / global_info[field]
        assert abs(ratio - 1.2) < 1e-9, f"{field}: us-gov / global ratio is {ratio}, expected 1.2"


# The us-gov.anthropic.* cross-region inference profile is the only us-gov
# entry that carries the 1M-context `_above_200k_tokens` pricing tier — the
# bedrock/us-gov-{east,west}-1/ entries are capped at 200k tokens.
USGOV_CROSS_REGION_KEY = "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0"

EXPECTED_USGOV_ABOVE_200K = {
    "input_cost_per_token_above_200k_tokens": 7.2e-06,
    "output_cost_per_token_above_200k_tokens": 2.7e-05,
    "cache_creation_input_token_cost_above_200k_tokens": 9.0e-06,
    "cache_creation_input_token_cost_above_1hr_above_200k_tokens": 1.44e-05,
    "cache_read_input_token_cost_above_200k_tokens": 7.2e-07,
}


@pytest.mark.parametrize("field,expected", EXPECTED_USGOV_ABOVE_200K.items())
def test_usgov_cross_region_above_200k_carries_gov_premium(model_data, field, expected):
    """The `_above_200k_tokens` tier on the us-gov cross-region inference
    profile must also carry the +20% GovCloud uplift. The original PR
    corrected the base rates but left the 200k-tier fields at the +10%
    commercial-US rates, undercharging long-context requests.
    """
    info = model_data[USGOV_CROSS_REGION_KEY]
    assert field in info, f"{USGOV_CROSS_REGION_KEY}: missing field {field}"
    assert info[field] == expected, f"{USGOV_CROSS_REGION_KEY}: {field} should be {expected} (got {info[field]})"


def test_usgov_cross_region_above_200k_ratio_to_global(model_data):
    """Cross-check via the property-based invariant: every `_above_200k_tokens`
    field on the us-gov cross-region profile must equal 1.2x the global
    anthropic.* rate, the same GovCloud uplift the base tier carries.
    """
    global_key = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    global_info = model_data[global_key]
    usgov_info = model_data[USGOV_CROSS_REGION_KEY]
    for field in EXPECTED_USGOV_ABOVE_200K:
        ratio = usgov_info[field] / global_info[field]
        assert abs(ratio - 1.2) < 1e-9, f"{field}: us-gov / global ratio is {ratio}, expected 1.2"


CLAUDE_GOV_EXPECTED = {
    "anthropic.claude-sonnet-5": {
        "input_cost_per_token": 2.4e-06,
        "output_cost_per_token": 1.2e-05,
        "cache_creation_input_token_cost": 3e-06,
        "cache_creation_input_token_cost_above_1hr": 4.8e-06,
        "cache_read_input_token_cost": 2.4e-07,
    },
    "anthropic.claude-opus-4-8": {
        "input_cost_per_token": 6e-06,
        "output_cost_per_token": 3e-05,
        "cache_creation_input_token_cost": 7.5e-06,
        "cache_creation_input_token_cost_above_1hr": 1.2e-05,
        "cache_read_input_token_cost": 6e-07,
    },
}


@pytest.mark.parametrize("base_key", CLAUDE_GOV_EXPECTED)
@pytest.mark.parametrize("region", ["us-gov-east-1", "us-gov-west-1"])
def test_usgov_claude_sonnet5_opus48_pricing(model_data, region, base_key):
    """Sonnet 5 and Opus 4.8 gov entries must match the rates AWS publishes
    for both GovCloud regions on the Bedrock pricing page (1.2x global).
    """
    gov_key = f"bedrock/{region}/{base_key}"
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    for field, expected in CLAUDE_GOV_EXPECTED[base_key].items():
        assert info[field] == expected, f"{gov_key}: {field} should be {expected} (got {info[field]})"
        ratio = info[field] / model_data[base_key][field]
        assert abs(ratio - 1.2) < 1e-9, f"{gov_key}: {field} gov/global ratio is {ratio}, expected 1.2"


CONVERSE_GOV_EXPECTED = {
    "nvidia.nemotron-nano-3-30b": (7.2e-08, 2.88e-07),
    "nvidia.nemotron-nano-12b-v2": (2.4e-07, 7.2e-07),
    "nvidia.nemotron-super-3-120b": (1.8e-07, 7.8e-07),
    "openai.gpt-oss-20b-1:0": (8.4e-08, 3.6e-07),
    "openai.gpt-oss-120b-1:0": (1.8e-07, 7.2e-07),
}


@pytest.mark.parametrize("base_key", CONVERSE_GOV_EXPECTED)
@pytest.mark.parametrize("region", ["us-gov-east-1", "us-gov-west-1"])
def test_usgov_converse_model_pricing(model_data, region, base_key):
    """Nemotron and gpt-oss gov entries must match the AWS Bedrock offer file,
    which prices both GovCloud regions identically at 1.2x commercial.
    """
    gov_key = f"bedrock/{region}/{base_key}"
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    expected_input, expected_output = CONVERSE_GOV_EXPECTED[base_key]
    assert info["input_cost_per_token"] == expected_input
    assert info["output_cost_per_token"] == expected_output
    assert info["litellm_provider"] == "bedrock"
    base = model_data[base_key]
    assert abs(info["input_cost_per_token"] / base["input_cost_per_token"] - 1.2) < 1e-9
    assert abs(info["output_cost_per_token"] / base["output_cost_per_token"] - 1.2) < 1e-9


def test_usgov_west_llama3_8b_output_price_fixed(model_data):
    """The us-gov-west-1 llama3-8b entry carried the 70B output rate ($2.65/MTok);
    the AWS Bedrock offer file prices output at $0.60/MTok. AWS lists the model
    in us-gov-west-1 only, so there is no east entry to check.
    """
    info = model_data["bedrock/us-gov-west-1/meta.llama3-8b-instruct-v1:0"]
    assert info["input_cost_per_token"] == 3e-07
    assert info["output_cost_per_token"] == 6e-07


MANTLE_GOV_TIERED_EXPECTED = {
    "openai.gpt-5.6-luna": {
        "input_cost_per_token": 2.64e-07,
        "input_cost_per_token_above_272k_tokens": 5.28e-07,
        "cache_creation_input_token_cost": 3.3e-07,
        "cache_creation_input_token_cost_above_272k_tokens": 6.6e-07,
        "cache_read_input_token_cost": 2.64e-08,
        "cache_read_input_token_cost_above_272k_tokens": 5.28e-08,
        "output_cost_per_token": 1.584e-06,
        "output_cost_per_token_above_272k_tokens": 2.376e-06,
    },
    "openai.gpt-5.6-terra": {
        "input_cost_per_token": 2.64e-06,
        "input_cost_per_token_above_272k_tokens": 5.28e-06,
        "cache_creation_input_token_cost": 3.3e-06,
        "cache_creation_input_token_cost_above_272k_tokens": 6.6e-06,
        "cache_read_input_token_cost": 2.64e-07,
        "cache_read_input_token_cost_above_272k_tokens": 5.28e-07,
        "output_cost_per_token": 1.584e-05,
        "output_cost_per_token_above_272k_tokens": 2.376e-05,
    },
}


@pytest.mark.parametrize("model", MANTLE_GOV_TIERED_EXPECTED)
def test_usgov_west_mantle_terra_luna_pricing(model_data, model):
    """Terra and Luna carry 1.2x commercial across every tier in the
    us-gov-west-1 offer file; the us-gov-east-1 offer file has no SKUs for them.
    """
    gov_key = f"bedrock_mantle/us-gov-west-1/{model}"
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    for field, expected in MANTLE_GOV_TIERED_EXPECTED[model].items():
        assert info[field] == expected, f"{gov_key}: {field} should be {expected} (got {info[field]})"
    assert info["litellm_provider"] == "bedrock_mantle"
    assert f"bedrock_mantle/us-gov-east-1/{model}" not in model_data


@pytest.mark.parametrize("region", ["us-gov-east-1", "us-gov-west-1"])
def test_usgov_mantle_gpt_5_4_pricing_has_no_long_context_tier(model_data, region):
    """gpt-5.4 gov rates come from the offer file, which publishes only the
    standard tier in GovCloud: no long-context SKUs exist there, unlike commercial.
    """
    gov_key = f"bedrock_mantle/{region}/openai.gpt-5.4"
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    assert info["input_cost_per_token"] == 3.3e-06
    assert info["cache_read_input_token_cost"] == 3.3e-07
    assert info["output_cost_per_token"] == 1.98e-05
    assert not any(field.endswith("_above_272k_tokens") for field in info)


def test_usgov_mantle_grok_4_3_west_only(model_data):
    """grok-4.3 is priced in the us-gov-west-1 offer file only; the east offer
    file carries grok-4.6 instead.
    """
    info = model_data["bedrock_mantle/us-gov-west-1/xai.grok-4.3"]
    assert info["input_cost_per_token"] == 1.5e-06
    assert info["output_cost_per_token"] == 3e-06
    assert info["cache_read_input_token_cost"] == 2.4e-07
    assert "bedrock_mantle/us-gov-east-1/xai.grok-4.3" not in model_data


AZURE_GOV_EXPECTED = {
    "azure/us-gov/gpt-5.1": {
        "input_cost_per_token": 1.71875e-06,
        "cache_read_input_token_cost": 1.71875e-07,
        "output_cost_per_token": 1.375e-05,
    },
    "azure/us-gov/o3-mini": {
        "input_cost_per_token": 1.513e-06,
        "cache_read_input_token_cost": 7.57e-07,
        "output_cost_per_token": 6.05e-06,
    },
    "azure/us-gov/text-embedding-3-large": {"input_cost_per_token": 1.63e-07},
    "azure/us-gov/text-embedding-3-small": {"input_cost_per_token": 2.5e-08},
}


@pytest.mark.parametrize("gov_key", AZURE_GOV_EXPECTED)
def test_azure_usgov_pricing(model_data, gov_key):
    """Azure Government meters from the Azure retail prices API
    (usgovvirginia/usgovarizona, serviceName 'Foundry Models'). No Government
    retirement schedule is published, so these entries carry no deprecation_date.
    """
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    for field, expected in AZURE_GOV_EXPECTED[gov_key].items():
        assert info[field] == expected, f"{gov_key}: {field} should be {expected} (got {info[field]})"
    assert info["litellm_provider"] == "azure"
    assert "deprecation_date" not in info
