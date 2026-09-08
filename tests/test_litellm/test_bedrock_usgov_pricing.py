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
    "anthropic.claude-opus-5": {
        "input_cost_per_token": 6e-06,
        "output_cost_per_token": 3e-05,
        "cache_creation_input_token_cost": 7.5e-06,
        "cache_creation_input_token_cost_above_1hr": 1.2e-05,
        "cache_read_input_token_cost": 6e-07,
    },
    "anthropic.claude-fable-5-1": {
        "input_cost_per_token": 1.2e-05,
        "output_cost_per_token": 6e-05,
        "cache_creation_input_token_cost": 1.5e-05,
        "cache_creation_input_token_cost_above_1hr": 2.4e-05,
        "cache_read_input_token_cost": 3e-07,
    },
}


USGOV_CLAUDE_KEY_TEMPLATES = {
    "bedrock/us-gov-east-1/{base_key}": "bedrock",
    "bedrock/us-gov-west-1/{base_key}": "bedrock",
    "us-gov.{base_key}": "bedrock_converse",
}


@pytest.mark.parametrize("base_key", CLAUDE_GOV_EXPECTED)
@pytest.mark.parametrize("key_template,expected_provider", USGOV_CLAUDE_KEY_TEMPLATES.items())
def test_usgov_claude_pricing(model_data, key_template, expected_provider, base_key):
    """Sonnet 5, Opus 4.8, Opus 5, and Fable 5.1 gov entries, both in-region keys
    and the us-gov. geo inference profile the model cards list for GovCloud, must
    carry the 1.2x GovCloud premium over the global anthropic.* rates. No public
    AWS source (offer files, pricing page) lists Claude GovCloud rows; the premium
    is the one AWS quotes for Opus 4.8 in GovCloud ($6/$30 per million).
    """
    gov_key = key_template.format(base_key=base_key)
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    assert info["litellm_provider"] == expected_provider
    assert "search_context_cost_per_query" not in info
    for field, expected in CLAUDE_GOV_EXPECTED[base_key].items():
        assert info[field] == expected, f"{gov_key}: {field} should be {expected} (got {info[field]})"
        ratio = info[field] / model_data[base_key][field]
        assert abs(ratio - 1.2) < 1e-9, f"{gov_key}: {field} gov/global ratio is {ratio}, expected 1.2"


CONVERSE_GOV_EXPECTED = {
    "nvidia.nemotron-nano-3-30b": (7.2e-08, 2.88e-07),
    "nvidia.nemotron-nano-9b-v2": (7.2e-08, 2.76e-07),
    "nvidia.nemotron-nano-12b-v2": (2.4e-07, 7.2e-07),
    "nvidia.nemotron-super-3-120b": (1.8e-07, 7.8e-07),
    "openai.gpt-oss-20b-1:0": (8.4e-08, 3.6e-07),
    "openai.gpt-oss-120b-1:0": (1.8e-07, 7.2e-07),
}


@pytest.mark.parametrize("base_key", CONVERSE_GOV_EXPECTED)
@pytest.mark.parametrize("key_template,expected_provider", USGOV_CLAUDE_KEY_TEMPLATES.items())
def test_usgov_converse_model_pricing(model_data, key_template, expected_provider, base_key):
    """Nemotron and gpt-oss gov entries, in-region and the us-gov. geo inference
    profile both GovCloud regions list as ACTIVE, must match the AWS Bedrock
    offer file, which prices both regions identically at 1.2x commercial.
    """
    gov_key = key_template.format(base_key=base_key)
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    expected_input, expected_output = CONVERSE_GOV_EXPECTED[base_key]
    assert info["input_cost_per_token"] == expected_input
    assert info["output_cost_per_token"] == expected_output
    assert info["litellm_provider"] == expected_provider
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


def test_usgov_east_haiku_profile_mirrors_in_region_row(model_data):
    """us-gov-east-1 serves claude-3-haiku through the us-gov. inference profile
    only, so the profile row must bill exactly like the in-region gov row.
    """
    profile = model_data["us-gov.anthropic.claude-3-haiku-20240307-v1:0"]
    in_region = model_data["bedrock/us-gov-east-1/anthropic.claude-3-haiku-20240307-v1:0"]
    assert profile["litellm_provider"] == "bedrock_converse"
    assert {k: v for k, v in profile.items() if k != "litellm_provider"} == {
        k: v for k, v in in_region.items() if k != "litellm_provider"
    }


GROK_4_6_GOV_KEYS = {
    "us-gov.xai.grok-4.6": ("us.xai.grok-4.6", "bedrock_converse"),
    "bedrock_mantle/us-gov-west-1/xai.grok-4.6": ("bedrock_mantle/xai.grok-4.6", "bedrock_mantle"),
    "bedrock_mantle/us-gov-east-1/xai.grok-4.6": ("bedrock_mantle/xai.grok-4.6", "bedrock_mantle"),
}


@pytest.mark.parametrize("gov_key", GROK_4_6_GOV_KEYS)
def test_usgov_grok_4_6_pricing(model_data, gov_key):
    """Both GovCloud regions serve grok-4.6 through the us-gov. profile only, and
    both offer files price its standard SKU at 1.2x the commercial US rate.
    """
    base_key, expected_provider = GROK_4_6_GOV_KEYS[gov_key]
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    assert info["litellm_provider"] == expected_provider
    assert info["input_cost_per_token"] == 2.64e-06
    assert info["output_cost_per_token"] == 7.92e-06
    assert info["cache_read_input_token_cost"] == 6.6e-07
    for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"):
        assert abs(info[field] / model_data[base_key][field] - 1.2) < 1e-9


NOVA_GOV_WEST_EXPECTED = {
    "amazon.nova-lite-v1:0": (7.2e-08, 2.88e-07),
    "amazon.nova-micro-v1:0": (4.2e-08, 1.68e-07),
}


@pytest.mark.parametrize("base_key", NOVA_GOV_WEST_EXPECTED)
def test_usgov_west_nova_lite_micro_pricing(model_data, base_key):
    """Nova Lite and Micro are on-demand in us-gov-west-1 only; the offer file
    prices them at 1.2x commercial, like the Nova Pro row that was already there.
    """
    gov_key = f"bedrock/us-gov-west-1/{base_key}"
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    expected_input, expected_output = NOVA_GOV_WEST_EXPECTED[base_key]
    assert info["litellm_provider"] == "bedrock"
    assert info["input_cost_per_token"] == expected_input
    assert info["output_cost_per_token"] == expected_output
    assert abs(info["input_cost_per_token"] / model_data[base_key]["input_cost_per_token"] - 1.2) < 1e-9
    assert abs(info["output_cost_per_token"] / model_data[base_key]["output_cost_per_token"] - 1.2) < 1e-9
    assert f"bedrock/us-gov-east-1/{base_key}" not in model_data


def test_usgov_west_nova_2_multimodal_embeddings_pricing(model_data):
    """Every meter of the multimodal embedding model (tokens, images, audio and
    video seconds) carries the 1.2x uplift the us-gov-west-1 offer file lists.
    """
    gov_key = "bedrock/us-gov-west-1/amazon.nova-2-multimodal-embeddings-v1:0"
    assert gov_key in model_data, f"Missing model entry: {gov_key}"
    info = model_data[gov_key]
    assert info["litellm_provider"] == "bedrock"
    assert info["mode"] == "embedding"
    assert info["input_cost_per_token"] == 1.62e-07
    assert info["input_cost_per_image"] == 7.2e-05
    assert info["input_cost_per_audio_per_second"] == 0.000168
    assert info["input_cost_per_video_per_second"] == 0.00084
    assert "bedrock/us-gov-east-1/amazon.nova-2-multimodal-embeddings-v1:0" not in model_data


MANTLE_GOV_FLAT_EXPECTED = {
    "google.gemma-4-e2b": (4.8e-08, 9.6e-08, ("us-gov-west-1",)),
    "google.gemma-4-26b-a4b": (1.56e-07, 4.8e-07, ("us-gov-west-1",)),
    "google.gemma-4-31b": (1.68e-07, 4.8e-07, ("us-gov-west-1",)),
    "openai.gpt-oss-20b": (8.4e-08, 3.6e-07, ("us-gov-west-1", "us-gov-east-1")),
    "openai.gpt-oss-120b": (1.8e-07, 7.2e-07, ("us-gov-west-1", "us-gov-east-1")),
}


@pytest.mark.parametrize("model", MANTLE_GOV_FLAT_EXPECTED)
def test_usgov_mantle_gemma_and_gpt_oss_pricing(model_data, model):
    """Gemma 4 is priced in the us-gov-west-1 offer file only and gpt-oss in both;
    each Mantle gov row carries the offer file's standard SKU, and no row exists
    for a region whose offer file has no SKU.
    """
    expected_input, expected_output, regions = MANTLE_GOV_FLAT_EXPECTED[model]
    for region in ("us-gov-west-1", "us-gov-east-1"):
        gov_key = f"bedrock_mantle/{region}/{model}"
        if region not in regions:
            assert gov_key not in model_data
            continue
        assert gov_key in model_data, f"Missing model entry: {gov_key}"
        info = model_data[gov_key]
        assert info["litellm_provider"] == "bedrock_mantle"
        assert info["input_cost_per_token"] == expected_input
        assert info["output_cost_per_token"] == expected_output


GOV_ROW_SOURCES = {
    "us-gov.anthropic.claude-fable-5-1": "anthropic.claude-fable-5-1",
    "bedrock/us-gov-west-1/anthropic.claude-fable-5-1": "anthropic.claude-fable-5-1",
    "bedrock/us-gov-east-1/anthropic.claude-fable-5-1": "anthropic.claude-fable-5-1",
    "us-gov.nvidia.nemotron-nano-9b-v2": "nvidia.nemotron-nano-9b-v2",
    "bedrock/us-gov-west-1/nvidia.nemotron-nano-9b-v2": "nvidia.nemotron-nano-9b-v2",
    "bedrock/us-gov-east-1/nvidia.nemotron-nano-9b-v2": "nvidia.nemotron-nano-9b-v2",
    "us-gov.xai.grok-4.6": "us.xai.grok-4.6",
    "bedrock_mantle/us-gov-west-1/xai.grok-4.6": "bedrock_mantle/xai.grok-4.6",
    "bedrock_mantle/us-gov-east-1/xai.grok-4.6": "bedrock_mantle/xai.grok-4.6",
    "bedrock/us-gov-west-1/amazon.nova-2-multimodal-embeddings-v1:0": "amazon.nova-2-multimodal-embeddings-v1:0",
    "bedrock/us-gov-west-1/amazon.nova-lite-v1:0": "amazon.nova-lite-v1:0",
    "bedrock/us-gov-west-1/amazon.nova-micro-v1:0": "amazon.nova-micro-v1:0",
    "bedrock_mantle/us-gov-west-1/google.gemma-4-e2b": "bedrock_mantle/google.gemma-4-e2b",
    "bedrock_mantle/us-gov-west-1/google.gemma-4-26b-a4b": "bedrock_mantle/google.gemma-4-26b-a4b",
    "bedrock_mantle/us-gov-west-1/google.gemma-4-31b": "bedrock_mantle/google.gemma-4-31b",
    "bedrock_mantle/us-gov-west-1/openai.gpt-oss-20b": "bedrock_mantle/openai.gpt-oss-20b",
    "bedrock_mantle/us-gov-east-1/openai.gpt-oss-20b": "bedrock_mantle/openai.gpt-oss-20b",
    "bedrock_mantle/us-gov-west-1/openai.gpt-oss-120b": "bedrock_mantle/openai.gpt-oss-120b",
    "bedrock_mantle/us-gov-east-1/openai.gpt-oss-120b": "bedrock_mantle/openai.gpt-oss-120b",
}


def _non_pricing_fields(info):
    return {k: v for k, v in info.items() if "cost" not in k and k not in ("litellm_provider", "source")}


@pytest.mark.parametrize("gov_key", GOV_ROW_SOURCES)
def test_usgov_rows_keep_commercial_limits_and_capabilities(model_data, gov_key):
    """A gov row differs from the commercial row it mirrors only in price and
    provider: context limits, mode, and capability flags stay identical, so a
    hand-copied row cannot silently drop tool calling or shrink the context window.
    """
    gov = model_data[gov_key]
    assert _non_pricing_fields(gov) == _non_pricing_fields(model_data[GOV_ROW_SOURCES[gov_key]])
    assert "search_context_cost_per_query" not in gov
    assert "source" not in gov


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
