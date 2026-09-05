"""
Pin the commercial Bedrock Mantle gpt-oss rows to the standard-tier Mantle SKUs in the AWS
us-east-1 Bedrock offer file (USE1-openai.gpt-oss-*-mantle-{input,output}-tokens-standard):
https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.csv
"""

from functools import cache
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
PRICE_FILES: Final = (
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
)
PRICE_FILE_IDS: Final = ("main", "backup")

PRICE_MAP: Final = TypeAdapter(dict[str, dict[str, object]])

MANTLE_GPT_OSS_OFFER_FILE_RATES: Final = {
    "openai.gpt-oss-20b": (7e-08, 3e-07),
    "openai.gpt-oss-120b": (1.5e-07, 6e-07),
    "openai.gpt-oss-safeguard-20b": (7e-08, 2e-07),
    "openai.gpt-oss-safeguard-120b": (1.5e-07, 6e-07),
}

CONVERSE_TWIN: Final = {
    "openai.gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "openai.gpt-oss-safeguard-20b": "openai.gpt-oss-safeguard-20b",
    "openai.gpt-oss-safeguard-120b": "openai.gpt-oss-safeguard-120b",
}


@cache
def load_price_map(path: Path) -> dict[str, dict[str, object]]:
    return PRICE_MAP.validate_json(path.read_bytes())


@pytest.mark.parametrize("price_file", PRICE_FILES, ids=PRICE_FILE_IDS)
@pytest.mark.parametrize("model", MANTLE_GPT_OSS_OFFER_FILE_RATES)
def test_mantle_gpt_oss_rows_match_offer_file(price_file: Path, model: str) -> None:
    info: Final = load_price_map(price_file)[f"bedrock_mantle/{model}"]
    expected_input, expected_output = MANTLE_GPT_OSS_OFFER_FILE_RATES[model]
    assert info["input_cost_per_token"] == expected_input
    assert info["output_cost_per_token"] == expected_output
    assert info["litellm_provider"] == "bedrock_mantle"


@pytest.mark.parametrize("price_file", PRICE_FILES, ids=PRICE_FILE_IDS)
@pytest.mark.parametrize("model", CONVERSE_TWIN)
def test_mantle_gpt_oss_rows_match_converse_rows(price_file: Path, model: str) -> None:
    model_data: Final = load_price_map(price_file)
    mantle: Final = model_data[f"bedrock_mantle/{model}"]
    converse: Final = model_data[CONVERSE_TWIN[model]]
    assert mantle["input_cost_per_token"] == converse["input_cost_per_token"]
    assert mantle["output_cost_per_token"] == converse["output_cost_per_token"]
