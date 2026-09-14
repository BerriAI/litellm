import os
from typing import Final

from bedrock_edge import TEST_ACCESS_KEY, TEST_SECRET_KEY, recording_credentials
from e2e_config import FIXTURE_MODE_RAW, provider_edge_base
from fixture_mode import parse_fixture_mode
from models import LiteLLMParamsBody

BEDROCK_BACKEND: Final = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"


def bedrock_params() -> LiteLLMParamsBody:
    if parse_fixture_mode(FIXTURE_MODE_RAW) == "live":
        return LiteLLMParamsBody(
            model=BEDROCK_BACKEND,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
        )
    if parse_fixture_mode(FIXTURE_MODE_RAW) == "record":
        recording_credentials()
    region: Final = os.environ.get("E2E_BEDROCK_REGION", "us-east-1")
    return LiteLLMParamsBody(
        model=BEDROCK_BACKEND,
        api_base=provider_edge_base(f"bedrock-{region}"),
        aws_access_key_id=TEST_ACCESS_KEY,
        aws_secret_access_key=TEST_SECRET_KEY,
        aws_region_name=region,
    )
