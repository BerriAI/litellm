import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.bedrock.common_utils import (
    BedrockModelInfo,
    apply_embedded_bedrock_region_from_model_path,
)


def test_apply_embedded_bedrock_region_strips_prefix_and_sets_region():
    optional_params: dict = {}
    out = apply_embedded_bedrock_region_from_model_path(
        "bedrock/us-west-2/mistral.mistral-7b-instruct-v0:2", optional_params
    )
    assert out == "mistral.mistral-7b-instruct-v0:2"
    assert optional_params.get("aws_region_name") == "us-west-2"


def test_apply_embedded_bedrock_region_respects_explicit_aws_region_name():
    optional_params = {"aws_region_name": "us-east-1"}
    out = apply_embedded_bedrock_region_from_model_path(
        "bedrock/us-west-2/mistral.mistral-7b-instruct-v0:2", optional_params
    )
    assert out == "mistral.mistral-7b-instruct-v0:2"
    assert optional_params.get("aws_region_name") == "us-east-1"


def test_deepseek_cris():
    bedrock_model_info = BedrockModelInfo
    bedrock_route = bedrock_model_info.get_bedrock_route(
        model="bedrock/us.deepseek.r1-v1:0"
    )
    assert bedrock_route == "converse"
