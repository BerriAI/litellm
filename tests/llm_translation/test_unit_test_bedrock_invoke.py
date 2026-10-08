import traceback
from dotenv import load_dotenv
import litellm.types
import pytest
from litellm import AmazonInvokeConfig
import json

load_dotenv()
import io

from unittest.mock import AsyncMock, Mock, patch


# Initialize the transformer
@pytest.fixture
def bedrock_transformer():
    return AmazonInvokeConfig()


def test_transform_request_meta_llama(bedrock_transformer):
    """Test request transformation for Meta/Llama"""
    messages = [{"role": "user", "content": "Hello"}]

    result = bedrock_transformer.transform_request(
        model="meta.llama2-70b",
        messages=messages,
        optional_params={"max_gen_len": 2048},
        litellm_params={},
        headers={},
    )

    print("transformed request for invoke meta llama=", json.dumps(result, indent=4))
    expected_result = {"prompt": "Hello", "max_gen_len": 2048}
    assert result == expected_result
