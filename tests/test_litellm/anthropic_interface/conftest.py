import os

import pytest


@pytest.fixture(autouse=True)
def skip_live_bedrock_tests_without_credentials(request):
    if (
        request.node.path.name == "test_bedrock_rust_bridge_e2e.py"
        and not os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    ):
        pytest.skip("requires AWS_BEARER_TOKEN_BEDROCK")
