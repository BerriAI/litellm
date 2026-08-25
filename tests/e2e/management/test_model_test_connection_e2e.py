"""Live e2e for POST /health/test_connection, the API behind the Admin UI's
Test Connection button on the add-model form.

The covered cell is a responses-mode Bedrock Mantle deployment: exactly this
shape 500ed on a functools.partial acompletion conflict before v1.91.0 while
every chat-mode probe stayed green, so the happy path asserts a real success
verdict from the live provider rather than just a 200 envelope. The region is a
literal because the endpoint rejects request-supplied os.environ/ references;
credentials fall through to the proxy's own environment (bearer token locally,
pod identity in CI).
"""

from __future__ import annotations

import pytest

from e2e_http import unwrap
from management_client import ManagementClient
from models import ConnectionTestBody, LiteLLMParamsBody

pytestmark = pytest.mark.e2e

MANTLE_RESPONSES_BACKEND = "bedrock_mantle/openai.gpt-5.6-luna"
MANTLE_REGION = "us-east-1"


class TestModelTestConnection:
    @pytest.mark.covers("mgmt.model.test_connection.happy_path")
    def test_bedrock_mantle_responses_connection_succeeds(self, client: ManagementClient) -> None:
        response = unwrap(
            client.connection_test(
                ConnectionTestBody(
                    litellm_params=LiteLLMParamsBody(
                        model=MANTLE_RESPONSES_BACKEND, aws_region_name=MANTLE_REGION
                    ),
                    mode="responses",
                )
            )
        )

        error = response.result.error if response.result else None
        assert response.status == "success", f"test_connection reported an error: {error}"
