"""Live e2e for POST /health/test_connection, the API behind the Admin UI's
Test Connection button on the add-model form.

The covered cell is a responses-mode Bedrock Mantle deployment: exactly this
shape 500ed on a functools.partial acompletion conflict before v1.91.0 while
every chat-mode probe stayed green, so the happy path asserts a real success
verdict from the live provider rather than just a 200 envelope. The region is a
literal because the endpoint rejects request-supplied os.environ/ references;
credentials fall through to the proxy's own environment (bearer token locally,
pod identity in CI).

The endpoint caps every probe at HEALTH_CHECK_TIMEOUT_SECONDS and answers a
timed-out probe with HTTP 200 and an in-body "Timeout exceeded", which the
harness's status-code retry policy cannot see. A Mantle probe can hit that cap
transiently while the rest of the suite saturates the same AWS account, so only
that exact error is retried here; any other error verdict fails immediately.
"""

from __future__ import annotations

import time

import pytest

from e2e_http import unwrap
from management_client import ManagementClient
from models import ConnectionTestBody, ConnectionTestResponse, LiteLLMParamsBody

pytestmark = pytest.mark.e2e

MANTLE_RESPONSES_BACKEND = "bedrock_mantle/openai.gpt-5.6-luna"
MANTLE_REGION = "us-east-1"
PROBE_TIMEOUT_ERROR = "Timeout exceeded"
PROBE_ATTEMPTS = 3
PROBE_RETRY_SLEEP_SECONDS = 30


def _probe_mantle(client: ManagementClient) -> ConnectionTestResponse:
    return unwrap(
        client.connection_test(
            ConnectionTestBody(
                litellm_params=LiteLLMParamsBody(
                    model=MANTLE_RESPONSES_BACKEND, aws_region_name=MANTLE_REGION
                ),
                mode="responses",
            )
        )
    )


class TestModelTestConnection:
    @pytest.mark.covers("mgmt.model.test_connection.happy_path")
    def test_bedrock_mantle_responses_connection_succeeds(self, client: ManagementClient) -> None:
        for attempt in range(1, PROBE_ATTEMPTS + 1):
            response = _probe_mantle(client)
            if response.status == "success":
                return
            error = response.result.error if response.result else None
            assert error == PROBE_TIMEOUT_ERROR, f"test_connection reported an error: {error}"
            if attempt < PROBE_ATTEMPTS:
                print(
                    f"test_connection probe timed out; retry {attempt}/{PROBE_ATTEMPTS - 1}"
                    f" in {PROBE_RETRY_SLEEP_SECONDS}s",
                    flush=True,
                )
                time.sleep(PROBE_RETRY_SLEEP_SECONDS)
        pytest.fail(f"test_connection timed out on all {PROBE_ATTEMPTS} attempts")
