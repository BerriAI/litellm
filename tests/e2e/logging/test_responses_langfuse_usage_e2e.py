"""Live e2e: Langfuse token usage on the /v1/responses path.

Covers logging.langfuse.success.logs_usage: a key-scoped ``langfuse``
(classic SDK) callback must log non-zero input/output tokens to the Langfuse
generation for a /v1/responses call. A ResponsesAPIResponse carries
ResponseAPIUsage (input_tokens/output_tokens), which the logger must
normalize before building usage_details; without the normalization Langfuse
shows 0 input / 0 output while cost is still right.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from lifecycle import ResourceManager
from logging_client import (
    LangfuseCreds,
    LoggingClient,
    first_ok,
    load_langfuse_creds,
)
from models import (
    KeyLoggingCallback,
    KeyLoggingCallbackVars,
    KeyMetadata,
    ResponsesApiResponse,
)

pytestmark = pytest.mark.e2e


class _LangfuseUsageDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input: int | None = None
    output: int | None = None
    total: int | None = None
    cache_read_input_tokens: int | None = None


_USAGE_DETAILS_ADAPTER: TypeAdapter[_LangfuseUsageDetails] = TypeAdapter(_LangfuseUsageDetails)


@pytest.fixture(scope="session")
def langfuse_creds() -> LangfuseCreds:
    return load_langfuse_creds()


class TestResponsesLangfuseUsage:
    @pytest.mark.covers("logging.langfuse.success.logs_usage", exercised_on=["responses"])
    def test_responses_call_logs_nonzero_token_usage(
        self, client: LoggingClient, langfuse_creds: LangfuseCreds, resources: ResourceManager
    ) -> None:
        key_alias = f"lf-resp-key-{unique_marker()}"
        key = client.key_with_alias(
            key_alias,
            models=[CHEAP_OPENAI_MODEL],
            metadata=KeyMetadata(
                logging=[
                    KeyLoggingCallback(
                        callback_name="langfuse",
                        callback_type="success",
                        callback_vars=KeyLoggingCallbackVars(
                            langfuse_public_key=langfuse_creds.public_key,
                            langfuse_secret_key=langfuse_creds.secret_key,
                            langfuse_host=langfuse_creds.host,
                        ),
                    )
                ]
            ),
        )
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.responses_raw(
                key, CHEAP_OPENAI_MODEL, f"reply with one word {marker}", max_output_tokens=64
            ),
        )
        response = ResponsesApiResponse.model_validate_json(outcome.body)
        assert response.usage is not None and response.usage.input_tokens > 0, (
            f"the /v1/responses body must report input_tokens, got {outcome.body[:300]}"
        )

        observation = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=marker,
        )
        assert observation is not None, (
            f"the key's /v1/responses call (marker {marker}) never reached Langfuse within the deadline"
        )
        assert observation.usage_details is not None, (
            f"the Langfuse generation must carry usageDetails, got {observation.model_dump_json()}"
        )
        details = _USAGE_DETAILS_ADAPTER.validate_python(observation.usage_details)
        cache_read = details.cache_read_input_tokens or 0
        assert details.input is not None and details.input > 0, (
            f"usageDetails.input must be non-zero, got {details.model_dump()}"
        )
        assert details.input + cache_read == response.usage.input_tokens, (
            f"usageDetails.input {details.input} + cache_read {cache_read} must equal the proxy's "
            f"input_tokens {response.usage.input_tokens}"
        )
        assert details.output == response.usage.output_tokens, (
            f"usageDetails.output {details.output} must equal the proxy's "
            f"output_tokens {response.usage.output_tokens}"
        )
