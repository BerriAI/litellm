"""Live e2e: s3_v2 log delivery for successful and failed calls.

Covers logging.s3.success.writes_object and logging.s3.failure.writes_object:
one /chat/completions call must land in the real S3 bucket as exactly one
StandardLoggingPayload object (the primary audit trail; the batch flush must
neither drop nor duplicate it), and a failed call must be persisted the same
way for compliance. Delivery is judged on what is actually in the bucket: the
proxy writes with its production credentials and the test lists and reads the
objects back.

Both halves of the contract are asserted: the recorded state (the proxy
reports the S3Logger callback active via /health/readiness/details) and the
enforced behavior (the object in the bucket, with the cost cross-checked
against the x-litellm-response-cost header of the very response the caller
received).

The suite requires ``s3_callback_params.s3_use_key_prefix: true`` on the proxy,
which keys objects as ``{key_alias}/{date}/time-..._{id}.json`` - a unique key
alias per test turns the poll into a cheap prefix listing.
"""

from __future__ import annotations

import math
import time

import pytest

from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from lifecycle import ResourceManager
from logging_client import (
    INVALID_UPSTREAM_API_KEY,
    LoggingClient,
    completion_response_id,
    first_ok,
    readiness_details_body,
)
from models import LiteLLMParamsBody
from s3_reader import S3LogReader, build_s3_reader

pytestmark = pytest.mark.e2e

#: The active s3_v2 callback's name in /health/readiness/details success_callbacks.
S3_LOGGER_NAME = "S3Logger"


@pytest.fixture(scope="session")
def s3_logs() -> S3LogReader:
    return build_s3_reader()


def _assert_s3_configured(client: LoggingClient) -> None:
    """Recorded state: the proxy reports the s3_v2 callback among its active
    callbacks, so a missing destination config fails here, before any
    delivery-based assertion can time out confusingly."""
    body = readiness_details_body(client)
    assert S3_LOGGER_NAME in body, (
        f"the proxy must report the {S3_LOGGER_NAME} callback active "
        f"(litellm_settings.callbacks: ['s3_v2'] + s3_callback_params in the proxy config); "
        f"got: {body[:400]}"
    )


class TestS3LogDelivery:
    @pytest.mark.covers("logging.s3.success.writes_object", exercised_on=["chat_completions"])
    def test_chat_completions_writes_one_success_object(
        self, client: LoggingClient, s3_logs: S3LogReader, resources: ResourceManager
    ) -> None:
        """One successful non-streaming /chat/completions call must land in
        the bucket as exactly one payload object carrying the model group, the
        token counts, and the same cost the caller's response header reported."""
        _assert_s3_configured(client)

        alias = f"s3-chat-{unique_marker()}"
        key = client.key_with_alias(alias, models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.chat_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=16),
        )
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
        )
        body_id = completion_response_id(outcome.body)
        assert body_id is not None, "the completion body must carry an id (it names the s3 object)"

        records = s3_logs.poll_records(prefix=f"{alias}/", predicate=lambda r: r.id == body_id)
        assert records, (
            f"no s3 object for response {body_id} under prefix {alias}/ reached the bucket within the deadline"
        )
        assert len(records) == 1, (
            f"expected exactly ONE s3 object for the call, got {len(records)} - "
            "more than one object for one call is the duplicate-delivery bug"
        )
        record = records[0]
        assert record.status == "success", f"payload status must be success, got {record.status!r}"
        assert record.model_group == CHEAP_ANTHROPIC_MODEL, (
            f"payload model_group must be {CHEAP_ANTHROPIC_MODEL!r}, got {record.model_group!r}"
        )
        assert record.total_tokens is not None and record.total_tokens > 0, (
            f"payload must count real tokens, got {record.total_tokens!r}"
        )
        assert record.response_cost is not None and math.isclose(
            record.response_cost, outcome.response_cost, rel_tol=1e-9
        ), f"payload response_cost {record.response_cost!r} must equal the header cost {outcome.response_cost}"

    @pytest.mark.covers("logging.s3.failure.writes_object", exercised_on=["chat_completions"])
    def test_chat_completions_failure_writes_one_object(
        self, client: LoggingClient, s3_logs: S3LogReader, resources: ResourceManager
    ) -> None:
        """A call that fails at the provider must be persisted to the bucket as
        exactly one failure payload carrying the provider error - failed calls
        are part of the audit trail, not an exemption from it.

        A deployment with an invalid upstream key lets the request pass proxy
        auth and fail at the provider (the same lever as the OTEL error test).
        Proxy-side rejections during key/model propagation can also ship
        failure payloads under this alias, but without a model_group and
        without the provider error, so the read-back keys on both: only
        provider-reaching calls carry them, and with this key every one of
        those is the AnthropicException that ends the send loop."""
        _assert_s3_configured(client)

        model_name = f"s3-err-{unique_marker()}"
        model_id = client.create_model(
            model_name,
            LiteLLMParamsBody(model="anthropic/claude-haiku-4-5", api_key=INVALID_UPSTREAM_API_KEY),
        )
        resources.defer(lambda: client.delete_model(model_id))
        alias = f"s3-err-key-{unique_marker()}"
        key = client.key_with_alias(alias, models=[model_name])
        resources.defer(lambda: client.delete_key(key))

        deadline = time.monotonic() + client.proxy.poll_timeout
        while True:
            outcome = client.chat_raw(key, model_name, "trigger an upstream auth failure", max_tokens=16)
            assert not outcome.ok, "the call must fail; the deployment's upstream key is invalid"
            assert outcome.status_code != -1, (
                "network failure between the test and the proxy while provoking the provider "
                "failure; retrying now could double-log the failure payload and falsely trip "
                f"the exactly-one assertion - fix the rig connectivity first: {outcome.body[:200]}"
            )
            if "AnthropicException" in outcome.body or time.monotonic() >= deadline:
                break
            time.sleep(client.proxy.poll_interval)
        assert "AnthropicException" in outcome.body, (
            "never saw the upstream provider failure before the deadline; the key may still be "
            f"propagating - last outcome {outcome.status_code}: {outcome.body[:200]}"
        )
        assert outcome.status_code == 401, (
            f"an upstream auth failure must map to 401, got {outcome.status_code}: {outcome.body[:200]}"
        )

        records = s3_logs.poll_records(
            prefix=f"{alias}/",
            predicate=lambda r: (
                r.status == "failure" and r.model_group == model_name and "AnthropicException" in (r.error_str or "")
            ),
        )
        assert records, (
            f"no failure object for {model_name} under prefix {alias}/ reached the bucket within the deadline"
        )
        assert len(records) == 1, f"expected exactly ONE failure object for the call, got {len(records)}"
        record = records[0]
        assert record.error_str is not None and "AnthropicException" in record.error_str, (
            f"the persisted failure must carry the provider error, got error_str={record.error_str!r}"
        )
        assert not record.response_cost, f"a failed call must not be billed, got response_cost={record.response_cost!r}"
