"""Live e2e: gcs_bucket log delivery for successful calls.

Covers logging.gcs_bucket.success.writes_object: one successful
/chat/completions call must land in the real GCS bucket as exactly one
StandardLoggingPayload record (GCS is the audit-trail parallel to S3 for GCP
deployments). Delivery is judged on what is actually readable in the bucket:
the proxy writes with its production service account, and the test reads the
record back through the GCS JSON API - covering both the batched NDJSON layout
(the default) and the per-request object layout.

Both halves of the contract are asserted: the recorded state (the proxy
reports the GCSBucketLogger callback active via /health/readiness/details -
note gcs_bucket is enterprise-gated, so this also requires a license) and the
enforced behavior (the record in the bucket, cost cross-checked against the
x-litellm-response-cost header of the very response the caller received).
"""

from __future__ import annotations

import math

import pytest

from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from gcs_reader import GcsLogReader, build_gcs_reader, utc_now
from lifecycle import ResourceManager
from logging_client import LoggingClient, completion_response_id, first_ok, readiness_details_body

pytestmark = pytest.mark.e2e

#: The active gcs_bucket callback's name in /health/readiness/details success_callbacks.
GCS_LOGGER_NAME = "GCSBucketLogger"


@pytest.fixture(scope="session")
def gcs_logs() -> GcsLogReader:
    return build_gcs_reader()


def _assert_gcs_configured(client: LoggingClient) -> None:
    """Recorded state: the proxy reports the gcs_bucket callback among its
    active callbacks, so a missing destination config (or a missing enterprise
    license - gcs_bucket refuses to initialize without one) fails here, before
    any delivery-based assertion can time out confusingly."""
    body = readiness_details_body(client)
    assert GCS_LOGGER_NAME in body, (
        f"the proxy must report the {GCS_LOGGER_NAME} callback active "
        f"(litellm_settings.callbacks: ['gcs_bucket'] + GCS_BUCKET_NAME env + enterprise license); "
        f"got: {body[:400]}"
    )


class TestGcsLogDelivery:
    @pytest.mark.covers("logging.gcs_bucket.success.writes_object", exercised_on=["chat_completions"])
    def test_chat_completions_writes_one_success_record(
        self, client: LoggingClient, gcs_logs: GcsLogReader, resources: ResourceManager
    ) -> None:
        """One successful non-streaming /chat/completions call must be
        readable back from the bucket as exactly one payload record carrying
        the model group, the token counts, and the same cost the caller's
        response header reported."""
        _assert_gcs_configured(client)

        alias = f"gcs-chat-{unique_marker()}"
        key = client.key_with_alias(alias, models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        since = utc_now()
        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.chat_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=16),
        )
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
        )
        body_id = completion_response_id(outcome.body)
        assert body_id is not None, "the completion body must carry an id (it names the gcs record)"

        records = gcs_logs.poll_records_for_response_id(body_id, since=since)
        assert records, f"no gcs record for response {body_id} was readable from the bucket within the deadline"
        assert len(records) == 1, (
            f"expected exactly ONE gcs record for the call, got {len(records)} - "
            "more than one record for one call is the duplicate-delivery bug"
        )
        record = records[0]
        assert record.id == body_id, f"record id must be the response id, got {record.id!r}"
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
