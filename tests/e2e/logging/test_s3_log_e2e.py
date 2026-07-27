"""Live e2e: S3 object delivery for successful and for failed calls.

Covers logging.s3.success.writes_object and logging.s3.failure.writes_object.
The S3 bucket is the audit trail: for a lot of deployments it, not the database,
is the record of what was asked and what it cost, and a compliance answer needs
the failed calls in it too, because a call that died upstream still left the
prompt with a provider. So the promise is an object per call, holding that
call's payload - and the integration batches uploads behind a flush interval and
swallows upload errors, which is exactly the shape of a bug that looks healthy
from the proxy and loses every record.

Delivery is judged on what is in the bucket. The proxy uploads with its own AWS
credentials as in production, and the tests fetch the object back with the AWS
SDK (see s3_reader.py), so a dropped upload, a misaddressed bucket, or a payload
that lost the error fails here. Both halves of the contract are asserted for
each case: the recorded state (the proxy reports the s3_v2 callback registered
for that event) and the enforced behavior (the object in the bucket, keyed by
the id of the very call the caller made, with the cost cross-checked against
that response's x-litellm-response-cost header on the success path and the
provider's error preserved on the failure path).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from callback_config import callback_enabled, registered_callbacks
from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from logging_client import (
    INVALID_UPSTREAM_API_KEY,
    LoggingClient,
    completion_response_id,
    first_ok,
)
from models import LiteLLMParamsBody
from s3_reader import S3LogObject, S3LogsReader, build_s3_logs_reader

pytestmark = pytest.mark.e2e

#: The integration's name in litellm's callback settings.
S3_CALLBACK_NAME = "s3_v2"
#: A deployment litellm routes to OpenAI, so the invalid key is rejected by
#: OpenAI rather than by litellm's own request validation.
UPSTREAM_MODEL = "openai/gpt-4o-mini"
#: Present only in OpenAI's own rejection, never in the gateway's auth error, so
#: it tells an upstream 401 (the behavior under test) apart from a proxy 401.
UPSTREAM_REJECTION = "OpenAIException"


@pytest.fixture(scope="session")
def s3_logs() -> S3LogsReader:
    """Read-back client for the real bucket the proxy's s3_callback_params name."""
    return build_s3_logs_reader()


@pytest.fixture(scope="module")
def s3_logging(client: LoggingClient) -> Iterator[None]:
    """The proxy must ship both successful and failed calls to S3 for the
    duration of this module. A deployment that already does is left as it is."""
    yield from callback_enabled(client, S3_CALLBACK_NAME, events=("success", "failure"))


def _first_upstream_rejection(client: LoggingClient, key: str, model: str, prompt: str) -> StreamingResponse:
    """The first response that is the provider's rejection rather than the
    gateway's own. A freshly created key can briefly 401 at the gateway until the
    data plane's auth cache picks it up, and that 401 never reaches the provider,
    so it would leave nothing for the integration to write; retry to a deadline
    until the body carries the upstream error.

    A gateway 401 is itself logged as a failure and carries the same prompt, so a
    discarded attempt is indistinguishable from the accepted one by prompt alone.
    The caller must therefore correlate on the returned response's
    x-litellm-call-id, which is unique per attempt and is what the object key
    ends in."""
    deadline = time.monotonic() + client.proxy.poll_timeout
    while True:
        outcome = client.chat_raw(key, model, prompt, max_tokens=16)
        if UPSTREAM_REJECTION in outcome.body:
            return outcome
        if time.monotonic() >= deadline:
            pytest.fail(
                f"the call never reached the provider (status {outcome.status_code}): {outcome.body[:300]}"
            )
        time.sleep(client.proxy.poll_interval)


def _sole_object(
    s3_logs: S3LogsReader, resources: ResourceManager, call_id: str
) -> S3LogObject:
    """The one object the bucket holds for the call, queued for deletion the
    moment it is found so a later assertion failure still cleans up."""
    written = s3_logs.poll_objects_for_call(call_id)
    for entry in written:
        resources.defer(lambda key=entry.key: s3_logs.delete(key))
    assert written, f"no S3 object for call {call_id} appeared in the bucket within the deadline"
    assert len(written) == 1, (
        f"expected exactly ONE S3 object for the call, got {len(written)}: {[o.key for o in written]}"
    )
    return written[0]


class TestS3LogDelivery:
    @pytest.mark.covers("logging.s3.success.writes_object", exercised_on=["chat_completions"])
    def test_chat_completions_success_writes_one_object(
        self,
        client: LoggingClient,
        s3_logs: S3LogsReader,
        resources: ResourceManager,
        s3_logging: None,
    ) -> None:
        """One successful /chat/completions call must leave exactly one object in
        the bucket, keyed by that completion's id, holding the prompt, the token
        counts and the same cost the caller was charged."""
        assert S3_CALLBACK_NAME in registered_callbacks(client, "success"), (
            "the proxy must report the s3_v2 callback registered for successful calls "
            "before delivery can be asserted"
        )

        marker = unique_marker()
        key = client.key_with_alias(f"s3-success-{marker}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        prompt = f"reply with one word {marker}"
        outcome = first_ok(
            client, lambda: client.chat_raw(key, CHEAP_ANTHROPIC_MODEL, prompt, max_tokens=16)
        )
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"the response must report x-litellm-response-cost, got {outcome.response_cost!r}"
        )
        completion_id = completion_response_id(outcome.body)
        assert completion_id is not None, f"the completion must carry an id: {outcome.body[:300]}"

        record = _sole_object(s3_logs, resources, completion_id).record
        assert record.id == completion_id, (
            f"the object must hold the payload of this completion, got id {record.id!r}"
        )
        assert record.status == "success", f"payload status must be success, got {record.status!r}"
        assert record.call_type == "acompletion", f"payload call_type must be acompletion, got {record.call_type!r}"
        assert record.model_group == CHEAP_ANTHROPIC_MODEL, (
            f"payload model_group must be {CHEAP_ANTHROPIC_MODEL!r}, got {record.model_group!r}"
        )
        assert record.total_tokens > 0, f"payload must count real tokens, got {record.total_tokens}"
        assert record.response_cost == outcome.response_cost, (
            f"the stored cost {record.response_cost} must equal the cost the caller was charged "
            f"{outcome.response_cost}"
        )
        assert any(marker in message.content for message in record.messages), (
            f"the stored object must carry this call's own prompt, got {record.messages!r}"
        )

    @pytest.mark.covers("logging.s3.failure.writes_object", exercised_on=["chat_completions"])
    def test_chat_completions_failure_writes_one_object(
        self,
        client: LoggingClient,
        s3_logs: S3LogsReader,
        resources: ResourceManager,
        s3_logging: None,
    ) -> None:
        """A /chat/completions call the provider rejects must still leave exactly
        one object in the bucket, keyed by the litellm call id the caller was
        handed, recording the failure and the upstream error rather than being
        dropped with the response."""
        assert S3_CALLBACK_NAME in registered_callbacks(client, "failure"), (
            "the proxy must report the s3_v2 callback registered for failed calls "
            "before delivery can be asserted"
        )

        marker = unique_marker()
        model_name = f"s3-failure-{marker}"
        model_id = client.create_model(
            model_name, LiteLLMParamsBody(model=UPSTREAM_MODEL, api_key=INVALID_UPSTREAM_API_KEY)
        )
        resources.defer(lambda: client.delete_model(model_id))

        key = client.key_with_alias(f"s3-failure-{marker}", models=[model_name])
        resources.defer(lambda: client.delete_key(key))

        prompt = f"reply with one word {marker}"
        outcome = _first_upstream_rejection(client, key, model_name, prompt)
        assert outcome.status_code == 401, (
            f"an invalid upstream key must surface as the provider's 401, got {outcome.status_code}: "
            f"{outcome.body[:300]}"
        )
        assert outcome.call_id is not None, "the failed response must still carry x-litellm-call-id"

        record = _sole_object(s3_logs, resources, outcome.call_id).record
        assert record.id == outcome.call_id, (
            f"the object must hold the payload of this call, got id {record.id!r}"
        )
        assert record.status == "failure", f"payload status must be failure, got {record.status!r}"
        assert record.call_type == "acompletion", f"payload call_type must be acompletion, got {record.call_type!r}"
        assert record.model_group == model_name, (
            f"payload model_group must be {model_name!r}, got {record.model_group!r}"
        )
        assert record.error_information.error_code == "401", (
            f"the stored payload must carry the provider's status code, got "
            f"{record.error_information.error_code!r}"
        )
        assert record.error_information.error_class == "AuthenticationError", (
            f"the stored payload must name the upstream error class, got "
            f"{record.error_information.error_class!r}"
        )
        assert record.error_information.llm_provider == "openai", (
            f"the stored payload must name the provider that rejected the call, got "
            f"{record.error_information.llm_provider!r}"
        )
        assert record.error_str is not None and UPSTREAM_REJECTION in record.error_str, (
            f"the stored payload's error_str must carry the provider's own message, got {record.error_str!r}"
        )
        assert record.response_cost == 0, (
            "a failed call bills nothing, so a non-zero cost means the failure was archived as if it "
            f"had succeeded; got {record.response_cost}"
        )
        assert record.total_tokens == 0, (
            f"a failed call consumes no tokens, got {record.total_tokens}"
        )
        assert any(marker in message.content for message in record.messages), (
            f"the stored object must carry the failed call's own prompt, got {record.messages!r}"
        )
