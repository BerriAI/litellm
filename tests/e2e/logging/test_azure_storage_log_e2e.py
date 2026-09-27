"""Live e2e: azure_storage log delivery for successful, streamed, and failed calls.

Covers logging.azure_storage.success.writes_object,
logging.azure_storage.stream.writes_object, and
logging.azure_storage.failure.writes_object: one /chat/completions or /v1/messages
call must land in the real Azure Data Lake filesystem as exactly one
StandardLoggingPayload object under {date}/{response_id}.json (the primary audit
trail; the batch flush must neither drop nor duplicate it), and a failed call
must be persisted the same way for compliance. Delivery is judged on what is
actually in the filesystem: the proxy writes with its production credentials and
the test lists and reads the objects back.

Both halves of the contract are asserted: the recorded state (the proxy reports
the AzureBlobStorageLogger callback active via /health/readiness/details) and
the enforced behavior (the object in the filesystem, with the cost cross-checked
against the x-litellm-response-cost header of the very response the caller
received).

The lane is opt-in (marker azure_storage, env E2E_AZURE_STORAGE): the gateway
must be booted with litellm_settings.callbacks: ["azure_storage"], a license,
and the AZURE_STORAGE_* env; the same tests prove whichever credential the
gateway carries (account key or Entra ID).
"""

from __future__ import annotations

import json
import time
from typing import cast

import pytest

from azure_storage_reader import AzureStorageLogReader, build_azure_storage_reader
from e2e_config import CHEAP_ANTHROPIC_MODEL, unique_marker
from lifecycle import ResourceManager
from logging_client import (
    INVALID_UPSTREAM_API_KEY,
    LoggingClient,
    completion_response_id,
    costs_agree,
    first_ok,
    readiness_details_body,
)
from models import LiteLLMParamsBody

pytestmark = [pytest.mark.e2e, pytest.mark.azure_storage]

#: The active azure_storage callback's name in /health/readiness/details success_callbacks.
AZURE_LOGGER_NAME = "AzureBlobStorageLogger"


@pytest.fixture(scope="session")
def azure_logs() -> AzureStorageLogReader:
    return build_azure_storage_reader()


def _assert_azure_configured(client: LoggingClient) -> None:
    """Recorded state: the proxy reports the azure_storage callback among its
    active callbacks, so a missing destination config fails here, before any
    delivery-based assertion can time out confusingly."""
    body = readiness_details_body(client)
    assert AZURE_LOGGER_NAME in body, (
        f"the proxy must report the {AZURE_LOGGER_NAME} callback active "
        f"(litellm_settings.callbacks: ['azure_storage'] + AZURE_STORAGE_* env on the proxy); "
        f"got: {body[:400]}"
    )


def _stream_response_id(stream_events: list[str]) -> str | None:
    for payload in stream_events:
        try:
            chunk: dict[str, object] = cast(dict[str, object], json.loads(payload))
        except json.JSONDecodeError:
            continue
        chunk_id = chunk.get("id")
        if isinstance(chunk_id, str) and chunk_id:
            return chunk_id
    return None


class TestAzureStorageLogDelivery:
    @pytest.mark.covers("logging.azure_storage.success.writes_object", exercised_on=["chat_completions"])
    def test_chat_completions_writes_one_success_object(
        self, client: LoggingClient, azure_logs: AzureStorageLogReader, resources: ResourceManager
    ) -> None:
        """One successful non-streaming /chat/completions call must land in the
        filesystem as exactly one payload object carrying the model group, the
        token counts, and the same cost the caller's response header reported."""
        _assert_azure_configured(client)

        key = client.key_with_alias(f"az-chat-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
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
        assert body_id is not None, "the completion body must carry an id (it names the azure object)"

        record = azure_logs.poll_record(body_id)
        assert record.id == body_id, f"payload id must be {body_id!r}, got {record.id!r}"
        assert record.status == "success", f"payload status must be success, got {record.status!r}"
        assert record.model_group == CHEAP_ANTHROPIC_MODEL, (
            f"payload model_group must be {CHEAP_ANTHROPIC_MODEL!r}, got {record.model_group!r}"
        )
        assert record.response_cost is not None and costs_agree(outcome.response_cost, record.response_cost), (
            f"payload response_cost {record.response_cost!r} must equal the header cost {outcome.response_cost}"
        )
        assert azure_logs.count_objects_for_id(body_id) == 1, (
            f"expected exactly ONE azure object for the call, got {azure_logs.count_objects_for_id(body_id)} - "
            "more than one object for one call is the duplicate-delivery bug"
        )

    @pytest.mark.covers("logging.azure_storage.success.writes_object", exercised_on=["messages"])
    def test_messages_writes_one_success_object(
        self, client: LoggingClient, azure_logs: AzureStorageLogReader, resources: ResourceManager
    ) -> None:
        """One successful non-streaming /v1/messages call must land in the
        filesystem as exactly one success payload object under the call's id."""
        _assert_azure_configured(client)

        key = client.key_with_alias(f"az-messages-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.messages_raw(key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", max_tokens=16),
        )
        body_id = completion_response_id(outcome.body)
        assert body_id is not None, "the messages body must carry an id (it names the azure object)"

        record = azure_logs.poll_record(body_id)
        assert record.id == body_id, f"payload id must be {body_id!r}, got {record.id!r}"
        assert record.status == "success", f"payload status must be success, got {record.status!r}"
        assert azure_logs.count_objects_for_id(body_id) == 1, (
            f"expected exactly ONE azure object for the call, got {azure_logs.count_objects_for_id(body_id)}"
        )

    @pytest.mark.covers("logging.azure_storage.stream.writes_object", exercised_on=["chat_completions"])
    def test_chat_completions_stream_writes_one_success_object(
        self, client: LoggingClient, azure_logs: AzureStorageLogReader, resources: ResourceManager
    ) -> None:
        """One successful STREAMED /chat/completions call must land in the
        filesystem as exactly one success payload object under the id the stream
        chunks carried."""
        _assert_azure_configured(client)

        key = client.key_with_alias(f"az-stream-{unique_marker()}", models=[CHEAP_ANTHROPIC_MODEL])
        resources.defer(lambda: client.delete_key(key))

        marker = unique_marker()
        outcome = first_ok(
            client,
            lambda: client.chat_raw(
                key, CHEAP_ANTHROPIC_MODEL, f"reply with one word {marker}", stream=True, max_tokens=16
            ),
        )
        assert outcome.chunks > 0, f"a streamed call must deliver chunks, got {outcome.chunks}"
        body_id = _stream_response_id(outcome.stream_events)
        assert body_id is not None, "the stream chunks must carry an id (it names the azure object)"

        record = azure_logs.poll_record(body_id)
        assert record.id == body_id, f"payload id must be {body_id!r}, got {record.id!r}"
        assert record.status == "success", f"payload status must be success, got {record.status!r}"
        assert azure_logs.count_objects_for_id(body_id) == 1, (
            f"expected exactly ONE azure object for the call, got {azure_logs.count_objects_for_id(body_id)}"
        )

    @pytest.mark.covers("logging.azure_storage.failure.writes_object", exercised_on=["chat_completions"])
    def test_failed_chat_completion_writes_one_failure_object(
        self, client: LoggingClient, azure_logs: AzureStorageLogReader, resources: ResourceManager
    ) -> None:
        """A call that fails at the provider must be persisted to the filesystem
        as one failure payload carrying the provider error - failed calls are
        part of the audit trail, not an exemption from it.

        A deployment with an invalid upstream key lets the request pass proxy
        auth and fail at the provider (the same lever as the s3 failure test).
        The persisted payload is named by the litellm call id on failure, not
        the response body id (a failed call has no completion id)."""
        _assert_azure_configured(client)

        model_name = f"az-err-{unique_marker()}"
        model_id = client.create_model(
            model_name,
            LiteLLMParamsBody(model="anthropic/claude-haiku-4-5", api_key=INVALID_UPSTREAM_API_KEY),
        )
        resources.defer(lambda: client.delete_model(model_id))
        key = client.key_with_alias(f"az-err-key-{unique_marker()}", models=[model_name])
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
        assert outcome.call_id is not None, "the failed call must carry x-litellm-call-id"

        record = azure_logs.poll_record(outcome.call_id)
        assert record.status == "failure", f"payload status must be failure, got {record.status!r}"
        assert record.error_str is not None and "AnthropicException" in record.error_str, (
            f"the persisted failure must carry the provider error, got error_str={record.error_str!r}"
        )
        assert azure_logs.count_objects_for_id(outcome.call_id) == 1, (
            f"expected exactly ONE failure object for the call, got {azure_logs.count_objects_for_id(outcome.call_id)}"
        )
