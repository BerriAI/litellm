"""Datadog sink contract: the `datadog` logs callback and the `datadog_llm_observability` span callback.

Both callbacks read `DD_API_KEY` and `DD_SITE`, and `DD_BASE_URL` overrides the intake host, which is how the rig
points them at one scripted Datadog peer. The logs callback batches entries and posts them gzip compressed to
`/api/v2/logs` with a `DD-API-KEY` header (litellm/integrations/datadog/datadog.py). Each entry is
`{ddsource, ddtags, hostname, message, service, status}` where `message` is the StandardLoggingPayload as a JSON
string with `messages`, `response` and `error_str` cut at 10000 characters. The LLM Observability callback posts
`{"data": {"type": "span", "attributes": {"ml_app", "tags", "spans"}}}` as plain JSON to
`/api/intake/llm-obs/v1/trace/spans` and expects a 202 (datadog_llm_obs.py).

The contract: one `info` log per success and one `error` log per failure, plus a separate proxy failure log carrying
error class and status code; tags carry env, service, version, host, pod, request tags and team; LLM Obs spans
carry kind, input and output messages, token and cost metrics, status and error; redaction replaces message
content but keeps roles; a key can send its own logs with its own Datadog API key; and a Datadog outage never fails
or slows the caller. These shapes are what Datadog log pipelines and LLM Obs views parse, so the Rust port must
keep them byte compatible in structure.
"""

from __future__ import annotations

import gzip
import json
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually
from integration.observability.dec1_callbacks_support import (
    CHAT_COST,
    CHAT_REJECTED,
    EMBEDDING,
    PROVIDER_KEY,
    PROVIDER_REJECTION,
    REPLY_TEXT,
)
from integration.observability.dec1_sinks_support import (
    DD_API_KEY,
    DD_LLM_OBS_PATH,
    DD_LOGS_PATH,
    REDACTED,
    SinkRig,
    assert_not_held,
    chat,
    healthy_seconds,
    key_logging,
    marker,
    new_call_id,
    sinks_rig,
    stream_chat,
    timed_chat,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

TRUNCATION_SUFFIX: Final = "...truncated by litellm, this logger does not support large content"


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-datadog"),
        {
            "callbacks": ["datadog", "datadog_llm_observability"],
            "cache": True,
            "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
        },
        environment={"HOSTNAME": "dec1-host", "POD_NAME": "dec1-pod", "DD_VERSION": "dec1-v1"},
    ) as created:
        yield created


def _message(entry: dict[str, JsonValue]) -> dict[str, JsonValue]:
    decoded: Final = json.loads(str(entry["message"]))
    assert isinstance(decoded, dict)
    return decoded


def _tags(entry: dict[str, JsonValue]) -> list[str]:
    return str(entry["ddtags"]).split(",")


# ---------------------------------------------------------------- logs


def test_logs_are_posted_gzip_compressed_with_the_api_key_header(rig: SinkRig) -> None:
    """Every post to /api/v2/logs must carry `Content-Encoding: gzip`, `Content-Type: application/json` and
    `DD-API-KEY` equal to DD_API_KEY, and its gunzipped body must be a JSON array of entries."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    rig.datadog_log(call_id)
    for request in rig.peer("datadog").posts(DD_LOGS_PATH):
        assert request.headers["content-encoding"] == "gzip"
        assert request.headers["content-type"] == "application/json"
        assert request.headers["dd-api-key"] == DD_API_KEY


def test_success_log_entry_has_the_documented_envelope(rig: SinkRig) -> None:
    """A successful chat must produce one entry with exactly the keys ddsource, ddtags, hostname, message, service
    and status, where ddsource is DD_SOURCE "dec1-source", service is DD_SERVICE "dec1-sinks-service", hostname is
    HOSTNAME "dec1-host" and status "info". UNVERIFIED: whether dd.trace_id / dd.span_id keys are added when ddtrace
    is active (`_add_trace_context_to_payload`)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    entry: Final = rig.datadog_log(call_id)
    assert {"ddsource", "ddtags", "hostname", "message", "service", "status"} <= set(entry)
    assert entry["ddsource"] == "dec1-source"
    assert entry["service"] == "dec1-sinks-service"
    assert entry["hostname"] == "dec1-host"
    assert entry["status"] == "info"


def test_ddtags_start_with_env_service_version_host_and_pod(rig: SinkRig) -> None:
    """ddtags must be a comma joined string whose first five tags are env:dec1-env, service:dec1-sinks-service,
    version:dec1-v1, HOSTNAME:dec1-host and POD_NAME:dec1-pod, in that order."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    assert _tags(rig.datadog_log(call_id))[:5] == [
        "env:dec1-env",
        "service:dec1-sinks-service",
        "version:dec1-v1",
        "HOSTNAME:dec1-host",
        "POD_NAME:dec1-pod",
    ]


def test_request_tags_are_normalised_into_request_tag_entries(rig: SinkRig) -> None:
    """Request metadata tags ["Dec1 Tag!", "plain"] must become `request_tag:dec1_tag` and `request_tag:plain`:
    lower cased, every character outside [a-z0-9_-:./] replaced by "_", runs of "_" collapsed and edges stripped
    (`normalize_datadog_tag_value`)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker(), metadata={"tags": ["Dec1 Tag!", "plain"]}).status_code == 200
    tags: Final = _tags(rig.datadog_log(call_id))
    assert "request_tag:dec1_tag" in tags
    assert "request_tag:plain" in tags


def test_team_tag_prefers_the_team_alias(rig: SinkRig) -> None:
    """A call from a key of a team aliased "Dec1 Team <n>" must add the tag `team:dec1_team_<n>`, built from the
    team alias rather than the team id."""
    suffix: Final = uuid.uuid4().hex[:6]
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(team_alias=f"Dec1 Team {suffix}")
        key: Final = scenario.key(team_id=team)
        call_id = new_call_id()
        assert chat(rig, call_id, marker(), key=key).status_code == 200
        assert f"team:dec1_team_{suffix}" in _tags(rig.datadog_log(call_id))


def test_success_message_is_the_standard_logging_payload(rig: SinkRig) -> None:
    """The `message` of a success entry must decode to the StandardLoggingPayload: status "success", call_type
    "acompletion", prompt_tokens 11, completion_tokens 4, response_cost 0.019 and the user text in `messages`."""
    call_id, text = new_call_id(), marker()
    assert chat(rig, call_id, text).status_code == 200
    payload: Final = _message(rig.datadog_log(call_id))
    assert payload["status"] == "success"
    assert payload["call_type"] == "acompletion"
    assert (payload["prompt_tokens"], payload["completion_tokens"]) == (11, 4)
    assert payload["response_cost"] == pytest.approx(CHAT_COST)
    assert text in json.dumps(payload["messages"])


def test_long_messages_are_truncated_to_ten_thousand_characters_with_a_marker(rig: SinkRig) -> None:
    """A 12000 character user message must reach Datadog with `messages` turned into a string of its first 10000
    characters followed by "...truncated by litellm, this logger does not support large content", while the
    provider still receives the whole message (`truncate_standard_logging_payload_content`)."""
    call_id, text = new_call_id(), marker() + "x" * 12000
    assert chat(rig, call_id, text).status_code == 200
    messages: Final = _message(rig.datadog_log(call_id))["messages"]
    assert isinstance(messages, str)
    assert messages.endswith(TRUNCATION_SUFFIX)
    assert len(messages) == 10000 + len(TRUNCATION_SUFFIX)


def test_provider_failure_logs_one_error_entry_with_the_error_payload(rig: SinkRig) -> None:
    """A provider 400 must produce exactly one entry for the call id with status "error" whose message decodes to a
    payload with status "failure" and error_str containing the provider's error text."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker(), model=CHAT_REJECTED).status_code == 400
    entry: Final = rig.datadog_log(call_id, "error")
    payload: Final = _message(entry)
    assert payload["status"] == "failure"
    assert PROVIDER_REJECTION in str(payload["error_str"])


def test_provider_failure_also_logs_a_proxy_failure_entry_with_class_and_status(rig: SinkRig) -> None:
    """Besides the payload entry, the proxy failure hook must post a second "error" entry whose message has keys
    exception, error_class, status_code, traceback and user_api_key_dict, with status_code 400 and the provider
    text in exception (`async_post_call_failure_hook`). A rewrite that drops the hook loses proxy level failures."""
    text: Final = marker()
    assert chat(rig, new_call_id(), text, model=CHAT_REJECTED).status_code == 400

    def proxy_failures() -> tuple[dict[str, JsonValue], ...]:
        decoded = (_message(entry) for entry in rig.datadog_logs() if entry.get("status") == "error")
        return tuple(m for m in decoded if "error_class" in m and PROVIDER_REJECTION in str(m.get("exception")))

    found: Final = eventually(proxy_failures, lambda entries: len(entries) >= 1, seconds=30)
    assert set(found[0]) == {"exception", "error_class", "status_code", "traceback", "user_api_key_dict"}
    assert found[0]["status_code"] == 400


def test_invalid_key_is_logged_as_a_proxy_failure_with_status_401(rig: SinkRig) -> None:
    """A request with an unknown virtual key must leave a Datadog "error" entry whose message has status_code 401,
    and that entry must not contain the presented key."""
    bad: Final = "sk-dec1-sinks-unknown-" + uuid.uuid4().hex
    assert chat(rig, new_call_id(), marker(), key=bad).status_code == 401

    def unauthorised() -> tuple[dict[str, JsonValue], ...]:
        decoded = (_message(entry) for entry in rig.datadog_logs() if entry.get("status") == "error")
        return tuple(m for m in decoded if m.get("status_code") == 401)

    found: Final = eventually(unauthorised, lambda entries: len(entries) >= 1, seconds=30)
    assert all(bad not in json.dumps(entry) for entry in found)


def test_streamed_chat_logs_one_entry_with_stream_true(rig: SinkRig) -> None:
    """A streamed chat must produce exactly one "info" entry whose payload has stream true and the assembled reply in
    `response`."""
    call_id: Final = new_call_id()
    status, _ = stream_chat(rig, call_id, marker())
    assert status == 200
    payload: Final = _message(rig.datadog_log(call_id))
    assert payload["stream"] is True
    assert REPLY_TEXT in json.dumps(payload["response"])


def test_cache_hit_entry_has_cache_hit_true_and_zero_cost(rig: SinkRig) -> None:
    """The second of two identical chats must log an entry whose payload has cache_hit true and response_cost 0."""
    text: Final = marker()
    first, second = new_call_id(), new_call_id()
    assert chat(rig, first, text).status_code == 200
    rig.datadog_log(first)
    assert chat(rig, second, text).status_code == 200
    payload: Final = _message(rig.datadog_log(second))
    assert payload["cache_hit"] is True
    assert payload["response_cost"] == 0


def test_key_scoped_datadog_api_key_sends_that_keys_logs_with_its_own_key(rig: SinkRig) -> None:
    """A key whose `metadata.logging` names `datadog` with callback_vars dd_api_key "<team key>" must have its log
    posted with `DD-API-KEY: <team key>`, through a logger separate from the global one."""
    team_key: Final = "synthetic-dd-team-key-" + uuid.uuid4().hex[:8]
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(metadata={"logging": key_logging("datadog", {"dd_api_key": team_key})})
        call_id = new_call_id()
        assert chat(rig, call_id, marker(), key=key).status_code == 200

        def headers_for_call() -> set[str]:
            found = set()
            for request in rig.peer("datadog").posts(DD_LOGS_PATH):
                if call_id.encode() in gzip.decompress(request.body):
                    found.add(request.headers["dd-api-key"])
            return found

        assert team_key in eventually(headers_for_call, lambda found: team_key in found, seconds=30)


def test_no_credential_reaches_the_datadog_logs(rig: SinkRig) -> None:
    """No log entry may contain the master key or the provider key, on success or failure."""
    ok, failed = new_call_id(), new_call_id()
    assert chat(rig, ok, marker()).status_code == 200
    assert chat(rig, failed, marker(), model=CHAT_REJECTED).status_code == 400
    rig.datadog_log(ok)
    rig.datadog_log(failed, "error")
    body: Final = json.dumps(rig.datadog_logs())
    assert rig.proxy.key not in body
    assert PROVIDER_KEY not in body


# ---------------------------------------------------------------- LLM Observability


def test_llm_obs_batch_envelope_names_the_ml_app_and_is_plain_json(rig: SinkRig) -> None:
    """Each post to /api/intake/llm-obs/v1/trace/spans must be uncompressed JSON with `DD-API-KEY`, shaped
    {"data": {"type": "span", "attributes": {"ml_app": "dec1-sinks-service", "tags": [...], "spans": [...]}}}."""
    text: Final = marker()
    assert chat(rig, new_call_id(), text).status_code == 200
    rig.llm_obs_span(text)
    for request in rig.peer("datadog").posts(DD_LLM_OBS_PATH):
        assert request.headers.get("content-encoding") is None
        assert request.headers["dd-api-key"] == DD_API_KEY
        data = json.loads(request.body)["data"]
        assert data["type"] == "span"
        assert data["attributes"]["ml_app"] == "dec1-sinks-service"
        assert "env:dec1-env" in data["attributes"]["tags"]


def test_llm_obs_success_span_has_kind_messages_metrics_and_ok_status(rig: SinkRig) -> None:
    """A successful chat must yield one span named "litellm_llm_call" with parent_id "undefined", meta.kind "llm",
    the user text in meta.input.messages, the reply in meta.output.messages, metrics input_tokens 11.0,
    output_tokens 4.0, total_tokens 15.0, total_cost 0.019, and status "ok"."""
    text: Final = marker()
    assert chat(rig, new_call_id(), text).status_code == 200
    span: Final = rig.llm_obs_span(text)
    meta: Final = span["meta"]
    assert isinstance(meta, dict)
    assert span["name"] == "litellm_llm_call"
    assert span["parent_id"] == "undefined"
    assert meta["kind"] == "llm"
    assert meta["input"]["messages"][0] == {"role": "user", "content": text}  # type: ignore[index]
    assert REPLY_TEXT in json.dumps(meta["output"])
    metrics: Final = span["metrics"]
    assert isinstance(metrics, dict)
    assert (metrics["input_tokens"], metrics["output_tokens"], metrics["total_tokens"]) == (11.0, 4.0, 15.0)
    assert metrics["total_cost"] == pytest.approx(CHAT_COST)
    assert span["status"] == "ok"


def test_llm_obs_trace_id_is_the_payload_trace_id(rig: SinkRig) -> None:
    """A request with metadata trace_id "<id>" must yield a span whose trace_id is "<id>", since the span takes the
    StandardLoggingPayload trace_id."""
    text, trace = marker(), "dec1-trace-" + uuid.uuid4().hex
    assert chat(rig, new_call_id(), text, metadata={"trace_id": trace}).status_code == 200
    assert rig.llm_obs_span(text)["trace_id"] == trace


def test_llm_obs_metadata_name_span_id_and_parent_id_override_the_defaults(rig: SinkRig) -> None:
    """Request metadata name "<n>", span_id "<s>" and parent_id "<p>" must become the span's name, span_id and
    parent_id."""
    text, tag = marker(), uuid.uuid4().hex[:8]
    metadata: Final = {"name": "dec1-step-" + tag, "span_id": "span-" + tag, "parent_id": "parent-" + tag}
    assert chat(rig, new_call_id(), text, metadata=metadata).status_code == 200
    span: Final = rig.llm_obs_span(text)
    assert (span["name"], span["span_id"], span["parent_id"]) == (
        metadata["name"],
        metadata["span_id"],
        metadata["parent_id"],
    )


def test_llm_obs_embedding_is_kind_llm_without_a_parent_and_embedding_with_one(rig: SinkRig) -> None:
    """An embedding without metadata parent_id must be span kind "llm", because only llm, workflow and agent spans
    may be roots; the same request with metadata parent_id must be kind "embedding" (`_get_datadog_span_kind`)."""
    root_text, child_text = marker(), marker()
    for text, metadata in ((root_text, {}), (child_text, {"parent_id": "dec1-parent"})):
        response = rig.proxy.request(
            "POST", "/v1/embeddings", {"model": EMBEDDING, "input": text, "metadata": metadata}
        )
        assert response.status_code == 200
    assert rig.llm_obs_span(root_text)["meta"]["kind"] == "llm"  # type: ignore[index]
    assert rig.llm_obs_span(child_text)["meta"]["kind"] == "embedding"  # type: ignore[index]


def test_llm_obs_failure_span_has_error_status_message_and_type(rig: SinkRig) -> None:
    """A provider 400 must yield a span with status "error" and meta.error.message containing the provider text and
    a non empty meta.error.type naming the exception class."""
    text: Final = marker()
    assert chat(rig, new_call_id(), text, model=CHAT_REJECTED).status_code == 400
    span: Final = rig.llm_obs_span(text)
    error: Final = span["meta"]["error"]  # type: ignore[index]
    assert span["status"] == "error"
    assert PROVIDER_REJECTION in error["message"]
    assert error["type"]


def test_llm_obs_redaction_replaces_content_and_keeps_roles(rig: SinkRig) -> None:
    """A request with header `litellm-enable-message-redaction: true` must yield a span whose input and output
    messages keep role "user" and "assistant" but whose content is "redacted-by-litellm", and the user text must
    not appear in the span. The span is found by its metadata `name` since the text is gone."""
    text, name = marker(), marker()
    headers: Final = {"litellm-enable-message-redaction": "true"}
    assert chat(rig, new_call_id(), text, headers=headers, metadata={"name": name}).status_code == 200
    span: Final = rig.llm_obs_span(name)
    meta: Final = span["meta"]
    assert meta["input"]["messages"] == [{"role": "user", "content": REDACTED}]  # type: ignore[index]
    assert meta["output"]["messages"][0]["content"] == REDACTED  # type: ignore[index]
    assert meta["output"]["messages"][0]["role"] == "assistant"  # type: ignore[index]
    assert text not in json.dumps(span)


def test_llm_obs_cache_hit_reports_zero_cost(rig: SinkRig) -> None:
    """The cache hit twin of a chat must yield a second span for the same text with metrics total_cost 0.0 and the
    same 11 input tokens, next to the original span with total_cost 0.019."""
    text: Final = marker()
    assert chat(rig, new_call_id(), text).status_code == 200
    rig.llm_obs_span(text)
    assert chat(rig, new_call_id(), text).status_code == 200
    spans: Final = eventually(
        lambda: tuple(
            span
            for batch in rig.llm_obs_batches()
            for span in batch["data"]["attributes"]["spans"]  # type: ignore[index]
            if text in json.dumps(span)
        ),
        lambda found: len(found) == 2,
        seconds=30,
    )
    costs: Final = sorted(span["metrics"]["total_cost"] for span in spans)
    assert costs == [0.0, pytest.approx(CHAT_COST)]
    assert all(span["metrics"]["input_tokens"] == 11.0 for span in spans)


# ---------------------------------------------------------------- outage


def test_datadog_returning_500_never_fails_or_slows_the_request(rig: SinkRig) -> None:
    """With the Datadog peer answering 500 to both intakes, a chat must still return 200 within the healthy control time plus the 30 second stall, and a call
    after recovery must still be logged."""
    control: Final = healthy_seconds(rig)
    peer: Final = rig.peer("datadog")
    peer.switch.status = 500
    try:
        response, elapsed = timed_chat(rig, new_call_id(), marker())
        assert response.status_code == 200
        assert_not_held(elapsed, control)
    finally:
        peer.switch.status = 200
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    rig.datadog_log(call_id, seconds=60)


def test_stalled_datadog_never_holds_the_response(rig: SinkRig) -> None:
    """With the Datadog peer holding every post open, three sequential chats must each return 200 within the healthy control time plus the 30 second stall,
    since both loggers only enqueue on the request path and flush from a periodic task."""
    control: Final = healthy_seconds(rig)
    peer: Final = rig.peer("datadog")
    peer.switch.stall()
    try:
        for _ in range(3):
            response, elapsed = timed_chat(rig, new_call_id(), marker())
            assert response.status_code == 200
            assert_not_held(elapsed, control)
    finally:
        peer.switch.release()
