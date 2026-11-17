"""Langfuse sink contract: what the classic `langfuse` callback sends for each outcome, and what it must hide.

The `langfuse` callback no longer talks to the v2 ingestion API. It builds one OpenTelemetry span per generation and
posts it as OTLP protobuf to `<LANGFUSE_HOST>/api/public/otel/v1/traces` with HTTP Basic auth of the public and
secret key and `x-langfuse-ingestion-version: 4` (litellm/integrations/langfuse/langfuse.py `_log_langfuse_v2`,
langfuse_sdk.py). Trace level fields ride on the same span as `langfuse.trace.*`, `user.id` and `session.id`
attributes, and generation fields as `langfuse.observation.*`. The scripted Langfuse peer records the posts and the
tests decode the spans.

The contract pinned here: the default trace name is `litellm-<call_type>`, the default generation name is
`litellm:<key alias>` and falls back to `litellm-<call_type>`; caller metadata (`trace_name`, `generation_name`,
`session_id`, `tags`, `existing_trace_id`, `version`, `mask_input`, `mask_output`) steers those fields; a non hex
caller trace id is hashed to 32 hex characters; failures are level ERROR with the error text as status message and
output; streamed calls log once with the assembled reply; cache hits are flagged in metadata; redaction replaces the
input and output; a key or team can send its own requests to its own Langfuse project; and a Langfuse outage never
fails or slows the caller. The Rust CustomLoggerV2 port must keep this wire shape, because Langfuse dashboards and
saved filters are built on these attribute names.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Iterator
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import eventually
from integration.observability.dec1_callbacks_support import (
    CHAT_COST,
    CHAT_OUTAGE,
    CHAT_REJECTED,
    EMBEDDING,
    PROVIDER_KEY,
    PROVIDER_OUTAGE,
    PROVIDER_REJECTION,
    REPLY_TEXT,
)
from integration.observability.dec1_sinks_support import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_TRACES_PATH,
    REDACTED,
    SinkRig,
    assert_not_held,
    attribute,
    chat,
    healthy_seconds,
    key_logging,
    marker,
    new_call_id,
    parsed,
    sinks_rig,
    stream_chat,
    timed_chat,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SinkRig]:
    with sinks_rig(
        tmp_path_factory.mktemp("dec1-sinks-langfuse"),
        {
            "success_callback": ["langfuse"],
            "failure_callback": ["langfuse"],
            "langfuse_default_tags": ["cache_hit"],
            "cache": True,
            "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
        },
    ) as created:
        yield created


def _basic(public: str, secret: str) -> str:
    return "Basic " + base64.b64encode(f"{public}:{secret}".encode()).decode()


def test_generation_post_uses_basic_auth_protobuf_and_ingestion_version_4(rig: SinkRig) -> None:
    """One successful chat must reach the Langfuse peer only as POSTs to /api/public/otel/v1/traces with
    content-type application/x-protobuf, header x-langfuse-ingestion-version 4, and Basic auth of
    LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY. A rewrite that posts to the legacy /api/public/ingestion batch API
    would be silently dropped by a v4 only Langfuse."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    rig.generation(call_id)
    posts: Final = rig.peer("langfuse").posts()
    assert {request.target.split("?")[0] for request in posts} == {LANGFUSE_TRACES_PATH}
    for request in posts:
        assert request.headers["authorization"] == _basic(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
        assert request.headers["content-type"] == "application/x-protobuf"
        assert request.headers["x-langfuse-ingestion-version"] == "4"


def test_default_trace_name_is_litellm_dash_call_type(rig: SinkRig) -> None:
    """With no `trace_name` in the request metadata, the generation span must carry `langfuse.trace.name`
    "litellm-acompletion", the proxy's call type for chat (`trace_name = f"litellm-{call_type}"`)."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    assert attribute(rig.generation(call_id), "langfuse.trace.name") == "litellm-acompletion"


def test_default_generation_name_is_litellm_colon_key_alias(rig: SinkRig) -> None:
    """A key created with key_alias "<alias>" and no `generation_name` in the request must produce a generation span
    named "litellm:<alias>"."""
    alias: Final = "dec1-sinks-alias-" + uuid.uuid4().hex[:8]
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias=alias)
        call_id = new_call_id()
        assert chat(rig, call_id, marker(), key=key).status_code == 200
        assert rig.generation(call_id)["name"] == f"litellm:{alias}"


def test_caller_generation_name_wins_over_the_key_alias(rig: SinkRig) -> None:
    """`metadata.generation_name` set by the caller must become the span name even when the key has an alias."""
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(key_alias="dec1-sinks-named-" + uuid.uuid4().hex[:8])
        call_id = new_call_id()
        name: Final = marker()
        assert chat(rig, call_id, marker(), key=key, metadata={"generation_name": name}).status_code == 200
        assert rig.generation(call_id)["name"] == name


def test_caller_trace_name_session_id_tags_and_version_reach_the_trace_attributes(rig: SinkRig) -> None:
    """Request metadata `trace_name`, `session_id`, `tags` and `version` must appear as `langfuse.trace.name`,
    `session.id`, the `langfuse.trace.tags` list and `langfuse.version` on the generation span."""
    call_id, tag = new_call_id(), marker()
    metadata: Final = {"trace_name": tag + "-trace", "session_id": tag + "-session", "tags": [tag], "version": "v7"}
    assert chat(rig, call_id, marker(), metadata=metadata).status_code == 200
    span: Final = rig.generation(call_id)
    assert attribute(span, "langfuse.trace.name") == tag + "-trace"
    assert attribute(span, "session.id") == tag + "-session"
    assert tag in (attribute(span, "langfuse.trace.tags") or [])
    assert attribute(span, "langfuse.version") == "v7"


def test_request_user_field_becomes_the_trace_user_id(rig: SinkRig) -> None:
    """The OpenAI `user` field names the end user, and the trace's `user.id` attribute must equal it, because
    `trace_params["user_id"]` is the payload's `user_api_key_end_user_id`."""
    call_id, end_user = new_call_id(), "dec1-end-user-" + uuid.uuid4().hex[:8]
    assert chat(rig, call_id, marker(), user=end_user).status_code == 200
    assert attribute(rig.generation(call_id), "user.id") == end_user


def test_hex_caller_trace_id_is_used_verbatim_and_a_uuid_loses_its_dashes(rig: SinkRig) -> None:
    """A caller `metadata.trace_id` that is a UUID must become the span's trace id with dashes removed and lower
    cased, since v4 needs 32 hex characters (`resolve_trace_id`)."""
    call_id, trace = new_call_id(), str(uuid.uuid4()).upper()
    assert chat(rig, call_id, marker(), metadata={"trace_id": trace}).status_code == 200
    assert rig.generation(call_id)["trace_id"] == trace.lower().replace("-", "")


def test_non_hex_caller_trace_id_is_hashed_to_the_first_16_bytes_of_its_sha256(rig: SinkRig) -> None:
    """A caller trace id that is not 32 hex characters must be mapped to the first 16 bytes of its sha256 as hex,
    so repeated calls with the same caller id land in one Langfuse trace (`resolve_trace_id`)."""
    call_id, trace = new_call_id(), "dec1 human readable trace " + uuid.uuid4().hex
    assert chat(rig, call_id, marker(), metadata={"trace_id": trace}).status_code == 200
    assert rig.generation(call_id)["trace_id"] == sha256(trace.encode()).digest()[:16].hex()


def test_existing_trace_id_appends_to_that_trace_without_renaming_it(rig: SinkRig) -> None:
    """`metadata.existing_trace_id` continues a trace the caller already created: the generation must sit in that
    trace id and carry no `langfuse.trace.name`, so the earlier trace's name is never overwritten."""
    call_id, existing = new_call_id(), uuid.uuid4().hex
    assert chat(rig, call_id, marker(), metadata={"existing_trace_id": existing}).status_code == 200
    span: Final = rig.generation(call_id)
    assert span["trace_id"] == existing
    assert attribute(span, "langfuse.trace.name") is None


def test_success_generation_carries_model_usage_cost_and_ids(rig: SinkRig) -> None:
    """A successful chat (provider usage 11 in, 4 out) must log observation type "generation", model name
    "openai/gpt-4o-mini", usage_details {input 11, output 4, total 15, cache tokens 0}, cost_details {total 0.019},
    and metadata `response_id` equal to the id in the caller's response."""
    call_id: Final = new_call_id()
    response: Final = chat(rig, call_id, marker())
    span: Final = rig.generation(call_id)
    assert attribute(span, "langfuse.observation.type") == "generation"
    assert attribute(span, "langfuse.observation.model.name") == "openai/gpt-4o-mini"
    assert parsed(attribute(span, "langfuse.observation.usage_details")) == {
        "input": 11,
        "output": 4,
        "total": 15,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert parsed(attribute(span, "langfuse.observation.cost_details")) == {"total": pytest.approx(CHAT_COST)}
    assert attribute(span, "langfuse.observation.metadata.response_id") == response.json()["id"]


def test_success_generation_input_has_the_messages_and_output_the_reply(rig: SinkRig) -> None:
    """The generation's `langfuse.observation.input` JSON must contain the caller's user text and its
    `langfuse.observation.output` must contain the provider's reply text."""
    call_id, text = new_call_id(), marker()
    assert chat(rig, call_id, text).status_code == 200
    span: Final = rig.generation(call_id)
    assert text in str(attribute(span, "langfuse.observation.input"))
    assert REPLY_TEXT in str(attribute(span, "langfuse.observation.output"))


def test_mask_input_redacts_only_the_input(rig: SinkRig) -> None:
    """`metadata.mask_input: true` must replace the generation input with "redacted-by-litellm" while the
    output still carries the reply, and the user text must appear nowhere in the posted bytes."""
    call_id, text = new_call_id(), marker()
    assert chat(rig, call_id, text, metadata={"mask_input": True}).status_code == 200
    span: Final = rig.generation(call_id)
    assert attribute(span, "langfuse.observation.input") == REDACTED
    assert REPLY_TEXT in str(attribute(span, "langfuse.observation.output"))
    assert all(text.encode() not in request.body for request in rig.peer("langfuse").posts())


def test_mask_output_redacts_only_the_output(rig: SinkRig) -> None:
    """`metadata.mask_output: true` must replace the generation output with "redacted-by-litellm" and keep the
    input."""
    call_id, text = new_call_id(), marker()
    assert chat(rig, call_id, text, metadata={"mask_output": True}).status_code == 200
    span: Final = rig.generation(call_id)
    assert attribute(span, "langfuse.observation.output") == REDACTED
    assert text in str(attribute(span, "langfuse.observation.input"))


def test_per_request_message_redaction_hides_the_prompt_and_the_reply(rig: SinkRig) -> None:
    """A request with header `litellm-enable-message-redaction: true` must reach Langfuse with neither the user text
    nor the reply text in any posted byte, and with usage still present. UNVERIFIED: the exact redacted input
    shape (a messages list whose content is "redacted-by-litellm"), see redact_messages.py perform_redaction."""
    call_id, text = new_call_id(), marker()
    headers: Final = {"litellm-enable-message-redaction": "true"}
    assert chat(rig, call_id, text, headers=headers).status_code == 200
    span: Final = rig.generation(call_id)
    body: Final = json.dumps(span["attributes"])
    assert text not in body
    assert REPLY_TEXT not in body
    assert REDACTED in body
    assert attribute(span, "langfuse.observation.usage_details") is not None


def test_provider_failure_logs_an_error_generation_with_the_error_text(rig: SinkRig) -> None:
    """A provider 400 must produce one generation with `langfuse.observation.level` "ERROR", a status message
    containing the provider's error text, the same text as the output, and the request in the input."""
    call_id, text = new_call_id(), marker()
    assert chat(rig, call_id, text, model=CHAT_REJECTED).status_code == 400
    span: Final = rig.generation(call_id)
    assert attribute(span, "langfuse.observation.level") == "ERROR"
    assert PROVIDER_REJECTION in str(attribute(span, "langfuse.observation.status_message"))
    assert PROVIDER_REJECTION in str(attribute(span, "langfuse.observation.output"))
    assert text in str(attribute(span, "langfuse.observation.input"))


def test_provider_failure_generation_has_no_usage_or_cost_details(rig: SinkRig) -> None:
    """A failed call has no response object, so the generation must carry no `usage_details` and no
    `cost_details` attribute rather than zeros."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker(), model=CHAT_OUTAGE).status_code == 500
    span: Final = rig.generation(call_id)
    assert PROVIDER_OUTAGE in str(attribute(span, "langfuse.observation.status_message"))
    assert attribute(span, "langfuse.observation.usage_details") is None
    assert attribute(span, "langfuse.observation.cost_details") is None


def test_success_generation_level_is_default(rig: SinkRig) -> None:
    """A successful call must log `langfuse.observation.level` "DEFAULT" and no status message."""
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    span: Final = rig.generation(call_id)
    assert attribute(span, "langfuse.observation.level") == "DEFAULT"
    assert attribute(span, "langfuse.observation.status_message") is None


def test_streamed_chat_logs_one_generation_with_the_assembled_reply_and_usage(rig: SinkRig) -> None:
    """A streamed chat read to the end must log exactly one generation whose output has the whole reply text, whose
    usage is 11 in and 4 out, and which carries `langfuse.observation.completion_start_time`."""
    call_id: Final = new_call_id()
    status, _ = stream_chat(rig, call_id, marker())
    assert status == 200
    span: Final = rig.generation(call_id)
    assert REPLY_TEXT in str(attribute(span, "langfuse.observation.output"))
    usage: Final = parsed(attribute(span, "langfuse.observation.usage_details"))
    assert isinstance(usage, dict)
    assert (usage["input"], usage["output"]) == (11, 4)
    assert attribute(span, "langfuse.observation.completion_start_time") is not None


def test_cache_hit_is_flagged_in_metadata_and_tagged(rig: SinkRig) -> None:
    """Two identical requests: the second is served from the cache, and its generation must carry metadata
    `cache_hit` true, the trace tag "cache_hit:True" (from `langfuse_default_tags: [cache_hit]`), and
    cost_details total 0."""
    text: Final = marker()
    first, second = new_call_id(), new_call_id()
    assert chat(rig, first, text).status_code == 200
    rig.generation(first)
    assert chat(rig, second, text).status_code == 200
    span: Final = rig.generation(second)
    assert attribute(span, "langfuse.observation.metadata.cache_hit") is True
    assert "cache_hit:True" in (attribute(span, "langfuse.trace.tags") or [])
    assert parsed(attribute(span, "langfuse.observation.cost_details")) == {"total": 0.0}


def test_embedding_generation_has_the_input_and_no_output(rig: SinkRig) -> None:
    """An embedding request must log a generation named by trace "litellm-aembedding" whose input carries the
    embedded text and which has no `langfuse.observation.output`, since vectors are never logged."""
    call_id, text = new_call_id(), marker()
    response: Final = rig.proxy.request(
        "POST", "/v1/embeddings", {"model": EMBEDDING, "input": text}, headers={"x-litellm-call-id": call_id}
    )
    assert response.status_code == 200
    span: Final = rig.generation(call_id)
    assert attribute(span, "langfuse.trace.name") == "litellm-aembedding"
    assert text in str(attribute(span, "langfuse.observation.input"))
    assert attribute(span, "langfuse.observation.output") is None


def test_no_credential_reaches_the_langfuse_peer(rig: SinkRig) -> None:
    """No post to Langfuse may contain the proxy master key, the deployment's provider key or the caller's
    Authorization header value, on success and on failure alike."""
    ok, failed = new_call_id(), new_call_id()
    assert chat(rig, ok, marker()).status_code == 200
    assert chat(rig, failed, marker(), model=CHAT_REJECTED).status_code == 400
    rig.generation(ok)
    rig.generation(failed)
    for request in rig.peer("langfuse").posts():
        assert rig.proxy.key.encode() not in request.body
        assert PROVIDER_KEY.encode() not in request.body


def test_key_logging_langfuse_credentials_send_that_keys_traces_with_its_own_auth(rig: SinkRig) -> None:
    """A key whose `metadata.logging` names `langfuse` with its own public and secret key must have its generation
    posted with Basic auth of those credentials, while a plain key's generation keeps the operator credentials."""
    public, secret = "pk-lf-key-" + uuid.uuid4().hex[:8], "sk-lf-key-" + uuid.uuid4().hex[:8]
    variables: Final = {
        "langfuse_public_key": public,
        "langfuse_secret_key": secret,
        "langfuse_host": rig.peer("langfuse").url,
    }
    with rig.proxy.scenario() as scenario:
        key: Final = scenario.key(metadata={"logging": key_logging("langfuse", variables)})
        call_id = new_call_id()
        assert chat(rig, call_id, marker(), key=key).status_code == 200
        rig.generation(call_id)
        auths = eventually(
            lambda: {
                request.headers["authorization"]
                for request in rig.peer("langfuse").posts()
                if call_id.encode() in request.body
            },
            lambda found: len(found) >= 1,
            seconds=30,
        )
        assert auths == {_basic(public, secret)}


def test_team_logging_langfuse_credentials_apply_to_every_key_of_the_team(rig: SinkRig) -> None:
    """A team whose `metadata.logging` names `langfuse` with its own credentials must have a generation from any of
    its keys posted with Basic auth of the team's public and secret key."""
    public, secret = "pk-lf-team-" + uuid.uuid4().hex[:8], "sk-lf-team-" + uuid.uuid4().hex[:8]
    variables: Final = {
        "langfuse_public_key": public,
        "langfuse_secret_key": secret,
        "langfuse_host": rig.peer("langfuse").url,
    }
    with rig.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"logging": key_logging("langfuse", variables)})
        key: Final = scenario.key(team_id=team)
        call_id = new_call_id()
        assert chat(rig, call_id, marker(), key=key).status_code == 200
        auths = eventually(
            lambda: {
                request.headers["authorization"]
                for request in rig.peer("langfuse").posts()
                if call_id.encode() in request.body
            },
            lambda found: len(found) >= 1,
            seconds=30,
        )
        assert _basic(public, secret) in auths


def test_langfuse_returning_500_never_fails_or_slows_the_request(rig: SinkRig) -> None:
    """With the Langfuse peer answering 500 to every post, a chat must still return 200 within the healthy control time plus the 30 second stall and the
    next chat after recovery must still be logged."""
    control: Final = healthy_seconds(rig)
    peer: Final = rig.peer("langfuse")
    peer.switch.status = 500
    try:
        response, elapsed = timed_chat(rig, new_call_id(), marker())
        assert response.status_code == 200
        assert_not_held(elapsed, control)
    finally:
        peer.switch.status = 200
    call_id: Final = new_call_id()
    assert chat(rig, call_id, marker()).status_code == 200
    rig.generation(call_id, seconds=60)


def test_stalled_langfuse_never_holds_the_response(rig: SinkRig) -> None:
    """With the Langfuse peer holding every post open, three sequential chats must each return 200 within the healthy control time plus the 30 second stall,
    because export happens on a background exporter, never on the request path."""
    control: Final = healthy_seconds(rig)
    peer: Final = rig.peer("langfuse")
    peer.switch.stall()
    try:
        for _ in range(3):
            response, elapsed = timed_chat(rig, new_call_id(), marker())
            assert response.status_code == 200
            assert_not_held(elapsed, control)
    finally:
        peer.switch.release()
