"""Live end-to-end spend-tracking tests against a running proxy.

Run against a proxy started with the gateway config. Coverage rationale:
SPEND_TRACKING_COVERAGE_MATRIX.md.

Model names are literals from that config: chat tests hit "gemini-2.5-flash",
embedding tests hit "openai-text-embedding-3-small".

Every test: fresh scoped key (isolation) -> real provider call -> unwrap (hard
fail if the proxy couldn't make a call it should) -> poll /spend/logs to a
deadline (rows land ~60s later via proxy_batch_write_at) -> assert invariants on
the real row (spend, token arithmetic, status, cache).

Assertions target invariants, not literals: a regression in the spend pipeline
fails the test; a pricing or token-count drift does not.
"""

import time
from collections.abc import Callable
from math import isclose
from typing import Final

import pytest
from e2e_http import RateLimitedError, Success
from lifecycle import ResourceManager
from models import KeyGenerateBody, LiteLLMParamsBody, SpendLogs, SpendLogsParams
from spend_e2e_client import (
    ClientAttributionHeaders,
    SpendClient,
    SpendLogRow,
    is_ok,
    unique_marker,
    unwrap,
)

pytestmark = pytest.mark.e2e


def _approx_equal(actual: float, expected: float) -> bool:
    """Within 1% or 1e-9 absolute - spend math, not exact float identity."""
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


def _summarize(rows: list[SpendLogRow]) -> list[dict[str, object]]:
    fields = {
        "request_id",
        "model",
        "spend",
        "status",
        "cache_hit",
        "call_type",
        "custom_llm_provider",
        "model_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    return [row.model_dump(include=fields) for row in rows]


def _require_row(
    rows: list[SpendLogRow], predicate: Callable[[SpendLogRow], bool], what: str
) -> SpendLogRow:
    matches = [r for r in rows if predicate(r)]
    assert matches, (
        f"no SpendLogs row {what} after polling; saw {len(rows)} row(s): "
        f"{_summarize(rows)}"
    )
    return matches[0]


@pytest.mark.covers("quota_management.spend_tracking.chat_completions.logs_cost")
def test_chat_completion_writes_nonzero_spend_row(
    client: SpendClient, scoped_key: str
) -> None:
    chat = unwrap(
        client.chat(
            scoped_key,
            "gemini-2.5-flash",
            f"reply with one word {unique_marker()}",
            max_tokens=16,
        )
    )

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(r.status == "success" for r in rs)
    )
    row = _require_row(rows, lambda r: r.status == "success", "for the chat call")

    assert (row.spend or 0) > 0, f"chat row should cost > 0: {_summarize(rows)}"
    assert row.status == "success"
    assert row.cache_hit != "True", "fresh call must not be a cache hit"
    assert "gemini-2.5-flash" in (row.model or "")

    prompt = row.prompt_tokens or 0
    completion = row.completion_tokens or 0
    total = row.total_tokens or 0
    assert prompt > 0 and completion > 0
    assert total == prompt + completion, f"token arithmetic broken: {_summarize(rows)}"

    if chat.id:
        assert any(
            r.request_id == chat.id for r in rows
        ), f"row request_id != client response.id ({chat.id})"


@pytest.mark.covers("quota_management.spend_tracking.stream.logs_cost")
def test_streaming_chat_completion_tracks_spend(
    client: SpendClient, scoped_key: str
) -> None:
    result = client.chat_stream(
        scoped_key,
        "gemini-2.5-flash",
        f"count to three {unique_marker()}",
        max_tokens=64,
    )
    assert (
        result.ok
    ), f"stream failed (status {result.status_code}): {result.body[:300]}"

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any((r.spend or 0) > 0 for r in rs)
    )
    row = _require_row(
        rows, lambda r: (r.spend or 0) > 0, "with nonzero spend for the stream"
    )
    prompt = row.prompt_tokens or 0
    completion = row.completion_tokens or 0
    assert (
        prompt > 0 and completion > 0
    ), f"streaming tokens not tracked: {_summarize(rows)}"
    assert (row.total_tokens or 0) == prompt + completion


@pytest.mark.covers("quota_management.spend_tracking.messages_bridge.logs_cost")
def test_streaming_messages_via_responses_bridge_tracks_spend(
    client: SpendClient, scoped_key: str
) -> None:
    """A streaming anthropic-format /v1/messages request served by an openai-provider
    model is bridged through litellm's anthropic-messages -> Responses adapter, and
    consuming the whole SSE stream writes exactly one costed spend row.

    The deployment is a Responses-only OpenAI model (gpt-5.3-codex, exposed only on
    /v1/responses), so a served call could not have taken the chat-completions bridge:
    that path would 404 at OpenAI on an endpoint the model does not have. The row
    proving the Responses path carries custom_llm_provider "openai" (the openai
    backend served it) under a call_type that keeps the /v1/messages billing identity
    (never a chat call_type), with nonzero cost and prompt/completion tokens that the
    bridge must aggregate out of the consumed stream.
    """
    result = client.messages_stream(
        scoped_key,
        "openai-responses-codex",
        f"reply with exactly one word {unique_marker()}",
        max_tokens=64,
    )
    assert (
        result.ok
    ), f"bridged /v1/messages stream failed (status {result.status_code}): {result.body[:300]}"
    assert result.is_streaming, (
        f"expected an SSE stream from /v1/messages, got content-type "
        f"{result.content_type!r}"
    )
    assert result.chunks > 0, "no SSE events were consumed from the /v1/messages stream"
    assert (
        result.stream_error is None
    ), f"the /v1/messages stream carried an error event: {result.stream_error}"

    def is_bridged_costed(row: SpendLogRow) -> bool:
        return (row.spend or 0) > 0 and "anthropic_messages" in (row.call_type or "")

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(is_bridged_costed(r) for r in rs)
    )
    costed = [r for r in rows if (r.spend or 0) > 0]
    bridged = [r for r in costed if is_bridged_costed(r)]
    assert bridged == costed, (
        f"a costed row was not billed as a /v1/messages call (wrong call_type); "
        f"the bridge must keep the messages billing identity: {_summarize(rows)}"
    )
    assert len(bridged) == 1, (
        f"expected exactly one costed row for the bridged stream, saw {_summarize(rows)}"
    )

    row = bridged[0]
    assert row.custom_llm_provider == "openai", (
        f"bridged row not attributed to the openai Responses backend "
        f"(custom_llm_provider {row.custom_llm_provider!r}): {_summarize(rows)}"
    )
    assert "codex" in (row.model or ""), (
        f"row model {row.model!r} is not the Responses-only codex deployment"
    )

    prompt = row.prompt_tokens or 0
    completion = row.completion_tokens or 0
    assert (
        prompt > 0 and completion > 0
    ), f"bridged stream tokens not tracked: {_summarize(rows)}"
    assert (row.total_tokens or 0) == prompt + completion, (
        f"token arithmetic broken on the bridged row: {_summarize(rows)}"
    )


@pytest.mark.covers("quota_management.spend_tracking.embeddings.logs_cost")
@pytest.mark.covers("llm.embeddings.openai.basic.nonstream.cost_logged")
def test_embedding_writes_nonzero_spend_row(
    client: SpendClient, scoped_key: str
) -> None:
    _ = unwrap(
        client.embed(
            scoped_key,
            "openai-text-embedding-3-small",
            f"vectorize this sentence {unique_marker()}",
        )
    )

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any((r.spend or 0) > 0 for r in rs)
    )
    row = _require_row(
        rows, lambda r: (r.spend or 0) > 0, "with nonzero spend for the embedding"
    )
    assert (row.prompt_tokens or 0) > 0
    assert (row.completion_tokens or 0) == 0, "embeddings have no completion tokens"
    assert "text-embedding-3-small" in (row.model or "")


@pytest.mark.covers("quota_management.spend_tracking.cache_hit.zero_cost")
def test_cache_hit_is_zero_cost_and_suffixed(
    client: SpendClient, scoped_key: str
) -> None:
    # Unique marker shared by both calls: call 1 is a guaranteed cache MISS (fresh
    # content, paid), call 2 repeats the identical request and HITS the cache just
    # populated. The marker keeps each run isolated - a fixed prompt would persist
    # in the shared response cache across runs and make both calls hit (flaky).
    prompt = f"What is the capital of France? Answer in one word. {unique_marker()}"
    _ = unwrap(client.chat(scoped_key, "gemini-2.5-flash", prompt, max_tokens=16, cache=None))
    _ = unwrap(client.chat(scoped_key, "gemini-2.5-flash", prompt, max_tokens=16, cache=None))

    rows = client.poll_logs_for_key(
        scoped_key,
        predicate=lambda rs: any(r.cache_hit == "True" for r in rs)
        and any(r.cache_hit != "True" for r in rs),
    )
    cache_row = _require_row(
        rows,
        lambda r: r.cache_hit == "True",
        "with cache_hit=True (caching is enabled on the e2e proxy, so an identical "
        "repeat call must hit the cache)",
    )
    assert (
        cache_row.spend or 0
    ) == 0.0, f"cache hit was charged (double-charge regression): {_summarize(rows)}"
    assert "_cache_hit" in (cache_row.request_id or ""), (
        "cache-hit row missing the _cache_hit request_id suffix; "
        "duplicate-key collisions will silently drop rows"
    )
    paid_rows = [r for r in rows if r.cache_hit != "True"]
    assert any(
        (r.spend or 0) > 0 for r in paid_rows
    ), f"the non-cached call should still be charged: {_summarize(rows)}"


@pytest.mark.covers("quota_management.spend_tracking.key_rollup.matches_sum_of_logs")
def test_key_spend_equals_sum_of_logs(client: SpendClient, scoped_key: str) -> None:
    for _ in range(2):
        _ = unwrap(
            client.chat(
                scoped_key,
                "gemini-2.5-flash",
                f"say hi {unique_marker()}",
                max_tokens=16,
            )
        )

    rows = client.poll_logs_for_key(
        scoped_key,
        min_rows=2,
        predicate=lambda rs: sum((r.spend or 0) for r in rs) > 0,
    )
    assert len(rows) >= 2, f"expected >=2 rows for the key, saw {_summarize(rows)}"
    logs_total = sum((r.spend or 0) for r in rows)
    assert logs_total > 0

    key_spend = client.poll_key_spend(scoped_key, minimum=logs_total * 0.999)
    assert _approx_equal(
        key_spend, logs_total
    ), f"key aggregate {key_spend} != sum of logs {logs_total}; rows: {_summarize(rows)}"


@pytest.mark.replayable
@pytest.mark.covers("quota_management.spend_tracking.concurrent_burst.loses_no_spend")
def test_burst_of_concurrent_calls_loses_no_spend(
    client: SpendClient, resources: ResourceManager
) -> None:
    from spend_reconciliation import TeamTraffic, assert_logs_match, create_traffic

    traffic: Final = create_traffic(client, resources)

    def assert_team(team: TeamTraffic) -> None:
        assert_logs_match(client, team)
        key_spend: Final = client.poll_key_spend(team.key, minimum=team.spend * 0.999999)
        assert isclose(key_spend, team.spend, rel_tol=1e-6, abs_tol=1e-9)

    for team in traffic:
        assert_team(team)


@pytest.mark.covers("quota_management.spend_tracking.pagination.keeps_total")
def test_spend_logs_v2_pagination_caps_pages_and_keeps_total(
    client: SpendClient, scoped_key: str
) -> None:
    """/spend/logs/v2 pagination contract for the key filter: page_size caps the
    rows returned, total counts every row for the filter (so with page_size=1,
    total_pages == total), a page past the end returns no rows while reporting
    the same total (an out-of-range page must not reset the count the UI
    paginates by), and a filter matching nothing reports zero without erroring.

    Unlike /spend/logs, the v2 filter matches the hashed token exactly as stored
    on the row (the form the UI passes), not the raw sk- key, so the filter value
    is read off the rows the poll returned."""
    for _ in range(2):
        _ = unwrap(
            client.chat(
                scoped_key,
                "gemini-2.5-flash",
                f"page fodder {unique_marker()}",
                max_tokens=16,
            )
        )
    rows = client.poll_logs_for_key(
        scoped_key, min_rows=2, predicate=lambda rs: sum((r.spend or 0) for r in rs) > 0
    )
    hashed_key = rows[0].api_key
    assert hashed_key, f"polled rows carry no api_key: {_summarize(rows)}"

    first = client.spend_logs_page(api_key=hashed_key, page=1, page_size=1)
    assert first.total >= 2, f"expected >=2 rows for the key, got total={first.total}"
    assert len(first.data) == 1, f"page_size=1 returned {len(first.data)} rows"
    assert first.total_pages == first.total, (
        f"page_size=1 must give one page per row: "
        f"total={first.total} total_pages={first.total_pages}"
    )

    beyond = client.spend_logs_page(
        api_key=hashed_key, page=first.total_pages + 7, page_size=1
    )
    assert beyond.data == [], f"out-of-range page returned rows: {beyond.data}"
    assert beyond.total == first.total, (
        f"out-of-range page changed the total: {beyond.total} != {first.total}"
    )

    nomatch = client.spend_logs_page(
        api_key=f"sk-no-such-key-{unique_marker()}", page=1, page_size=1
    )
    assert nomatch.total == 0 and nomatch.data == [], (
        f"filter matching nothing must report zero: "
        f"total={nomatch.total} rows={len(nomatch.data)}"
    )


@pytest.mark.covers("quota_management.spend_tracking.tags.attributes_spend")
def test_request_tags_round_trip(client: SpendClient, scoped_key: str) -> None:
    tag = f"e2e-spend-{unique_marker()}"
    _ = unwrap(
        client.chat(
            scoped_key, "gemini-2.5-flash", "tagged request", tags=[tag], max_tokens=16
        )
    )

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(tag in (r.request_tags or []) for r in rs)
    )
    _require_row(
        rows, lambda r: tag in (r.request_tags or []), f"carrying request tag {tag!r}"
    )


@pytest.mark.covers("quota_management.spend_tracking.tags.attributes_spend")
def test_tag_spend_matches_sum_of_tagged_logs(
    client: SpendClient, scoped_key: str
) -> None:
    # Unique tag so /spend/tags can't be polluted by other rows; unique content
    # per call so both are fresh misses (paid), not cache hits.
    tag = f"e2e-tagspend-{unique_marker()}"
    for _ in range(2):
        _ = unwrap(
            client.chat(
                scoped_key,
                "gemini-2.5-flash",
                f"hi {unique_marker()}",
                tags=[tag],
                max_tokens=16,
            )
        )

    rows = client.poll_logs_for_key(
        scoped_key,
        min_rows=2,
        predicate=lambda rs: sum((r.spend or 0) for r in rs) > 0,
    )
    tagged = [r for r in rows if tag in (r.request_tags or [])]
    assert len(tagged) >= 2, f"expected 2 tagged rows, saw {_summarize(rows)}"
    logs_total = sum((r.spend or 0) for r in tagged)
    assert logs_total > 0

    entry = client.poll_tag_spend(tag, minimum=logs_total * 0.999)
    assert entry is not None, f"tag {tag!r} never appeared in /spend/tags"
    assert _approx_equal(entry.total_spend or 0, logs_total), (
        f"/spend/tags total_spend {entry} != sum of tagged rows {logs_total}"
    )
    assert (entry.log_count or 0) == len(tagged), (
        f"/spend/tags log_count {entry.log_count} != tagged rows {len(tagged)}"
    )


@pytest.mark.covers("quota_management.spend_tracking.end_user.attributes_spend")
def test_end_user_spend_attributed_on_row(
    client: SpendClient, scoped_key: str, resources: ResourceManager
) -> None:
    customer = resources.customer(f"e2e-cust-{unique_marker()}")
    _ = unwrap(
        client.chat(scoped_key, "gemini-2.5-flash", "hi", user=customer, max_tokens=16)
    )

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(r.end_user == customer for r in rs)
    )
    row = _require_row(
        rows, lambda r: r.end_user == customer, f"attributed to end_user {customer!r}"
    )
    assert (row.spend or 0) > 0, f"end-user row should cost > 0: {_summarize(rows)}"


@pytest.mark.covers("quota_management.spend_tracking.end_user.attributes_responses_header")
@pytest.mark.parametrize("header", ["x-litellm-customer-id", "x-litellm-end-user-id"])
def test_end_user_header_attributes_responses_row(
    client: SpendClient, scoped_key: str, resources: ResourceManager, header: str
) -> None:
    """Codex CLI has no body field for the end user, so its config.toml http_headers
    attach the customer header (and x-litellm-tags) to every /v1/responses call.
    A regression that stops reading either header on the Responses route, drops the
    tags, costs the row at zero, or leaves the customer's own spend total behind the
    row fails here."""
    customer = resources.customer(f"e2e-codex-{unique_marker()}")
    tag = f"codex-{unique_marker()}"
    headers = ClientAttributionHeaders.model_validate(
        {"authorization": f"Bearer {scoped_key}", header: customer, "x-litellm-tags": tag}
    )
    sent = client.send_responses_with_headers(
        headers, "openai-responses-codex", f"one word {unique_marker()}"
    )
    assert sent.ok, f"/v1/responses failed with {sent.status_code}: {sent.body[:300]}"

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(r.end_user == customer for r in rs)
    )
    row = _require_row(
        rows, lambda r: r.end_user == customer, f"attributed to end_user {customer!r} via {header}"
    )
    assert row.call_type == "aresponses", f"row is not a Responses row: {_summarize(rows)}"
    assert tag in (row.request_tags or []), f"tag {tag!r} missing from {row.request_tags}"
    assert (row.spend or 0) > 0, f"end-user row should cost > 0: {_summarize(rows)}"
    customer_total = client.poll_customer_spend(customer)
    assert _approx_equal(customer_total, row.spend or 0), (
        f"/customer/info spend {customer_total} != the row's {row.spend}: {_summarize(rows)}"
    )


@pytest.mark.covers("quota_management.spend_tracking.per_model.writes_own_rows")
def test_each_model_on_a_shared_key_gets_its_own_row(
    client: SpendClient, scoped_key: str
) -> None:
    """One key calling two different models, on two providers, gets one spend row per
    call - each carrying its own model and a nonzero cost, under distinct request_ids
    that match the call's response id. Pins per-model/per-provider attribution: a
    regression that stamps the wrong model on the row, bills a call's cost to the
    sibling deployment, or collapses both calls onto one request_id fails here."""
    gemini = unwrap(
        client.chat(
            scoped_key, "gemini-2.5-flash", f"one word {unique_marker()}", max_tokens=16
        )
    )
    claude = unwrap(
        client.chat(
            scoped_key, "claude-haiku-4-5", f"one word {unique_marker()}", max_tokens=16
        )
    )

    def both_models_costed(rows: list[SpendLogRow]) -> bool:
        costed = [r.model or "" for r in rows if (r.spend or 0) > 0]
        return any("gemini-2.5-flash" in m for m in costed) and any(
            "claude-haiku-4-5" in m for m in costed
        )

    rows = client.poll_logs_for_key(scoped_key, min_rows=2, predicate=both_models_costed)
    gemini_row = _require_row(
        rows, lambda r: "gemini-2.5-flash" in (r.model or ""), "for the gemini call"
    )
    claude_row = _require_row(
        rows, lambda r: "claude-haiku-4-5" in (r.model or ""), "for the claude call"
    )

    assert (gemini_row.spend or 0) > 0, f"gemini row should cost > 0: {_summarize(rows)}"
    assert (claude_row.spend or 0) > 0, f"claude row should cost > 0: {_summarize(rows)}"
    assert (
        gemini_row.request_id != claude_row.request_id
    ), f"two distinct calls collapsed onto one request_id: {_summarize(rows)}"
    if gemini.id:
        assert (
            gemini_row.request_id == gemini.id
        ), f"gemini row request_id {gemini_row.request_id} != response id {gemini.id}"
    if claude.id:
        assert (
            claude_row.request_id == claude.id
        ), f"claude row request_id {claude_row.request_id} != response id {claude.id}"


@pytest.mark.covers("quota_management.spend_tracking.failure.writes_failure_row")
def test_failure_call_writes_failure_status_row(
    client: SpendClient, resources: ResourceManager, scoped_key: str
) -> None:
    model = f"e2e-spend-failure-{unique_marker()}"
    model_id = client.proxy.create_model(
        model,
        LiteLLMParamsBody(model="openai/gpt-5.5", api_key="sk-invalid-e2e-failure-row"),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))

    result = client.chat(scoped_key, model, f"trigger failure {unique_marker()}", max_tokens=1)
    assert not is_ok(result), (
        f"a call to a deployment with an invalid upstream key must fail, not succeed: {result}"
    )

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(r.status == "failure" for r in rs)
    )
    failure_row = _require_row(
        rows, lambda r: r.status == "failure", "with status=failure for the rejected call"
    )
    assert (failure_row.spend or 0) == 0.0, "failed call must not be charged"


@pytest.mark.covers("quota_management.spend_tracking.failure.writes_normalized_error")
def test_failure_rows_share_normalized_error_across_provider_wording(
    client: SpendClient, resources: ResourceManager, scoped_key: str
) -> None:
    """Two upstream auth failures with different provider wording land as failure rows
    whose metadata.error_information keeps each provider's own error_message and
    carries the same stable normalized_error cluster key."""
    marker = unique_marker()
    deployments: Final = (
        (f"e2e-norm-openai-{marker}", "openai/gpt-5.5"),
        (f"e2e-norm-anthropic-{marker}", "anthropic/claude-haiku-4-5"),
    )
    for name, provider_model in deployments:
        model_id = client.proxy.create_model(
            name, LiteLLMParamsBody(model=provider_model, api_key=f"sk-invalid-{marker}")
        )
        resources.defer(lambda model_id=model_id: client.proxy.delete_model(model_id))
        result = client.chat(scoped_key, name, f"normalize failure {marker}", max_tokens=1)
        assert not is_ok(result), f"{name}: invalid upstream key must fail the call, got {result}"

    rows = client.poll_logs_for_key(
        scoped_key,
        min_rows=2,
        predicate=lambda rs: sum(1 for r in rs if r.status == "failure") >= 2,
    )
    failure_rows = [r for r in rows if r.status == "failure"]
    assert len(failure_rows) == 2, f"expected one failure row per deployment: {_summarize(rows)}"

    infos = [r.metadata.error_information if r.metadata else None for r in failure_rows]
    assert all(info is not None for info in infos), (
        f"failure rows must carry metadata.error_information: {[r.model_dump() for r in failure_rows]}"
    )
    messages = {info.error_message for info in infos if info is not None}
    assert len(messages) == 2, f"provider wording must stay distinct in error_message: {messages}"
    normalized = {info.normalized_error for info in infos if info is not None}
    assert normalized == {"401_AUTHENTICATION_FAILED"}, (
        f"both auth failures must share one normalized_error cluster key; saw {normalized} "
        f"for messages {messages}"
    )


@pytest.mark.covers("quota_management.spend_tracking.failure.attributes_provider")
def test_pre_call_rejection_row_attributes_provider_and_model_id(
    client: SpendClient, resources: ResourceManager
) -> None:
    """A request the proxy rejects before the router picks a deployment (here the
    key's rpm limit, a pre_call_hook 429) never reaches the code that stamps the
    deployment onto the log. The failure row must still carry the provider and
    model_id of the model group's only deployment, so per-provider failure reports
    can count it."""
    model = f"e2e-spend-precall-{unique_marker()}"
    model_id = client.proxy.create_model(
        model, LiteLLMParamsBody(model="openai/gpt-5.5", api_key="os.environ/OPENAI_API_KEY")
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    key = client.proxy.generate_key(KeyGenerateBody(models=[model], rpm_limit=1))
    resources.defer(lambda: client.proxy.delete_key(key))

    unwrap(client.chat(key, model, f"reply with one word {unique_marker()}", max_tokens=8))
    rejected = client.chat(key, model, f"over the rpm limit {unique_marker()}", max_tokens=8)
    assert isinstance(rejected, RateLimitedError), (
        f"the second call on an rpm_limit=1 key must be rejected with 429 before routing, got {rejected}"
    )

    rows = client.poll_logs_for_key(
        key,
        min_rows=2,
        predicate=lambda rs: {r.status for r in rs} >= {"success", "failure"},
    )
    success_row = _require_row(rows, lambda r: r.status == "success", "for the served call")
    failure_row = _require_row(rows, lambda r: r.status == "failure", "for the rate-limited call")

    assert failure_row.custom_llm_provider == success_row.custom_llm_provider, (
        f"rejected call lost its provider: failure row {failure_row.custom_llm_provider!r} vs "
        f"served row {success_row.custom_llm_provider!r}; {_summarize(rows)}"
    )
    assert failure_row.model_id == model_id, (
        f"rejected call lost its deployment: failure row model_id {failure_row.model_id!r} vs "
        f"registered {model_id!r}; {_summarize(rows)}"
    )


@pytest.mark.covers("quota_management.spend_tracking.spend_calculate.returns_cost")
def test_spend_calculate_returns_nonzero_cost(client: SpendClient) -> None:
    cost = client.calculate_spend(
        "gemini-2.5-flash", "estimate the cost of this request"
    )
    assert cost > 0, (
        "/spend/calculate returned 0 for gemini-2.5-flash; "
        "cost map may be missing this model"
    )


def test_spend_logs_endpoint_returns_spend(
    client: SpendClient, scoped_key: str
) -> None:
    """The /spend/logs read endpoint returns a 200 carrying the key's spend, never a
    5xx. Regression for intermittent 500s (DB query / serialization errors under load)
    on this endpoint: every poll asserts a success response, not just a truthy row
    list, so a 500 fails loudly instead of being swallowed as 'no rows yet'; the
    call's nonzero spend must surface before the deadline."""
    unwrap(
        client.chat(
            scoped_key, "gemini-2.5-flash", f"spend logs {unique_marker()}", max_tokens=16
        )
    )

    proxy = client.proxy
    deadline = time.monotonic() + proxy.poll_timeout
    while True:
        result = proxy.transport.get(
            "/spend/logs",
            headers=proxy.transport.master,
            params=SpendLogsParams(api_key=scoped_key),
            response_type=SpendLogs,
        )
        assert isinstance(result, Success), f"/spend/logs did not return 200 OK: {result}"
        rows = result.data.root
        if sum((r.spend or 0) for r in rows) > 0:
            return
        if time.monotonic() >= deadline:
            pytest.fail(
                f"/spend/logs never surfaced the key's spend before the deadline; "
                f"saw {_summarize(rows)}"
            )
        time.sleep(proxy.poll_interval)
