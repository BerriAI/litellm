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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from e2e_http import Result, Success
from lifecycle import ResourceManager
from models import ChatResponse, LiteLLMParamsBody, SpendLogsPage, SpendLogsPageParams
from spend_e2e_client import SpendClient, SpendLogRow, is_ok, unique_marker, unwrap

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


@pytest.mark.covers("quota_management.spend_tracking.embeddings.logs_cost")
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
    _ = unwrap(client.chat(scoped_key, "gemini-2.5-flash", prompt, max_tokens=16))
    _ = unwrap(client.chat(scoped_key, "gemini-2.5-flash", prompt, max_tokens=16))

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(r.cache_hit == "True" for r in rs)
    )
    cache_rows = [r for r in rows if r.cache_hit == "True"]
    if not cache_rows:
        pytest.skip(
            "no cache-hit row observed; caching may be disabled on this proxy. "
            f"rows seen: {_summarize(rows)}"
        )

    cache_row = cache_rows[0]
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


@pytest.mark.covers("quota_management.spend_tracking.concurrent_burst.loses_no_spend")
def test_burst_of_concurrent_calls_loses_no_spend(
    client: SpendClient, scoped_key: str
) -> None:
    """Six concurrent calls on one key: every call lands its own spend row under a
    distinct request_id and the key aggregate equals the sum of the rows.
    Sequential accuracy is covered by test_key_spend_equals_sum_of_logs; this pins
    the concurrent increment path (parallel writers racing on one key's counter),
    where a lost update can never be reproduced by sequential calls."""
    burst = 6

    def call(idx: int) -> Result[ChatResponse]:
        return client.chat(
            scoped_key,
            "gemini-2.5-flash",
            f"burst call {idx} {unique_marker()}",
            max_tokens=16,
        )

    with ThreadPoolExecutor(max_workers=burst) as pool:
        results = tuple(pool.map(call, range(burst)))
    failed = [r for r in results if not is_ok(r)]
    assert not failed, f"{len(failed)}/{burst} burst calls failed; first: {failed[0]}"

    rows = client.poll_logs_for_key(
        scoped_key,
        min_rows=burst,
        predicate=lambda rs: len([r for r in rs if (r.spend or 0) > 0]) >= burst,
    )
    costed = [r for r in rows if (r.spend or 0) > 0]
    assert len(costed) >= burst, (
        f"only {len(costed)}/{burst} burst calls produced a costed row - "
        f"rows lost under concurrency: {_summarize(rows)}"
    )
    request_ids = [r.request_id for r in costed]
    assert len(set(request_ids)) == len(request_ids), (
        f"concurrent rows collapsed onto shared request_ids: {_summarize(rows)}"
    )

    logs_total = sum((r.spend or 0) for r in rows)
    key_spend = client.poll_key_spend(scoped_key, minimum=logs_total * 0.999)
    assert _approx_equal(key_spend, logs_total), (
        f"key aggregate {key_spend} != sum of {len(rows)} rows {logs_total} - "
        f"spend increments lost under concurrency: {_summarize(rows)}"
    )


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
    client: SpendClient, resources: ResourceManager
) -> None:
    """A call the provider rejects writes a failure-status spend row at zero cost.

    The failure is induced deterministically: a throwaway deployment credentialed with
    an invalid provider key, so the call is a guaranteed 401 from the provider. The
    previous trigger sent an empty prompt and hoped gemini would reject it; gemini
    accepts it, so the test skipped on every run and this cell asserted nothing while
    still counting as covered. Both skips are gone on purpose - a failure row that
    stops being written is a real spend-tracking regression (an un-logged failure is a
    hole in the audit trail), not an environment quirk to shrug at."""
    model_name = f"e2e-badkey-{unique_marker()}"
    model_id = client.gateway.create_model(
        model_name,
        LiteLLMParamsBody(model="openai/gpt-5.5", api_key="sk-e2e-deliberately-invalid"),
    )
    resources.defer(lambda: client.gateway.delete_model(model_id))
    key = resources.key(models=[model_name])

    result = client.chat(key, model_name, "trigger a provider failure", max_tokens=8)
    assert not is_ok(
        result
    ), f"a deployment with an invalid provider key must fail, got success: {result}"

    rows = client.poll_logs_for_key(
        key, predicate=lambda rs: any(r.status == "failure" for r in rs)
    )
    row = _require_row(
        rows, lambda r: r.status == "failure", "with status=failure for the rejected call"
    )
    assert (
        row.spend or 0
    ) == 0.0, f"a failed call must not be charged: {_summarize(rows)}"


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
    """The spend read endpoint returns a 200 carrying the key's spend, never a 5xx.
    Regression for intermittent 500s (DB query / serialization errors under load): each
    poll asserts a success response, not just a truthy row list, so a 500 fails loudly
    instead of being swallowed as 'no rows yet'.

    Reads /spend/logs/v2, not /spend/logs: the latter is unbounded and on a long-lived
    environment answers with the whole spend table, which has OOM-killed the runner.
    v2 takes an explicit window and pages, and it filters on the hashed token rather
    than the raw sk- key (the raw value silently matches nothing)."""
    unwrap(
        client.chat(
            scoped_key, "gemini-2.5-flash", f"spend logs {unique_marker()}", max_tokens=16
        )
    )

    gateway = client.gateway
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%d %H:%M:%S"
    deadline = time.monotonic() + gateway.poll_timeout
    while True:
        result = gateway.transport.get(
            "/spend/logs/v2",
            headers=gateway.transport.master,
            params=SpendLogsPageParams(
                start_date=(now - timedelta(days=1)).strftime(fmt),
                end_date=(now + timedelta(days=1)).strftime(fmt),
                page=1,
                page_size=100,
                api_key=gateway.hash_token(scoped_key),
            ),
            response_type=SpendLogsPage,
        )
        assert isinstance(
            result, Success
        ), f"/spend/logs/v2 did not return 200 OK: {result}"
        rows = result.data.data
        if sum((r.spend or 0) for r in rows) > 0:
            return
        if time.monotonic() >= deadline:
            pytest.fail(
                f"/spend/logs/v2 never surfaced the key's spend before the deadline; "
                f"saw {_summarize(rows)}"
            )
        time.sleep(gateway.poll_interval)
