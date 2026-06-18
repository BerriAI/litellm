"""Live end-to-end spend-tracking tests against a running proxy.

Run against a proxy started with tests/e2e_tests/docker-config.yaml (or the
gateway config). Coverage rationale: SPEND_TRACKING_COVERAGE_MATRIX.md.

Model names are literals from that config: chat tests hit "gemini-2.5-flash",
embedding tests hit "openai-text-embedding-3-small".

Every test: fresh scoped key (isolation) -> real provider call ->
require_successful_call (skip iff env can't make it) -> poll /spend/logs to a
deadline (rows land ~60s later via proxy_batch_write_at) -> assert invariants on
the real row (spend, token arithmetic, status, cache).

Assertions target invariants, not literals: a regression in the spend pipeline
fails the test; a pricing or token-count drift does not.
"""

from typing import Callable, Dict, List

import pytest

from lifecycle import ResourceManager
from spend_e2e_client import (
    SpendE2EClient,
    SpendLogRow,
    require_successful_call,
    unique_marker,
)

pytestmark = pytest.mark.e2e


def _f(value: object) -> float:
    return float(value) if value is not None else 0.0  # type: ignore[arg-type]


def _i(value: object) -> int:
    return int(value) if value is not None else 0  # type: ignore[arg-type]


def _s(value: object) -> str:
    return str(value) if value is not None else ""


def _summarize(rows: List[SpendLogRow]) -> List[Dict[str, object]]:
    keys = (
        "request_id",
        "model",
        "spend",
        "status",
        "cache_hit",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )
    return [{k: r.get(k) for k in keys} for r in rows]


def _require_row(
    rows: List[SpendLogRow], predicate: Callable[[SpendLogRow], bool], what: str
) -> SpendLogRow:
    matches = [r for r in rows if predicate(r)]
    assert matches, (
        f"no SpendLogs row {what} after polling; saw {len(rows)} row(s): "
        f"{_summarize(rows)}"
    )
    return matches[0]


def test_chat_completion_writes_nonzero_spend_row(
    client: SpendE2EClient, scoped_key: str
) -> None:
    result = client.chat(
        scoped_key,
        "gemini-2.5-flash",
        f"reply with one word {unique_marker()}",
        extra_body={"max_tokens": 16},
    )
    require_successful_call(result)

    rows = client.poll_logs_for_key(
        scoped_key,
        predicate=lambda rs: any(_s(r.get("status")) == "success" for r in rs),
    )
    row = _require_row(
        rows, lambda r: _s(r.get("status")) == "success", "for the chat call"
    )

    assert _f(row.get("spend")) > 0, f"chat row should cost > 0: {_summarize(rows)}"
    assert _s(row.get("status")) == "success"
    assert _s(row.get("cache_hit")) != "True", "fresh call must not be a cache hit"
    assert "gemini-2.5-flash" in _s(row.get("model"))

    prompt = _i(row.get("prompt_tokens"))
    completion = _i(row.get("completion_tokens"))
    total = _i(row.get("total_tokens"))
    assert prompt > 0 and completion > 0
    assert total == prompt + completion, f"token arithmetic broken: {_summarize(rows)}"

    if result.response_id:
        assert any(
            _s(r.get("request_id")) == result.response_id for r in rows
        ), f"row request_id != client response.id ({result.response_id})"


def test_streaming_chat_completion_tracks_spend(
    client: SpendE2EClient, scoped_key: str
) -> None:
    result = client.chat(
        scoped_key,
        "gemini-2.5-flash",
        f"count to three {unique_marker()}",
        stream=True,
        extra_body={"max_tokens": 64},
    )
    require_successful_call(result)

    rows = client.poll_logs_for_key(
        scoped_key,
        predicate=lambda rs: any(_f(r.get("spend")) > 0 for r in rs),
    )
    row = _require_row(
        rows, lambda r: _f(r.get("spend")) > 0, "with nonzero spend for the stream"
    )
    prompt = _i(row.get("prompt_tokens"))
    completion = _i(row.get("completion_tokens"))
    assert prompt > 0 and completion > 0, f"streaming tokens not tracked: {_summarize(rows)}"
    assert _i(row.get("total_tokens")) == prompt + completion


def test_embedding_writes_nonzero_spend_row(
    client: SpendE2EClient, scoped_key: str
) -> None:
    result = client.embed(
        scoped_key,
        "openai-text-embedding-3-small",
        f"vectorize this sentence {unique_marker()}",
    )
    require_successful_call(result)

    rows = client.poll_logs_for_key(
        scoped_key, predicate=lambda rs: any(_f(r.get("spend")) > 0 for r in rs)
    )
    row = _require_row(
        rows, lambda r: _f(r.get("spend")) > 0, "with nonzero spend for the embedding"
    )
    assert _i(row.get("prompt_tokens")) > 0
    assert _i(row.get("completion_tokens")) == 0, "embeddings have no completion tokens"
    assert "text-embedding-3-small" in _s(row.get("model"))


def test_cache_hit_is_zero_cost_and_suffixed(
    client: SpendE2EClient, scoped_key: str
) -> None:
    # Unique marker shared by both calls: call 1 is a guaranteed cache MISS (fresh
    # content, paid), call 2 repeats the identical request and HITS the cache just
    # populated. The marker keeps each run isolated - a fixed prompt would persist
    # in the shared response cache across runs and make both calls hit (flaky).
    prompt = f"What is the capital of France? Answer in one word. {unique_marker()}"
    first = client.chat(
        scoped_key, "gemini-2.5-flash", prompt, extra_body={"max_tokens": 16}
    )
    require_successful_call(first)
    second = client.chat(
        scoped_key, "gemini-2.5-flash", prompt, extra_body={"max_tokens": 16}
    )
    require_successful_call(second)

    rows = client.poll_logs_for_key(
        scoped_key,
        predicate=lambda rs: any(_s(r.get("cache_hit")) == "True" for r in rs),
    )
    cache_rows = [r for r in rows if _s(r.get("cache_hit")) == "True"]
    if not cache_rows:
        pytest.skip(
            "no cache-hit row observed; caching may be disabled on this proxy. "
            f"rows seen: {_summarize(rows)}"
        )

    cache_row = cache_rows[0]
    assert _f(cache_row.get("spend")) == 0.0, (
        f"cache hit was charged (double-charge regression): {_summarize(rows)}"
    )
    assert "_cache_hit" in _s(cache_row.get("request_id")), (
        "cache-hit row missing the _cache_hit request_id suffix; "
        "duplicate-key collisions will silently drop rows"
    )
    paid_rows = [r for r in rows if _s(r.get("cache_hit")) != "True"]
    assert any(_f(r.get("spend")) > 0 for r in paid_rows), (
        f"the non-cached call should still be charged: {_summarize(rows)}"
    )


def test_key_spend_equals_sum_of_logs(
    client: SpendE2EClient, scoped_key: str
) -> None:
    for _ in range(2):
        result = client.chat(
            scoped_key,
            "gemini-2.5-flash",
            f"say hi {unique_marker()}",
            extra_body={"max_tokens": 16},
        )
        require_successful_call(result)

    rows = client.poll_logs_for_key(
        scoped_key,
        min_rows=2,
        predicate=lambda rs: sum(_f(r.get("spend")) for r in rs) > 0,
    )
    assert len(rows) >= 2, f"expected >=2 rows for the key, saw {_summarize(rows)}"
    logs_total = sum(_f(r.get("spend")) for r in rows)
    assert logs_total > 0

    key_spend = client.poll_key_spend(scoped_key, minimum=logs_total * 0.999)
    assert key_spend == pytest.approx(logs_total, rel=1e-2, abs=1e-9), (
        f"key aggregate {key_spend} != sum of logs {logs_total}; "
        f"rows: {_summarize(rows)}"
    )


def test_request_tags_round_trip(
    client: SpendE2EClient, scoped_key: str
) -> None:
    tag = f"e2e-spend-{unique_marker()}"
    result = client.chat(
        scoped_key,
        "gemini-2.5-flash",
        "tagged request",
        metadata={"tags": [tag]},
        extra_body={"max_tokens": 16},
    )
    require_successful_call(result)

    rows = client.poll_logs_for_key(
        scoped_key,
        predicate=lambda rs: any(tag in _s(r.get("request_tags")) for r in rs),
    )
    _require_row(
        rows,
        lambda r: tag in _s(r.get("request_tags")),
        f"carrying request tag {tag!r}",
    )


def test_tag_spend_matches_sum_of_tagged_logs(
    client: SpendE2EClient, scoped_key: str
) -> None:
    # Unique tag so /spend/tags can't be polluted by other rows; unique content
    # per call so both are fresh misses (paid), not cache hits.
    tag = f"e2e-tagspend-{unique_marker()}"
    for _ in range(2):
        result = client.chat(
            scoped_key,
            "gemini-2.5-flash",
            f"hi {unique_marker()}",
            metadata={"tags": [tag]},
            extra_body={"max_tokens": 16},
        )
        require_successful_call(result)

    rows = client.poll_logs_for_key(
        scoped_key,
        min_rows=2,
        predicate=lambda rs: sum(_f(r.get("spend")) for r in rs) > 0,
    )
    tagged = [r for r in rows if tag in _s(r.get("request_tags"))]
    assert len(tagged) >= 2, f"expected 2 tagged rows, saw {_summarize(rows)}"
    logs_total = sum(_f(r.get("spend")) for r in tagged)
    assert logs_total > 0

    entry = client.poll_tag_spend(tag, minimum=logs_total * 0.999)
    assert entry is not None, f"tag {tag!r} never appeared in /spend/tags"
    assert _f(entry.get("total_spend")) == pytest.approx(
        logs_total, rel=1e-2, abs=1e-9
    ), f"/spend/tags total_spend {entry} != sum of tagged rows {logs_total}"
    assert _i(entry.get("log_count")) == len(tagged), (
        f"/spend/tags log_count {entry.get('log_count')} != tagged rows {len(tagged)}"
    )


def test_end_user_spend_attributed_on_row(
    client: SpendE2EClient, scoped_key: str, resources: ResourceManager
) -> None:
    customer = resources.customer(f"e2e-cust-{unique_marker()}")
    result = client.chat(
        scoped_key,
        "gemini-2.5-flash",
        "hi",
        extra_body={"user": customer, "max_tokens": 16},
    )
    require_successful_call(result)

    rows = client.poll_logs_for_key(
        scoped_key,
        predicate=lambda rs: any(_s(r.get("end_user")) == customer for r in rs),
    )
    row = _require_row(
        rows,
        lambda r: _s(r.get("end_user")) == customer,
        f"attributed to end_user {customer!r}",
    )
    assert _f(row.get("spend")) > 0, f"end-user row should cost > 0: {_summarize(rows)}"


def test_failure_call_writes_failure_status_row(
    client: SpendE2EClient, scoped_key: str
) -> None:
    result = client.chat(
        scoped_key, "gemini-2.5-flash", "", extra_body={"max_tokens": 1}
    )
    if result.ok:
        pytest.skip("call unexpectedly succeeded; could not induce a failure row")

    rows = client.poll_logs_for_key(
        scoped_key,
        predicate=lambda rs: any(_s(r.get("status")) == "failure" for r in rs),
    )
    failure_rows = [r for r in rows if _s(r.get("status")) == "failure"]
    if not failure_rows:
        pytest.skip(
            "no failure-status row was logged for the rejected call; "
            "failure logging is environment-specific"
        )
    assert _f(failure_rows[0].get("spend")) == 0.0, "failed call must not be charged"


def test_spend_calculate_returns_nonzero_cost(client: SpendE2EClient) -> None:
    cost = client.calculate_spend(
        "gemini-2.5-flash", "estimate the cost of this request"
    )
    assert cost > 0, (
        "/spend/calculate returned 0 for gemini-2.5-flash; "
        "cost map may be missing this model"
    )
