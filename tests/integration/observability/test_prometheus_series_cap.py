"""Prometheus series cap on the live proxy: label sets past prometheus_metrics_max_series_per_metric share one
`other` series on every labeled counter and histogram and stay out of the gauges, idle series expire under
prometheus_metrics_ttl_seconds in single-process mode only, and a setting that is not a positive number is
ignored with a warning instead of silencing the metrics."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import eventually, object_value, string_value
from integration._support.prometheus_series import (
    AGENT_HEADERS,
    CACHE_HITS,
    FAILED_FALLBACKS,
    OVERFLOW,
    PROVIDER_OUTAGE,
    PROXY_FAILURES,
    PROXY_REQUESTS,
    REMAINING_REQUESTS,
    REQUESTS,
    SUCCESSFUL_FALLBACKS,
    Call,
    CapRig,
    Key,
    Sample,
    WorkerSamples,
    alias_values,
    chat_once,
    expect_spend_rows,
    families_over,
    gauge_samples,
    label_values,
    overflow_total,
    received_markers,
    scrape,
    series_cap_rig,
    sse_data,
    worker_samples,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(240)

CAP: Final = 3
TTL_SECONDS: Final = 2
CLEANUP_SECONDS: Final = 1
PRIMARY: Final = "primary"
FALLBACK: Final = "fallback"
TTL_IGNORED_WARNING: Final = "prometheus_metrics_ttl_seconds is ignored while PROMETHEUS_MULTIPROC_DIR is set"


def _grew(before: Sequence[Sample], after: Sequence[Sample], name: str, by: int) -> bool:
    return overflow_total(after, name) - overflow_total(before, name) >= by


def _overflowed(rig: CapRig, key: Key, before: Sequence[Sample], requests: int) -> tuple[Sample, ...]:
    """The scrape once the key's requests landed on `other` for both request counters, or as soon as the key got
    a series of its own, so the caller's assertion fails fast on a proxy without the cap."""
    return eventually(
        lambda: scrape(rig.gateway),
        lambda after: (
            (_grew(before, after, REQUESTS, requests) and _grew(before, after, PROXY_REQUESTS, requests))
            or key.alias in alias_values(after, REQUESTS)
        ),
        seconds=60,
    )


def _expect_other(
    rig: CapRig, key: Key, calls: Sequence[Call], response_ids: Sequence[str], before: Sequence[Sample]
) -> None:
    samples: Final = _overflowed(rig, key, before, len(calls))
    assert key.alias not in alias_values(samples, REQUESTS) | alias_values(samples, PROXY_REQUESTS), key.alias
    assert not families_over(samples, CAP), families_over(samples, CAP)
    assert overflow_total(samples, REQUESTS) - overflow_total(before, REQUESTS) == len(calls)
    expect_spend_rows(key.alias, response_ids)
    markers: Final = received_markers(rig.provider)
    assert all(call.marker in markers for call in calls), (calls, markers)


def _bearer(key: Key, call: Call) -> dict[str, str]:
    return {**AGENT_HEADERS, **call.headers, "Authorization": f"Bearer {key.token}"}


class TestCapped:
    def test_openai_sync_chat_past_the_cap_counts_on_other(self, capped: CapRig) -> None:
        """H1: two OpenAI SDK chat completions from a fourth key count on `other` and keep their spend rows."""
        key: Final = capped.key("h1")
        calls: Final = (Call.new(), Call.new())
        before: Final = scrape(capped.gateway)
        client: Final = openai.OpenAI(
            base_url=capped.openai_base, api_key=key.token, default_headers=dict(AGENT_HEADERS), max_retries=0
        )
        completions: Final = tuple(
            client.chat.completions.create(model=capped.model, messages=[call.message], extra_headers=call.headers)
            for call in calls
        )
        assert tuple(completion.choices[0].message.content for completion in completions) == tuple(
            call.answer for call in calls
        )
        _expect_other(capped, key, calls, tuple(completion.id for completion in completions), before)

    async def test_openai_async_stream_past_the_cap_counts_on_other(self, capped: CapRig) -> None:
        """H2: a streamed AsyncOpenAI chat completion from a fourth key counts on `other` once the stream ends."""
        key: Final = capped.key("h2")
        call: Final = Call.new()
        before: Final = scrape(capped.gateway)
        client: Final = openai.AsyncOpenAI(
            base_url=capped.openai_base, api_key=key.token, default_headers=dict(AGENT_HEADERS), max_retries=0
        )
        stream: Final = await client.chat.completions.create(
            model=capped.model, messages=[call.message], stream=True, extra_headers=call.headers
        )
        chunks: Final = tuple([chunk async for chunk in stream])
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == call.answer
        ids: Final = frozenset(chunk.id for chunk in chunks)
        assert len(ids) == 1, ids
        _expect_other(capped, key, (call,), tuple(ids), before)

    def test_anthropic_sync_messages_past_the_cap_counts_on_other(self, capped: CapRig) -> None:
        """H3: an Anthropic SDK /v1/messages call from a fourth key counts on `other`."""
        key: Final = capped.key("h3")
        call: Final = Call.new()
        before: Final = scrape(capped.gateway)
        client: Final = anthropic.Anthropic(
            base_url=capped.base_url, api_key=key.token, default_headers=dict(AGENT_HEADERS), max_retries=0
        )
        message: Final = client.messages.create(
            model=capped.model, max_tokens=64, messages=[call.message], extra_headers=call.headers
        )
        assert "".join(block.text for block in message.content if block.type == "text") == call.answer
        _expect_other(capped, key, (call,), (message.id,), before)

    async def test_anthropic_async_stream_past_the_cap_counts_on_other(self, capped: CapRig) -> None:
        """H4: a streamed AsyncAnthropic /v1/messages call from a fourth key counts on `other`."""
        key: Final = capped.key("h4")
        call: Final = Call.new()
        before: Final = scrape(capped.gateway)
        client: Final = anthropic.AsyncAnthropic(
            base_url=capped.base_url, api_key=key.token, default_headers=dict(AGENT_HEADERS), max_retries=0
        )
        async with client.messages.stream(
            model=capped.model, max_tokens=64, messages=[call.message], extra_headers=call.headers
        ) as stream:
            final: Final = await stream.get_final_message()
        assert "".join(block.text for block in final.content if block.type == "text") == call.answer
        _expect_other(capped, key, (call,), (final.id,), before)

    def test_openai_sync_responses_past_the_cap_counts_on_other(self, capped: CapRig) -> None:
        """H5: an OpenAI SDK /v1/responses call from a fourth key counts on `other`."""
        key: Final = capped.key("h5")
        call: Final = Call.new()
        before: Final = scrape(capped.gateway)
        client: Final = openai.OpenAI(
            base_url=capped.openai_base, api_key=key.token, default_headers=dict(AGENT_HEADERS), max_retries=0
        )
        response: Final = client.responses.create(model=capped.model, input=call.text, extra_headers=call.headers)
        assert response.output_text == call.answer
        _expect_other(capped, key, (call,), (response.id,), before)

    def test_raw_responses_stream_past_the_cap_counts_on_other(self, capped: CapRig) -> None:
        """H6: a raw httpx streamed /v1/responses call from a fourth key counts on `other`."""
        key: Final = capped.key("h6")
        call: Final = Call.new()
        before: Final = scrape(capped.gateway)
        with httpx.Client(base_url=capped.base_url, timeout=60, trust_env=False) as client:
            response: Final = client.post(
                "/v1/responses",
                json={"model": capped.model, "input": call.text, "stream": True},
                headers=_bearer(key, call),
            )
        assert response.status_code == 200, response.text
        events: Final = sse_data(response.text)
        deltas: Final = tuple(event for event in events if event.get("type") == "response.output_text.delta")
        assert "".join(string_value(event["delta"]) for event in deltas) == call.answer
        completed: Final = tuple(event for event in events if event.get("type") == "response.completed")
        assert len(completed) == 1, events
        response_id: Final = string_value(object_value(completed[0]["response"])["id"])
        _expect_other(capped, key, (call,), (response_id,), before)

    def test_gauges_never_get_an_other_series(self, capped: CapRig) -> None:
        """H7: a fourth key's request leaves no gauge sample for it and no gauge sample labeled `other`."""
        key: Final = capped.key("h7")
        before: Final = scrape(capped.gateway)
        assert capped.chat(key, Call.new()).status_code == 200
        samples: Final = _overflowed(capped, key, before, 1)
        assert key.alias not in label_values(samples)
        gauges: Final = gauge_samples(samples)
        assert not any(OVERFLOW in gauge.labels.values() for gauge in gauges), gauges
        for alias in capped.warm_aliases:
            assert any(
                gauge.name == REMAINING_REQUESTS and gauge.labels.get("api_key_alias") == alias for gauge in gauges
            ), alias

    def test_cache_hits_past_the_cap_count_on_other(self, capped: CapRig) -> None:
        """H8: the cache-hit twin: one populating call, hits from the warm keys, then a fourth key's hit on `other`."""
        shared: Final = Call.new()
        first, second, third = capped.warm
        extra: Final = capped.key("h8")
        capped.provider.drain()
        before: Final = scrape(capped.gateway)
        for key in (first, first, second, third, extra):
            response = capped.chat(key, shared)
            assert response.status_code == 200 and shared.answer in response.text, response.text
        samples: Final = eventually(
            lambda: scrape(capped.gateway),
            lambda after: _grew(before, after, CACHE_HITS, 1) or extra.alias in alias_values(after, CACHE_HITS),
            seconds=60,
        )
        assert alias_values(samples, CACHE_HITS) == capped.warm_aliases
        assert overflow_total(samples, CACHE_HITS) - overflow_total(before, CACHE_HITS) == 1
        assert received_markers(capped.provider).count(shared.marker) == 1

    def test_both_workers_share_the_admitted_series(self, capped: CapRig) -> None:
        """H9: fresh connections reach both workers, and each worker's own sample file names only the warm aliases
        while counting the fourth key on `other`, since the admitted sets live in the shared directory."""
        extra: Final = capped.key("h9")

        def send_on_a_fresh_connection() -> tuple[WorkerSamples, ...]:
            assert capped.chat(extra, Call.new()).status_code == 200
            return worker_samples(capped.prom_dir, REQUESTS)

        workers: Final = eventually(
            send_on_a_fresh_connection,
            lambda found: (
                sum(1 for worker in found if worker.overflow > 0) >= 2
                or any(extra.alias in worker.aliases for worker in found)
            ),
            seconds=90,
        )
        assert all(extra.alias not in worker.aliases for worker in workers), workers
        assert sum(1 for worker in workers if worker.overflow > 0) >= 2, workers
        assert frozenset().union(*(worker.aliases for worker in workers)) == capped.warm_aliases, workers

    def test_failures_past_the_cap_count_on_other(self, capped: CapRig) -> None:
        """F1: provider failures fill the failure counter's cap with the warm keys, a fourth key's lands on `other`."""
        key: Final = capped.key("f1")
        call: Final = Call.new()
        before: Final = scrape(capped.gateway)
        capped.outage.set()
        try:
            for warm in capped.warm:
                assert capped.chat(warm, Call.new()).status_code == 500
            response: Final = capped.chat(key, call)
        finally:
            capped.outage.clear()
        assert response.status_code == 500 and PROVIDER_OUTAGE in response.text, response.text
        samples: Final = eventually(
            lambda: scrape(capped.gateway),
            lambda after: _grew(before, after, PROXY_FAILURES, 1) or key.alias in alias_values(after, PROXY_FAILURES),
            seconds=60,
        )
        assert key.alias not in label_values(samples)
        assert overflow_total(samples, PROXY_FAILURES) - overflow_total(before, PROXY_FAILURES) == 1
        assert alias_values(samples, PROXY_FAILURES) == capped.warm_aliases
        expect_spend_rows(key.alias, (), (call.call_id,))

    def test_config_update_cannot_lift_a_yaml_cap(self, capped: CapRig) -> None:
        """E1: /config/update refuses the YAML-owned cap, so a fourth key still lands on `other`."""
        response: Final = capped.gateway.client.post(
            "/config/update",
            json={"litellm_settings": {"prometheus_metrics_max_series_per_metric": 50}},
            headers={"Authorization": f"Bearer {capped.gateway.key}"},
        )
        assert response.status_code == 400, response.text
        key: Final = capped.key("e1")
        before: Final = scrape(capped.gateway)
        assert capped.chat(key, Call.new()).status_code == 200
        samples: Final = _overflowed(capped, key, before, 1)
        assert key.alias not in label_values(samples)


@pytest.fixture(scope="class")
def ttl(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapRig]:
    with series_cap_rig(
        tmp_path_factory.mktemp("series-ttl"),
        {
            "prometheus_metrics_max_series_per_metric": 2,
            "prometheus_metrics_ttl_seconds": TTL_SECONDS,
            "prometheus_metrics_cleanup_interval_seconds": CLEANUP_SECONDS,
        },
        workers=1,
        warm_keys=2,
    ) as rig:
        yield rig


class TestTtl:
    def test_idle_series_expire_and_free_their_slot(self, ttl: CapRig) -> None:
        """T1: a third key lands on `other`; once the idle first key expires, a new key gets its own series."""
        first, second = ttl.warm
        extra: Final = ttl.key("t1-extra")
        assert ttl.chat(extra, Call.new()).status_code == 200
        samples: Final = eventually(
            lambda: scrape(ttl.gateway),
            lambda after: overflow_total(after, REQUESTS) >= 1 or extra.alias in alias_values(after, REQUESTS),
            seconds=60,
        )
        assert extra.alias not in label_values(samples)

        def keep_second_busy() -> tuple[Sample, ...]:
            assert ttl.chat(second, Call.new()).status_code == 200
            return scrape(ttl.gateway)

        expired: Final = eventually(keep_second_busy, lambda after: first.alias not in label_values(after), seconds=30)
        assert second.alias in alias_values(expired, REQUESTS)
        late: Final = ttl.key("t1-late")
        assert ttl.chat(late, Call.new()).status_code == 200
        named: Final = eventually(
            lambda: scrape(ttl.gateway), lambda after: late.alias in alias_values(after, REQUESTS), seconds=30
        )
        assert late.alias in alias_values(named, REQUESTS)


@pytest.fixture(scope="class")
def ttl_multiproc(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapRig]:
    with series_cap_rig(
        tmp_path_factory.mktemp("series-ttl-multiproc"),
        {
            "prometheus_metrics_max_series_per_metric": CAP,
            "prometheus_metrics_ttl_seconds": TTL_SECONDS,
            "prometheus_metrics_cleanup_interval_seconds": CLEANUP_SECONDS,
        },
        workers=2,
        warm_keys=3,
    ) as rig:
        yield rig


class TestTtlMultiproc:
    def test_ttl_is_ignored_with_two_workers_while_the_cap_applies(self, ttl_multiproc: CapRig) -> None:
        """M1: with two workers an idle key keeps its series past the TTL, the cap still applies, and the log says so."""
        first, second, _ = ttl_multiproc.warm
        deadline: Final = time.monotonic() + 2 * TTL_SECONDS
        while time.monotonic() < deadline:
            assert ttl_multiproc.chat(second, Call.new()).status_code == 200
        assert first.alias in alias_values(scrape(ttl_multiproc.gateway), REQUESTS)
        extra: Final = ttl_multiproc.key("m1")
        before: Final = scrape(ttl_multiproc.gateway)
        assert ttl_multiproc.chat(extra, Call.new()).status_code == 200
        samples: Final = _overflowed(ttl_multiproc, extra, before, 1)
        assert extra.alias not in label_values(samples)
        assert TTL_IGNORED_WARNING in ttl_multiproc.proxy.log.read_text()


@pytest.fixture(scope="class")
def ignored(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapRig]:
    with series_cap_rig(
        tmp_path_factory.mktemp("series-ignored"),
        {"prometheus_metrics_max_series_per_metric": "five", "prometheus_metrics_ttl_seconds": ""},
        workers=1,
        warm_keys=5,
    ) as rig:
        yield rig


class TestIgnored:
    def test_settings_that_are_not_positive_numbers_are_ignored_with_a_warning(self, ignored: CapRig) -> None:
        """I1: a cap of "five" and an empty TTL leave every key its own series and each warning names its setting."""
        samples: Final = scrape(ignored.gateway)
        assert alias_values(samples, REQUESTS) >= ignored.warm_aliases
        assert not any(sample.is_overflow() for sample in samples)
        log: Final = ignored.proxy.log.read_text()
        assert (
            "prometheus_metrics_max_series_per_metric is ignored because it is not a number greater than 0 (got 'five')"
            in log
        )
        assert "prometheus_metrics_ttl_seconds is ignored because it is not a number greater than 0 (got '')" in log


@pytest.fixture(scope="class")
def ignored_interval(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapRig]:
    with series_cap_rig(
        tmp_path_factory.mktemp("series-ignored-interval"),
        {
            "prometheus_metrics_max_series_per_metric": CAP,
            "prometheus_metrics_ttl_seconds": TTL_SECONDS,
            "prometheus_metrics_cleanup_interval_seconds": "sixty",
        },
        workers=1,
        warm_keys=CAP,
    ) as rig:
        yield rig


class TestIgnoredInterval:
    def test_a_cleanup_interval_that_is_not_a_number_is_ignored_with_a_warning_while_the_cap_and_ttl_apply(
        self, ignored_interval: CapRig
    ) -> None:
        """I2: a cleanup interval of "sixty" next to a TTL is ignored for the default, so the first labeled emit
        still counts (it raised inside the logging callback before) and a fourth key lands on `other`."""
        key: Final = ignored_interval.key("i2")
        call: Final = Call.new()
        before: Final = scrape(ignored_interval.gateway)
        response: Final = ignored_interval.chat(key, call)
        assert response.status_code == 200 and call.answer in response.text, response.text
        _expect_other(ignored_interval, key, (call,), (string_value(object_value(response.json())["id"]),), before)
        assert (
            "prometheus_metrics_cleanup_interval_seconds is ignored because it is not a number of at least 0 "
            "(got 'sixty'). Idle series are checked every 60.0 seconds"
        ) in ignored_interval.proxy.log.read_text()


def _fallback_deployments(provider_url: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "model_name": name,
            "litellm_params": {
                "model": f"openai/gpt-{name}",
                "api_base": provider_url + "/v1",
                "api_key": "synthetic-provider-key",
            },
        }
        for name in (PRIMARY, FALLBACK)
    )


@pytest.fixture(scope="class")
def excluded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapRig]:
    with series_cap_rig(
        tmp_path_factory.mktemp("series-excluded"),
        {"prometheus_metrics_max_series_per_metric": CAP, "prometheus_exclude_labels": ["api_key_alias"]},
        workers=1,
        warm_keys=0,
        failing_models=frozenset({f"gpt-{PRIMARY}"}),
        deployments=_fallback_deployments,
        router_settings={"fallbacks": [{PRIMARY: [FALLBACK]}]},
    ) as rig:
        yield rig


def _owned_fallback_series(samples: Sequence[Sample], name: str) -> tuple[Sample, ...]:
    return tuple(sample for sample in samples if sample.name == name and not sample.is_overflow())


def _fallback_counter_settled(samples: Sequence[Sample], name: str) -> bool:
    """The counter has admitted the cap and sent the next key to `other`, or has handed out more series than the
    cap, which is what a proxy without the cap does and what the caller's assertion then reports."""
    owned: Final = _owned_fallback_series(samples, name)
    return len(owned) > CAP or (len(owned) == CAP and overflow_total(samples, name) >= 1)


def _expect_capped_fallback_counter(rig: CapRig, name: str) -> None:
    samples: Final = eventually(
        lambda: scrape(rig.gateway), lambda seen: _fallback_counter_settled(seen, name), seconds=60
    )
    owned: Final = _owned_fallback_series(samples, name)
    assert len(owned) == CAP, owned
    assert overflow_total(samples, name) == 1, samples
    assert all(sample.labels.get("fallback_model") == FALLBACK for sample in owned), owned
    assert len({sample.labels["hashed_api_key"] for sample in owned}) == CAP, owned
    assert all("api_key_alias" not in sample.labels for sample in samples if sample.name == name), samples


class TestExcluded:
    def test_fallback_counters_are_capped_and_drop_excluded_labels(self, excluded: CapRig) -> None:
        """X1: the successful and failed fallback counters are capped like every other metric (their label names
        reached the factory positionally before, so the cap never wrapped them) and keep honoring
        prometheus_exclude_labels, which get_labels_for_metric already applied to them."""
        keys: Final = tuple(excluded.key("x1") for _ in range(CAP + 1))
        for key in keys:
            call = Call.new()
            response = chat_once(excluded.base_url, key, PRIMARY, call)
            assert response.status_code == 200 and call.answer in response.text, response.text
        _expect_capped_fallback_counter(excluded, SUCCESSFUL_FALLBACKS)
        excluded.outage.set()
        try:
            for key in keys:
                failed = chat_once(excluded.base_url, key, PRIMARY, Call.new())
                assert failed.status_code == 500, failed.text
        finally:
            excluded.outage.clear()
        _expect_capped_fallback_counter(excluded, FAILED_FALLBACKS)


@pytest.fixture(scope="class")
def nocap(tmp_path_factory: pytest.TempPathFactory) -> Iterator[CapRig]:
    with series_cap_rig(
        tmp_path_factory.mktemp("series-nocap"),
        {"prometheus_metrics_max_series_per_metric": None},
        workers=2,
        warm_keys=4,
    ) as rig:
        yield rig


class TestNoCap:
    def test_a_null_cap_keeps_every_series(self, nocap: CapRig) -> None:
        """N1: an explicit null cap and a missing TTL leave every key its own series and no `other` series."""
        samples: Final = scrape(nocap.gateway)
        assert alias_values(samples, REQUESTS) >= nocap.warm_aliases
        assert not any(sample.is_overflow() for sample in samples)
