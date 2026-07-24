# What this tests ?
## Makes sure the number of callbacks on the proxy don't increase over time
## Num callbacks should be a fixed number at t=0 and t=10, t=20
"""
PROD TEST - DO NOT Delete this Test
"""

import pytest
import asyncio
import aiohttp
import os
import re
import dotenv
from collections import Counter
from dotenv import load_dotenv
import pytest

load_dotenv()

# A *leak* is sustained, monotonic growth of one callback TYPE across the whole
# sampling window. A one-time bump that then plateaus is benign pollution from
# other tests sharing this proxy (this suite runs `pytest -n 4` against a single
# proxy container, so other workers legitimately add team/key-scoped callbacks
# while this test sleeps). We therefore sample N times and only flag a type
# whose normalized count never decreases, grows in >=2 distinct intervals, and
# nets >= LEAK_MIN_NET_GROWTH overall.
NUM_SAMPLES = 4
SAMPLE_INTERVAL_SECONDS = 20
LEAK_MIN_NET_GROWTH = 5
LEAK_MIN_GROWING_INTERVALS = 2
# A routing-strategy switch / alerting config is a *known, bounded, one-time*
# registration (CCI diagnostic 2026-05-16: total 85->95 on the first interval
# after switching to latency-based-routing, then flat at 95 for 2.5 min under
# load). We absorb that step by settling before the baseline sample, so only
# growth *after* the deliberate perturbation can count as a leak.
SETTLE_SECONDS = 30

# Strip instance-identity noise so N leaked instances of one class collapse to
# one rising counter instead of N opaque, unrelated-looking strings.
_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+")
_OBJ_RE = re.compile(r"<([\w.]+) object")


def _normalize_callback(cb_str: str) -> str:
    """Reduce a callback's str() to a stable type key (drops 0x… addresses)."""
    s = _ADDR_RE.sub("", cb_str)
    m = _OBJ_RE.search(s)
    if m:
        return m.group(1).split(".")[-1]
    # bound methods: "<bound method Cls.m of <... at 0x..>>" -> "Cls.m"
    bm = re.search(r"bound method ([\w.]+)", s)
    if bm:
        return bm.group(1)
    return s.strip()


def _summarize(all_litellm_callbacks) -> Counter:
    return Counter(_normalize_callback(str(c)) for c in all_litellm_callbacks)


def _detect_leaks(samples):
    """
    samples: list[Counter] taken in time order.

    Returns {callback_type: [counts across samples]} for types that grew
    monotonically (never decreased), in >=LEAK_MIN_GROWING_INTERVALS intervals,
    and netted >=LEAK_MIN_NET_GROWTH overall — i.e. a real leak, not a one-shot
    step from a parallel test.
    """
    leaks = {}
    all_types = set().union(*[set(s) for s in samples]) if samples else set()
    for t in all_types:
        series = [s.get(t, 0) for s in samples]
        deltas = [b - a for a, b in zip(series, series[1:])]
        net = series[-1] - series[0]
        non_decreasing = all(d >= 0 for d in deltas)
        growing_intervals = sum(1 for d in deltas if d > 0)
        if (
            non_decreasing
            and net >= LEAK_MIN_NET_GROWTH
            and growing_intervals >= LEAK_MIN_GROWING_INTERVALS
        ):
            leaks[t] = series
    return leaks


def _terminal_suspects(samples):
    """
    Types whose net growth clears the threshold monotonically but is confined
    to the *final* interval — `growing_intervals == 1` with that one growing
    interval being the last. `_detect_leaks`' `>= 2` guard silently passes
    these, so a real leak that accumulates entirely in the last sampled window
    is indistinguishable from a one-time terminal step *without one more
    sample*. Returns the set of such types so the caller can re-confirm.
    """
    suspects = set()
    all_types = set().union(*[set(s) for s in samples]) if samples else set()
    for t in all_types:
        series = [s.get(t, 0) for s in samples]
        deltas = [b - a for a, b in zip(series, series[1:])]
        if not deltas:
            continue
        net = series[-1] - series[0]
        non_decreasing = all(d >= 0 for d in deltas)
        growing = [i for i, d in enumerate(deltas) if d > 0]
        if (
            non_decreasing
            and net >= LEAK_MIN_NET_GROWTH
            and growing == [len(deltas) - 1]
        ):
            suspects.add(t)
    return suspects


async def _detect_leaks_confirmed(session, samples):
    """
    `_detect_leaks`, plus a single confirmation sample when growth is confined
    to the final interval (see `_terminal_suspects`). A genuine ongoing leak
    keeps climbing -> now grows in >= 2 intervals -> flagged; a one-time
    terminal registration plateaus -> still 1 growing interval -> ignored.
    Returns `(leaks, samples)` (samples may have one extra entry appended).
    """
    leaks = _detect_leaks(samples)
    if not leaks and _terminal_suspects(samples):
        await asyncio.sleep(SAMPLE_INTERVAL_SECONDS)
        _, _, all_cb = await get_active_callbacks(session=session)
        samples = samples + [_summarize(all_cb)]
        leaks = _detect_leaks(samples)
    return leaks, samples


def _format_report(samples, leaks) -> str:
    lines = ["Callback count per type across samples (time order):"]
    all_types = sorted(set().union(*[set(s) for s in samples]))
    for t in all_types:
        series = [s.get(t, 0) for s in samples]
        marker = "  <-- LEAK" if t in leaks else ""
        lines.append(f"  {t}: {series}{marker}")
    totals = [sum(s.values()) for s in samples]
    lines.append(f"TOTAL callbacks per sample: {totals}")
    if leaks:
        lines.append(
            "Leaking callback types (sustained monotonic growth): "
            + ", ".join(sorted(leaks))
        )
    return "\n".join(lines)


async def _sample_callbacks(session, num_samples, interval):
    """Take `num_samples` callback snapshots `interval`s apart."""
    samples = []
    alerts = []
    for i in range(num_samples):
        if i > 0:
            await asyncio.sleep(interval)
        num_cb, num_alert, all_cb = await get_active_callbacks(session=session)
        samples.append(_summarize(all_cb))
        alerts.append(num_alert)
    return samples, alerts


async def config_update(session, routing_strategy=None):
    url = "http://0.0.0.0:4000/config/update"
    headers = {"Authorization": "Bearer sk-1234", "Content-Type": "application/json"}
    print("routing_strategy: ", routing_strategy)
    data = {
        "router_settings": {
            "routing_strategy": routing_strategy,
        },
        "general_settings": {
            "alert_to_webhook_url": {"llm_exceptions": "example-slack-webhook-url"},
            "alert_types": ["llm_exceptions", "db_exceptions"],
        },
    }

    async with session.post(url, headers=headers, json=data) as response:
        status = response.status
        response_text = await response.text()

        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")
        return await response.json()


async def get_active_callbacks(session):
    url = "http://0.0.0.0:4000/active/callbacks"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print("response from /active/callbacks")
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        _json_response = await response.json()

        _num_callbacks = _json_response["num_callbacks"]
        _num_alerts = _json_response["num_alerting"]
        all_litellm_callbacks = _json_response["all_litellm_callbacks"]

        print("current number of callbacks: ", _num_callbacks)
        print("current number of alerts: ", _num_alerts)
        return _num_callbacks, _num_alerts, all_litellm_callbacks


async def get_current_routing_strategy(session):
    url = "http://0.0.0.0:4000/get/config/callbacks"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234",
    }

    async with session.get(url, headers=headers) as response:
        status = response.status
        response_text = await response.text()
        print(response_text)
        print()

        if status != 200:
            raise Exception(f"Request did not return a 200 status code: {status}")

        _json_response = await response.json()
        print("JSON response: ", _json_response)

        router_settings = _json_response["router_settings"]
        print("Router settings: ", router_settings)
        routing_strategy = router_settings["routing_strategy"]
        return routing_strategy


@pytest.mark.asyncio
@pytest.mark.order1
@pytest.mark.flaky(reruns=2, reruns_delay=5)
async def test_check_num_callbacks():
    """
    PROD invariant: no callback TYPE should grow without bound over time.

    This suite runs `pytest -n 4` against one shared proxy, so the raw count is
    noisy — other workers legitimately add team/key-scoped callbacks that then
    plateau. We settle first, then sample several times, and only fail on
    *sustained, monotonic* per-type growth (a genuine leak), naming the type.
    """
    async with aiohttp.ClientSession() as session:
        # Absorb proxy warmup / in-flight parallel registration before baseline.
        await asyncio.sleep(SETTLE_SECONDS)

        samples, _ = await _sample_callbacks(
            session, NUM_SAMPLES, SAMPLE_INTERVAL_SECONDS
        )

        assert sum(samples[0].values()) > 0, "expected some callbacks registered"

        leaks, samples = await _detect_leaks_confirmed(session, samples)
        report = _format_report(samples, leaks)
        print(report)
        assert not leaks, f"Callback leak detected.\n{report}"


@pytest.mark.asyncio
@pytest.mark.order2
@pytest.mark.flaky(reruns=2, reruns_delay=5)
async def test_check_num_callbacks_on_lowest_latency():
    """
    Same PROD invariant as test_check_num_callbacks, but after switching the
    router to latency-based-routing. That switch is a *known, bounded* one-time
    registration (it adds the latency strategy handler + Slack alerting); we
    settle past it before baselining so only post-switch growth counts as a
    leak. Also asserts the alerting count is stable.
    """
    async with aiohttp.ClientSession() as session:
        await asyncio.sleep(30)

        original_routing_strategy = await get_current_routing_strategy(session=session)
        await config_update(session=session, routing_strategy="latency-based-routing")

        try:
            # Absorb the deliberate one-time config/update registration step.
            await asyncio.sleep(SETTLE_SECONDS)

            samples, alerts = await _sample_callbacks(
                session, NUM_SAMPLES, SAMPLE_INTERVAL_SECONDS
            )

            leaks, samples = await _detect_leaks_confirmed(session, samples)
            report = _format_report(samples, leaks)
            print(report)
            assert not leaks, f"Callback leak detected.\n{report}"
            assert (
                len(set(alerts)) == 1
            ), f"alerting count changed across samples: {alerts}"
        finally:
            await config_update(
                session=session, routing_strategy=original_routing_strategy
            )
