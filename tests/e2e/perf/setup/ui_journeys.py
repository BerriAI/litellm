from __future__ import annotations

import math
import re
import statistics
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator, Page, Response, expect, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ConfigDict

MONEY: Final = re.compile(r"^\$[\d,]+\.\d{2}$")
LOADING_LOGS: Final = "Loading request logs…"
LOADING_MODELS: Final = "Loading models…"
LOADING_KEYS: Final = "Loading keys..."


class Sample(BaseModel):
    model_config = ConfigDict(frozen=True)

    seconds: float
    rows: int = 0
    error: str = ""
    status: int = 0
    request: str = ""


class Timings(BaseModel):
    model_config = ConfigDict(frozen=True)

    first_seconds: float | None
    repeated_samples: int
    repeated_p50_seconds: float | None
    repeated_p95_seconds: float | None
    repeated_p99_seconds: float | None
    repeated_max_seconds: float | None


@dataclass(frozen=True, slots=True)
class Journey:
    """One dashboard interaction. ``start`` brings the page to the state the user acts from (untimed),
    ``act`` is the click or navigation being measured and ``settled`` blocks until the data the user
    was waiting for is on screen, returning the number of rows or tiles that rendered."""

    name: str
    surface: str
    start: Callable[[Page, float], None]
    act: Callable[[Page], None]
    settled: Callable[[Page, float], int]


@dataclass(frozen=True, slots=True)
class Dashboard:
    base_url: str
    username: str
    password: str
    anchor: datetime


def summarize(samples: tuple[Sample, ...]) -> Timings:
    assert samples, "No UI journeys were measured"
    repeated: Final = tuple(sorted(sample.seconds for sample in samples[1:] if not sample.error))
    return Timings(
        first_seconds=samples[0].seconds if not samples[0].error else None,
        repeated_samples=len(repeated),
        repeated_p50_seconds=statistics.median(repeated) if repeated else None,
        repeated_p95_seconds=repeated[math.ceil(len(repeated) * 0.95) - 1] if len(repeated) >= 20 else None,
        repeated_p99_seconds=repeated[math.ceil(len(repeated) * 0.99) - 1] if len(repeated) >= 100 else None,
        repeated_max_seconds=max(repeated) if repeated else None,
    )


def error_code(sample: Sample) -> str:
    if not sample.error:
        return ""
    if sample.status >= 400:
        return f"http_{sample.status}"
    if sample.error.startswith("wall_clock"):
        return "wall_clock"
    return "timeout" if sample.error == "timeout" else "ui"


def ui_junit_properties(samples: tuple[Sample, ...]) -> tuple[tuple[str, float], ...]:
    codes: Final = tuple(error_code(sample) for sample in samples if sample.error)
    timings: Final = summarize(samples)
    successful: Final = tuple(sorted(sample.seconds for sample in samples if not sample.error))
    latency: Final = (
        ("seconds_p50", statistics.median(successful) if successful else None),
        ("seconds_p90", successful[math.ceil(len(successful) * 0.90) - 1] if successful else None),
        ("seconds_max", max(successful) if successful else None),
        ("first_seconds", timings.first_seconds),
        ("repeated_seconds_p50", timings.repeated_p50_seconds),
        ("repeated_seconds_p95", timings.repeated_p95_seconds),
        ("repeated_seconds_p99", timings.repeated_p99_seconds),
        ("repeated_seconds_max", timings.repeated_max_seconds),
    )
    return (
        ("perf_ui_journeys", len(samples)),
        ("perf_ui_failures", len(codes)),
        ("perf_ui_samples", len(successful)),
        ("perf_ui_repeated_samples", timings.repeated_samples),
        *((f"perf_ui_error_{code}", codes.count(code)) for code in sorted(frozenset(codes))),
        *((f"perf_ui_{name}", value) for name, value in latency if value is not None),
    )


@dataclass(frozen=True, slots=True)
class Ceiling:
    p50_seconds: float
    p90_seconds: float


LOGS_CEILING: Final = Ceiling(p50_seconds=10, p90_seconds=20)
CHART_CEILING: Final = Ceiling(p50_seconds=20, p90_seconds=40)
TABLE_CEILING: Final = Ceiling(p50_seconds=5, p90_seconds=10)
UI_JOURNEY_LIMITS: Final = MappingProxyType(
    {
        "logs_open_24h": LOGS_CEILING,
        "logs_last_7_days": LOGS_CEILING,
        "logs_next_page": LOGS_CEILING,
        "logs_filter_key_hash": LOGS_CEILING,
        "logs_filter_key_alias": LOGS_CEILING,
        "logs_filter_team": LOGS_CEILING,
        "logs_filter_model": LOGS_CEILING,
        "usage_open": CHART_CEILING,
        "usage_last_30_days": CHART_CEILING,
        "usage_team_view": CHART_CEILING,
        "usage_customer_view": CHART_CEILING,
        "usage_model_activity_30_days": CHART_CEILING,
        "cache_open": CHART_CEILING,
        "models_open": TABLE_CEILING,
        "keys_open": TABLE_CEILING,
    }
)


def assert_within_ceiling(journey_name: str, samples: tuple[Sample, ...]) -> None:
    failed: Final = tuple(sample.error for sample in samples if sample.error)
    assert not failed, f"{len(failed)} of {len(samples)} {journey_name} samples failed: {failed[0]}"
    metrics: Final = dict(ui_junit_properties(samples))
    ceiling: Final = UI_JOURNEY_LIMITS[journey_name]
    assert metrics["perf_ui_seconds_p50"] <= ceiling.p50_seconds, (
        f"{journey_name} p50 {metrics['perf_ui_seconds_p50']:.1f}s over {ceiling.p50_seconds}s"
    )
    assert metrics["perf_ui_seconds_p90"] <= ceiling.p90_seconds, (
        f"{journey_name} p90 {metrics['perf_ui_seconds_p90']:.1f}s over {ceiling.p90_seconds}s"
    )


@contextmanager
def dashboard_page(dashboard: Dashboard, timeout: float) -> Generator[Page]:
    """A logged-in admin tab whose clock reads the seed anchor, so relative windows such as
    Last 24 Hours land on seeded rows. Live Tail is off so its 15 second refetch cannot land
    inside a sample. Playwright keeps a UTC timezone to match the seed."""
    with sync_playwright() as playwright:
        browser: Final = playwright.chromium.launch()
        context: Final = browser.new_context(
            base_url=dashboard.base_url, timezone_id="UTC", viewport={"width": 1600, "height": 1000}
        )
        context.set_default_timeout(timeout * 1000)
        context.add_init_script('sessionStorage.setItem("isLiveTail", "false")')
        page: Final = context.new_page()
        page.clock.set_fixed_time(dashboard.anchor)
        _login(page, dashboard)
        try:
            yield page
        finally:
            context.close()
            browser.close()


def _login(page: Page, dashboard: Dashboard) -> None:
    _ = page.goto("/ui/login")
    page.get_by_placeholder("Enter your username").fill(dashboard.username)
    page.get_by_placeholder("Enter your password").fill(dashboard.password)
    page.get_by_role("button", name="Login", exact=True).click()
    page.wait_for_url(lambda url: "/ui" in url and "/login" not in url)
    expect(page.get_by_label("LiteLLM home")).to_be_visible()


def _is_data_fetch(response: Response) -> bool:
    """API calls only. The Admin UI's own assets and Next.js prefetches live under /ui/ and /_next/."""
    path: Final = urlsplit(response.url).path
    return response.request.resource_type in ("fetch", "xhr") and not path.startswith(("/ui/", "/_next/"))


def _worst(responses: list[Response]) -> Response | None:
    return next((response for response in responses if response.status >= 400), responses[-1] if responses else None)


def _with_response(sample: Sample, responses: list[Response]) -> Sample:
    worst: Final = _worst(responses)
    if worst is None:
        return sample
    return sample.model_copy(
        update={"status": worst.status, "request": f"{worst.request.method} {urlsplit(worst.url).path}"[:120]}
    )


def measure(page: Page, journey: Journey, timeout: float) -> Sample:
    """Wall time from the user's action to the data being on screen. Every sample starts from the
    same pre-action state so repeats measure the same journey, not a warmed client cache."""
    try:
        journey.start(page, timeout)
    except (PlaywrightError, AssertionError) as error:
        return Sample(seconds=0.0, error=f"start {type(error).__name__}: {error}"[:300])
    responses: Final[list[Response]] = []  # mutable-ok: filled by the playwright response listener

    def record(response: Response) -> None:
        if _is_data_fetch(response):
            responses.append(response)

    page.on("response", record)
    started: Final = time.monotonic()
    try:
        with page.expect_response(_is_data_fetch, timeout=timeout * 1000):
            journey.act(page)
        rows: Final = journey.settled(page, timeout)
    except PlaywrightTimeoutError:
        return _with_response(Sample(seconds=time.monotonic() - started, error="timeout"), responses)
    except (PlaywrightError, AssertionError) as error:
        return _with_response(
            Sample(seconds=time.monotonic() - started, error=f"{type(error).__name__}: {error}"[:300]), responses
        )
    finally:
        page.remove_listener("response", record)
    return _with_response(Sample(seconds=time.monotonic() - started, rows=rows), responses)


def _visible(page: Page, test_id: str) -> Locator:
    return page.get_by_test_id(test_id).filter(visible=True).first


def _logs_rows(page: Page) -> Locator:
    return page.locator("table").filter(visible=True).first.locator("tbody tr")


def _logs_settled(page: Page, timeout: float) -> int:
    millis: Final = timeout * 1000
    rows: Final = _logs_rows(page).filter(has_not_text="No results")
    expect(rows.first).to_be_visible(timeout=millis)
    expect(_visible(page, "datatable-refresh")).to_be_enabled(timeout=millis)
    expect(page.get_by_text(LOADING_LOGS)).to_have_count(0, timeout=millis)
    return rows.count()


def _no_skeletons(page: Page, timeout: float) -> None:
    expect(page.locator('[data-slot="skeleton"]')).to_have_count(0, timeout=timeout * 1000)


def _usage_settled(page: Page, timeout: float) -> int:
    millis: Final = timeout * 1000
    expect(page.get_by_text("Total Spend", exact=True).first).to_be_visible(timeout=millis)
    _no_skeletons(page, timeout)
    money: Final = page.get_by_text(MONEY)
    expect(money.first).to_be_visible(timeout=millis)
    return money.count()


def _cache_settled(page: Page, timeout: float) -> int:
    millis: Final = timeout * 1000
    expect(page.get_by_text("Cache Hits", exact=True)).to_be_visible(timeout=millis)
    bars: Final = page.locator(".recharts-bar-rectangle")
    expect(bars.first).to_be_visible(timeout=millis)
    return bars.count()


def _table_settled(loading: str) -> Callable[[Page, float], int]:
    def settled(page: Page, timeout: float) -> int:
        millis: Final = timeout * 1000
        rows: Final = page.locator("[data-row-id]")
        expect(rows.first).to_be_visible(timeout=millis)
        expect(page.get_by_text(loading)).to_have_count(0, timeout=millis)
        return rows.count()

    return settled


def _open_logs(page: Page, timeout: float) -> None:
    _ = page.goto("/ui/logs")
    _ = _logs_settled(page, timeout)


def _open_usage(page: Page, timeout: float) -> None:
    _ = page.goto("/ui/usage")
    _ = _usage_settled(page, timeout)


def _filters_drawer(page: Page) -> Locator:
    _visible(page, "datatable-filters-trigger").click()
    drawer: Final = page.get_by_role("dialog", name="Filters")
    expect(drawer).to_be_visible()
    return drawer


def _apply_filters(page: Page) -> None:
    page.get_by_role("dialog", name="Filters").get_by_role("button", name="Apply Filters").click()


def _pick_first_option(page: Page, combobox: str) -> None:
    _filters_drawer(page).get_by_role("combobox", name=combobox).click()
    page.get_by_role("option").first.click()


def _open_time_range(page: Page, timeout: float) -> None:
    _open_logs(page, timeout)
    page.get_by_role("button", name="Last 24 Hours").click()


def _open_usage_range(page: Page, timeout: float) -> None:
    _open_usage(page, timeout)
    page.locator('[data-slot="advanced-date-picker-trigger"]').click()


def _model_activity_settled(page: Page, timeout: float) -> int:
    millis: Final = timeout * 1000
    expect(page.locator('[role="tab"][data-active]', has_text="Model Activity")).to_be_visible(timeout=millis)
    _no_skeletons(page, timeout)
    money: Final = page.get_by_text(MONEY).filter(visible=True)
    expect(money.first).to_be_visible(timeout=millis)
    return money.count()


def _open_model_activity_range(page: Page, timeout: float) -> None:
    _open_usage(page, timeout)
    page.get_by_role("tab", name="Model Activity").click()
    _ = _model_activity_settled(page, timeout)
    page.locator('[data-slot="advanced-date-picker-trigger"]').click()


def _pick_usage_preset(preset: str) -> Callable[[Page], None]:
    def act(page: Page) -> None:
        page.locator('[data-slot="advanced-date-picker-preset"]', has_text=preset).click()

    return act


def _select_usage_view(page: Page, view: str) -> None:
    page.locator('[data-slot="select-trigger"]').filter(has_text="Usage").first.click()
    page.get_by_role("option", name=re.compile(rf"^{re.escape(view)}")).click()


def _nothing(page: Page, timeout: float) -> None:
    return None


def _goto(path: str) -> Callable[[Page], None]:
    def act(page: Page) -> None:
        _ = page.goto(path)

    return act


def _logs_filter(name: str, prepare: Callable[[Page], None]) -> Journey:
    def start(page: Page, timeout: float) -> None:
        _open_logs(page, timeout)
        prepare(page)

    return Journey(name, "logs", start, _apply_filters, _logs_settled)


def journeys(api_key: str) -> tuple[Journey, ...]:
    return (
        Journey("logs_open_24h", "logs", _nothing, _goto("/ui/logs"), _logs_settled),
        Journey(
            "logs_last_7_days",
            "logs",
            _open_time_range,
            lambda page: page.get_by_role("button", name="Last 7 Days", exact=True).click(),
            _logs_settled,
        ),
        Journey(
            "logs_next_page", "logs", _open_logs, lambda page: _visible(page, "pagination-next").click(), _logs_settled
        ),
        _logs_filter(
            "logs_filter_key_hash",
            lambda page: _filters_drawer(page).get_by_placeholder("Enter key hash…").fill(api_key),
        ),
        _logs_filter("logs_filter_key_alias", lambda page: _pick_first_option(page, "Search a key alias")),
        _logs_filter("logs_filter_team", lambda page: _pick_first_option(page, "Search or select a team")),
        _logs_filter("logs_filter_model", lambda page: _pick_first_option(page, "Search a model")),
        Journey("usage_open", "usage", _nothing, _goto("/ui/usage"), _usage_settled),
        Journey("usage_last_30_days", "usage", _open_usage_range, _pick_usage_preset("Last 30 days"), _usage_settled),
        Journey(
            "usage_team_view", "usage", _open_usage, lambda page: _select_usage_view(page, "Team Usage"), _usage_settled
        ),
        Journey(
            "usage_customer_view",
            "usage",
            _open_usage,
            lambda page: _select_usage_view(page, "Customer Usage"),
            _usage_settled,
        ),
        Journey(
            "usage_model_activity_30_days",
            "usage",
            _open_model_activity_range,
            _pick_usage_preset("Last 30 days"),
            _model_activity_settled,
        ),
        Journey("cache_open", "caching", _nothing, _goto("/ui/caching"), _cache_settled),
        Journey("models_open", "models", _nothing, _goto("/ui/models-and-endpoints"), _table_settled(LOADING_MODELS)),
        Journey("keys_open", "keys", _nothing, _goto("/ui/api-keys"), _table_settled(LOADING_KEYS)),
    )


def database_failures(text: str) -> float:
    families: Final = tuple(text_string_to_metric_families(text))
    required: Final = frozenset(("litellm_postgres_failed_requests", "litellm_batch_write_to_db_failed_requests"))
    missing: Final = required - frozenset(family.name for family in families)
    all_samples: Final = tuple(chain.from_iterable(family.samples for family in families))
    has_activity: Final = any(
        sample.name == "litellm_postgres_total_requests_total" and math.isfinite(sample.value) and sample.value > 0
        for sample in all_samples
    )
    assert not missing or has_activity, f"Missing PostgreSQL failure counters {missing}"
    names: Final = frozenset(f"{name}_total" for name in required)
    values: Final = tuple(sample.value for sample in all_samples if sample.name in names)
    assert all(math.isfinite(value) and value >= 0 for value in values), "Invalid PostgreSQL failure counter"
    return sum(values)
