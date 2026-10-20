from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from itertools import chain
from typing import Final, Literal

from pydantic import ConfigDict

from setup.ui_queries import Contract, Document, Query, UiQueryCase, descend


class Params(Query):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    start_date: str | None = None
    timezone: str | None = None
    offset: int | None = None
    model_group: str | None = None
    by_model_group: str | None = None
    end_date: str | None = None
    page: int | None = None
    page_size: int | None = None
    sort_by: str | None = None
    sort_order: str | None = None
    group_by_session: Literal["true"] | None = None
    session_cursor: str | None = None
    api_key: str | None = None
    team_id: str | None = None
    team_ids: str | None = None
    user_id: str | None = None
    model: str | None = None
    model_id: str | None = None
    key_alias: str | None = None
    search: str | None = None
    status_filter: str | None = None
    cache_hit_filter: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    span_type: str | None = None
    exclude_internal_health_checks: Literal["true"] | None = None
    key_aliases: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    limit: int | None = None
    tags: str | None = None
    exclude_team_ids: str | None = None
    include_current_utc_day: Literal["true"] | None = None


def _timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def next_page(case: UiQueryCase, document: Document) -> UiQueryCase:
    params: Final = Params.model_validate(case.params.model_dump())
    rows: Final = descend(document.root, case.contract.collection)
    assert isinstance(rows, list), "Missing page-one rows"
    ids: Final = frozenset(str(row["request_id"]) for row in rows if isinstance(row, dict))
    contract: Final = replace(case.contract, excluded_ids=ids)
    cursor: Final = descend(document.root, ("next_session_cursor",))
    assert isinstance(cursor, str) and cursor, "Missing session cursor"
    return replace(case, params=params.model_copy(update={"page": 2, "session_cursor": cursor}), contract=contract)


_LOGS: Final = Contract(("data",), ("request_id", "startTime", "metadata"))


def log_cases(anchor: datetime, api_key: str) -> tuple[UiQueryCase, ...]:
    params: Final = Params(
        start_date=_timestamp(anchor - timedelta(days=1)),
        end_date=_timestamp(anchor),
        page=1,
        page_size=25,
        sort_by="startTime",
        sort_order="desc",
        group_by_session="true",
    )
    base: Final = UiQueryCase("logs_default_24h", "/spend/logs/ui", params, _LOGS)
    filtered: Final = params.model_copy(update={"start_date": _timestamp(anchor - timedelta(days=30))})
    filters: Final = (
        ("api_key", api_key),
        ("team_id", "perf-team-0"),
        ("user_id", "perf-user-0"),
        ("model", "perf-chat"),
        ("model_id", "perf-model-0"),
        ("key_alias", "perf-key-0"),
        ("status_filter", "failure"),
        ("cache_hit_filter", "hit"),
        ("error_code", "429"),
        ("error_message", "Synthetic"),
        ("span_type", "llm"),
        ("exclude_internal_health_checks", "true"),
    )
    return (
        base,
        replace(base, name="logs_cursor_page_2", params=params.model_copy(update={"page": 2}), cursor="session"),
        replace(
            base,
            name="logs_7d",
            params=params.model_copy(update={"start_date": _timestamp(anchor - timedelta(days=7))}),
        ),
        replace(
            base,
            name="logs_30d",
            params=params.model_copy(update={"start_date": _timestamp(anchor - timedelta(days=30))}),
        ),
        *(
            replace(base, name=f"logs_{name}_30d", params=filtered.model_copy(update={name: value}))
            for name, value in filters
        ),
        *(
            replace(base, name=f"logs_sort_{name}", params=params.model_copy(update={"sort_by": name}))
            for name in ("spend", "total_tokens", "request_duration_ms", "model", "ttft_ms")
        ),
    )


def analytics_cases(anchor: datetime) -> tuple[UiQueryCase, ...]:
    dates: Final = Params(
        start_date=(anchor - timedelta(days=7)).date().isoformat(), end_date=anchor.date().isoformat()
    )
    cache: Final = UiQueryCase(
        "cache_activity_7d", "/global/activity/cache_hits", dates, Contract(("groups",), ("call_type",))
    )
    return (
        UiQueryCase(
            "team_users_all_teams",
            "/team/spend/by_user",
            dates.model_copy(update={"team_ids": ",".join(f"perf-team-{i}" for i in range(100))}),
            Contract(("results",), ("team_id", "user_id", "spend"), ("api_requests",)),
        ),
        UiQueryCase(
            "team_users_filtered",
            "/team/spend/by_user",
            dates.model_copy(update={"team_ids": "perf-team-0"}),
            Contract(("results",), ("team_id", "user_id", "spend"), ("api_requests",)),
        ),
        cache,
        replace(
            cache,
            name="cache_activity_filtered",
            params=dates.model_copy(update={"key_aliases": ("perf-key-0", "perf-key-1"), "models": ("perf-chat",)}),
        ),
    )


def _entity_activity_cases(anchor: datetime, entity: str) -> tuple[UiQueryCase, ...]:
    dates: Final = Params(
        start_date=(anchor - timedelta(days=7)).date().isoformat(),
        end_date=anchor.date().isoformat(),
        timezone="0",
        exclude_team_ids="litellm-dashboard" if entity == "team" else None,
    )
    routes: Final = (
        ("aggregated", dates, Contract(("results",), ("date", "breakdown"))),
        (
            "aggregated/keys",
            dates.model_copy(update={"offset": 0, "limit": 50}),
            Contract(("api_keys",), ("api_key", "metrics", "metadata")),
        ),
        (
            "aggregated/search",
            dates.model_copy(update={"search": "perf-key"}),
            Contract(("api_keys",), ("api_key", "metrics", "metadata")),
        ),
        (
            "aggregated/model_top_keys",
            dates.model_copy(update={"model_group": "perf-chat", "by_model_group": "true"}),
            Contract(("api_keys",), ("api_key", "metrics", "metadata")),
        ),
    )
    return tuple(
        UiQueryCase(
            f"usage_{entity}_{route.replace('/', '_')}",
            f"/{entity}/daily/activity/{route}",
            params,
            contract,
        )
        for route, params, contract in routes
    )


def activity_cases(anchor: datetime) -> tuple[UiQueryCase, ...]:
    groups: Final = tuple(_entity_activity_cases(anchor, entity) for entity in ("user", "team", "tag", "customer"))
    return (
        *chain.from_iterable(groups),
        UiQueryCase(
            "cache_leakage_keys",
            "/user/daily/activity/aggregated/cache_leakage_keys",
            Params(
                start_date=(anchor - timedelta(days=30)).date().isoformat(),
                end_date=anchor.date().isoformat(),
                timezone="0",
                include_current_utc_day="true",
            ),
            Contract(("api_keys",), ("api_key", "metrics", "metadata")),
        ),
    )


def cases(anchor: datetime, api_key: str) -> tuple[UiQueryCase, ...]:
    return (
        *log_cases(anchor, api_key),
        *analytics_cases(anchor),
        *activity_cases(anchor),
    )
