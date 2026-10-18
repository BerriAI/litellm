from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from itertools import chain
from typing import Final, Literal

from pydantic import ConfigDict, Field

from setup.ui_queries import Contract, Document, Query, UiQueryCase, descend, orphan_key


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
    group_by_session: bool | None = None
    session_cursor: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    api_key: str | None = None
    team_id: str | None = None
    team_ids: str | None = None
    user_id: str | None = None
    end_user: str | None = None
    model: str | None = None
    model_id: str | None = None
    key_alias: str | None = None
    search: str | None = None
    status_filter: str | None = None
    cache_hit_filter: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    span_type: str | None = None
    exclude_internal_health_checks: bool | None = None
    used_client_oauth_token: bool | None = None
    lower: str | None = Field(default=None, alias="filter[startTime][gte]")
    upper: str | None = Field(default=None, alias="filter[startTime][lte]")
    q: str | None = None
    key_aliases: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    filter: Literal["all", "injected", "hits"] | None = None
    cursor_start_time: str | None = None
    cursor_request_id: str | None = None
    guardrail_id: str | None = None
    limit: int | None = None
    tags: str | None = None
    group_by: str | None = None


class EndUserBody(Query):
    startTime: str | None = None
    endTime: str | None = None
    api_key: str | None = None


def _timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def next_page(case: UiQueryCase, document: Document) -> UiQueryCase:
    params: Final = Params.model_validate(case.params.model_dump())
    rows: Final = descend(document.root, case.contract.collection)
    assert isinstance(rows, list), "Missing page-one rows"
    ids: Final = frozenset(str(row["request_id"]) for row in rows if isinstance(row, dict))
    contract: Final = replace(case.contract, excluded_ids=ids)
    if case.cursor == "session":
        cursor: Final = descend(document.root, ("next_session_cursor",))
        assert isinstance(cursor, str) and cursor, "Missing session cursor"
        return replace(case, params=params.model_copy(update={"page": 2, "session_cursor": cursor}), contract=contract)
    cursor_data: Final = descend(document.root, ("next_cursor",))
    assert isinstance(cursor_data, dict), "Missing prompt-cache cursor"
    start: Final = cursor_data.get("start_time")
    request_id: Final = cursor_data.get("request_id")
    assert isinstance(start, str) and isinstance(request_id, str), "Invalid prompt-cache cursor"
    return replace(
        case,
        contract=contract,
        params=params.model_copy(update={"page": None, "cursor_start_time": start, "cursor_request_id": request_id}),
    )


_LOGS: Final = Contract(("data",), ("request_id", "startTime", "metadata"))
_REQUESTS: Final = Contract(("requests",), ("request_id", "start_time", "cache_read_tokens", "gateway_injected"))
_ROWS: Final = Contract(("logs",), ("id", "timestamp", "model"))


def log_cases(anchor: datetime, api_key: str) -> tuple[UiQueryCase, ...]:
    params: Final = Params(
        start_date=_timestamp(anchor - timedelta(days=1)),
        end_date=_timestamp(anchor),
        page=1,
        page_size=25,
        sort_by="startTime",
        sort_order="desc",
        group_by_session=True,
    )
    base: Final = UiQueryCase("logs_default_24h", "/spend/logs/ui", params, _LOGS)
    filters: Final = (
        ("api_key", api_key),
        ("team_id", "perf-team-0"),
        ("user_id", "perf-user-0"),
        ("end_user", "perf-end-user-0"),
        ("model", "perf-chat"),
        ("model_id", "perf-model-0"),
        ("key_alias", "perf-key-0"),
        ("session_id", "perf-ui-session"),
        ("status_filter", "failure"),
        ("cache_hit_filter", "hit"),
        ("error_code", "429"),
        ("error_message", "Synthetic"),
        ("span_type", "llm"),
        ("search", "perf-ui-1"),
        ("exclude_internal_health_checks", True),
        ("used_client_oauth_token", False),
    )
    return (
        base,
        replace(base, name="logs_cursor_page_2", params=params.model_copy(update={"page": 2}), cursor="session"),
        replace(base, name="logs_offset_page_3", params=params.model_copy(update={"page": 3})),
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
            replace(base, name=f"logs_{name}", params=params.model_copy(update={name: value}))
            for name, value in filters
        ),
        *(
            replace(base, name=f"logs_sort_{name}", params=params.model_copy(update={"sort_by": name}))
            for name in ("spend", "total_tokens", "request_duration_ms", "model", "ttft_ms")
        ),
        UiQueryCase("log_id_lookup", "/spend/logs/ui", Params(request_id="perf-ui-1", page_size=1), _LOGS),
        UiQueryCase(
            "log_payload",
            "/spend/logs/ui/perf-ui-1",
            Params(start_date=params.start_date),
            Contract(object_fields=("messages", "response", "proxy_server_request")),
        ),
        UiQueryCase(
            "session_logs", "/spend/logs/session/ui", Params(session_id="perf-ui-session", page=1, page_size=100), _LOGS
        ),
    )


def facet_cases(anchor: datetime) -> tuple[UiQueryCase, ...]:
    params: Final = Params.model_validate(
        {"lower": _timestamp(anchor - timedelta(days=1)), "upper": _timestamp(anchor), "page_size": 50, "page": 1}
    )
    return tuple(
        chain.from_iterable(
            (
                UiQueryCase(f"facet_{entity}", f"/management/v1/spend_logs/{entity}", params, Contract(("data",))),
                UiQueryCase(
                    f"facet_{entity}_page_2",
                    f"/management/v1/spend_logs/{entity}",
                    params.model_copy(update={"page": 2}),
                    Contract(("data",)),
                ),
                UiQueryCase(
                    f"facet_{entity}_search",
                    f"/management/v1/spend_logs/{entity}",
                    params.model_copy(update={"q": "perf-"}),
                    Contract(("data",)),
                ),
            )
            for entity in ("users", "end_users")
        )
    )


def analytics_cases(anchor: datetime) -> tuple[UiQueryCase, ...]:
    dates: Final = Params(
        start_date=(anchor - timedelta(days=7)).date().isoformat(), end_date=anchor.date().isoformat()
    )
    prompt_dates: Final = Params(
        start_date=(anchor - timedelta(days=30)).isoformat(), end_date=anchor.isoformat(), filter="all", page_size=10
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
        *(
            UiQueryCase(
                f"prompt_cache_{filter}",
                "/cost_optimization/prompt_caching/requests",
                prompt_dates.model_copy(update={"filter": filter}),
                _REQUESTS,
            )
            for filter in ("all", "injected", "hits")
        ),
        UiQueryCase(
            "prompt_cache_page_2",
            "/cost_optimization/prompt_caching/requests",
            prompt_dates,
            _REQUESTS,
            cursor="prompt_cache",
        ),
        UiQueryCase(
            "guardrail_logs",
            "/guardrails/usage/logs",
            dates.model_copy(update={"guardrail_id": "perf-ui-guardrail", "page": 1, "page_size": 50}),
            _ROWS,
        ),
        UiQueryCase(
            "tool_logs",
            "/v1/tool/perf-ui-tool/logs",
            Params(
                start_date=_timestamp(anchor - timedelta(days=90)), end_date=_timestamp(anchor), page=1, page_size=50
            ),
            _ROWS,
        ),
    )


def old_usage_cases(anchor: datetime, api_key: str) -> tuple[UiQueryCase, ...]:
    month_start: Final = anchor.replace(day=1)
    next_month: Final = (month_start + timedelta(days=32)).replace(day=1)
    month: Final = Params(
        start_date=month_start.date().isoformat(), end_date=(next_month - timedelta(days=1)).date().isoformat()
    )
    week: Final = Params(
        start_date=(anchor - timedelta(days=7)).isoformat(), end_date=anchor.isoformat(), tags="all-tags"
    )
    return (
        UiQueryCase(
            "old_activity",
            "/global/activity",
            month,
            Contract(("daily_data",), ("date",), ("api_requests",)),
            "old_usage",
        ),
        UiQueryCase(
            "old_activity_models",
            "/global/activity/model",
            month,
            Contract(row_fields=("model", "daily_data"), positive=("sum_api_requests",)),
            "old_usage",
        ),
        UiQueryCase(
            "old_spend_provider",
            "/global/spend/provider",
            month,
            Contract(row_fields=("provider", "spend")),
            "old_usage",
        ),
        UiQueryCase(
            "old_tags",
            "/global/spend/tags",
            week,
            Contract(("spend_per_tag",), ("name", "spend"), ("log_count",)),
            "old_usage",
        ),
        UiQueryCase(
            "old_tags_filtered",
            "/global/spend/tags",
            week.model_copy(update={"tags": "project-0,production"}),
            Contract(("spend_per_tag",), ("name", "spend"), ("log_count",)),
            "old_usage",
        ),
        UiQueryCase("old_tag_names", "/global/spend/all_tag_names", Params(), Contract(("tag_names",)), "old_usage"),
        UiQueryCase(
            "old_spend_logs", "/global/spend/logs", Params(), Contract(row_fields=("date", "spend")), "old_usage"
        ),
        UiQueryCase(
            "old_keys",
            "/global/spend/keys",
            Params(limit=5),
            Contract(row_fields=("api_key", "key_alias"), positive=("total_spend",)),
            "old_usage",
        ),
        UiQueryCase("old_teams", "/global/spend/teams", Params(), Contract(("daily_spend",)), "old_usage"),
        UiQueryCase(
            "old_models",
            "/global/spend/models",
            Params(limit=5),
            Contract(row_fields=("model",), positive=("total_spend",)),
            "old_usage",
        ),
        UiQueryCase(
            "old_end_users",
            "/global/spend/end_users",
            Params(),
            Contract(row_fields=("end_user",), positive=("total_count",)),
            "old_usage",
            body=Query(),
        ),
        UiQueryCase(
            "old_end_users_filtered",
            "/global/spend/end_users",
            Params(),
            Contract(row_fields=("end_user",), positive=("total_count",)),
            "old_usage",
            body=EndUserBody(startTime=week.start_date, endTime=week.end_date, api_key=api_key),
        ),
    )


def report_cases(anchor: datetime, api_key: str) -> tuple[UiQueryCase, ...]:
    dates: Final = Params(
        start_date=(anchor - timedelta(days=7)).date().isoformat(), end_date=anchor.date().isoformat()
    )
    return (
        UiQueryCase(
            "agent_global_report_teams",
            "/global/spend/report",
            dates.model_copy(update={"group_by": "team"}),
            Contract(row_fields=("group_by_day", "teams")),
            "liteadmin",
        ),
        UiQueryCase(
            "agent_global_report_customers",
            "/global/spend/report",
            dates.model_copy(update={"group_by": "customer"}),
            Contract(row_fields=("group_by_day", "customers")),
            "liteadmin",
        ),
        UiQueryCase(
            "agent_global_report_keys",
            "/global/spend/report",
            dates.model_copy(update={"group_by": "api_key"}),
            Contract(row_fields=("api_key", "model_details"), positive=("total_cost",)),
            "liteadmin",
        ),
        UiQueryCase(
            "agent_key_report",
            "/key/spend/report",
            dates.model_copy(update={"api_key": api_key}),
            Contract(row_fields=("api_key", "model_details"), positive=("total_cost",)),
            "liteadmin",
        ),
        UiQueryCase(
            "agent_team_report",
            "/team/spend/report",
            dates.model_copy(update={"team_id": "perf-team-0"}),
            Contract(row_fields=("api_key", "model_details"), positive=("total_cost",)),
            "liteadmin",
        ),
    )


def cases(anchor: datetime, api_key: str) -> tuple[UiQueryCase, ...]:
    return (
        *log_cases(anchor, api_key),
        *facet_cases(anchor),
        *analytics_cases(anchor),
        *old_usage_cases(anchor, api_key),
        *report_cases(anchor, api_key),
        *recovery_cases(anchor),
    )


def _entity_recovery_cases(anchor: datetime, entity: str) -> tuple[UiQueryCase, ...]:
    dates: Final = Params(
        start_date=(anchor - timedelta(days=7)).date().isoformat(), end_date=anchor.date().isoformat(), timezone="0"
    )
    routes: Final = (
        ("aggregated", dates, Contract(("results",), ("date", "breakdown"))),
        (
            "aggregated/keys",
            dates.model_copy(update={"offset": 0, "limit": 50}),
            Contract(("api_keys",), ("api_key", "metrics", "metadata")),
        ),
        ("aggregated/search", dates, Contract(("api_keys",), ("api_key", "metrics", "metadata"))),
        (
            "aggregated/model_top_keys",
            dates.model_copy(update={"model_group": "perf-chat", "by_model_group": "true"}),
            Contract(("api_keys",), ("api_key", "metrics", "metadata")),
        ),
    )
    return tuple(_recovery_case(entity, route, params, contract) for route, params, contract in routes)


def _recovery_case(entity: str, route: str, params: Params, contract: Contract) -> UiQueryCase:
    name: Final = f"recovery_{entity}_{route.replace('/', '_')}"
    key: Final = orphan_key(name)
    return UiQueryCase(
        name,
        f"/{entity}/daily/activity/{route}",
        params.model_copy(update={"api_key": key, **({"search": key} if route.endswith("search") else {})}),
        replace(contract, recovered_alias="perf-ui-recovered"),
        "key_recovery",
    )


def recovery_cases(anchor: datetime) -> tuple[UiQueryCase, ...]:
    entities: Final = ("user", "team", "tag", "organization", "customer", "agent")
    groups: Final = tuple(_entity_recovery_cases(anchor, entity) for entity in entities)
    dates: Final = Params(
        start_date=(anchor - timedelta(days=7)).date().isoformat(), end_date=anchor.date().isoformat(), timezone="0"
    )
    return (
        *sum(groups, ()),
        _recovery_case(
            "user", "aggregated/cache_leakage_keys", dates, Contract(("api_keys",), ("api_key", "metrics", "metadata"))
        ),
    )
