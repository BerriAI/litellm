from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.constants import (
    BILLED_RETRY_ATTEMPT_METADATA_KEY,
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
    MODEL_INSIGHTS_DEFAULT_TASK,
    MODEL_INSIGHTS_TASK_TAG_PREFIX,
)
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import GuardrailCostByUnitEntry
from litellm.proxy.db.daily_spend_bulk_upsert import DAILY_SPEND_TABLES, daily_spend_entity_ids
from litellm.proxy.db.model_insights_tasks import load_model_insight_tasks
from litellm.proxy.db.model_usage_rollup import date_from_start_time
from litellm.proxy.route_llm_request import ROUTE_ENDPOINT_MAPPING
from litellm.proxy.spend_tracking.compression_savings import extract_compression_saved_tokens
from litellm.proxy.spend_tracking.savings import extract_cache_creation_tokens, extract_cache_read_tokens
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

_FIELDS: Final = TypeAdapter(dict[str, object])
_STRINGS: Final = TypeAdapter(tuple[str, ...])
_VALUES: Final = TypeAdapter(tuple[object, ...])
_ROWS: Final = TypeAdapter(tuple[dict[str, object], ...])
_TASK_TYPES: Final = frozenset(load_model_insight_tasks())
_MAX_INDEXED_TEXT_BYTES: Final = 2000
_MAX_COUNTER: Final = 2**40


@dataclass(frozen=True, slots=True)
class Contribution:
    table: str
    key: tuple[tuple[str, str], ...]
    increments: tuple[tuple[str, int | float], ...]
    values: tuple[tuple[str, str | None], ...] = ()


def _fields(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        return _FIELDS.validate_json(value)
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else MappingProxyType({})


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _number(value: object) -> int | float:
    return value if isinstance(value, (int, float)) else 0


def _tags(row: Mapping[str, object]) -> tuple[str, ...]:
    value: Final = row.get("request_tags")
    if value is None:
        return ()
    parsed: Final = _VALUES.validate_json(value) if isinstance(value, str) else _VALUES.validate_python(value)
    return tuple(tag for tag in parsed if isinstance(tag, str))


def _date(row: Mapping[str, object]) -> str | None:
    start: Final = row.get("startTime")
    return date_from_start_time(start) if isinstance(start, (datetime, str)) else None


def _spend(table: str, key: tuple[tuple[str, str], ...], spend: float) -> Contribution:
    return Contribution(table, key, (("spend", spend),))


def _entity_contributions(row: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[Contribution, ...]:
    spend: Final = _number(row.get("spend"))
    entities: Final = _fields(metadata.get("accounting_entities"))
    identities: Final = (
        ("LiteLLM_VerificationToken", "token", entities.get("api_key")),
        ("LiteLLM_UserTable", "user_id", entities.get("user")),
        ("LiteLLM_UserTable", "user_id", entities.get("proxy_budget_name")),
        ("LiteLLM_TeamTable", "team_id", entities.get("team_id")),
        ("LiteLLM_OrganizationTable", "organization_id", entities.get("organization_id")),
        ("LiteLLM_EndUserTable", "user_id", entities.get("end_user")),
        ("LiteLLM_AgentsTable", "agent_id", entities.get("agent_id")),
        ("LiteLLM_ProjectTable", "project_id", entities.get("project_id")),
    )
    groups: Final = _STRINGS.validate_python(metadata.get("accounting_model_access_groups", ()))
    return (
        *(
            _spend(table, ((column, identity),), spend)
            for table, column, identity in identities
            if isinstance(identity, str)
        ),
        *(_spend("LiteLLM_TagTable", (("tag_name", tag),), spend) for tag in _tags(row) if tag),
        *(_spend("LiteLLM_ModelAccessGroupBudgetTable", (("access_group_name", group),), spend) for group in groups),
        *_membership_contributions(entities, spend),
    )


def _membership_contributions(row: Mapping[str, object], spend: float) -> tuple[Contribution, ...]:
    user: Final = row.get("user")
    if not isinstance(user, str):
        return ()
    owners: Final = (
        ("LiteLLM_TeamMembership", "team_id", row.get("team_id")),
        ("LiteLLM_OrganizationMembership", "organization_id", row.get("organization_id")),
    )
    return tuple(
        _spend(table, ((column, identity), ("user_id", user)), spend)
        for table, column, identity in owners
        if isinstance(identity, str)
    )


def _daily_counters(row: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[tuple[str, int | float], ...]:
    usage: Final = _fields(metadata.get("usage_object"))
    savings: Final = _fields(metadata.get("accounting_savings"))
    internal: Final = bool(metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY))
    caller: Final = not internal and metadata.get(BILLED_RETRY_ATTEMPT_METADATA_KEY) is not True
    success: Final = metadata.get("status") != "failure"
    duration: Final = row.get("request_duration_ms")
    timed: Final = not internal and success and isinstance(duration, int) and duration >= 0
    return (
        ("spend", _number(row.get("spend"))),
        ("prompt_tokens", _number(row.get("prompt_tokens"))),
        ("completion_tokens", _number(row.get("completion_tokens"))),
        ("api_requests", int(caller)),
        ("successful_requests", int(caller and success)),
        ("failed_requests", int(caller and not success)),
        ("cache_read_input_tokens", extract_cache_read_tokens(usage)),
        ("cache_creation_input_tokens", extract_cache_creation_tokens(usage)),
        ("compression_saved_tokens", extract_compression_saved_tokens(metadata)),
        ("compression_savings_spend", _number(savings.get("compression"))),
        ("prompt_caching_savings_spend", _number(savings.get("prompt_caching"))),
        ("gateway_injected_caching_savings_spend", _number(savings.get("gateway_injected_caching"))),
        ("autorouter_savings_spend", 0.0 if internal else _number(savings.get("autorouter"))),
        ("guardrail_spend", _number(metadata.get("accounting_guardrail_spend"))),
        ("total_response_time_ms", _number(duration) if timed else 0),
        ("timed_requests", int(timed)),
    )


def _daily_contributions(row: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[Contribution, ...]:
    date: Final = _date(row)
    if date is None or "api_key" not in row:
        return ()
    if "mcp_namespaced_tool_name" not in row and (
        "model" not in row or "custom_llm_provider" not in row or "model_group" not in row
    ):
        return ()
    counters: Final = _daily_counters(row, metadata)
    endpoint: Final = ROUTE_ENDPOINT_MAPPING.get(_text(row.get("call_type")))
    shared_key: Final = (
        ("date", date),
        ("api_key", _text(row.get("api_key"))),
        ("model", _text(row.get("model"))),
        ("custom_llm_provider", _text(row.get("custom_llm_provider"))),
        ("mcp_namespaced_tool_name", _text(row.get("mcp_namespaced_tool_name"))),
        ("endpoint", _text(endpoint)),
    )
    # The legacy daily organization writer uses the charged org argument, not
    # SpendLogs' normalized empty-string identity for a request with no org.
    daily_row: Final = {**row, "organization_id": _fields(metadata.get("accounting_entities")).get("organization_id")}
    return tuple(_daily_rows(daily_row, shared_key, counters))


def _daily_rows(
    row: Mapping[str, object],
    shared_key: tuple[tuple[str, str], ...],
    counters: tuple[tuple[str, int | float], ...],
) -> Iterator[Contribution]:
    for entity, table in DAILY_SPEND_TABLES.items():
        for identity in daily_spend_entity_ids(row, entity):
            yield Contribution(
                table.name,
                ((table.entity_id_column, _text(identity)), *shared_key),
                counters,
                (
                    ("model_group", None if row.get("model_group") is None else _text(row.get("model_group"))),
                    *((("request_id", _text(row.get("request_id"))),) if table.carries_request_id else ()),
                ),
            )


def _window_contributions(row: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[Contribution, ...]:
    windows: Final = _ROWS.validate_python(metadata.get("accounting_windows", ()))
    return tuple(
        Contribution(
            "LiteLLM_BudgetWindowSpend",
            tuple((column, _text(window[column])) for column in ("entity_type", "entity_id", "window_duration")),
            (("spend", _number(window.get("spend", row.get("spend")))),),
            (("window_start", _text(window["window_start"])), ("started_at", _text(row.get("startTime")))),
        )
        for window in windows
    )


def project(row: Mapping[str, object]) -> tuple[Contribution, ...]:
    metadata: Final = _fields(row.get("metadata"))
    contributions: Final = (
        *_entity_contributions(row, metadata),
        *_daily_contributions(row, metadata),
        *_window_contributions(row, metadata),
        *_model_contributions(row, metadata),
        *_guardrail_contributions(row, metadata),
    )
    return tuple(_capped(contribution) for contribution in contributions if _indexable(contribution))


def _indexable(contribution: Contribution) -> bool:
    texts: Final = (value for _, value in (*contribution.key, *contribution.values) if value is not None)
    return all(len(text.encode("utf-8", "surrogatepass")) <= _MAX_INDEXED_TEXT_BYTES for text in texts)


def _capped(contribution: Contribution) -> Contribution:
    """Counters are summed into bigint columns in the charge's transaction, so one capped far below that range
    can never overflow a day's total."""
    return replace(
        contribution,
        increments=tuple(
            (name, max(-_MAX_COUNTER, min(_MAX_COUNTER, value)) if isinstance(value, int) else value)
            for name, value in contribution.increments
        ),
    )


def _model_contributions(row: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[Contribution, ...]:
    date: Final = _date(row)
    origin: Final = metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY)
    if date is None or (origin and origin != BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN):
        return ()
    model: Final = _text(row.get("model")) or "unknown"
    caller: Final = not origin
    success: Final = row.get("status") == "success"
    return (
        Contribution(
            "LiteLLM_DailyModelUsage",
            (
                ("date", date),
                ("model_group", _text(row.get("model_group")) or model),
                ("model", model),
                ("custom_llm_provider", _text(row.get("custom_llm_provider")) or "unknown"),
                ("task_type", _model_task(row)),
            ),
            (
                ("spend", _number(row.get("spend"))),
                ("prompt_tokens", _number(row.get("prompt_tokens"))),
                ("completion_tokens", _number(row.get("completion_tokens"))),
                ("request_count", int(caller)),
                ("successful_requests", int(caller and success)),
                ("failed_requests", int(caller and not success)),
            ),
        ),
    )


def _utc_date(row: Mapping[str, object]) -> str | None:
    start: Final = row.get("startTime")
    try:
        parsed: Final = datetime.fromisoformat(start.replace("Z", "+00:00")) if isinstance(start, str) else start
    except ValueError:
        return None
    if not isinstance(parsed, datetime):
        return None
    return (
        (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed)
        .astimezone(timezone.utc)
        .date()
        .isoformat()
    )


def _guardrail_contributions(row: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[Contribution, ...]:
    date: Final = _utc_date(row)
    info: Final = metadata.get("guardrail_information") or metadata.get("standard_logging_guardrail_information")
    if not row.get("request_id") or date is None or not isinstance(info, list):
        return ()
    return tuple(chain.from_iterable(_guardrail_entry(row, date, entry) for entry in _ROWS.validate_python(info)))


def _guardrail_entry(row: Mapping[str, object], date: str, entry: Mapping[str, object]) -> tuple[Contribution, ...]:
    identity: Final = _text(entry.get("guardrail_id") or entry.get("guardrail_name"))
    usage: Final = entry.get("guardrail_usage")
    if not identity or not isinstance(usage, Mapping):
        return ()
    costs: Final = _guardrail_unit_costs(entry)
    return tuple(
        _guardrail_unit(row, (identity, date, unit), units, None if costs is None else costs.get(unit))
        for unit, units in _fields(usage).items()
        if isinstance(units, int) and not isinstance(units, bool) and units > 0
    )


def _guardrail_unit(
    row: Mapping[str, object], identity: tuple[str, str, str], units: int, cost: float | None
) -> Contribution:
    guardrail_id, date, unit = identity
    return Contribution(
        "LiteLLM_DailyGuardrailUsageUnits",
        (
            ("guardrail_id", guardrail_id),
            ("date", date),
            ("team_id", _text(row.get("team_id"))),
            ("api_key", _text(row.get("api_key"))),
            ("usage_unit", unit),
        ),
        (("units", units), ("cost", cost or 0.0), ("untracked_units", units if cost is None else 0)),
    )


def _model_task(row: Mapping[str, object]) -> str:
    return next(
        (
            task
            for tag in _tags(row)
            if tag.startswith(MODEL_INSIGHTS_TASK_TAG_PREFIX)
            and (task := tag.removeprefix(MODEL_INSIGHTS_TASK_TAG_PREFIX)) in _TASK_TYPES
        ),
        MODEL_INSIGHTS_DEFAULT_TASK,
    )


def _guardrail_unit_costs(entry: Mapping[str, object]) -> Mapping[str, float | None] | None:
    try:
        cost: Final = GuardrailCostByUnitEntry.model_validate(entry)
    except ValidationError:
        return None
    return None if cost.guardrail_cost_in_spend is False else cost.guardrail_cost_by_unit
