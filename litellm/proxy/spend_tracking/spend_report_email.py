import html
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger

EMPTY_MAP: Final[Mapping[str, object]] = MappingProxyType({})
_OBJECT_MAP_ADAPTER: Final = TypeAdapter(dict[str, object])
_ROW_LIST_ADAPTER: Final = TypeAdapter(list[dict[str, object]])
_STR_SEQUENCE_ADAPTER: Final = TypeAdapter(list[str])
LITELLM_LOGO_URL: Final = "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
LITELLM_SUPPORT_CONTACT: Final = "support@berri.ai"

SpendReportGroupBy: TypeAlias = Literal["team", "team_key", "team_model", "team_key_model"]
VALID_GROUP_BYS: Final[tuple[SpendReportGroupBy, ...]] = (
    "team",
    "team_key",
    "team_model",
    "team_key_model",
)


def get_server_timezone() -> str:
    tzname: Final = datetime.now().astimezone().tzname()
    if tzname:
        return tzname
    tzinfo: Final = datetime.now().astimezone().tzinfo
    if tzinfo:
        return str(tzinfo)
    return "UTC"


def _extract_group_by_candidates(v: object) -> Sequence[str]:
    if isinstance(v, str):
        return (v,)
    if isinstance(v, (list, tuple)):
        try:
            return _STR_SEQUENCE_ADAPTER.validate_python(v)
        except ValidationError:
            return ("team",)
    return ("team",)


def _normalize_group_by(v: object) -> tuple[SpendReportGroupBy, ...]:
    candidates: Final = _extract_group_by_candidates(v)
    filtered: Final[tuple[SpendReportGroupBy, ...]] = tuple(
        target for target in VALID_GROUP_BYS if target in candidates
    )
    return filtered if filtered else ("team",)


class SpendReportEmailAlert(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this alert",
    )
    name: str = Field(
        default="Spend Report Alert",
        description="Descriptive name for the alert",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this alert is enabled",
    )
    frequency: Literal["daily", "monthly"] = Field(
        default="daily",
        description="Alert frequency: 'daily' or 'monthly'",
    )
    send_time: str = Field(
        default="09:00",
        description="Send time in HH:MM (24-hour server format)",
    )
    recipient_emails: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Target email addresses for this alert",
    )
    group_by: tuple[SpendReportGroupBy, ...] = Field(
        default_factory=lambda: ("team",),
        description="Cumulative groupings to include: 'team', 'team_key', 'team_model', 'team_key_model'",
    )
    team_id: str | None = Field(
        default=None,
        description="Optional team filter to report only on a specific team",
    )

    @field_validator("group_by", mode="before")
    @classmethod
    def validate_group_by(cls, v: object) -> tuple[SpendReportGroupBy, ...]:
        return _normalize_group_by(v)


class SpendReportEmailSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    alerts: tuple[SpendReportEmailAlert, ...] = Field(
        default_factory=tuple,
        description="Configured spend report alert instances",
    )


class SpendReportEmailSendRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    recipient_emails: tuple[str, ...] | None = Field(
        default=None,
        description="Optional list of recipients. Defaults to configured recipient_emails",
    )
    frequency: Literal["daily", "monthly"] = Field(
        default="daily",
        description="Report frequency to generate: 'daily' or 'monthly'",
    )
    group_by: tuple[SpendReportGroupBy, ...] | None = Field(
        default=None,
        description="Optional list of groupings to include in the report",
    )
    start_date: str | None = Field(
        default=None,
        description="Optional YYYY-MM-DD start date override",
    )
    end_date: str | None = Field(
        default=None,
        description="Optional YYYY-MM-DD end date override",
    )
    team_id: str | None = Field(
        default=None,
        description="Optional team filter to report only on a specific team",
    )


class SpendReportEmailStatusResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    configured: bool = Field(
        default=False,
        description="Whether email delivery (SMTP, SendGrid, or Resend) is configured",
    )
    provider: Literal["smtp", "sendgrid", "resend"] | None = Field(
        default=None,
        description="Active email provider being used",
    )
    message: str = Field(
        default="",
        description="Informational message about email delivery readiness",
    )
    server_timezone: str = Field(
        default_factory=get_server_timezone,
        description="Server or container local timezone name or abbreviation",
    )


def _extract_env_vars() -> Mapping[str, object]:
    try:
        from litellm.proxy import proxy_server

        raw_settings: Final[object] = getattr(proxy_server, "general_settings", None)
    except (ImportError, AttributeError):
        return EMPTY_MAP
    if isinstance(raw_settings, dict):
        try:
            settings_map: Final = _OBJECT_MAP_ADAPTER.validate_python(raw_settings)
            raw_vars: Final[object] = settings_map.get("environment_variables")
            if isinstance(raw_vars, dict):
                return _OBJECT_MAP_ADAPTER.validate_python(raw_vars)
        except ValidationError:
            return EMPTY_MAP
    return EMPTY_MAP


def check_spend_report_email_status() -> SpendReportEmailStatusResponse:
    env_vars: Final = _extract_env_vars()
    server_tz: Final = get_server_timezone()
    if os.getenv("SENDGRID_API_KEY") or env_vars.get("SENDGRID_API_KEY"):
        return SpendReportEmailStatusResponse(
            configured=True,
            provider="sendgrid",
            message="SendGrid email delivery is configured",
            server_timezone=server_tz,
        )
    if os.getenv("RESEND_API_KEY") or env_vars.get("RESEND_API_KEY"):
        return SpendReportEmailStatusResponse(
            configured=True,
            provider="resend",
            message="Resend email delivery is configured",
            server_timezone=server_tz,
        )
    if os.getenv("SMTP_HOST") or env_vars.get("SMTP_HOST"):
        return SpendReportEmailStatusResponse(
            configured=True,
            provider="smtp",
            message="SMTP email delivery is configured",
            server_timezone=server_tz,
        )
    return SpendReportEmailStatusResponse(
        configured=False,
        provider=None,
        message="Email delivery is not configured. Configure SMTP server settings or set SENDGRID_API_KEY.",
        server_timezone=server_tz,
    )


async def _send_with_enterprise_logger(
    module_name: str,
    class_name: str,
    from_email: str,
    to_email: Sequence[str],
    subject: str,
    html: str,
) -> bool:
    try:
        module: Final[object] = __import__(module_name, fromlist=[class_name])
    except ImportError:
        return False
    logger_cls: Final[object] = getattr(module, class_name, None)
    if not callable(logger_cls):
        return False
    logger_instance: Final[object] = logger_cls()
    send_fn: Final[Callable[..., Awaitable[object]] | None] = getattr(logger_instance, "send_email", None)
    if send_fn is None:
        return False
    await send_fn(
        from_email=from_email,
        to_email=list(to_email),
        subject=subject,
        html_body=html,
    )
    return True


async def default_email_sender(
    receiver_email: str,
    subject: str,
    html: str,
) -> None:
    if os.getenv("SENDGRID_API_KEY"):
        sg_sender_email: Final = (
            os.getenv("SENDGRID_SENDER_EMAIL") or os.getenv("SMTP_SENDER_EMAIL") or "notifications@alerts.litellm.ai"
        )
        sg_sent: Final = await _send_with_enterprise_logger(
            module_name="litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email",
            class_name="SendGridEmailLogger",
            from_email=sg_sender_email,
            to_email=(receiver_email,),
            subject=subject,
            html=html,
        )
        if sg_sent:
            return
        verbose_proxy_logger.warning("litellm_enterprise not installed; falling back to SMTP")

    if os.getenv("RESEND_API_KEY"):
        resend_sender_email: Final = os.getenv("SMTP_SENDER_EMAIL") or "notifications@alerts.litellm.ai"
        resend_sent: Final = await _send_with_enterprise_logger(
            module_name="litellm_enterprise.enterprise_callbacks.send_emails.resend_email",
            class_name="ResendEmailLogger",
            from_email=resend_sender_email,
            to_email=(receiver_email,),
            subject=subject,
            html=html,
        )
        if resend_sent:
            return
        verbose_proxy_logger.warning("litellm_enterprise not installed; falling back to SMTP")

    from litellm.proxy.utils import send_email as send_smtp_email

    await send_smtp_email(
        receiver_email=receiver_email,
        subject=subject,
        html=html,
    )


class SpendReportSendResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status: str
    message: str
    recipients: tuple[str, ...]
    frequency: str
    start_date: str
    end_date: str
    total_spend: float
    total_requests: int
    team_id: str | None = None


class TeamSpendRow(TypedDict):
    team: ReadOnly[str]
    total_spend: ReadOnly[float]
    request_count: ReadOnly[int]


class TeamKeySpendRow(TypedDict):
    team: ReadOnly[str]
    api_key: ReadOnly[str]
    total_spend: ReadOnly[float]
    request_count: ReadOnly[int]


class TeamModelSpendRow(TypedDict):
    team: ReadOnly[str]
    model: ReadOnly[str]
    total_spend: ReadOnly[float]
    request_count: ReadOnly[int]


class TeamKeyModelSpendRow(TypedDict):
    team: ReadOnly[str]
    api_key: ReadOnly[str]
    model: ReadOnly[str]
    total_spend: ReadOnly[float]
    request_count: ReadOnly[int]


TEAM_SPEND_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
GROUP BY COALESCE(t.team_alias, s.team_id, 'No Team')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

TEAM_SPEND_WITH_TEAM_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
  AND (s.team_id = $3 OR t.team_alias = $3 OR t.team_id = $3)
GROUP BY COALESCE(t.team_alias, s.team_id, 'No Team')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

TEAM_KEY_SPEND_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key') AS api_key,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
LEFT JOIN "LiteLLM_VerificationToken" k ON s.api_key = k.token
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
GROUP BY 
    COALESCE(t.team_alias, s.team_id, 'No Team'),
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

TEAM_KEY_SPEND_WITH_TEAM_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key') AS api_key,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
LEFT JOIN "LiteLLM_VerificationToken" k ON s.api_key = k.token
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
  AND (s.team_id = $3 OR t.team_alias = $3 OR t.team_id = $3)
GROUP BY 
    COALESCE(t.team_alias, s.team_id, 'No Team'),
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

TEAM_MODEL_SPEND_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    COALESCE(NULLIF(s.model, ''), 'Unknown') AS model,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
GROUP BY 
    COALESCE(t.team_alias, s.team_id, 'No Team'),
    COALESCE(NULLIF(s.model, ''), 'Unknown')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

TEAM_MODEL_SPEND_WITH_TEAM_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    COALESCE(NULLIF(s.model, ''), 'Unknown') AS model,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
  AND (s.team_id = $3 OR t.team_alias = $3 OR t.team_id = $3)
GROUP BY 
    COALESCE(t.team_alias, s.team_id, 'No Team'),
    COALESCE(NULLIF(s.model, ''), 'Unknown')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

TEAM_KEY_MODEL_SPEND_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key') AS api_key,
    COALESCE(NULLIF(s.model, ''), 'Unknown') AS model,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
LEFT JOIN "LiteLLM_VerificationToken" k ON s.api_key = k.token
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
GROUP BY 
    COALESCE(t.team_alias, s.team_id, 'No Team'),
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key'),
    COALESCE(NULLIF(s.model, ''), 'Unknown')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

TEAM_KEY_MODEL_SPEND_WITH_TEAM_SQL: Final = """
SELECT 
    COALESCE(t.team_alias, s.team_id, 'No Team') AS team,
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key') AS api_key,
    COALESCE(NULLIF(s.model, ''), 'Unknown') AS model,
    ROUND(SUM(s.spend)::numeric, 2) AS total_spend,
    COUNT(*) AS request_count
FROM "LiteLLM_SpendLogs" s
LEFT JOIN "LiteLLM_TeamTable" t ON s.team_id = t.team_id
LEFT JOIN "LiteLLM_VerificationToken" k ON s.api_key = k.token
WHERE s."startTime" >= ($1::timestamptz AT TIME ZONE 'UTC')
  AND s."startTime" < (($2::timestamptz + INTERVAL '1 day') AT TIME ZONE 'UTC')
  AND (s.team_id = $3 OR t.team_alias = $3 OR t.team_id = $3)
GROUP BY 
    COALESCE(t.team_alias, s.team_id, 'No Team'),
    COALESCE(k.key_alias, k.key_name, NULLIF(s.api_key, ''), 'No Key'),
    COALESCE(NULLIF(s.model, ''), 'Unknown')
HAVING SUM(s.spend) > 0
ORDER BY total_spend DESC
"""

SPEND_REPORT_QUERIES: Final[Mapping[SpendReportGroupBy, str]] = {
    "team": TEAM_SPEND_SQL,
    "team_key": TEAM_KEY_SPEND_SQL,
    "team_model": TEAM_MODEL_SPEND_SQL,
    "team_key_model": TEAM_KEY_MODEL_SPEND_SQL,
}

SPEND_REPORT_QUERIES_WITH_TEAM: Final[Mapping[SpendReportGroupBy, str]] = {
    "team": TEAM_SPEND_WITH_TEAM_SQL,
    "team_key": TEAM_KEY_SPEND_WITH_TEAM_SQL,
    "team_model": TEAM_MODEL_SPEND_WITH_TEAM_SQL,
    "team_key_model": TEAM_KEY_MODEL_SPEND_WITH_TEAM_SQL,
}

GROUP_BY_TITLES: Final[Mapping[SpendReportGroupBy, str]] = {
    "team": "Spend by Team",
    "team_key": "Spend by Team and Key",
    "team_model": "Spend by Team and Model",
    "team_key_model": "Spend by Team, Key, and Model",
}

GROUP_BY_COLUMNS: Final[Mapping[SpendReportGroupBy, Sequence[tuple[str, str, str]]]] = {
    "team": (
        ("Team", "team", "left"),
        ("Total Spend", "total_spend", "right"),
        ("Requests", "request_count", "right"),
    ),
    "team_key": (
        ("Team", "team", "left"),
        ("Key / Token", "api_key", "left"),
        ("Total Spend", "total_spend", "right"),
        ("Requests", "request_count", "right"),
    ),
    "team_model": (
        ("Team", "team", "left"),
        ("Model", "model", "left"),
        ("Total Spend", "total_spend", "right"),
        ("Requests", "request_count", "right"),
    ),
    "team_key_model": (
        ("Team", "team", "left"),
        ("Key / Token", "api_key", "left"),
        ("Model", "model", "left"),
        ("Total Spend", "total_spend", "right"),
        ("Requests", "request_count", "right"),
    ),
}


def parse_send_time(time_str: str) -> tuple[int, int]:
    parts: Final = time_str.strip().split(":")
    if len(parts) != 2:
        return (9, 0)
    try:
        hour: Final = int(parts[0])
        minute: Final = int(parts[1])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return (hour, minute)
    except ValueError:
        return (9, 0)
    return (9, 0)


def get_daily_report_date_range(ref_date: date | None = None) -> tuple[date, date]:
    current_date: Final = ref_date or datetime.now().astimezone().date()
    yesterday: Final = current_date - timedelta(days=1)
    return (yesterday, yesterday)


def get_monthly_report_date_range(ref_date: date | None = None) -> tuple[date, date]:
    current_date: Final = ref_date or datetime.now().astimezone().date()
    first_of_current: Final = current_date.replace(day=1)
    last_of_prev: Final = first_of_current - timedelta(days=1)
    first_of_prev: Final = last_of_prev.replace(day=1)
    return (first_of_prev, last_of_prev)


def _settings_from_config(config: Mapping[str, object] | None) -> SpendReportEmailSettings | None:
    if config is None:
        return None
    raw_settings: Final[object] = config.get("litellm_settings")
    if not isinstance(raw_settings, dict):
        return None
    try:
        raw_settings_map: Final = _OBJECT_MAP_ADAPTER.validate_python(raw_settings)
    except ValidationError:
        return None
    email_data: Final[object] = raw_settings_map.get("spend_report_email_settings")
    if isinstance(email_data, (dict, Mapping)):
        return SpendReportEmailSettings.model_validate(email_data)
    return None


def _settings_from_memory() -> SpendReportEmailSettings | None:
    import litellm

    in_memory_val: Final = getattr(litellm, "spend_report_email_settings", None)
    if isinstance(in_memory_val, Mapping):
        return SpendReportEmailSettings.model_validate(in_memory_val)
    if isinstance(in_memory_val, SpendReportEmailSettings):
        return in_memory_val
    return None


def get_spend_report_email_settings(
    config: Mapping[str, object] | None = None,
) -> SpendReportEmailSettings:
    from_config: Final = _settings_from_config(config)
    if from_config is not None:
        return from_config

    from_mem: Final = _settings_from_memory()
    if from_mem is not None:
        return from_mem

    return SpendReportEmailSettings()


def _to_float(val: object) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


def _to_int(val: object) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val))
        except ValueError:
            return 0
    return 0


async def fetch_grouped_spend_data(
    prisma_client: object,
    group_by: SpendReportGroupBy,
    start_date: str,
    end_date: str,
    team_id: str | None = None,
) -> Sequence[Mapping[str, object]]:
    client: Final = getattr(prisma_client, "db", prisma_client)
    clean_team_id: Final = team_id.strip() if team_id is not None and team_id.strip() else None
    query_raw_fn: Final[Callable[..., Awaitable[object]] | None] = getattr(client, "query_raw", None)
    if query_raw_fn is None:
        return ()
    raw_rows: Final = (
        await query_raw_fn(SPEND_REPORT_QUERIES_WITH_TEAM[group_by], start_date, end_date, clean_team_id)
        if clean_team_id is not None
        else await query_raw_fn(SPEND_REPORT_QUERIES[group_by], start_date, end_date)
    )

    def _validate_rows(raw: object) -> Sequence[Mapping[str, object]]:
        if not isinstance(raw, (list, tuple)):
            return ()
        try:
            return _ROW_LIST_ADAPTER.validate_python(raw)
        except ValidationError:
            return ()

    rows_sequence: Final[Sequence[Mapping[str, object]]] = _validate_rows(raw_rows)

    def _sanitize_row(r: Mapping[str, object]) -> Mapping[str, object] | None:
        raw_spend: Final[object] = r.get("total_spend")
        spend_val: Final = _to_float(raw_spend) if raw_spend is not None else 0.0
        if spend_val <= 0:
            return None
        raw_requests: Final[object] = r.get("request_count")
        req_val: Final = _to_int(raw_requests) if raw_requests is not None else 0
        base_items: Final = tuple((k, v) for k, v in r.items() if k not in ("total_spend", "request_count"))
        return dict((*base_items, ("total_spend", round(spend_val, 2)), ("request_count", req_val)))

    sanitized_items: Final = (_sanitize_row(r) for r in rows_sequence)
    return tuple(clean_row for clean_row in sanitized_items if clean_row is not None)


def _render_table_html(
    group_by: SpendReportGroupBy,
    rows: Sequence[Mapping[str, object]],
) -> str:
    title: Final = GROUP_BY_TITLES[group_by]
    columns: Final = GROUP_BY_COLUMNS[group_by]
    header_cells_html: Final = "".join(
        f'<th style="padding: 10px 12px; font-weight: 600; color: #334155; text-align: {align};">{html.escape(label)}</th>'
        for label, _, align in columns
    )

    def _render_row(r: Mapping[str, object]) -> str:
        cells_html: Final = "".join(
            f'<td style="padding: 9px 12px; color: #1e293b; text-align: {align};">'
            + html.escape(
                f"${_to_float(r.get(field_key)):.2f}"
                if field_key == "total_spend"
                else f"{_to_int(r.get(field_key)):,}"
                if field_key == "request_count"
                else str(r.get(field_key) or "")
            )
            + "</td>"
            for _, field_key, align in columns
        )
        return f'<tr style="border-bottom: 1px solid #e2e8f0;">{cells_html}</tr>'

    tbody_html: Final = (
        f'<tr><td colspan="{len(columns)}" style="padding: 16px; text-align: center; color: #64748b;">'
        f"No spend recorded for this period</td></tr>"
        if not rows
        else "".join(_render_row(r) for r in rows)
    )

    return f"""
    <div style="margin-bottom: 28px;">
        <h3 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600; color: #1e293b;">{title}</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 13px; border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden;">
            <thead>
                <tr style="background-color: #f1f5f9; border-bottom: 2px solid #cbd5e1;">
                    {header_cells_html}
                </tr>
            </thead>
            <tbody>
                {tbody_html}
            </tbody>
        </table>
    </div>
    """


def build_spend_report_email_html(
    report_data: Mapping[SpendReportGroupBy, Sequence[Mapping[str, object]]],
    frequency: str,
    start_date: str,
    end_date: str,
    team_id: str | None = None,
) -> str:
    email_logo_url: Final = os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", LITELLM_LOGO_URL))
    email_support_contact: Final = os.getenv("EMAIL_SUPPORT_CONTACT", LITELLM_SUPPORT_CONTACT)

    first_rows: Final = next((rows for rows in report_data.values() if rows), ())
    overall_total_spend: Final = sum(_to_float(r.get("total_spend", 0.0)) for r in first_rows)
    overall_total_requests: Final = sum(_to_int(r.get("request_count", 0)) for r in first_rows)

    frequency_title: Final = frequency.capitalize()
    tables_html: Final = "".join(_render_table_html(group_by, rows) for group_by, rows in report_data.items())
    team_badge_html: Final = (
        f'<div style="font-size: 13px; color: #475569; margin-bottom: 6px;">Team Scope: <strong style="color: #0f172a;">{html.escape(team_id)}</strong></div>'
        if team_id
        else ""
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 20px; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <div style="max-width: 680px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="margin-bottom: 20px;">
                <img src="{email_logo_url}" alt="LiteLLM Logo" width="140" style="display: block;" />
            </div>

            <h2 style="margin: 0 0 6px 0; font-size: 20px; font-weight: 700; color: #0f172a;">
                LiteLLM {frequency_title} Spend Report
            </h2>
            {team_badge_html}
            <div style="font-size: 13px; color: #64748b; margin-bottom: 20px;">
                Period: <strong>{start_date}</strong> to <strong>{end_date}</strong>
            </div>

            <div style="margin-bottom: 24px; padding: 14px 16px; background-color: #f1f5f9; border-radius: 6px;">
                <div style="display: inline-block; width: 48%; vertical-align: top;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600;">Total Spend</div>
                    <div style="font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 4px;">${overall_total_spend:.2f}</div>
                </div>
                <div style="display: inline-block; width: 48%; vertical-align: top;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600;">Total Requests</div>
                    <div style="font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 4px;">{overall_total_requests:,}</div>
                </div>
            </div>

            {tables_html}

            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;" />
            <p style="font-size: 13px; color: #64748b; margin: 0 0 8px 0;">
                If you have any questions, please contact <a href="mailto:{email_support_contact}" style="color: #2563eb; text-decoration: none;">{email_support_contact}</a>
            </p>
            <p style="font-size: 13px; color: #64748b; margin: 0;">
                Best,<br />
                The LiteLLM Team
            </p>
        </div>
    </body>
    </html>
    """


def _resolve_report_date_range(
    frequency: str,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str]:
    if start_date is not None and end_date is not None:
        return (start_date, end_date)
    if frequency == "daily":
        d_start, d_end = get_daily_report_date_range()
        return (d_start.strftime("%Y-%m-%d"), d_end.strftime("%Y-%m-%d"))
    m_start, m_end = get_monthly_report_date_range()
    return (m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))


def _resolve_prisma_client(provided: object | None) -> object:
    if provided is not None:
        return provided
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is not None:
            return prisma_client
    except (ImportError, AttributeError):
        pass
    raise RuntimeError("Prisma client is not connected")


async def send_spend_report_email(
    frequency: Literal["daily", "monthly"] = "daily",
    recipient_emails: Sequence[str] | None = None,
    group_by: Sequence[SpendReportGroupBy] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    team_id: str | None = None,
    prisma_client: object | None = None,
    email_sender: Callable[..., Awaitable[None]] | None = None,
) -> SpendReportSendResult:
    if not recipient_emails:
        raise ValueError("No recipient emails configured for spend report email")

    resolved_recipients: Final[Sequence[str]] = recipient_emails

    resolved_dates: Final = _resolve_report_date_range(frequency, start_date, end_date)
    resolved_start_date: Final = resolved_dates[0]
    resolved_end_date: Final = resolved_dates[1]

    resolved_groupings: Final[Sequence[SpendReportGroupBy]] = (
        group_by if group_by is not None and len(group_by) > 0 else ("team",)
    )

    active_prisma_client: Final = _resolve_prisma_client(prisma_client)

    report_data: Final[Mapping[SpendReportGroupBy, Sequence[Mapping[str, object]]]] = {
        g: await fetch_grouped_spend_data(
            active_prisma_client, g, resolved_start_date, resolved_end_date, team_id=team_id
        )
        for g in resolved_groupings
    }

    html_content: Final = build_spend_report_email_html(
        report_data=report_data,
        frequency=frequency,
        start_date=resolved_start_date,
        end_date=resolved_end_date,
        team_id=team_id,
    )

    actual_sender: Final = email_sender if email_sender is not None else default_email_sender

    team_subject_suffix: Final = f" - Team: {team_id}" if team_id else ""
    subject: Final = f"LiteLLM {frequency.capitalize()} Spend Report{team_subject_suffix} ({resolved_start_date} to {resolved_end_date})"
    for email_addr in resolved_recipients:
        await actual_sender(
            receiver_email=email_addr,
            subject=subject,
            html=html_content,
        )

    first_group_rows: Final = next(iter(report_data.values()), ())
    total_spend: Final = (
        sum(_to_float(r.get("total_spend", 0.0)) for r in first_group_rows) if first_group_rows else 0.0
    )
    total_requests: Final = sum(_to_int(r.get("request_count", 0)) for r in first_group_rows) if first_group_rows else 0

    return SpendReportSendResult(
        status="success",
        message=f"Spend report email sent to {len(resolved_recipients)} recipient(s)",
        recipients=tuple(resolved_recipients),
        frequency=frequency,
        start_date=resolved_start_date,
        end_date=resolved_end_date,
        total_spend=round(total_spend, 2),
        total_requests=total_requests,
        team_id=team_id,
    )


def _safe_remove_job(remove_fn: Callable[[str], object] | None, job_id: str) -> None:
    if remove_fn is None or not job_id:
        return
    try:
        remove_fn(job_id)
    except Exception as exc:  # noqa: BLE001  # scheduler job removal is best-effort cleanup
        verbose_proxy_logger.debug("Spend report job %s could not be removed: %s", job_id, exc)


async def _resolve_current_alert(alert_id: str) -> tuple[bool, SpendReportEmailAlert | None]:
    from litellm.proxy.proxy_server import proxy_config

    get_config_fn: Final[Callable[[], Awaitable[object]] | None] = getattr(proxy_config, "get_config", None)
    if get_config_fn is not None:
        try:
            raw_config: Final[object] = await get_config_fn()
            config_map: Final[Mapping[str, object] | None] = (
                _OBJECT_MAP_ADAPTER.validate_python(raw_config) if isinstance(raw_config, (dict, Mapping)) else None
            )
            from_cfg: Final = _settings_from_config(config_map)
            if from_cfg is not None:
                matching: Final = next((a for a in from_cfg.alerts if a.id == alert_id), None)
                return (True, matching)
        except Exception as exc:  # noqa: BLE001  # fallback to memory if config query fails
            verbose_proxy_logger.debug("Failed to query current config for alert resolution: %s", exc)

    from_mem: Final = _settings_from_memory()
    if from_mem is not None:
        matching_mem: Final = next((a for a in from_mem.alerts if a.id == alert_id), None)
        return (True, matching_mem)

    return (False, None)


def _select_target_alert(
    captured_alert: SpendReportEmailAlert,
    settings_found: bool,
    current_alert: SpendReportEmailAlert | None,
) -> SpendReportEmailAlert:
    if settings_found and current_alert is not None:
        return current_alert
    return captured_alert


async def run_scheduled_email_spend_alert(
    alert: SpendReportEmailAlert,
) -> None:
    resolution_result: Final = await _resolve_current_alert(alert.id)
    settings_found: Final = resolution_result[0]
    current_alert: Final = resolution_result[1]
    if settings_found and (current_alert is None or not current_alert.enabled or not current_alert.recipient_emails):
        verbose_proxy_logger.info(
            "Skipping scheduled spend report alert %s: deleted, disabled, or has no recipients",
            alert.id,
        )
        return

    target_alert: Final = _select_target_alert(alert, settings_found, current_alert)

    from litellm.proxy.proxy_server import proxy_logging_obj

    job_id: Final = f"email_spend_report_{target_alert.id}"
    db_writer: Final[object] = getattr(proxy_logging_obj, "db_spend_update_writer", None)
    pod_lock_manager: Final[object] = getattr(db_writer, "pod_lock_manager", None) if db_writer is not None else None
    if pod_lock_manager is not None:
        acquire_lock_fn: Final[Callable[..., Awaitable[bool]] | None] = getattr(pod_lock_manager, "acquire_lock", None)
        if acquire_lock_fn is not None:
            has_lock: Final = await acquire_lock_fn(cronjob_id=job_id, ttl=3600, allow_reentrant=False)
            if not has_lock:
                verbose_proxy_logger.info("Email spend report lock already acquired by another pod: %s", job_id)
                return

    try:
        await send_spend_report_email(
            frequency=target_alert.frequency,
            recipient_emails=target_alert.recipient_emails,
            group_by=target_alert.group_by,
            team_id=target_alert.team_id,
        )
    except Exception as e:  # noqa: BLE001  # scheduled job runner must catch top-level exceptions to prevent crash
        verbose_proxy_logger.error("Failed to run scheduled email spend alert (%s): %s", target_alert.name, e)


_registered_spend_report_job_ids: Final[set[str]] = set()  # mutable-ok: active background job registry


def _setup_alert_jobs(
    add_job_fn: Callable[..., object],
    alerts: Sequence[SpendReportEmailAlert],
) -> None:
    for alert in alerts:
        if not alert.enabled or not alert.recipient_emails:
            continue

        job_id = f"email_spend_report_{alert.id}"
        hour, minute = parse_send_time(alert.send_time)

        def _build_runner(target_alert: SpendReportEmailAlert) -> Callable[[], Awaitable[None]]:
            async def _runner() -> None:
                await run_scheduled_email_spend_alert(target_alert)

            return _runner

        runner_fn = _build_runner(alert)

        if alert.frequency == "daily":
            add_job_fn(
                runner_fn,
                "cron",
                hour=hour,
                minute=minute,
                id=job_id,
                replace_existing=True,
            )
            _registered_spend_report_job_ids.add(job_id)
        elif alert.frequency == "monthly":
            add_job_fn(
                runner_fn,
                "cron",
                day=1,
                hour=hour,
                minute=minute,
                id=job_id,
                replace_existing=True,
            )
            _registered_spend_report_job_ids.add(job_id)


def _fetch_current_jobs(get_jobs_fn: Callable[[], Sequence[object]]) -> Sequence[object]:
    try:
        return get_jobs_fn()
    except Exception as exc:  # noqa: BLE001  # scheduler job inspection is best-effort
        verbose_proxy_logger.debug("Could not inspect current scheduler jobs: %s", exc)
        return ()


def setup_email_spend_report_jobs(
    scheduler: object,
    settings: SpendReportEmailSettings | None = None,
) -> None:
    if scheduler is None:
        return

    add_job_fn: Final[Callable[..., object] | None] = getattr(scheduler, "add_job", None)
    remove_job_fn: Final[Callable[[str], object] | None] = getattr(scheduler, "remove_job", None)
    if add_job_fn is None or remove_job_fn is None:
        return

    resolved_settings: Final = settings or get_spend_report_email_settings()

    for jid in _registered_spend_report_job_ids:
        _safe_remove_job(remove_job_fn, jid)
    _registered_spend_report_job_ids.clear()

    get_jobs_fn: Final[Callable[[], Sequence[object]] | None] = getattr(scheduler, "get_jobs", None)
    if get_jobs_fn is not None:
        current_jobs: Final = _fetch_current_jobs(get_jobs_fn)
        for job in current_jobs:
            jid_str = str(getattr(job, "id", ""))
            if jid_str.startswith("email_spend_report_"):
                _safe_remove_job(remove_job_fn, jid_str)

    _setup_alert_jobs(add_job_fn, resolved_settings.alerts)
