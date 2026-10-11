from collections.abc import Mapping, Sequence
from datetime import date
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import (
    DAILY_EMAIL_SPEND_REPORT_JOB_ID,
    MONTHLY_EMAIL_SPEND_REPORT_JOB_ID,
)
from litellm.proxy.spend_tracking.spend_report_email import (
    SpendReportEmailAlert,
    SpendReportEmailSettings,
    build_spend_report_email_html,
    check_spend_report_email_status,
    default_email_sender,
    fetch_grouped_spend_data,
    get_daily_report_date_range,
    get_monthly_report_date_range,
    parse_send_time,
    send_spend_report_email,
    setup_email_spend_report_jobs,
)


def test_parse_send_time() -> None:
    assert parse_send_time("09:00") == (9, 0)
    assert parse_send_time("00:00") == (0, 0)
    assert parse_send_time("23:59") == (23, 59)
    assert parse_send_time("14:30") == (14, 30)

    assert parse_send_time("24:00") == (9, 0)
    assert parse_send_time("12:60") == (9, 0)
    assert parse_send_time("invalid") == (9, 0)
    assert parse_send_time("") == (9, 0)
    assert parse_send_time("12:-1") == (9, 0)


def test_get_daily_report_date_range() -> None:
    start_date, end_date = get_daily_report_date_range(ref_date=date(2026, 10, 9))
    assert start_date == date(2026, 10, 8)
    assert end_date == date(2026, 10, 8)

    jan1_start, jan1_end = get_daily_report_date_range(ref_date=date(2026, 1, 1))
    assert jan1_start == date(2025, 12, 31)
    assert jan1_end == date(2025, 12, 31)


def test_get_monthly_report_date_range() -> None:
    oct_start, oct_end = get_monthly_report_date_range(ref_date=date(2026, 10, 1))
    assert oct_start == date(2026, 9, 1)
    assert oct_end == date(2026, 9, 30)

    jan_start, jan_end = get_monthly_report_date_range(ref_date=date(2026, 1, 1))
    assert jan_start == date(2025, 12, 1)
    assert jan_end == date(2025, 12, 31)

    leap_start, leap_end = get_monthly_report_date_range(ref_date=date(2024, 3, 1))
    assert leap_start == date(2024, 2, 1)
    assert leap_end == date(2024, 2, 29)


def test_spend_report_email_settings_validation() -> None:
    default_settings: Final = SpendReportEmailSettings()
    assert default_settings.enabled is False
    assert default_settings.frequency == "daily"
    assert default_settings.daily_send_time == "09:00"
    assert default_settings.monthly_send_time == "09:00"
    assert default_settings.recipient_emails == []
    assert default_settings.group_by == ["team"]

    string_group_by: Final = SpendReportEmailSettings(group_by="team_key")  # pyright: ignore[reportArgumentType]
    assert string_group_by.group_by == ["team_key"]

    multiple_group_by: Final = SpendReportEmailSettings(group_by=["team", "team_key", "team_model", "team_key_model"])
    assert multiple_group_by.group_by == ["team", "team_key", "team_model", "team_key_model"]

    invalid_filtered: Final = SpendReportEmailSettings(group_by=["invalid", "team_model"])  # pyright: ignore[reportArgumentType]
    assert invalid_filtered.group_by == ["team_model"]


@pytest.mark.asyncio
async def test_fetch_grouped_spend_data() -> None:
    mock_prisma: Final = MagicMock()
    mock_db: Final = MagicMock()
    mock_prisma.db = mock_db

    mock_db.query_raw = AsyncMock(
        return_value=[
            {"team": "Engineering", "total_spend": 18.879, "request_count": 3},
            {"team": "Sales", "total_spend": 1.0, "request_count": 2},
            {"team": "ZeroSpend", "total_spend": 0.0, "request_count": 5},
        ]
    )

    rows: Final = await fetch_grouped_spend_data(
        prisma_client=mock_prisma,
        group_by="team",
        start_date="2026-10-01",
        end_date="2026-10-08",
    )

    assert len(rows) == 2
    assert rows[0]["team"] == "Engineering"
    assert rows[0]["total_spend"] == 18.88
    assert rows[0]["request_count"] == 3
    assert rows[1]["team"] == "Sales"
    assert rows[1]["total_spend"] == 1.0
    assert rows[1]["request_count"] == 2


def test_build_spend_report_email_html() -> None:
    report_data: Final[Mapping[str, Sequence[Mapping[str, object]]]] = {
        "team": [
            {"team": "Engineering", "total_spend": 18.88, "request_count": 3},
            {"team": "Sales", "total_spend": 1.00, "request_count": 2},
        ],
        "team_key": [
            {"team": "Engineering", "api_key": "eng-key-1", "total_spend": 18.88, "request_count": 3},
        ],
    }

    html_out: Final = build_spend_report_email_html(
        report_data=report_data,  # pyright: ignore[reportArgumentType]
        frequency="daily",
        start_date="2026-10-08",
        end_date="2026-10-08",
    )

    assert "LiteLLM Daily Spend Report" in html_out
    assert "2026-10-08" in html_out
    assert "$19.88" in html_out
    assert "Engineering" in html_out
    assert "Sales" in html_out
    assert "eng-key-1" in html_out
    assert "Spend by Team" in html_out
    assert "Spend by Team and Key" in html_out


def test_build_spend_report_email_html_empty() -> None:
    report_data: Final[Mapping[str, Sequence[Mapping[str, object]]]] = {
        "team": [],
    }

    html_out: Final = build_spend_report_email_html(
        report_data=report_data,  # pyright: ignore[reportArgumentType]
        frequency="monthly",
        start_date="2026-09-01",
        end_date="2026-09-30",
    )

    assert "LiteLLM Monthly Spend Report" in html_out
    assert "$0.00" in html_out
    assert "No spend recorded for this period" in html_out


@pytest.mark.asyncio
async def test_send_spend_report_email() -> None:
    mock_prisma: Final = MagicMock()
    mock_db: Final = MagicMock()
    mock_prisma.db = mock_db
    mock_db.query_raw = AsyncMock(
        return_value=[
            {"team": "Engineering", "total_spend": 15.50, "request_count": 10},
        ]
    )

    sent_emails: Final[list[dict[str, str]]] = []

    async def mock_sender(receiver_email: str, subject: str, html: str) -> None:
        sent_emails.append({"to": receiver_email, "subject": subject, "html": html})

    result: Final = await send_spend_report_email(
        frequency="daily",
        recipient_emails=["alerts@example.com", "finance@example.com"],
        group_by=["team"],
        start_date="2026-10-08",
        end_date="2026-10-08",
        prisma_client=mock_prisma,
        email_sender=mock_sender,
    )

    assert result.status == "success"
    assert result.total_spend == 15.50
    assert result.total_requests == 10
    assert len(sent_emails) == 2
    assert sent_emails[0]["to"] == "alerts@example.com"
    assert sent_emails[1]["to"] == "finance@example.com"
    assert "LiteLLM Daily Spend Report (2026-10-08 to 2026-10-08)" in sent_emails[0]["subject"]
    assert "Engineering" in sent_emails[0]["html"]


@pytest.mark.asyncio
async def test_send_spend_report_email_no_recipients_raises() -> None:
    mock_prisma: Final = MagicMock()

    with pytest.raises(ValueError, match="No recipient emails configured"):
        await send_spend_report_email(
            frequency="daily",
            recipient_emails=[],
            prisma_client=mock_prisma,
            email_sender=AsyncMock(),
        )


def test_setup_email_spend_report_jobs() -> None:
    mock_scheduler: Final = MagicMock()
    added_jobs: Final[dict[str, dict[str, object]]] = {}

    def fake_add_job(func: object, trigger: str, **kwargs: object) -> None:
        job_id = str(kwargs.get("id"))
        added_jobs[job_id] = {"trigger": trigger, **kwargs}

    mock_scheduler.add_job.side_effect = fake_add_job
    mock_scheduler.remove_job = MagicMock()

    settings_daily: Final = SpendReportEmailSettings(
        enabled=True,
        frequency="daily",
        daily_send_time="09:00",
        recipient_emails=["team@example.com"],
    )
    setup_email_spend_report_jobs(mock_scheduler, settings=settings_daily)

    assert DAILY_EMAIL_SPEND_REPORT_JOB_ID in added_jobs
    assert added_jobs[DAILY_EMAIL_SPEND_REPORT_JOB_ID]["hour"] == 9
    assert added_jobs[DAILY_EMAIL_SPEND_REPORT_JOB_ID]["minute"] == 0
    mock_scheduler.remove_job.assert_called_with(MONTHLY_EMAIL_SPEND_REPORT_JOB_ID)

    added_jobs.clear()
    settings_monthly: Final = SpendReportEmailSettings(
        enabled=True,
        frequency="monthly",
        monthly_send_time="10:30",
        recipient_emails=["team@example.com"],
    )
    setup_email_spend_report_jobs(mock_scheduler, settings=settings_monthly)

    assert MONTHLY_EMAIL_SPEND_REPORT_JOB_ID in added_jobs
    assert added_jobs[MONTHLY_EMAIL_SPEND_REPORT_JOB_ID]["day"] == 1
    assert added_jobs[MONTHLY_EMAIL_SPEND_REPORT_JOB_ID]["hour"] == 10
    assert added_jobs[MONTHLY_EMAIL_SPEND_REPORT_JOB_ID]["minute"] == 30
    mock_scheduler.remove_job.assert_called_with(DAILY_EMAIL_SPEND_REPORT_JOB_ID)

    added_jobs.clear()
    settings_disabled: Final = SpendReportEmailSettings(
        enabled=False,
        recipient_emails=["team@example.com"],
    )
    setup_email_spend_report_jobs(mock_scheduler, settings=settings_disabled)
    mock_scheduler.remove_job.assert_any_call(DAILY_EMAIL_SPEND_REPORT_JOB_ID)
    mock_scheduler.remove_job.assert_any_call(MONTHLY_EMAIL_SPEND_REPORT_JOB_ID)


def test_get_spend_report_email_settings_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import app, proxy_config

    async def mock_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test")

    async def mock_get_config() -> dict[str, object]:
        return {
            "litellm_settings": {
                "spend_report_email_settings": {
                    "enabled": True,
                    "frequency": "both",
                    "daily_send_time": "08:30",
                    "monthly_send_time": "10:00",
                    "recipient_emails": ["finance@example.com"],
                    "group_by": ["team", "team_key"],
                }
            }
        }

    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    app.dependency_overrides[user_api_key_auth] = mock_auth
    try:
        client: Final = TestClient(app)
        response: Final = client.get("/get/spend_report_email_settings")
        assert response.status_code == 200
        data: Final = response.json()
        assert "values" in data
        assert "field_schema" in data
        assert data["values"]["enabled"] is True
        assert data["values"]["frequency"] == "both"
        assert data["values"]["daily_send_time"] == "08:30"
        assert data["values"]["monthly_send_time"] == "10:00"
        assert data["values"]["recipient_emails"] == ["finance@example.com"]
        assert data["values"]["group_by"] == ["team", "team_key"]
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


def test_update_spend_report_email_settings_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from litellm.proxy import proxy_server
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import app, proxy_config

    async def mock_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test")

    stored_config: dict[str, object] = {"litellm_settings": {}}

    async def mock_get_config() -> dict[str, object]:
        return stored_config

    async def mock_save_config(new_config: dict[str, object]) -> dict[str, object]:
        nonlocal stored_config
        stored_config = new_config
        return new_config

    monkeypatch.setattr(proxy_server, "store_model_in_db", True)
    monkeypatch.setattr(proxy_config, "get_config", mock_get_config)
    monkeypatch.setattr(proxy_config, "save_config", mock_save_config)
    app.dependency_overrides[user_api_key_auth] = mock_auth

    try:
        client: Final = TestClient(app)
        payload: Final = {
            "enabled": True,
            "frequency": "daily",
            "daily_send_time": "09:15",
            "monthly_send_time": "09:00",
            "recipient_emails": ["dev@example.com"],
            "group_by": ["team", "team_model"],
        }
        response: Final = client.patch("/update/spend_report_email_settings", json=payload)
        assert response.status_code == 200
        data: Final = response.json()
        assert data["status"] == "success"
        assert data["settings"]["enabled"] is True
        assert data["settings"]["daily_send_time"] == "09:15"
        assert data["settings"]["group_by"] == ["team", "team_model"]
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


def test_trigger_spend_report_email_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import app
    from litellm.proxy.spend_tracking import spend_report_email
    from litellm.proxy.spend_tracking.spend_report_email import SpendReportSendResult

    async def mock_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test")

    mock_send = AsyncMock(
        return_value=SpendReportSendResult(
            status="success",
            message="Spend report email sent to 1 recipient(s)",
            recipients=["test@example.com"],
            frequency="daily",
            start_date="2026-10-08",
            end_date="2026-10-08",
            total_spend=12.34,
            total_requests=5,
        )
    )
    monkeypatch.setattr(spend_report_email, "send_spend_report_email", mock_send)
    app.dependency_overrides[user_api_key_auth] = mock_auth

    try:
        client: Final = TestClient(app)
        payload: Final = {
            "frequency": "daily",
            "recipient_emails": ["test@example.com"],
            "group_by": ["team"],
        }
        response: Final = client.post("/spend/report/email/send", json=payload)
        assert response.status_code == 200
        data: Final = response.json()
        assert data["status"] == "success"
        assert data["total_spend"] == 12.34
        assert data["total_requests"] == 5
        assert data["recipients"] == ["test@example.com"]
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


def test_trigger_spend_report_email_requires_admin() -> None:
    from fastapi.testclient import TestClient

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import app

    async def mock_non_admin_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-test")

    app.dependency_overrides[user_api_key_auth] = mock_non_admin_auth
    try:
        client: Final = TestClient(app)
        response: Final = client.post("/spend/report/email/send", json={"frequency": "daily"})
        assert response.status_code == 403
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


def test_get_spend_report_email_status_endpoint() -> None:
    from fastapi.testclient import TestClient

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.proxy_server import app

    async def mock_admin_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test")

    async def mock_non_admin_auth() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-test")

    client: Final = TestClient(app)

    app.dependency_overrides[user_api_key_auth] = mock_non_admin_auth
    try:
        non_admin_response: Final = client.get("/spend/report/email/status")
        assert non_admin_response.status_code == 403
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)

    app.dependency_overrides[user_api_key_auth] = mock_admin_auth
    try:
        admin_response: Final = client.get("/spend/report/email/status")
        assert admin_response.status_code == 200
        assert "configured" in admin_response.json()
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_fetch_grouped_spend_data_with_team_id() -> None:
    captured_queries: Final[list[tuple[str, tuple[object, ...]]]] = []

    class MockPrismaDb:
        async def query_raw(self, query: str, *args: object) -> list[dict[str, object]]:
            captured_queries.append((query, args))
            return [{"team": "engineering", "total_spend": 25.5, "request_count": 10}]

    mock_prisma = MockPrismaDb()
    rows = await fetch_grouped_spend_data(mock_prisma, "team", "2026-10-01", "2026-10-08", team_id="engineering")
    assert len(rows) == 1
    assert rows[0]["team"] == "engineering"
    assert rows[0]["total_spend"] == 25.5
    assert len(captured_queries) == 1
    query_str, params = captured_queries[0]
    assert "/* TEAM_FILTER */" not in query_str
    assert "AND (s.team_id = $3 OR t.team_alias = $3 OR t.team_id = $3)" in query_str
    assert params == ("2026-10-01", "2026-10-08", "engineering")


def test_setup_email_spend_report_jobs_multiple_alerts() -> None:
    mock_scheduler: Final = MagicMock()
    added_jobs: Final[dict[str, dict[str, object]]] = {}

    def fake_add_job(func: object, trigger: str, **kwargs: object) -> None:
        job_id = str(kwargs.get("id"))
        added_jobs[job_id] = {"trigger": trigger, **kwargs}

    mock_scheduler.add_job.side_effect = fake_add_job
    mock_scheduler.remove_job = MagicMock()
    mock_scheduler.get_jobs.return_value = []

    alert1 = SpendReportEmailAlert(
        id="alert-daily-finance",
        name="Daily Finance",
        enabled=True,
        frequency="daily",
        send_time="08:15",
        recipient_emails=["finance@example.com"],
        group_by=["team"],
    )
    alert2 = SpendReportEmailAlert(
        id="alert-monthly-exec",
        name="Monthly Exec",
        enabled=True,
        frequency="monthly",
        send_time="10:00",
        recipient_emails=["exec@example.com"],
        group_by=["team_key"],
    )
    alert3 = SpendReportEmailAlert(
        id="alert-daily-eng",
        name="Daily Engineering",
        enabled=True,
        frequency="daily",
        send_time="09:00",
        recipient_emails=["eng@example.com"],
        team_id="eng_team",
    )
    alert_disabled = SpendReportEmailAlert(
        id="alert-disabled",
        name="Disabled Alert",
        enabled=False,
        frequency="daily",
        recipient_emails=["none@example.com"],
    )

    settings = SpendReportEmailSettings(
        enabled=True,
        alerts=[alert1, alert2, alert3, alert_disabled],
    )
    setup_email_spend_report_jobs(mock_scheduler, settings=settings)

    assert "email_spend_report_alert-daily-finance" in added_jobs
    assert added_jobs["email_spend_report_alert-daily-finance"]["hour"] == 8
    assert added_jobs["email_spend_report_alert-daily-finance"]["minute"] == 15

    assert "email_spend_report_alert-monthly-exec" in added_jobs
    assert added_jobs["email_spend_report_alert-monthly-exec"]["day"] == 1
    assert added_jobs["email_spend_report_alert-monthly-exec"]["hour"] == 10
    assert added_jobs["email_spend_report_alert-monthly-exec"]["minute"] == 0

    assert "email_spend_report_alert-daily-eng" in added_jobs
    assert "email_spend_report_alert-disabled" not in added_jobs


def test_check_spend_report_email_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("SMTP_HOST", raising=False)

    unconf: Final = check_spend_report_email_status()
    assert unconf.configured is False
    assert unconf.provider is None
    assert unconf.server_timezone is not None
    assert len(unconf.server_timezone) > 0

    monkeypatch.setenv("SENDGRID_API_KEY", "sg-test-123")
    sg_status: Final = check_spend_report_email_status()
    assert sg_status.configured is True
    assert sg_status.provider == "sendgrid"

    monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
    monkeypatch.setenv("RESEND_API_KEY", "re-test-123")
    resend_status: Final = check_spend_report_email_status()
    assert resend_status.configured is True
    assert resend_status.provider == "resend"

    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    smtp_status: Final = check_spend_report_email_status()
    assert smtp_status.configured is True
    assert smtp_status.provider == "smtp"


@pytest.mark.asyncio
async def test_default_email_sender_sendgrid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENDGRID_API_KEY", "sg-test-key")
    monkeypatch.setenv("SENDGRID_SENDER_EMAIL", "sender@example.com")

    mock_send: Final = AsyncMock()
    mock_logger: Final = MagicMock()
    mock_logger.send_email = mock_send

    monkeypatch.setattr(
        "litellm_enterprise.enterprise_callbacks.send_emails.sendgrid_email.SendGridEmailLogger",
        lambda: mock_logger,
    )

    await default_email_sender(
        receiver_email="receiver@example.com",
        subject="Test Report",
        html="<p>Test</p>",
    )

    mock_send.assert_awaited_once_with(
        from_email="sender@example.com",
        to_email=["receiver@example.com"],
        subject="Test Report",
        html_body="<p>Test</p>",
    )


@pytest.mark.asyncio
async def test_default_email_sender_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
    monkeypatch.delenv("RESEND_API_KEY", raising=False)

    mock_smtp_send: Final = AsyncMock()
    monkeypatch.setattr("litellm.proxy.utils.send_email", mock_smtp_send)

    await default_email_sender(
        receiver_email="receiver@example.com",
        subject="SMTP Report",
        html="<p>SMTP</p>",
    )

    mock_smtp_send.assert_awaited_once_with(
        receiver_email="receiver@example.com",
        subject="SMTP Report",
        html="<p>SMTP</p>",
    )
