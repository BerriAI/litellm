"""Retirement reminder schedule, durable dedupe, ownership scope and safe rendering."""

import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations import model_deprecation_email as module
from litellm.integrations.email_templates.model_deprecation_email import (
    render_model_deprecation_email,
)
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LiteLLM_VerificationToken
from litellm.repositories.config_repository import ConfigParam
from litellm.types.integrations.model_deprecation_email import (
    DeprecationEmailMilestone,
    ModelDeprecationEmail,
)
from litellm.types.proxy.model_deprecation import (
    ModelDeprecationInfo,
    ModelDeprecationResponse,
    SuccessorModelSource,
)

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def info(days=30, model="gemini/gemini-2.5-pro"):
    return ModelDeprecationInfo(
        model_name="research",
        litellm_model=model,
        deprecation_date=NOW.date() + timedelta(days=days),
        days_until_deprecation=days,
        status="imminent",
        successor_model="gemini/gemini-3.8-flash",
        successor_model_source=SuccessorModelSource.MODEL_COST_MAP,
        deprecation_source="https://ai.google.dev/gemini-api/docs/deprecations",
    )


class MemoryConfig:
    def __init__(self):
        self.rows = {"general_settings": {"email_settings": {"Model Deprecations": True}}}

    async def model_deprecations_enabled(self):
        return self.rows.get("model_deprecation_alerts_enabled", True)

    async def get_param(self, key):
        return ConfigParam(key, self.rows[key]) if key in self.rows else None

    async def set_param_if_revision(self, key, value, revision):
        current = self.rows.get(key, {}).get("revision", 0)
        if current != revision:
            return False
        self.rows[key] = value
        return True

    async def delete_params_older_than(self, prefix, before):
        return 0


@pytest.fixture()
def sender(monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "alerts@example.com")
    instance = module.ModelDeprecationEmailSender(MagicMock())
    instance.config = MemoryConfig()
    instance._impacts = AsyncMock(
        return_value=MappingProxyType(
            {"owner@example.com": module._Impact(key_count=3, requests=12, teams=frozenset({"Research"}))}
        )
    )
    send = AsyncMock(return_value=True)
    monkeypatch.setattr("litellm.proxy.utils.send_email", send)
    return instance, send


def schedule(monkeypatch, models):
    snapshot = ModelDeprecationResponse(
        deprecated=[], imminent=models, upcoming=[], warn_within_days=30, checked_at=NOW
    )
    monkeypatch.setattr(module, "collect_model_deprecations", lambda **kwargs: snapshot)


@pytest.mark.asyncio
async def test_should_send_only_the_two_milestones_and_keep_receipts_across_restart(sender, monkeypatch):
    instance, send = sender
    schedule(monkeypatch, [info()])
    assert await instance.run(MagicMock(), now=NOW) == 1
    assert await instance.run(MagicMock(), now=NOW) == 0
    restarted = module.ModelDeprecationEmailSender(MagicMock())
    restarted.config = instance.config
    restarted._impacts = instance._impacts
    assert await restarted.run(MagicMock(), now=NOW) == 0
    final_day = NOW + timedelta(days=29)
    schedule(monkeypatch, [info().model_copy(update={"days_until_deprecation": 1})])
    assert await restarted.run(MagicMock(), now=final_day) == 1
    assert await restarted.run(MagicMock(), now=final_day) == 0
    assert send.await_count == 2
    assert "in 30 days" in send.await_args_list[0].kwargs["subject"]
    assert "tomorrow" in send.await_args_list[1].kwargs["subject"]
    assert send.await_args_list[0].kwargs["message_id"] != send.await_args_list[1].kwargs["message_id"]
    assert "<html" in send.await_args_list[0].kwargs["html"]
    assert "3 keys | 12 requests in the last 30 days" in send.await_args_list[0].kwargs["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("days", [31, 29, 2, 0, -1])
async def test_should_not_send_on_other_days(sender, monkeypatch, days):
    instance, send = sender
    schedule(monkeypatch, [info(days)])
    assert await instance.run(MagicMock(), now=NOW) == 0
    send.assert_not_awaited()
    instance._impacts.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [{}, {"email_settings": {}}, {"email_settings": {"Model Deprecations": False}}],
)
async def test_should_require_explicit_opt_in(sender, settings):
    instance, send = sender
    instance.config.rows["general_settings"] = settings
    assert await instance.run(MagicMock(), now=NOW) == 0
    send.assert_not_awaited()
    instance._impacts.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_claim_delivery_once_across_concurrent_workers(sender, monkeypatch):
    instance, send = sender
    schedule(monkeypatch, [info()])
    second = module.ModelDeprecationEmailSender(MagicMock())
    second.config = instance.config
    second._impacts = instance._impacts
    assert sum(await asyncio.gather(instance.run(MagicMock(), now=NOW), second.run(MagicMock(), now=NOW))) == 1
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_should_retry_failed_delivery_after_the_lease_expires(sender, monkeypatch):
    instance, send = sender
    schedule(monkeypatch, [info()])
    send.side_effect = [False, True]
    assert await instance.run(MagicMock(), now=NOW) == 0
    assert await instance.run(MagicMock(), now=NOW + timedelta(minutes=1)) == 0
    assert await instance.run(MagicMock(), now=NOW + timedelta(minutes=6)) == 1
    assert await instance.run(MagicMock(), now=NOW + timedelta(minutes=7)) == 0
    assert send.await_count == 2
    assert send.await_args_list[0].kwargs["message_id"] == send.await_args_list[1].kwargs["message_id"]


@pytest.mark.asyncio
async def test_should_dedupe_model_aliases_and_skip_models_with_no_observed_owners(sender, monkeypatch):
    instance, send = sender
    schedule(monkeypatch, [info(), info().model_copy(update={"model_name": "second-alias"})])
    assert await instance.run(MagicMock(), now=NOW) == 1
    send.assert_awaited_once()
    instance.config.rows = {"general_settings": {"email_settings": {"Model Deprecations": True}}}
    instance._impacts.return_value = {}
    assert await instance.run(MagicMock(), now=NOW) == 0
    assert send.await_count == 1


@pytest.mark.asyncio
async def test_should_scope_usage_to_current_owners_and_admins_excluding_former_admins_across_pages(
    monkeypatch,
):
    instance = module.ModelDeprecationEmailSender(MagicMock())
    monkeypatch.setattr(module, "_PAGE_SIZE", 2)
    grouped = AsyncMock(
        side_effect=[
            [
                {"api_key": "one", "_sum": {"api_requests": 10}},
                {"api_key": "two", "_sum": {"api_requests": 20}},
            ],
            [{"api_key": "three", "_sum": {"api_requests": 30}}],
        ]
    )
    monkeypatch.setattr(
        module.DailyUserSpendRepository,
        "table",
        property(lambda _: MagicMock(group_by=grouped)),
    )
    keys = [
        LiteLLM_VerificationToken(token="one", user_id="alice", team_id="research"),
        LiteLLM_VerificationToken(token="two", user_id="alice"),
        LiteLLM_VerificationToken(token="three", user_id="bob", team_id="research"),
    ]

    async def key_rows(_, field, values):
        ids = set(values)
        return [key for key in keys if key.token in ids]

    monkeypatch.setattr(module.VerificationTokenRepository, "find_many_in", key_rows)
    team = LiteLLM_TeamTable(
        team_id="research",
        team_alias="Research",
        admins=["bob", "mallory", "removed-admin"],
        members_with_roles=[
            {"user_id": "bob", "user_email": "BOB@example.com", "role": "admin"},
            {"user_id": "mallory", "user_email": "mallory@example.com", "role": "user"},
        ],
    )

    async def team_rows(_, field, values):
        return [team] if "research" in values else []

    monkeypatch.setattr(module.TeamRepository, "find_many_in", team_rows)

    async def user_rows(_, field, values):
        return [LiteLLM_UserTable(user_id=user_id, user_email=f"{user_id}@example.com") for user_id in values]

    monkeypatch.setattr(module.UserRepository, "find_many_in", user_rows)
    impacts = await instance._impacts("gemini/gemini-2.5-pro", date(2026, 10, 3), NOW.date())
    assert set(impacts) == {"alice@example.com", "bob@example.com"}
    assert impacts["alice@example.com"].requests == 30
    assert impacts["alice@example.com"].key_count == 2
    assert impacts["bob@example.com"].requests == 40
    assert impacts["bob@example.com"].key_count == 2
    assert impacts["bob@example.com"].teams == frozenset({"Research"})
    assert grouped.await_args_list[0].kwargs["where"] == {
        "model": "gemini/gemini-2.5-pro",
        "date": {"gte": "2026-10-03", "lte": "2026-11-01"},
        "api_requests": {"gt": 0},
    }


def test_should_render_the_current_ui_logo_safe_model_text_and_actionable_links():
    model = info().model_copy(
        update={
            "successor_model": '<script>alert("x")</script>',
            "deprecation_source": "javascript:alert(1)",
        }
    )
    email = ModelDeprecationEmail(
        model=model,
        usage_model=model.litellm_model,
        recipient_email="owner@example.com",
        milestone=DeprecationEmailMilestone.THIRTY_DAYS,
        key_count=3,
        requests=12,
        team_names=("Research",),
        start_date=date(2026, 10, 3),
        end_date=NOW.date(),
    )
    rendered = render_model_deprecation_email(email, base_url="https://gateway.example")
    assert "/public/assets/logos/litellm_logo.png" in rendered.html
    assert "litellm-listing.s3.amazonaws.com" not in rendered.html
    assert "<script>" not in rendered.html
    assert "javascript:" not in rendered.html
    assert "&lt;script&gt;" in rendered.html
    assert "/ui/usage?tab=models&amp;from=2026-10-03&amp;to=2026-11-01" in rendered.html
    assert "Recommendation source: LiteLLM model cost map" in rendered.text
    assert "Google is deprecating this model on 01 Dec 2026 (UTC)." in rendered.html
    assert "Google is deprecating this model on 01 Dec 2026 (UTC)." in rendered.text
    assert "google_pse.png" in rendered.html
    assert 'alt="Google"' in rendered.html
    assert "BEFORE &middot; MODEL TO REPLACE" in rendered.html
    assert "AFTER &middot; RECOMMENDED MODEL" in rendered.html
    assert "What to do" in rendered.html
    assert "check that requests to the old model reach zero" in rendered.text
    missing = render_model_deprecation_email(
        replace(email, model=model.model_copy(update={"successor_model": None})),
        base_url="https://gateway.example",
    )
    assert "No recommended model listed. Choose a supported replacement." in missing.html
    assert "Choose a supported replacement and test it with your workload." in missing.text
    tomorrow = render_model_deprecation_email(
        replace(email, milestone=DeprecationEmailMilestone.ONE_DAY),
        base_url="https://gateway.example",
    )
    assert "01 DAY UNTIL DEPRECATION" in tomorrow.html
    assert "Model being deprecated" in rendered.html
    assert "Model being deprecated" in rendered.subject
    assert "Model being deprecated" in tomorrow.html
    assert "Model being deprecated" in tomorrow.subject
    assert "FINAL REMINDER" in tomorrow.html
    assert "tomorrow" in tomorrow.subject


@pytest.mark.asyncio
async def test_should_keep_the_email_loop_running_after_a_delivery_error(monkeypatch):
    from litellm.proxy import proxy_server

    router = MagicMock()
    database = MagicMock()
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    runner = MagicMock()
    runner.run = AsyncMock(side_effect=[RuntimeError("temporary SMTP failure"), 0])
    monkeypatch.setattr(module, "ModelDeprecationEmailSender", lambda _: runner)
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock(side_effect=[None, asyncio.CancelledError()]))
    with pytest.raises(asyncio.CancelledError):
        await module.run_scheduled_model_deprecation_emails()
    assert runner.run.await_count == 2
    assert all(call.args == (router,) for call in runner.run.await_args_list)


@pytest.mark.asyncio
async def test_gateway_opt_out_stops_owner_usage_scans_and_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "alerts@example.com")
    client: Final = MagicMock()
    client.writer_db.litellm_config.find_unique = AsyncMock(return_value=MagicMock(param_value=False))
    client.db.litellm_dailyuserspend.group_by = AsyncMock()
    instance: Final = module.ModelDeprecationEmailSender(client)
    assert await instance.run(MagicMock(), now=NOW) == 0
    client.db.litellm_dailyuserspend.group_by.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ("SMTP_HOST", "SMTP_SENDER_EMAIL"))
async def test_missing_email_connection_does_not_scan_owner_usage(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "alerts@example.com")
    monkeypatch.delenv(missing)
    client: Final = MagicMock()
    client.writer_db.litellm_config.find_unique = AsyncMock()
    client.db.litellm_dailyuserspend.group_by = AsyncMock()
    instance: Final = module.ModelDeprecationEmailSender(client)
    assert await instance.run(MagicMock(), now=NOW) == 0
    client.writer_db.litellm_config.find_unique.assert_not_awaited()
    client.db.litellm_dailyuserspend.group_by.assert_not_awaited()
