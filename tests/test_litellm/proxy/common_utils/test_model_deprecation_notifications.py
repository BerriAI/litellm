"""Tests for the model deprecation email notifications module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from types import SimpleNamespace
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.caching.caching import DualCache
from litellm.constants import EMAIL_MODEL_DEPRECATION_LOCK_ID
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.common_utils.model_deprecation_notifications import (
    AffectedModel,
    DeprecationEmailContext,
    TeamNotification,
    build_team_notifications,
    email_sent_key,
    make_email_deliverer,
    render_model_deprecation_email,
    resolve_affected_teams,
    select_milestone,
    send_model_deprecation_emails,
)
from litellm.types.integrations.slack_alerting import SlackAlertingArgs, SlackAlertingCacheKeys
from litellm.types.proxy.model_deprecation import DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS, ModelDeprecationInfo


class TestSelectMilestone:
    @pytest.mark.parametrize(
        ("days_until", "expected"),
        [(45, None), (30, 30), (25, 30), (7, 7), (3, 7), (0, 0), (-12, 0)],
    )
    def test_should_pick_the_most_urgent_threshold_reached(self, days_until, expected):
        assert select_milestone(days_until, (30, 7, 0)) == expected

    def test_should_return_none_when_no_thresholds_configured(self):
        assert select_milestone(-5, ()) is None

    def test_should_not_depend_on_threshold_order(self):
        assert select_milestone(5, (0, 7, 30)) == 7


def _info(model_name: str, days_until: int, provider: str | None = "openai") -> ModelDeprecationInfo:
    return ModelDeprecationInfo(
        model_name=model_name,
        litellm_model=model_name,
        deprecation_date=date(2026, 10, 1),
        days_until_deprecation=days_until,
        status="deprecated" if days_until < 0 else ("imminent" if days_until <= 30 else "upcoming"),
        litellm_provider=provider,
    )


def _router(deployments: Sequence[Mapping[str, object]]) -> MagicMock:
    router: Final = MagicMock()
    router.get_model_list.return_value = deployments
    router.get_model_access_groups.return_value = {}
    router.model_group_alias = {}
    return router


def _team(team_id: str, models: Sequence[str] = (), blocked: bool = False) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id=team_id, models=list(models), blocked=blocked)


THRESHOLDS: Final = (30, 7, 0)


class TestResolveAffectedTeams:
    @pytest.mark.asyncio
    async def test_should_include_team_listing_the_model_explicitly(self):
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5),), _router([]), (_team("t1", ("gpt-old",)), _team("t2", ("other",))), THRESHOLDS
        )
        assert set(affected) == {"t1"}
        assert affected["t1"] == (AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),)

    @pytest.mark.asyncio
    async def test_should_include_unrestricted_team(self):
        affected: Final = await resolve_affected_teams((_info("gpt-old", 5),), _router([]), (_team("t1"),), THRESHOLDS)
        assert set(affected) == {"t1"}

    @pytest.mark.asyncio
    async def test_should_match_wildcard_allowlists(self):
        affected: Final = await resolve_affected_teams(
            (_info("openai/gpt-old", 5),),
            _router([]),
            (_team("t1", ("openai/*",)), _team("t2", ("azure/*",))),
            THRESHOLDS,
        )
        assert set(affected) == {"t1"}

    @pytest.mark.asyncio
    async def test_should_skip_blocked_teams(self):
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5),), _router([]), (_team("t1", blocked=True),), THRESHOLDS
        )
        assert dict(affected) == {}

    @pytest.mark.asyncio
    async def test_should_skip_models_that_have_not_reached_a_threshold(self):
        affected: Final = await resolve_affected_teams((_info("gpt-old", 45),), _router([]), (_team("t1"),), THRESHOLDS)
        assert dict(affected) == {}

    @pytest.mark.asyncio
    async def test_should_attribute_team_scoped_deployment_to_its_owner_only(self):
        deployments: Final = [
            {
                "model_name": "model_name_t2_abc",
                "litellm_params": {"model": "openai/gpt-old"},
                "model_info": {"id": "1", "team_id": "t2", "team_public_model_name": "gpt-old"},
            }
        ]
        affected: Final = await resolve_affected_teams(
            (_info("model_name_t2_abc", -3),), _router(deployments), (_team("t1"), _team("t2")), THRESHOLDS
        )
        assert set(affected) == {"t2"}
        assert affected["t2"][0].display_name == "gpt-old"
        assert affected["t2"][0].milestone == 0

    @pytest.mark.asyncio
    async def test_should_collect_every_reachable_model_per_team(self):
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5), _info("gpt-older", -1)),
            _router([]),
            (_team("t1", ("gpt-old", "gpt-older")),),
            THRESHOLDS,
        )
        assert [m.info.model_name for m in affected["t1"]] == ["gpt-old", "gpt-older"]

    @pytest.mark.asyncio
    async def test_should_treat_an_ordinary_deployment_as_shared(self):
        deployments: Final = [
            {"model_name": "gpt-old", "litellm_params": {"model": "openai/gpt-old"}, "model_info": {"id": "1"}}
        ]
        affected: Final = await resolve_affected_teams(
            (_info("gpt-old", 5),), _router(deployments), (_team("t1", ("gpt-old",)),), THRESHOLDS
        )
        assert set(affected) == {"t1"}
        assert affected["t1"][0].display_name == "gpt-old"

    @pytest.mark.asyncio
    async def test_should_fall_back_to_the_internal_name_when_a_scoped_deployment_has_no_public_name(self):
        deployments: Final = [
            {
                "model_name": "model_name_t2_abc",
                "litellm_params": {"model": "openai/gpt-old"},
                "model_info": {"id": "1", "team_id": "t2"},
            }
        ]
        affected: Final = await resolve_affected_teams(
            (_info("model_name_t2_abc", -3),), _router(deployments), (_team("t2"),), THRESHOLDS
        )
        assert affected["t2"][0].display_name == "model_name_t2_abc"

    @pytest.mark.asyncio
    async def test_should_deny_a_team_whose_allowlist_is_malformed_and_keep_going(self):
        affected: Final = await resolve_affected_teams(
            (_info("openai/gpt-old", 5),),
            _router([]),
            (_team("t1", ("openai/[*",)), _team("t2", ("openai/gpt-old",))),
            THRESHOLDS,
        )
        assert set(affected) == {"t2"}


class TestRenderModelDeprecationEmail:
    def _notification(self, models: tuple[AffectedModel, ...]) -> TeamNotification:
        return TeamNotification(team_id="t1", team_alias="Data Team", recipients=("a@example.com",), models=models)

    def test_should_render_one_row_per_model_with_escaped_names(self):
        hostile: Final = _info("<img src=x onerror=alert(1)>", 5, provider="op&en")
        subject, html = render_model_deprecation_email(
            self._notification((AffectedModel(info=hostile, display_name=hostile.model_name, milestone=7),)),
            email_logo_url="https://logo",
            email_support_contact="help@example.com",
        )
        assert "<img src=x" not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html
        assert "op&amp;en" in html
        assert "2026-10-01" in html
        assert "<td>5d</td>" in html
        assert "https://logo" in html
        assert "help@example.com" in html
        assert "Data Team" in html
        assert subject == "[LiteLLM] 1 model(s) deprecating for team Data Team"

    def test_should_say_deprecated_in_subject_once_a_model_reaches_day_zero(self):
        models: Final = (
            AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),
            AffectedModel(info=_info("gpt-older", -4), display_name="gpt-older", milestone=0),
        )
        subject, html = render_model_deprecation_email(
            self._notification(models), email_logo_url="https://logo", email_support_contact="help@example.com"
        )
        assert subject == "[LiteLLM] 2 model(s) deprecated for team Data Team"
        assert "deprecated 4d ago" in html
        assert "<td>deprecated</td>" in html
        assert html.index("gpt-old") < html.index("gpt-older")

    def test_should_fall_back_to_team_id_when_alias_missing(self):
        notification: Final = TeamNotification(
            team_id="t1",
            team_alias=None,
            recipients=("a@example.com",),
            models=(AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),),
        )
        subject, _ = render_model_deprecation_email(
            notification, email_logo_url="https://logo", email_support_contact="help@example.com"
        )
        assert subject.endswith("for team t1")

    def test_should_escape_the_team_alias(self):
        notification: Final = TeamNotification(
            team_id="t1",
            team_alias="<b>Ops</b>",
            recipients=("a@example.com",),
            models=(AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),),
        )
        _, html = render_model_deprecation_email(
            notification, email_logo_url="https://logo", email_support_contact="help@example.com"
        )
        assert "<b>Ops</b>" not in html
        assert "&lt;b&gt;Ops&lt;/b&gt;" in html

    def test_should_render_unknown_when_the_provider_is_missing(self):
        models: Final = (AffectedModel(info=_info("gpt-old", 5, provider=None), display_name="gpt-old", milestone=7),)
        _, html = render_model_deprecation_email(
            self._notification(models), email_logo_url="https://logo", email_support_contact="help@example.com"
        )
        assert "<td>unknown</td>" in html

    def test_should_describe_day_zero_as_today_and_still_deprecating(self):
        models: Final = (AffectedModel(info=_info("gpt-old", 0), display_name="gpt-old", milestone=0),)
        subject, html = render_model_deprecation_email(
            self._notification(models), email_logo_url="https://logo", email_support_contact="help@example.com"
        )
        assert subject == "[LiteLLM] 1 model(s) deprecating for team Data Team"
        assert "<td>today</td>" in html
        assert "<td>imminent</td>" in html

    def test_should_append_the_shared_footer_and_escape_the_support_contact(self):
        models: Final = (AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),)
        _, html = render_model_deprecation_email(
            self._notification(models), email_logo_url="https://logo", email_support_contact="<help>@example.com"
        )
        assert "github.com/BerriAI/litellm" in html
        assert "&lt;help&gt;@example.com" in html
        assert "<help>@example.com" not in html

    def test_should_escape_the_logo_url(self):
        models: Final = (AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),)
        _, html = render_model_deprecation_email(
            self._notification(models), email_logo_url='https://logo" onerror="x', email_support_contact="h@x.io"
        )
        assert '" onerror="' not in html
        assert "https://logo&quot; onerror=&quot;x" in html

    def test_should_collapse_newlines_in_the_subject_team_name(self):
        notification: Final = TeamNotification(
            team_id="t1",
            team_alias="Data\r\n Team",
            recipients=("a@example.com",),
            models=(AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),),
        )
        subject, _ = render_model_deprecation_email(
            notification, email_logo_url="https://logo", email_support_contact="h@x.io"
        )
        assert subject == "[LiteLLM] 1 model(s) deprecating for team Data Team"


def _prisma_with_users(rows: Sequence[Mapping[str, object]]) -> SimpleNamespace:
    async def find_many(where=None, take=None, skip=None, order=None):
        wanted: Final = where["user_id"]["in"]
        return [row for row in rows if row["user_id"] in wanted]

    return SimpleNamespace(db=SimpleNamespace(litellm_usertable=SimpleNamespace(find_many=find_many)))


ADMIN_USERS: Final = ({"user_id": "u1", "user_email": "admin@example.com"},)


def _admin_team(team_id: str, alias: str | None = None) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id=team_id, team_alias=alias, members_with_roles=[{"user_id": "u1", "role": "admin"}])


class TestBuildTeamNotifications:
    @pytest.mark.asyncio
    async def test_should_build_one_digest_per_team_with_admin_recipients(self):
        models: Final = (AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),)
        notifications: Final = await build_team_notifications(
            {"t1": models}, (_admin_team("t1", "Data"),), DualCache(), _prisma_with_users(ADMIN_USERS)
        )
        assert notifications == (
            TeamNotification(team_id="t1", team_alias="Data", recipients=("admin@example.com",), models=models),
        )

    @pytest.mark.asyncio
    async def test_should_skip_milestones_already_sent(self):
        cache: Final = DualCache()
        await cache.async_set_cache(key=email_sent_key("t1", "gpt-old", 7), value=1.0)
        sent: Final = AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7)
        fresh: Final = AffectedModel(info=_info("gpt-older", -1), display_name="gpt-older", milestone=0)
        notifications: Final = await build_team_notifications(
            {"t1": (sent, fresh)}, (_admin_team("t1"),), cache, _prisma_with_users(ADMIN_USERS)
        )
        assert notifications[0].models == (fresh,)

    @pytest.mark.asyncio
    async def test_should_drop_team_when_everything_was_already_sent(self):
        cache: Final = DualCache()
        await cache.async_set_cache(key=email_sent_key("t1", "gpt-old", 7), value=1.0)
        models: Final = (AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),)
        notifications: Final = await build_team_notifications(
            {"t1": models}, (_admin_team("t1"),), cache, _prisma_with_users(ADMIN_USERS)
        )
        assert notifications == ()

    @pytest.mark.asyncio
    async def test_should_drop_team_without_admin_emails(self):
        models: Final = (AffectedModel(info=_info("gpt-old", 5), display_name="gpt-old", milestone=7),)
        no_admins: Final = LiteLLM_TeamTable(team_id="t1", members_with_roles=[{"user_id": "u1", "role": "user"}])
        notifications: Final = await build_team_notifications(
            {"t1": models}, (no_admins,), DualCache(), _prisma_with_users(ADMIN_USERS)
        )
        assert notifications == ()

    @pytest.mark.asyncio
    async def test_should_key_the_sent_marker_by_team_model_and_milestone(self):
        assert email_sent_key("t1", "gpt-old", 7) == "model_deprecation_email:t1:gpt-old:7"
        assert email_sent_key("t1", "gpt-old", 7) != email_sent_key("t2", "gpt-old", 7)
        assert email_sent_key("t1", "gpt-old", 7) != email_sent_key("t1", "gpt-old", 0)


DEAD_DEPLOYMENT: Final = {
    "model_name": "dead-alias",
    "litellm_params": {"model": "openai/dead-model"},
    "model_info": {"id": "1", "deprecation_date": "2020-01-01", "litellm_provider": "openai"},
}
FRESH_DEPLOYMENT: Final = {
    "model_name": "fresh",
    "litellm_params": {"model": "openai/fresh-model-with-no-cost-map-entry"},
    "model_info": {"id": "x"},
}


class _Deliverer:
    def __init__(self, fail_for: Sequence[str] = ()):
        self.fail_for = fail_for
        self.sent = []

    async def __call__(self, recipients: Sequence[str], subject: str, html_body: str) -> None:
        if any(recipient in self.fail_for for recipient in recipients):
            raise ConnectionError("smtp down")
        self.sent.append((tuple(recipients), subject))


class _Lock:
    def __init__(self, result: bool | None):
        self.result = result
        self.calls = []

    async def acquire_lock(self, cronjob_id: str, ttl: int | None = None, allow_reentrant: bool = True) -> bool | None:
        self.calls.append({"cronjob_id": cronjob_id, "ttl": ttl, "allow_reentrant": allow_reentrant})
        return self.result


def _team_row(team_id: str, alias: str = "Team") -> Mapping[str, object]:
    return {
        "team_id": team_id,
        "team_alias": alias,
        "models": [],
        "members_with_roles": [{"user_id": f"admin-{team_id}", "role": "admin"}],
    }


def _prisma(team_rows: Sequence[Mapping[str, object]], user_rows: Sequence[Mapping[str, object]]) -> SimpleNamespace:
    async def find_many_teams(where=None, take=None, skip=None, order=None):
        return list(team_rows)

    async def find_many_users(where=None, take=None, skip=None, order=None):
        wanted: Final = where["user_id"]["in"]
        return [row for row in user_rows if row["user_id"] in wanted]

    return SimpleNamespace(
        db=SimpleNamespace(
            litellm_teamtable=SimpleNamespace(find_many=find_many_teams),
            litellm_usertable=SimpleNamespace(find_many=find_many_users),
        )
    )


TWO_TEAMS: Final = (_team_row("t1", "Alpha"), _team_row("t2", "Beta"))
TWO_ADMINS: Final = (
    {"user_id": "admin-t1", "user_email": "a@x.io"},
    {"user_id": "admin-t2", "user_email": "b@x.io"},
)


def _context(router, prisma, deliver, cache=None, pod_lock_manager=None) -> DeprecationEmailContext:
    return DeprecationEmailContext(
        llm_router=router,
        prisma_client=prisma,
        cache=cache or DualCache(),
        alerting_args=SlackAlertingArgs(),
        pod_lock_manager=pod_lock_manager,
        deliver=deliver,
    )


class TestSendModelDeprecationEmails:
    @pytest.mark.asyncio
    async def test_should_email_each_affected_team_once_and_stamp_the_milestones(self):
        cache: Final = DualCache()
        deliverer: Final = _Deliverer()
        ctx: Final = _context(_router([DEAD_DEPLOYMENT]), _prisma(TWO_TEAMS, TWO_ADMINS), deliverer, cache)

        assert await send_model_deprecation_emails(ctx) == 2
        assert sorted(recipients for recipients, _ in deliverer.sent) == [("a@x.io",), ("b@x.io",)]
        assert all("deprecated" in subject for _, subject in deliverer.sent)
        assert await cache.async_get_cache(key=email_sent_key("t1", "dead-alias", 0)) is not None
        assert await cache.async_get_cache(key=SlackAlertingCacheKeys.deprecation_email_pass_key.value) is not None

        assert await send_model_deprecation_emails(ctx) == 0
        assert len(deliverer.sent) == 2

    @pytest.mark.asyncio
    async def test_should_stamp_the_pass_and_touch_nothing_else_when_no_model_deprecates(self):
        cache: Final = DualCache()
        deliverer: Final = _Deliverer()
        lock: Final = _Lock(True)

        async def explode(where=None, take=None, skip=None, order=None):
            raise AssertionError("teams must not be loaded when nothing deprecates")

        prisma: Final = _prisma((), ())
        prisma.db.litellm_teamtable.find_many = explode

        assert (
            await send_model_deprecation_emails(_context(_router([FRESH_DEPLOYMENT]), prisma, deliverer, cache, lock))
            == 0
        )
        assert deliverer.sent == []
        assert lock.calls == []
        assert await cache.async_get_cache(key=SlackAlertingCacheKeys.deprecation_email_pass_key.value) is not None

    @pytest.mark.asyncio
    async def test_should_not_claim_the_lock_when_every_milestone_was_already_sent(self):
        cache: Final = DualCache()
        await cache.async_set_cache(key=email_sent_key("t1", "dead-alias", 0), value=1.0)
        lock: Final = _Lock(True)
        deliverer: Final = _Deliverer()
        ctx: Final = _context(_router([DEAD_DEPLOYMENT]), _prisma(TWO_TEAMS[:1], TWO_ADMINS), deliverer, cache, lock)

        assert await send_model_deprecation_emails(ctx) == 0
        assert deliverer.sent == []
        assert lock.calls == []

    @pytest.mark.parametrize(
        ("lock_result", "expected_sent"),
        [(True, 1), (None, 1), (False, 0)],
        ids=["lock won", "no redis lock", "another pod holds the lock"],
    )
    @pytest.mark.asyncio
    async def test_should_send_only_from_the_pod_holding_the_daily_lock(self, lock_result, expected_sent):
        lock: Final = _Lock(lock_result)
        deliverer: Final = _Deliverer()
        ctx: Final = _context(_router([DEAD_DEPLOYMENT]), _prisma(TWO_TEAMS[:1], TWO_ADMINS), deliverer, None, lock)

        assert await send_model_deprecation_emails(ctx) == expected_sent
        assert len(deliverer.sent) == expected_sent
        assert lock.calls == [
            {
                "cronjob_id": EMAIL_MODEL_DEPRECATION_LOCK_ID,
                "ttl": DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
                "allow_reentrant": False,
            }
        ]

    @pytest.mark.asyncio
    async def test_should_keep_going_and_leave_keys_unstamped_when_one_team_fails(self):
        cache: Final = DualCache()
        deliverer: Final = _Deliverer(fail_for=("a@x.io",))
        ctx: Final = _context(_router([DEAD_DEPLOYMENT]), _prisma(TWO_TEAMS, TWO_ADMINS), deliverer, cache)

        assert await send_model_deprecation_emails(ctx) == 1
        assert [recipients for recipients, _ in deliverer.sent] == [("b@x.io",)]
        assert await cache.async_get_cache(key=email_sent_key("t1", "dead-alias", 0)) is None
        assert await cache.async_get_cache(key=email_sent_key("t2", "dead-alias", 0)) is not None


class TestMakeEmailDeliverer:
    @pytest.mark.asyncio
    async def test_should_use_the_configured_email_logger_when_present(self):
        calls: Final = []

        async def send_email(from_email, to_email, subject, html_body):
            calls.append((from_email, tuple(to_email), subject, html_body))

        logger: Final = SimpleNamespace(DEFAULT_LITELLM_EMAIL="noreply@litellm.ai", send_email=send_email)
        deliver: Final = make_email_deliverer(logger)

        await deliver(("a@x.io", "b@x.io"), "subj", "<p>hi</p>")
        assert calls == [("noreply@litellm.ai", ("a@x.io", "b@x.io"), "subj", "<p>hi</p>")]

    @pytest.mark.asyncio
    async def test_should_fall_back_to_smtp_per_recipient_without_a_logger(self):
        calls: Final = []

        async def smtp_send(*, receiver_email, subject, html):
            calls.append((receiver_email, subject, html))

        deliver: Final = make_email_deliverer(None, smtp_send=smtp_send)

        await deliver(("a@x.io", "b@x.io"), "subj", "<p>hi</p>")
        assert calls == [("a@x.io", "subj", "<p>hi</p>"), ("b@x.io", "subj", "<p>hi</p>")]

    @pytest.mark.asyncio
    async def test_should_raise_when_smtp_is_not_configured(self, monkeypatch):
        monkeypatch.delenv("SMTP_SENDER_EMAIL", raising=False)
        deliver: Final = make_email_deliverer(None)

        with pytest.raises(ValueError, match="SMTP_SENDER_EMAIL"):
            await deliver(("a@x.io",), "subj", "<p>hi</p>")
