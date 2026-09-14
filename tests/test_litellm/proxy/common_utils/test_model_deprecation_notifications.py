"""Tests for the model deprecation email notifications module."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from types import SimpleNamespace
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.caching.caching import DualCache
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.common_utils.model_deprecation_notifications import (
    AffectedModel,
    TeamNotification,
    build_team_notifications,
    email_sent_key,
    render_model_deprecation_email,
    resolve_affected_teams,
    select_milestone,
)
from litellm.types.proxy.model_deprecation import ModelDeprecationInfo


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
