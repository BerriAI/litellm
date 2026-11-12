"""Tests for the model deprecation helper module.

These tests focus on the helper itself — not on the proxy endpoint or
Slack integration — so they can run without the full proxy stack.
"""

import logging
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm import Router
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.model_cost_table import AliasedCostEntry
from litellm.proxy.common_utils.model_deprecation import (
    _classify,
    _parse_deprecation_date,
    collect_model_deprecations,
    format_deprecation_alert_message,
)


def _make_router(deployments):
    router = MagicMock()
    router.registry.return_value.deployments.return_value = deployments
    return router


class TestParseDeprecationDate:
    def test_should_parse_iso_string(self):
        assert _parse_deprecation_date("2026-12-31") == date(2026, 12, 31)

    def test_should_pass_through_date_object(self):
        d = date(2026, 1, 1)
        assert _parse_deprecation_date(d) == d

    def test_should_return_none_for_documentation_sentinel(self):
        # The JSON map ships a sentinel string under the "sample_spec" key.
        assert _parse_deprecation_date("date when the model becomes deprecated in the format YYYY-MM-DD") is None

    def test_should_return_none_for_none(self):
        assert _parse_deprecation_date(None) is None

    def test_should_return_none_for_unsupported_type(self):
        assert _parse_deprecation_date(12345) is None

    def test_should_narrow_datetime_to_date(self):
        assert _parse_deprecation_date(datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)) == date(2026, 12, 31)


class TestClassify:
    def test_should_classify_past_dates_as_deprecated(self):
        assert _classify(-1, warn_within_days=30) == "deprecated"
        assert _classify(-365, warn_within_days=30) == "deprecated"

    def test_should_classify_inside_window_as_imminent(self):
        assert _classify(0, warn_within_days=30) == "imminent"
        assert _classify(15, warn_within_days=30) == "imminent"
        assert _classify(30, warn_within_days=30) == "imminent"

    def test_should_classify_outside_window_as_upcoming(self):
        assert _classify(31, warn_within_days=30) == "upcoming"
        assert _classify(365, warn_within_days=30) == "upcoming"


class TestCollectModelDeprecations:
    def test_should_return_empty_response_when_router_is_none(self):
        snapshot = collect_model_deprecations(llm_router=None)
        assert snapshot.deprecated == []
        assert snapshot.imminent == []
        assert snapshot.upcoming == []

    def test_should_skip_models_without_deprecation_metadata(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = _make_router(
            [
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o"},
                    "model_info": {"id": "abc"},
                }
            ]
        )
        snapshot = collect_model_deprecations(llm_router=router)
        assert snapshot.deprecated == []
        assert snapshot.imminent == []
        assert snapshot.upcoming == []

    def test_should_classify_into_three_buckets(self, monkeypatch):
        today = date(2026, 6, 1)
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                "deprecated-model": {
                    "deprecation_date": "2026-01-01",
                    "litellm_provider": "openai",
                },
                "imminent-model": {
                    "deprecation_date": "2026-06-15",
                    "litellm_provider": "openai",
                },
                "upcoming-model": {
                    "deprecation_date": "2027-01-01",
                    "litellm_provider": "openai",
                },
            },
        )
        router = _make_router(
            [
                {
                    "model_name": "deprecated-alias",
                    "litellm_params": {"model": "openai/deprecated-model"},
                    "model_info": {"id": "1"},
                },
                {
                    "model_name": "imminent-alias",
                    "litellm_params": {"model": "imminent-model"},
                    "model_info": {"id": "2"},
                },
                {
                    "model_name": "upcoming-alias",
                    "litellm_params": {"model": "openai/upcoming-model"},
                    "model_info": {"id": "3"},
                },
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=today)

        assert [m.model_name for m in snapshot.deprecated] == ["deprecated-alias"]
        assert [m.model_name for m in snapshot.imminent] == ["imminent-alias"]
        assert [m.model_name for m in snapshot.upcoming] == ["upcoming-alias"]

        assert snapshot.deprecated[0].days_until_deprecation < 0
        assert snapshot.imminent[0].days_until_deprecation == 14
        assert snapshot.upcoming[0].days_until_deprecation > 30

    def test_should_prefer_explicit_deployment_override(self, monkeypatch):
        today = date(2026, 6, 1)
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {"some-model": {"deprecation_date": "2030-01-01"}},
        )
        router = _make_router(
            [
                {
                    "model_name": "my-alias",
                    "litellm_params": {"model": "some-model"},
                    "model_info": {
                        "id": "x",
                        "deprecation_date": "2026-06-10",
                    },
                }
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=today)

        assert len(snapshot.imminent) == 1
        assert snapshot.imminent[0].deprecation_date == date(2026, 6, 10)

    def test_should_dedupe_duplicate_deployments_in_same_group(self, monkeypatch):
        today = date(2026, 6, 1)
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {"shared-model": {"deprecation_date": "2026-06-10"}},
        )
        router = _make_router(
            [
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "shared-model"},
                    "model_info": {"id": "1"},
                },
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "shared-model"},
                    "model_info": {"id": "2"},
                },
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=today)

        assert len(snapshot.imminent) == 1

    def test_should_resolve_via_unprefixed_model_name(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {"gpt-4o": {"deprecation_date": "2026-06-10"}},
        )
        router = _make_router(
            [
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "openai/gpt-4o"},
                    "model_info": {"id": "1"},
                }
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))

        assert [m.litellm_model for m in snapshot.imminent] == ["gpt-4o"]

    def test_should_keep_both_dates_when_group_has_conflicting_dates(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {"shared-model": {"deprecation_date": "2026-06-10"}},
        )
        router = _make_router(
            [
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "shared-model"},
                    "model_info": {"id": "1"},
                },
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "shared-model"},
                    "model_info": {"id": "2", "deprecation_date": "2027-01-01"},
                },
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))

        assert len(snapshot.imminent) == 1
        assert len(snapshot.upcoming) == 1

    def test_should_resolve_via_base_model(self, monkeypatch):
        today = date(2026, 6, 1)
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {"base-thing": {"deprecation_date": "2026-06-10"}},
        )
        router = _make_router(
            [
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "azure/some-deployment-name"},
                    "model_info": {"id": "1", "base_model": "base-thing"},
                }
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=today)

        assert len(snapshot.imminent) == 1
        assert snapshot.imminent[0].litellm_model == "base-thing"


class TestProviderScopedResolution:
    CATALOG = {
        "gpt-4o": {"deprecation_date": "2027-01-14", "litellm_provider": "openai"},
        "azure/gpt-4o": {"deprecation_date": "2027-04-14", "litellm_provider": "azure"},
        "github_copilot/gpt-4o": {"litellm_provider": "github_copilot"},
        "azure_ai/gpt-4.1": AliasedCostEntry(
            {"deprecation_date": "2027-05-01", "litellm_provider": "azure_ai", "mode": "chat"}, ()
        ),
        "custom-base": {"deprecation_date": "2026-12-01"},
        "gpt-4.1": {"deprecation_date": "2027-01-14", "litellm_provider": "openai"},
        "azure/gpt-4.1": {"deprecation_date": "2027-04-14", "litellm_provider": "azure"},
        "gpt-4o-2024-05-13": {"deprecation_date": "2026-10-23", "litellm_provider": "openai"},
        "azure/gpt-4o-2024-05-13": {"deprecation_date": "2026-12-09", "litellm_provider": "azure"},
        "azure/aliased-base": AliasedCostEntry(
            {"deprecation_date": "2026-11-20"}, ("azure/aliased-base", "azure/aliased-base-v1")
        ),
        "aliased-base": {"deprecation_date": "2027-02-01"},
    }
    COST_MAP = {
        **CATALOG,
        "azure_ai/gpt-4o": AliasedCostEntry(
            {"deprecation_date": "2027-01-14", "litellm_provider": "azure", "mode": "chat"}, ()
        ),
        "registered-base": AliasedCostEntry({"deprecation_date": "2026-12-15"}, (), frozenset({"deprecation_date"})),
    }

    @pytest.mark.parametrize(
        ("litellm_params", "model_info", "expected"),
        [
            ({"model": "gpt-4o"}, {}, ("gpt-4o", date(2027, 1, 14), "openai")),
            ({"model": "openai/gpt-4o"}, {}, ("gpt-4o", date(2027, 1, 14), "openai")),
            (
                {"model": "gpt-4o", "custom_llm_provider": "azure"},
                {},
                ("azure/gpt-4o", date(2027, 4, 14), "azure"),
            ),
            ({"model": "azure/gpt-4o"}, {}, ("azure/gpt-4o", date(2027, 4, 14), "azure")),
            (
                {"model": "azure/prod-chat"},
                {"base_model": "gpt-4o"},
                ("azure/gpt-4o", date(2027, 4, 14), "azure"),
            ),
            (
                {"model": "gpt-4o", "custom_llm_provider": "github_copilot"},
                {},
                ("gpt-4o", date(2027, 1, 14), "openai"),
            ),
            ({"model": "azure_ai/gpt-4o"}, {}, ("azure/gpt-4o", date(2027, 4, 14), "azure")),
            (
                {"model": "azure_ai/prod-chat"},
                {"base_model": "gpt-4o"},
                ("azure/gpt-4o", date(2027, 4, 14), "azure"),
            ),
            (
                {"model": "azure_ai/gpt-4o"},
                {"base_model": "azure_ai/gpt-4o"},
                ("azure/gpt-4o", date(2027, 4, 14), "azure"),
            ),
            (
                {"model": "azure_ai/prod-chat"},
                {"base_model": "azure_ai/gpt-4o"},
                ("azure/gpt-4o", date(2027, 4, 14), "azure"),
            ),
            ({"model": "azure_ai/gpt-4.1"}, {}, ("azure_ai/gpt-4.1", date(2027, 5, 1), "azure_ai")),
            ({"model": "azure/gpt-4o"}, {"base_model": "custom-base"}, ("custom-base", date(2026, 12, 1), None)),
            (
                {"model": "azure/gpt-4o"},
                {"base_model": "registered-base"},
                ("registered-base", date(2026, 12, 15), None),
            ),
            (
                {"model": "azure/prod-chat"},
                {"base_model": "aliased-base"},
                ("azure/aliased-base", date(2026, 11, 20), None),
            ),
            (
                {"model": "openai/gpt-4o-2024-05-13"},
                {"base_model": "gpt-4o"},
                ("gpt-4o-2024-05-13", date(2026, 10, 23), "openai"),
            ),
            (
                {"model": "azure/gpt-4o-2024-05-13"},
                {"base_model": "gpt-4o"},
                ("azure/gpt-4o-2024-05-13", date(2026, 12, 9), "azure"),
            ),
        ],
        ids=[
            "bare",
            "openai-prefix",
            "declared-azure",
            "azure-prefix",
            "azure-base-model",
            "undated-own-row",
            "azure-ai-reads-azure",
            "azure-ai-base-model-reads-azure",
            "azure-ai-copied-base-model-yields-to-azure",
            "azure-ai-prefixed-base-model-on-custom-deployment-reads-azure",
            "azure-ai-own-row-wins",
            "custom-base-model-row-wins",
            "registered-base-model-row-wins",
            "aliased-base-model-row-wins",
            "openai-snapshot-shuts-down-before-its-base-model",
            "azure-snapshot-shuts-down-before-its-base-model",
        ],
    )
    def test_should_prefer_the_deployments_own_provider_date(
        self,
        monkeypatch: pytest.MonkeyPatch,
        litellm_params: Mapping[str, str],
        model_info: Mapping[str, str],
        expected: tuple[str, date, str | None],
    ) -> None:
        monkeypatch.setattr(litellm, "model_cost", self.COST_MAP)
        router: Final = _make_router(
            [{"model_name": "alias", "litellm_params": litellm_params, "model_info": {"id": "1", **model_info}}]
        )

        snapshot: Final = collect_model_deprecations(
            llm_router=router, warn_within_days=30, today=date(2026, 6, 1), loaded_catalog=lambda: self.CATALOG
        )

        assert [(m.litellm_model, m.deprecation_date, m.litellm_provider) for m in snapshot.upcoming] == [expected]


class TestFormatDeprecationAlertMessage:
    def test_should_return_none_when_nothing_to_alert(self):
        snapshot = collect_model_deprecations(llm_router=None)
        assert format_deprecation_alert_message(snapshot) is None

    def test_should_render_imminent_and_deprecated_sections(self, monkeypatch):
        today = date(2026, 6, 1)
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                "dead-model": {
                    "deprecation_date": "2026-01-01",
                    "litellm_provider": "openai",
                },
                "soon-model": {
                    "deprecation_date": "2026-06-15",
                    "litellm_provider": "anthropic",
                },
                "later-model": {
                    "deprecation_date": "2027-01-01",
                    "litellm_provider": "anthropic",
                },
            },
        )
        router = _make_router(
            [
                {
                    "model_name": "dead",
                    "litellm_params": {"model": "dead-model"},
                    "model_info": {"id": "1"},
                },
                {
                    "model_name": "soon",
                    "litellm_params": {"model": "soon-model"},
                    "model_info": {"id": "2"},
                },
                {
                    "model_name": "later",
                    "litellm_params": {"model": "later-model"},
                    "model_info": {"id": "3"},
                },
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=today)
        message = format_deprecation_alert_message(snapshot)

        assert message is not None
        assert "Already deprecated" in message
        assert "Deprecating within 30 days" in message
        assert "`dead`" in message
        assert "`soon`" in message
        # Upcoming models must NOT be in the alert (avoid alert fatigue).
        assert "`later`" not in message

    def test_should_neutralize_slack_markup_from_model_metadata(self):
        today = date(2026, 6, 1)
        router = _make_router(
            [
                {
                    "model_name": "<!channel> pwned",
                    "litellm_params": {"model": "openai/whatever"},
                    "model_info": {
                        "id": "1",
                        "deprecation_date": "2026-06-10",
                        "litellm_provider": "<https://evil.example|openai> & co",
                    },
                }
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=today)
        message = format_deprecation_alert_message(snapshot)

        assert message is not None
        assert "<!channel>" not in message
        assert "<https://evil.example|openai>" not in message
        assert "&lt;!channel&gt; pwned" in message
        assert "&lt;https://evil.example|openai&gt; &amp; co" in message


class TestSuccessorModel:
    def test_unmatched_successor_logs_warning_by_default(self, monkeypatch, caplog):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = Router(
            model_list=[
                {
                    "model_name": "old",
                    "litellm_params": {"model": "openai/old-model"},
                    "model_info": {
                        "id": "1",
                        "deprecation_date": "2026-06-10",
                        "successor_model": "missing-model",
                    },
                }
            ]
        )

        with caplog.at_level(logging.DEBUG, logger=verbose_logger.name):
            collect_model_deprecations(llm_router=router, today=date(2026, 6, 1))

        successor_records: Final = tuple(
            record for record in caplog.records if "names successor missing-model" in record.getMessage()
        )
        assert len(successor_records) == 1
        assert successor_records[0].levelno == logging.WARNING

    def test_empty_team_public_model_name_is_normalized_to_none(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = Router(
            model_list=[
                {
                    "model_name": "team-model",
                    "litellm_params": {"model": "openai/team-model"},
                    "model_info": {
                        "id": "1",
                        "team_id": "team-1",
                        "team_public_model_name": "",
                        "deprecation_date": "2026-06-10",
                    },
                }
            ]
        )

        snapshot: Final = collect_model_deprecations(llm_router=router, today=date(2026, 6, 1))

        assert snapshot.imminent[0].team_public_model_name is None

    def test_should_surface_the_successor_set_on_the_deployment(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = Router(
            model_list=[
                {
                    "model_name": "old",
                    "litellm_params": {"model": "openai/old-model"},
                    "model_info": {"id": "1", "deprecation_date": "2026-06-10", "successor_model": "new"},
                },
                {"model_name": "new", "litellm_params": {"model": "openai/new-model"}, "model_info": {"id": "2"}},
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))

        assert snapshot.imminent[0].successor_model == "new"

    def test_should_name_the_deployment_successor_when_the_date_comes_from_the_cost_map(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {"old-model": {"deprecation_date": "2026-06-10"}})
        router = Router(
            model_list=[
                {
                    "model_name": "old",
                    "litellm_params": {"model": "openai/old-model"},
                    "model_info": {"id": "1", "successor_model": "my-pick"},
                },
                {"model_name": "my-pick", "litellm_params": {"model": "openai/pick"}, "model_info": {"id": "2"}},
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))

        assert snapshot.imminent[0].successor_model == "my-pick"

    def test_should_ignore_blank_or_non_string_successors(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = _make_router(
            [
                {
                    "model_name": "blank",
                    "litellm_params": {"model": "openai/a"},
                    "model_info": {"id": "1", "deprecation_date": "2026-06-10", "successor_model": "   "},
                },
                {
                    "model_name": "number",
                    "litellm_params": {"model": "openai/b"},
                    "model_info": {"id": "2", "deprecation_date": "2026-06-10", "successor_model": 7},
                },
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))

        assert [m.successor_model for m in snapshot.imminent] == [None, None]

    def test_should_report_successors_as_set_and_warn_when_no_model_in_the_team_scope_matches(
        self, monkeypatch, caplog
    ):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = Router(
            model_list=[
                {
                    "model_name": "model_name_t1_old",
                    "litellm_params": {"model": "openai/team-old"},
                    "model_info": {
                        "id": "1",
                        "team_id": "t1",
                        "team_public_model_name": "team-old",
                        "deprecation_date": "2026-06-10",
                        "successor_model": "team-new",
                    },
                },
                {
                    "model_name": "model_name_t1_new",
                    "litellm_params": {"model": "openai/team-new"},
                    "model_info": {"id": "2", "team_id": "t1", "team_public_model_name": "team-new"},
                },
                {
                    "model_name": "global-old",
                    "litellm_params": {"model": "openai/global-old"},
                    "model_info": {"id": "3", "deprecation_date": "2026-06-10", "successor_model": "team-new"},
                },
                {
                    "model_name": "typo-old",
                    "litellm_params": {"model": "openai/typo-old"},
                    "model_info": {"id": "4", "deprecation_date": "2026-06-10", "successor_model": "gpt-nwe"},
                },
                {
                    "model_name": "wildcard-old",
                    "litellm_params": {"model": "openai/wildcard-old"},
                    "model_info": {"id": "5", "deprecation_date": "2026-06-10", "successor_model": "anthropic/x"},
                },
                {"model_name": "anthropic/*", "litellm_params": {"model": "anthropic/*"}, "model_info": {"id": "6"}},
            ]
        )

        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))

        assert {m.model_name: m.successor_model for m in snapshot.imminent} == {
            "model_name_t1_old": "team-new",
            "global-old": "team-new",
            "typo-old": "gpt-nwe",
            "wildcard-old": "anthropic/x",
        }
        team_deprecation: Final = next(info for info in snapshot.imminent if info.model_name == "model_name_t1_old")
        assert team_deprecation.team_public_model_name == "team-old"
        warnings = [record.getMessage() for record in caplog.records if "names successor" in record.getMessage()]
        assert len(warnings) == 2
        assert any("global-old names successor team-new" in warning for warning in warnings)
        assert any("typo-old names successor gpt-nwe" in warning for warning in warnings)

    def test_should_keep_the_first_deployment_and_carry_the_group_successor_when_deduping(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = Router(
            model_list=[
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "openai/a"},
                    "model_info": {"id": "1", "deprecation_date": "2026-06-10"},
                },
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "openai/b"},
                    "model_info": {"id": "2", "deprecation_date": "2026-06-10", "successor_model": "alias-v2"},
                },
                {
                    "model_name": "alias",
                    "litellm_params": {"model": "openai/c"},
                    "model_info": {"id": "3", "deprecation_date": "2026-06-10", "successor_model": "alias-v3"},
                },
                {"model_name": "alias-v2", "litellm_params": {"model": "openai/v2"}, "model_info": {"id": "4"}},
                {"model_name": "alias-v3", "litellm_params": {"model": "openai/v3"}, "model_info": {"id": "5"}},
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))

        assert [(m.litellm_model, m.successor_model) for m in snapshot.imminent] == [("openai/a", "alias-v2")]

    def test_should_name_the_successor_in_the_slack_entry_and_escape_it(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {})
        router = Router(
            model_list=[
                {
                    "model_name": "old",
                    "litellm_params": {"model": "openai/old-model"},
                    "model_info": {
                        "id": "1",
                        "deprecation_date": "2026-06-10",
                        "successor_model": "<https://evil.example|new>",
                    },
                },
                {
                    "model_name": "<https://evil.example|new>",
                    "litellm_params": {"model": "openai/new-model"},
                    "model_info": {"id": "2"},
                },
                {
                    "model_name": "orphan",
                    "litellm_params": {"model": "openai/orphan-model"},
                    "model_info": {"id": "3", "deprecation_date": "2026-06-10"},
                },
            ]
        )

        snapshot = collect_model_deprecations(llm_router=router, warn_within_days=30, today=date(2026, 6, 1))
        message = format_deprecation_alert_message(snapshot)

        assert message is not None
        assert "migrate to `&lt;https://evil.example|new&gt;`" in message
        assert "<https://evil.example|new>" not in message
        assert message.count("migrate to") == 1


class TestRecommendationSources:
    def test_should_return_the_cost_map_recommendation_and_provider_notice(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                "old-model": {
                    "deprecation_date": "2026-06-10",
                    "successor_model": "new-model",
                    "deprecation_source": "https://provider.example/model-retirements",
                }
            },
        )
        router = _make_router(
            [{"model_name": "support", "litellm_params": {"model": "openai/old-model"}, "model_info": {}}]
        )
        info = collect_model_deprecations(llm_router=router, today=date(2026, 6, 1)).imminent[0]
        assert info.successor_model == "new-model"
        assert info.successor_model_source == "model_cost_map"
        assert info.deprecation_source == "https://provider.example/model-retirements"

    def test_should_keep_configured_recommendation_provenance_when_merging_deployments(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {"old-model": {"deprecation_date": "2026-06-10", "successor_model": "catalog-pick"}},
        )
        router = _make_router(
            [
                {"model_name": "support", "litellm_params": {"model": "openai/old-model"}, "model_info": {}},
                {
                    "model_name": "support",
                    "litellm_params": {"model": "openai/old-model"},
                    "model_info": {"successor_model": "configured-pick"},
                },
            ]
        )
        info = collect_model_deprecations(llm_router=router, today=date(2026, 6, 1)).imminent[0]
        assert info.successor_model == "configured-pick"
        assert info.successor_model_source == "deployment_config"
        assert info.deprecation_source is None

    def test_should_carry_the_notice_from_the_same_deployment_as_the_selected_recommendation(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                "a": {"deprecation_date": "2026-06-10", "deprecation_source": "https://provider.example/a"},
                "b": {
                    "deprecation_date": "2026-06-10",
                    "successor_model": "b-next",
                    "deprecation_source": "https://provider.example/b",
                },
            },
        )
        router = _make_router(
            [
                {"model_name": "support", "litellm_params": {"model": "openai/a"}, "model_info": {}},
                {"model_name": "support", "litellm_params": {"model": "openai/b"}, "model_info": {}},
            ]
        )
        info = collect_model_deprecations(llm_router=router, today=date(2026, 6, 1)).imminent[0]
        assert info.successor_model == "b-next"
        assert info.successor_model_source == "model_cost_map"
        assert info.deprecation_source == "https://provider.example/b"
