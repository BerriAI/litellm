"""Tests for the model deprecation helper module.

These tests focus on the helper itself — not on the proxy endpoint or
Slack integration — so they can run without the full proxy stack.
"""

import os
import sys
from datetime import date
from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.proxy.common_utils.model_deprecation import (
    _classify,
    _parse_deprecation_date,
    collect_model_deprecations,
    format_deprecation_alert_message,
)


def _make_router(deployments):
    router = MagicMock()
    router.get_model_list.return_value = deployments
    return router


class TestParseDeprecationDate:
    def test_should_parse_iso_string(self):
        assert _parse_deprecation_date("2026-12-31") == date(2026, 12, 31)

    def test_should_pass_through_date_object(self):
        d = date(2026, 1, 1)
        assert _parse_deprecation_date(d) == d

    def test_should_return_none_for_documentation_sentinel(self):
        # The JSON map ships a sentinel string under the "sample_spec" key.
        assert (
            _parse_deprecation_date(
                "date when the model becomes deprecated in the format YYYY-MM-DD"
            )
            is None
        )

    def test_should_return_none_for_none(self):
        assert _parse_deprecation_date(None) is None

    def test_should_return_none_for_unsupported_type(self):
        assert _parse_deprecation_date(12345) is None


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

        snapshot = collect_model_deprecations(
            llm_router=router, warn_within_days=30, today=today
        )

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

        snapshot = collect_model_deprecations(
            llm_router=router, warn_within_days=30, today=today
        )

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

        snapshot = collect_model_deprecations(
            llm_router=router, warn_within_days=30, today=today
        )

        assert len(snapshot.imminent) == 1

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

        snapshot = collect_model_deprecations(
            llm_router=router, warn_within_days=30, today=today
        )

        assert len(snapshot.imminent) == 1
        assert snapshot.imminent[0].litellm_model == "base-thing"


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

        snapshot = collect_model_deprecations(
            llm_router=router, warn_within_days=30, today=today
        )
        message = format_deprecation_alert_message(snapshot)

        assert message is not None
        assert "Already deprecated" in message
        assert "Deprecating within 30 days" in message
        assert "`dead`" in message
        assert "`soon`" in message
        # Upcoming models must NOT be in the alert (avoid alert fatigue).
        assert "`later`" not in message
