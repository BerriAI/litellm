"""
Tests for team-scoped New Relic metrics callback support.

Verifies that NewRelicMetricsLogger is instantiated with per-team credentials
(newrelic_api_key, newrelic_region) with no environment fallback, and that
NewRelicHandler correctly resolves and caches per-team loggers.
"""

import copy
from unittest.mock import patch

import pytest

from litellm.integrations.newrelic.newrelic_metrics import NewRelicMetricsLogger
from litellm.integrations.newrelic.newrelic_team_handler import NewRelicHandler
from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
    TRUSTED_CALLBACK_VARS_FIELD,
)
from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import (
    DynamicLoggingCache,
)
from litellm.types.integrations.newrelic import NEWRELIC_METRIC_ENDPOINT_BY_REGION
from litellm.types.utils import StandardCallbackDynamicParams

US_ENDPOINT = NEWRELIC_METRIC_ENDPOINT_BY_REGION["us"]
EU_ENDPOINT = NEWRELIC_METRIC_ENDPOINT_BY_REGION["eu"]


class TestNewRelicMetricsLoggerCredentialKwargs:
    """The logger takes credentials by injection only; env vars never leak in."""

    def test_init_with_explicit_credentials(self):
        with patch("asyncio.create_task"):
            logger = NewRelicMetricsLogger(newrelic_api_key="team_key", newrelic_region="eu")

        assert logger.newrelic_api_key == "team_key"
        assert logger.metric_api_url == EU_ENDPOINT

    def test_init_defaults_to_us_region(self):
        with patch("asyncio.create_task"):
            logger = NewRelicMetricsLogger(newrelic_api_key="team_key")

        assert logger.metric_api_url == US_ENDPOINT

    def test_unknown_region_falls_back_to_us(self):
        with patch("asyncio.create_task"):
            logger = NewRelicMetricsLogger(newrelic_api_key="team_key", newrelic_region="mars")

        assert logger.metric_api_url == US_ENDPOINT

    def test_region_is_case_insensitive(self):
        with patch("asyncio.create_task"):
            logger = NewRelicMetricsLogger(newrelic_api_key="team_key", newrelic_region="EU")

        assert logger.metric_api_url == EU_ENDPOINT

    def test_init_raises_without_api_key(self):
        with pytest.raises(ValueError, match="newrelic_api_key"):
            with patch("asyncio.create_task"):
                NewRelicMetricsLogger(newrelic_api_key="")

    def test_init_never_falls_back_to_env_license_key(self, monkeypatch):
        """A missing team key must fail, never silently reuse the operator's key."""
        monkeypatch.setenv("NEW_RELIC_LICENSE_KEY", "operator-license-key")

        with pytest.raises(ValueError, match="newrelic_api_key"):
            with patch("asyncio.create_task"):
                NewRelicMetricsLogger(newrelic_api_key="")


class TestNewRelicHandler:
    """The handler resolves the correct logger per team."""

    def test_creates_team_logger_with_dynamic_credentials(self):
        cache = DynamicLoggingCache()
        params = StandardCallbackDynamicParams(newrelic_api_key="team_a_key", newrelic_region="eu")

        with patch("asyncio.create_task"):
            result = NewRelicHandler.get_newrelic_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result.newrelic_api_key == "team_a_key"
        assert result.metric_api_url == EU_ENDPOINT

    def test_caches_team_logger(self):
        cache = DynamicLoggingCache()
        params = StandardCallbackDynamicParams(newrelic_api_key="team_b_key")

        with patch("asyncio.create_task"):
            result1 = NewRelicHandler.get_newrelic_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )
            result2 = NewRelicHandler.get_newrelic_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result1 is result2

    def test_different_teams_get_different_loggers(self):
        cache = DynamicLoggingCache()
        params_a = StandardCallbackDynamicParams(newrelic_api_key="team_a_key")
        params_b = StandardCallbackDynamicParams(newrelic_api_key="team_b_key")

        with patch("asyncio.create_task"):
            result_a = NewRelicHandler.get_newrelic_logger_for_request(
                standard_callback_dynamic_params=params_a,
                in_memory_dynamic_logger_cache=cache,
            )
            result_b = NewRelicHandler.get_newrelic_logger_for_request(
                standard_callback_dynamic_params=params_b,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result_a is not result_b
        assert result_a.newrelic_api_key == "team_a_key"
        assert result_b.newrelic_api_key == "team_b_key"

    def test_region_is_part_of_cache_key(self):
        """Same key, different region must not share a logger (different endpoints)."""
        cache = DynamicLoggingCache()

        with patch("asyncio.create_task"):
            result_us = NewRelicHandler.get_newrelic_logger_for_request(
                standard_callback_dynamic_params=StandardCallbackDynamicParams(newrelic_api_key="key"),
                in_memory_dynamic_logger_cache=cache,
            )
            result_eu = NewRelicHandler.get_newrelic_logger_for_request(
                standard_callback_dynamic_params=StandardCallbackDynamicParams(
                    newrelic_api_key="key", newrelic_region="eu"
                ),
                in_memory_dynamic_logger_cache=cache,
            )

        assert result_us is not result_eu
        assert result_us.metric_api_url == US_ENDPOINT
        assert result_eu.metric_api_url == EU_ENDPOINT

    def test_request_blocked_callback_params_includes_newrelic(self):
        from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
            _request_blocked_callback_params,
        )

        assert "newrelic_api_key" in _request_blocked_callback_params
        assert "newrelic_region" in _request_blocked_callback_params


class TestDynamicCredentialDetection:
    def test_no_credentials(self):
        params = StandardCallbackDynamicParams()
        assert NewRelicHandler._dynamic_newrelic_credentials_are_passed(params) is False

    def test_region_only_is_not_credentials(self):
        params = StandardCallbackDynamicParams(newrelic_region="eu")
        assert NewRelicHandler._dynamic_newrelic_credentials_are_passed(params) is False

    def test_api_key_is_credentials(self):
        params = StandardCallbackDynamicParams(newrelic_api_key="key")
        assert NewRelicHandler._dynamic_newrelic_credentials_are_passed(params) is True


class TestStandardCallbackDynamicParamsIncludesNewRelic:
    def test_newrelic_params_in_annotations(self):
        annotations = StandardCallbackDynamicParams.__annotations__
        assert "newrelic_api_key" in annotations
        assert "newrelic_region" in annotations


def _build_logging_obj(kwargs: dict, *, with_newrelic_callback: bool = True):
    from litellm.litellm_core_utils.litellm_logging import Logging

    with patch("asyncio.create_task"):
        return Logging(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="completion",
            start_time="2026-01-01",
            litellm_call_id="test-call-id",
            function_id="test-func",
            dynamic_success_callbacks=["newrelic"] if with_newrelic_callback else None,
            kwargs=kwargs,
        )


def _metrics_loggers(logging_obj) -> list[NewRelicMetricsLogger]:
    return [cb for cb in (logging_obj.dynamic_success_callbacks or []) if isinstance(cb, NewRelicMetricsLogger)]


class TestTeamCallbackFlowPassesNewRelicCredentials:
    """
    newrelic_* credentials reach NewRelicHandler only from the proxy-stamped trusted
    field. Anything the caller put in the request body must not, or a caller could
    pair its own newrelic_region with the team's ingest key.
    """

    def test_trusted_callback_vars_reach_newrelic_handler(self):
        logging_obj = _build_logging_obj(
            {
                TRUSTED_CALLBACK_VARS_FIELD: {"newrelic_api_key": "team-nr-key-123", "newrelic_region": "eu"},
                "model": "gpt-4",
                "litellm_params": {"metadata": {}},
            }
        )

        metrics_loggers = _metrics_loggers(logging_obj)
        assert len(metrics_loggers) == 1, "NewRelicMetricsLogger should be initialized from team callback_vars"
        assert metrics_loggers[0].newrelic_api_key == "team-nr-key-123"
        assert metrics_loggers[0].metric_api_url == EU_ENDPOINT

    def test_trace_logger_still_dispatched_alongside_metrics(self):
        """The metrics logger must not displace the trace logger for the same name."""
        logging_obj = _build_logging_obj(
            {
                TRUSTED_CALLBACK_VARS_FIELD: {"newrelic_api_key": "team-nr-key-123"},
                "model": "gpt-4",
                "litellm_params": {"metadata": {}},
            }
        )

        non_metrics = [
            cb for cb in (logging_obj.dynamic_success_callbacks or []) if not isinstance(cb, NewRelicMetricsLogger)
        ]
        assert len(non_metrics) == 1, "trace logger (OTel v2 or legacy agent) must remain in the dynamic list"
        assert len(_metrics_loggers(logging_obj)) == 1
        async_non_metrics = [
            cb
            for cb in (logging_obj.dynamic_async_success_callbacks or [])
            if not isinstance(cb, NewRelicMetricsLogger)
        ]
        assert len(async_non_metrics) == 1

    def test_request_kwargs_newrelic_params_are_ignored(self):
        logging_obj = _build_logging_obj(
            {
                "newrelic_api_key": "caller-nr-key",
                "newrelic_region": "eu",
                "model": "gpt-4",
                "litellm_params": {"metadata": {}},
            }
        )

        assert _metrics_loggers(logging_obj) == []

    def test_logging_object_stays_deepcopyable(self):
        logging_obj = _build_logging_obj(
            {
                TRUSTED_CALLBACK_VARS_FIELD: {"newrelic_api_key": "team-nr-key-123"},
                "model": "gpt-4",
                "litellm_params": {"metadata": {}},
            },
            with_newrelic_callback=False,
        )

        assert copy.deepcopy(logging_obj)._trusted_callback_vars == logging_obj._trusted_callback_vars

    def test_caller_cannot_redirect_team_credentials(self):
        """The exfil shape: caller's newrelic_region paired with the team's key."""
        logging_obj = _build_logging_obj(
            {
                TRUSTED_CALLBACK_VARS_FIELD: {"newrelic_api_key": "team-nr-key-123"},
                "newrelic_region": "eu",
                "model": "gpt-4",
                "litellm_params": {"metadata": {}},
            }
        )

        metrics_loggers = _metrics_loggers(logging_obj)
        assert len(metrics_loggers) == 1
        assert metrics_loggers[0].newrelic_api_key == "team-nr-key-123"
        assert metrics_loggers[0].metric_api_url == US_ENDPOINT
