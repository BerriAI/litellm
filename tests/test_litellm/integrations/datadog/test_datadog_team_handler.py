"""
Tests for team-scoped Datadog callback support.

Verifies that DataDogLogger can be instantiated with per-team credentials
(dd_api_key, dd_site) instead of relying solely on environment variables,
and that the DataDogHandler correctly resolves and caches per-team loggers.
"""

from unittest.mock import patch

import pytest

from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_team_handler import (
    DataDogHandler,
    DatadogLoggingConfig,
)
from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import (
    DynamicLoggingCache,
)
from litellm.types.utils import StandardCallbackDynamicParams


@pytest.fixture
def datadog_env(monkeypatch):
    """Set global DD env vars for the default/global logger."""
    monkeypatch.setenv("DD_API_KEY", "global_api_key")
    monkeypatch.setenv("DD_SITE", "us1.datadoghq.com")


class TestDataDogLoggerCredentialKwargs:
    """Test that DataDogLogger accepts credentials as kwargs."""

    def test_init_with_explicit_credentials(self):
        """Logger should use explicit kwargs instead of env vars."""
        with patch("asyncio.create_task"):
            logger = DataDogLogger(
                dd_api_key="team_api_key",
                dd_site="eu1.datadoghq.com",
            )

        assert logger.DD_API_KEY == "team_api_key"
        assert "eu1.datadoghq.com" in logger.intake_url

    def test_init_falls_back_to_env_vars(self, datadog_env):
        """Logger should fall back to env vars when no kwargs provided."""
        with patch("asyncio.create_task"):
            logger = DataDogLogger()

        assert logger.DD_API_KEY == "global_api_key"
        assert "us1.datadoghq.com" in logger.intake_url

    def test_init_kwargs_override_env_vars(self, datadog_env):
        """Explicit kwargs should take precedence over env vars."""
        with patch("asyncio.create_task"):
            logger = DataDogLogger(
                dd_api_key="override_key",
                dd_site="ap1.datadoghq.com",
            )

        assert logger.DD_API_KEY == "override_key"
        assert "ap1.datadoghq.com" in logger.intake_url

    def test_init_with_agent_credentials(self):
        """Logger should use agent mode when dd_agent_host is provided."""
        with patch("asyncio.create_task"):
            logger = DataDogLogger(
                dd_agent_host="dd-agent.local",
                dd_agent_port="8125",
                dd_api_key="agent_api_key",
            )

        assert "dd-agent.local:8125" in logger.intake_url
        assert logger.DD_API_KEY == "agent_api_key"

    def test_init_raises_without_credentials(self, monkeypatch):
        """Logger should raise if no credentials are available."""
        monkeypatch.delenv("DD_API_KEY", raising=False)
        monkeypatch.delenv("DD_SITE", raising=False)
        monkeypatch.delenv("LITELLM_DD_AGENT_HOST", raising=False)

        with pytest.raises(Exception, match="DD_API_KEY"):
            with patch("asyncio.create_task"):
                DataDogLogger()

    def test_agent_mode_does_not_leak_env_api_key_when_disallowed(self, datadog_env):
        """With allow_env_credentials=False, the agent logger must not pick up DD_API_KEY env var."""
        with patch("asyncio.create_task"):
            logger = DataDogLogger(
                dd_agent_host="attacker.example.com",
                allow_env_credentials=False,
            )

        assert logger.DD_API_KEY is None
        assert "attacker.example.com" in logger.intake_url

    def test_direct_api_mode_does_not_leak_env_api_key_when_disallowed(
        self, datadog_env
    ):
        """With allow_env_credentials=False and no explicit key, init must fail rather than reuse env key."""
        with pytest.raises(Exception, match="DD_API_KEY"):
            with patch("asyncio.create_task"):
                DataDogLogger(
                    dd_site="attacker.example.com",
                    allow_env_credentials=False,
                )


class TestDataDogHandler:
    """Test that DataDogHandler resolves the correct logger per team."""

    def test_creates_team_logger_with_dynamic_credentials(self, datadog_env):
        """Should create a new logger when team credentials are provided."""
        cache = DynamicLoggingCache()
        params = StandardCallbackDynamicParams(
            dd_api_key="team_a_key",
            dd_site="eu1.datadoghq.com",
        )

        with patch("asyncio.create_task"):
            result = DataDogHandler.get_datadog_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result.DD_API_KEY == "team_a_key"
        assert "eu1.datadoghq.com" in result.intake_url

    def test_caches_team_logger(self, datadog_env):
        """Same team credentials should return the same cached logger instance."""
        cache = DynamicLoggingCache()
        params = StandardCallbackDynamicParams(
            dd_api_key="team_b_key",
            dd_site="us5.datadoghq.com",
        )

        with patch("asyncio.create_task"):
            result1 = DataDogHandler.get_datadog_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )
            result2 = DataDogHandler.get_datadog_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result1 is result2

    def test_different_teams_get_different_loggers(self, datadog_env):
        """Different team credentials should create separate logger instances."""
        cache = DynamicLoggingCache()

        params_a = StandardCallbackDynamicParams(
            dd_api_key="team_a_key",
            dd_site="us1.datadoghq.com",
        )
        params_b = StandardCallbackDynamicParams(
            dd_api_key="team_b_key",
            dd_site="eu1.datadoghq.com",
        )

        with patch("asyncio.create_task"):
            result_a = DataDogHandler.get_datadog_logger_for_request(
                standard_callback_dynamic_params=params_a,
                in_memory_dynamic_logger_cache=cache,
            )
            result_b = DataDogHandler.get_datadog_logger_for_request(
                standard_callback_dynamic_params=params_b,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result_a is not result_b
        assert result_a.DD_API_KEY == "team_a_key"
        assert result_b.DD_API_KEY == "team_b_key"

    def test_partial_agent_config_does_not_leak_env_api_key(self, datadog_env):
        """A team-supplied dd_agent_host without dd_api_key must not exfiltrate the proxy DD_API_KEY."""
        cache = DynamicLoggingCache()
        params = StandardCallbackDynamicParams(
            dd_agent_host="attacker.example.com",
        )

        with patch("asyncio.create_task"):
            result = DataDogHandler.get_datadog_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result.DD_API_KEY is None
        assert "attacker.example.com" in result.intake_url

    def test_partial_site_config_does_not_leak_env_api_key(self, datadog_env):
        """A team-supplied dd_site without dd_api_key must not exfiltrate the proxy DD_API_KEY."""
        cache = DynamicLoggingCache()
        params = StandardCallbackDynamicParams(
            dd_site="attacker.example.com",
        )

        with pytest.raises(Exception, match="DD_API_KEY"):
            with patch("asyncio.create_task"):
                DataDogHandler.get_datadog_logger_for_request(
                    standard_callback_dynamic_params=params,
                    in_memory_dynamic_logger_cache=cache,
                )

    def test_full_team_config_still_uses_supplied_key(self, datadog_env):
        """When a team supplies its own key alongside a custom site, that key (not the env key) is used."""
        cache = DynamicLoggingCache()
        params = StandardCallbackDynamicParams(
            dd_api_key="team_key",
            dd_site="eu1.datadoghq.com",
        )

        with patch("asyncio.create_task"):
            result = DataDogHandler.get_datadog_logger_for_request(
                standard_callback_dynamic_params=params,
                in_memory_dynamic_logger_cache=cache,
            )

        assert result.DD_API_KEY == "team_key"
        assert "eu1.datadoghq.com" in result.intake_url

    def test_request_blocked_callback_params_includes_dd(self):
        """DD params should be blocked from request-level metadata (security)."""
        from litellm.litellm_core_utils.initialize_dynamic_callback_params import (
            _request_blocked_callback_params,
        )

        assert "dd_api_key" in _request_blocked_callback_params
        assert "dd_site" in _request_blocked_callback_params
        assert "dd_agent_host" in _request_blocked_callback_params
        assert "dd_agent_port" in _request_blocked_callback_params


class TestDynamicCredentialDetection:
    """Test that _dynamic_datadog_credentials_are_passed works correctly."""

    def test_no_credentials(self):
        params = StandardCallbackDynamicParams()
        assert DataDogHandler._dynamic_datadog_credentials_are_passed(params) is False

    def test_dd_api_key_only(self):
        params = StandardCallbackDynamicParams(dd_api_key="key")
        assert DataDogHandler._dynamic_datadog_credentials_are_passed(params) is True

    def test_dd_site_only(self):
        params = StandardCallbackDynamicParams(dd_site="site")
        assert DataDogHandler._dynamic_datadog_credentials_are_passed(params) is True

    def test_dd_agent_host_only(self):
        params = StandardCallbackDynamicParams(dd_agent_host="host")
        assert DataDogHandler._dynamic_datadog_credentials_are_passed(params) is True


class TestStandardCallbackDynamicParamsIncludesDatadog:
    """Verify that Datadog params are in the allow-list."""

    def test_dd_params_in_annotations(self):
        annotations = StandardCallbackDynamicParams.__annotations__
        assert "dd_api_key" in annotations
        assert "dd_site" in annotations
        assert "dd_agent_host" in annotations
        assert "dd_agent_port" in annotations
