"""
Tests for database connection idle lifetime configuration.

Verifies that max_idle_connection_lifetime is correctly added to
DATABASE_URL query parameters.

Regression tests for https://github.com/BerriAI/litellm/issues/22289
"""


from litellm.proxy.proxy_cli import (
    LiteLLMDatabaseConnectionPool,
    append_query_params,
)


class TestIdleConnectionLifetimeDefault:
    """LiteLLMDatabaseConnectionPool must have an idle lifetime default."""

    def test_default_value_is_60(self):
        assert (
            LiteLLMDatabaseConnectionPool.database_connection_idle_lifetime.value == 60
        )


class TestAppendQueryParamsIncludesIdleLifetime:
    """append_query_params must propagate max_idle_connection_lifetime."""

    def test_idle_lifetime_appended_to_url(self):
        url = "postgresql://user:pass@host:5432/db"
        params = {
            "connection_limit": 10,
            "pool_timeout": 60,
            "max_idle_connection_lifetime": 60,
        }
        result = append_query_params(url, params)
        assert "max_idle_connection_lifetime=60" in result

    def test_preserves_existing_params(self):
        url = "postgresql://user:pass@host:5432/db?sslmode=require"
        params = {
            "connection_limit": 10,
            "pool_timeout": 60,
            "max_idle_connection_lifetime": 120,
        }
        result = append_query_params(url, params)
        assert "sslmode=require" in result
        assert "max_idle_connection_lifetime=120" in result

    def test_does_not_duplicate_existing_idle_lifetime(self):
        url = "postgresql://user:pass@host:5432/db?max_idle_connection_lifetime=30"
        params = {
            "connection_limit": 10,
            "pool_timeout": 60,
            "max_idle_connection_lifetime": 60,
        }
        result = append_query_params(url, params)
        # The update should override the existing value
        assert "max_idle_connection_lifetime=60" in result
        assert result.count("max_idle_connection_lifetime") == 1

    def test_custom_lifetime_value(self):
        url = "postgresql://user:pass@host:5432/db"
        params = {
            "connection_limit": 10,
            "pool_timeout": 60,
            "max_idle_connection_lifetime": 300,
        }
        result = append_query_params(url, params)
        assert "max_idle_connection_lifetime=300" in result
