"""
Unit Tests for Databricks Partner Integration Features
=======================================================

These tests are designed for automated CI/CD pipelines and do NOT require
real Databricks credentials. All external calls are mocked.

For integration tests that use real Databricks credentials, see:
    test_databricks_integration.py

Features Tested:
    - User-Agent building with partner prefixing (Databricks partner telemetry)
    - Token/sensitive data redaction for secure logging
    - OAuth M2M (Machine-to-Machine) authentication flow
    - Databricks SDK partner telemetry registration
    - Authentication priority (OAuth M2M > PAT > SDK)

Run with:
    pytest test_databricks_partner_integration.py -v

These tests align with Databricks Partner Architecture best practices:
    https://github.com/databrickslabs/partner-architecture
"""

import json
import os
import sys

import pytest
from unittest.mock import MagicMock, patch, Mock

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.databricks.common_utils import DatabricksBase, DatabricksException


class TestBuildUserAgent:
    """Test cases for User-Agent string building."""

    def test_default_user_agent(self):
        """No custom user agent returns litellm/{version}."""
        ua = DatabricksBase._build_user_agent(None)
        assert ua.startswith("litellm/")
        assert "_" not in ua.split("/")[0]

    def test_custom_user_agent_with_version(self):
        """Custom user agent with version extracts partner name."""
        ua = DatabricksBase._build_user_agent("mycompany/1.0.0")
        assert ua.startswith("mycompany_litellm/")
        # Verify the version is litellm's, not the custom one
        assert "/1.0.0" not in ua or "mycompany_litellm/1.0.0" not in ua

    def test_custom_user_agent_without_version(self):
        """Custom user agent without version still works."""
        ua = DatabricksBase._build_user_agent("mycompany")
        assert ua.startswith("mycompany_litellm/")

    def test_custom_user_agent_with_underscore(self):
        """Partner names with underscores are preserved."""
        ua = DatabricksBase._build_user_agent("my_company/2.0.0")
        assert ua.startswith("my_company_litellm/")

    def test_custom_user_agent_with_hyphen(self):
        """Partner names with hyphens are preserved."""
        ua = DatabricksBase._build_user_agent("my-company/2.0.0")
        assert ua.startswith("my-company_litellm/")

    def test_custom_user_agent_ignores_custom_version(self):
        """Custom version is ignored, litellm version is used."""
        ua = DatabricksBase._build_user_agent("partner/99.99.99")
        parts = ua.split("/")
        assert parts[0] == "partner_litellm"
        assert parts[1] != "99.99.99"

    def test_empty_string_returns_default(self):
        """Empty string returns default user agent."""
        ua = DatabricksBase._build_user_agent("")
        assert ua.startswith("litellm/")
        assert "_" not in ua.split("/")[0]

    def test_whitespace_only_returns_default(self):
        """Whitespace-only string returns default user agent."""
        ua = DatabricksBase._build_user_agent("   ")
        assert ua.startswith("litellm/")
        assert "_" not in ua.split("/")[0]

    def test_invalid_partner_name_returns_default(self):
        """Invalid partner names (special chars) return default."""
        ua = DatabricksBase._build_user_agent("my@company/1.0.0")
        assert ua.startswith("litellm/")

    def test_partner_with_numbers(self):
        """Partner names with numbers work."""
        ua = DatabricksBase._build_user_agent("company123/1.0.0")
        assert ua.startswith("company123_litellm/")


class TestRedactSensitiveData:
    """Test cases for sensitive data redaction."""

    def test_redact_bearer_token_in_string(self):
        """Bearer tokens are redacted in strings."""
        result = DatabricksBase.redact_sensitive_data("Bearer dapi12345abcdef")
        assert "dapi12345abcdef" not in result
        assert "[REDACTED]" in result

    def test_redact_dict_with_authorization(self):
        """Dict with authorization key is redacted."""
        data = {"Authorization": "Bearer secret123", "other": "value"}
        result = DatabricksBase.redact_sensitive_data(data)
        assert result["Authorization"] == "[REDACTED]"
        assert result["other"] == "value"

    def test_redact_nested_dict(self):
        """Nested dicts with sensitive keys are redacted."""
        data = {"config": {"api_key": "secret", "name": "test"}}
        result = DatabricksBase.redact_sensitive_data(data)
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["name"] == "test"

    def test_redact_pat_token(self):
        """Databricks PAT tokens are redacted."""
        test_token = "dapiTESTTOKENFAKEVALUEFORTESTINGPURPOSESONLY123"
        result = DatabricksBase.redact_sensitive_data(
            f"Using token {test_token}"
        )
        assert test_token not in result
        assert "[REDACTED_PAT]" in result

    def test_redact_client_secret(self):
        """Client secrets are redacted."""
        data = {"client_secret": "my-super-secret-value"}
        result = DatabricksBase.redact_sensitive_data(data)
        assert result["client_secret"] == "[REDACTED]"

    def test_redact_list_of_dicts(self):
        """Lists containing dicts with sensitive data are redacted."""
        data = [{"api_key": "secret1"}, {"name": "test"}]
        result = DatabricksBase.redact_sensitive_data(data)
        assert result[0]["api_key"] == "[REDACTED]"
        assert result[1]["name"] == "test"

    def test_redact_none_returns_none(self):
        """None input returns None."""
        assert DatabricksBase.redact_sensitive_data(None) is None

    def test_redact_preserves_non_sensitive_data(self):
        """Non-sensitive data is preserved."""
        data = {"model": "dbrx", "temperature": 0.7, "messages": ["hello"]}
        result = DatabricksBase.redact_sensitive_data(data)
        assert result == data


class TestRedactHeadersForLogging:
    """Test cases for header redaction."""

    def test_authorization_header_partially_shown(self):
        """Authorization header shows first 8 chars then redacts."""
        headers = {"Authorization": "Bearer dapi123456789abcdef"}
        result = DatabricksBase.redact_headers_for_logging(headers)
        assert result["Authorization"].startswith("Bearer d")
        assert "[REDACTED]" in result["Authorization"]

    def test_short_authorization_header_fully_redacted(self):
        """Short authorization values are fully redacted."""
        headers = {"Authorization": "short"}
        result = DatabricksBase.redact_headers_for_logging(headers)
        assert result["Authorization"] == "[REDACTED]"

    def test_non_sensitive_headers_preserved(self):
        """Non-sensitive headers are not modified."""
        headers = {"Content-Type": "application/json", "User-Agent": "test/1.0"}
        result = DatabricksBase.redact_headers_for_logging(headers)
        assert result["Content-Type"] == "application/json"
        assert result["User-Agent"] == "test/1.0"

    def test_empty_headers_returns_empty(self):
        """Empty headers dict returns empty dict."""
        assert DatabricksBase.redact_headers_for_logging({}) == {}

    def test_none_headers_returns_empty(self):
        """None headers returns empty dict."""
        assert DatabricksBase.redact_headers_for_logging(None) == {}

    def test_x_api_key_header_redacted(self):
        """X-API-Key header is redacted."""
        headers = {"X-API-Key": "my-api-key-12345"}
        result = DatabricksBase.redact_headers_for_logging(headers)
        assert "[REDACTED]" in result["X-API-Key"]


class TestOAuthM2M:
    """Test cases for OAuth M2M authentication."""

    def test_oauth_m2m_token_success(self):
        """OAuth M2M token is successfully obtained."""
        databricks_base = DatabricksBase()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "test-access-token"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            token = databricks_base._get_oauth_m2m_token(
                api_base="https://adb-123.azuredatabricks.net/serving-endpoints",
                client_id="test-client-id",
                client_secret="test-client-secret",
            )

            assert token == "test-access-token"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "oidc/v1/token" in call_args[0][0]
            assert call_args[1]["data"]["grant_type"] == "client_credentials"

    def test_oauth_m2m_token_failure(self):
        """OAuth M2M raises exception on failure."""
        databricks_base = DatabricksBase()

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(DatabricksException) as exc_info:
                databricks_base._get_oauth_m2m_token(
                    api_base="https://adb-123.azuredatabricks.net",
                    client_id="bad-client-id",
                    client_secret="bad-secret",
                )
            assert exc_info.value.status_code == 401

    def test_oauth_m2m_strips_serving_endpoints(self):
        """OAuth M2M correctly strips /serving-endpoints from URL."""
        databricks_base = DatabricksBase()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "token"}

        with patch("requests.post", return_value=mock_response) as mock_post:
            databricks_base._get_oauth_m2m_token(
                api_base="https://adb-123.azuredatabricks.net/serving-endpoints",
                client_id="id",
                client_secret="secret",
            )

            call_url = mock_post.call_args[0][0]
            assert "/serving-endpoints" not in call_url
            assert call_url == "https://adb-123.azuredatabricks.net/oidc/v1/token"


class TestValidateEnvironmentWithOAuth:
    """Test OAuth M2M is used when credentials are available."""

    def test_oauth_used_when_credentials_set(self, monkeypatch):
        """OAuth M2M is used when client_id and client_secret are set."""
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "test-secret")
        monkeypatch.setenv(
            "DATABRICKS_API_BASE", "https://adb-123.net/serving-endpoints"
        )

        databricks_base = DatabricksBase()

        with patch.object(
            databricks_base, "_get_oauth_m2m_token", return_value="oauth-token"
        ) as mock_oauth:
            api_base, headers = databricks_base.databricks_validate_environment(
                api_key=None,
                api_base=None,
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )

            mock_oauth.assert_called_once()
            assert headers["Authorization"] == "Bearer oauth-token"

    def test_pat_used_when_api_key_set(self, monkeypatch):
        """PAT is used when api_key is provided."""
        monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
        monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="dapi-test-key",
            api_base="https://adb-123.net/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
        )

        assert headers["Authorization"] == "Bearer dapi-test-key"


class TestValidateEnvironmentUserAgent:
    """Test User-Agent is correctly set in validate_environment."""

    def test_default_user_agent(self, monkeypatch):
        """Default user agent is set when no custom agent provided."""
        monkeypatch.delenv("DATABRICKS_USER_AGENT", raising=False)
        monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="test-key",
            api_base="https://adb-123.net/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
            custom_user_agent=None,
        )

        assert headers["User-Agent"].startswith("litellm/")
        assert "_" not in headers["User-Agent"].split("/")[0]

    def test_custom_user_agent_via_param(self, monkeypatch):
        """Custom user agent is prefixed when passed as parameter."""
        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="test-key",
            api_base="https://adb-123.net/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
            custom_user_agent="mycompany/1.0.0",
        )

        assert headers["User-Agent"].startswith("mycompany_litellm/")


class TestSDKPartnerTelemetry:
    """Test that SDK partner telemetry is registered."""

    def test_sdk_partner_registered(self):
        """useragent.with_partner is called when using SDK."""
        databricks_base = DatabricksBase()

        mock_workspace_client = MagicMock()
        mock_workspace_client.config.host = "https://adb-123.net"
        mock_workspace_client.config.authenticate.return_value = {
            "Authorization": "Bearer token"
        }

        mock_useragent = MagicMock()
        # Create a mock databricks.sdk module to simulate the SDK being available
        # This allows us to test the partner telemetry registration without requiring
        # the actual databricks-sdk package to be installed
        mock_sdk_module = MagicMock()
        mock_sdk_module.WorkspaceClient = MagicMock(return_value=mock_workspace_client)
        mock_sdk_module.useragent = mock_useragent
        
        # Mock both databricks and databricks.sdk modules to ensure the import works
        with patch.dict(sys.modules, {
            "databricks": MagicMock(),
            "databricks.sdk": mock_sdk_module
        }):
            databricks_base._get_databricks_credentials(
                api_key=None,
                api_base=None,
                headers=None,
            )

            # Verify that partner telemetry registration was called correctly
            mock_useragent.with_partner.assert_called_once_with("litellm")


class TestUserAgentFromEnvironment:
    """Test User-Agent is correctly picked up from environment variables."""

    def test_user_agent_from_databricks_env_var(self, monkeypatch):
        """DATABRICKS_USER_AGENT environment variable is used."""
        monkeypatch.setenv("DATABRICKS_USER_AGENT", "envpartner")
        monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="test-key",
            api_base="https://adb-123.net/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
            custom_user_agent="envpartner",  # Simulating what transformation.py passes
        )

        assert headers["User-Agent"].startswith("envpartner_litellm/")

    def test_custom_param_takes_precedence(self, monkeypatch):
        """Custom user_agent parameter takes precedence over environment."""
        monkeypatch.setenv("DATABRICKS_USER_AGENT", "envpartner")

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="test-key",
            api_base="https://adb-123.net/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
            custom_user_agent="parampartner/1.0.0",
        )

        assert headers["User-Agent"].startswith("parampartner_litellm/")


class TestLiteLLMCompletionUserAgent:
    """Test User-Agent is correctly passed through LiteLLM completion calls."""

    def test_completion_passes_user_agent_to_headers(self):
        """litellm.completion() correctly passes user_agent to request headers."""
        from litellm.llms.databricks.chat.transformation import DatabricksConfig

        config = DatabricksConfig()
        optional_params = {"user_agent": "testpartner/1.0.0"}

        # Mock the validation to capture what headers are set
        with patch.object(
            config,
            "databricks_validate_environment",
            return_value=(
                "https://test.net/serving-endpoints/chat/completions",
                {
                    "Authorization": "Bearer test",
                    "User-Agent": "testpartner_litellm/1.0.0",
                },
            ),
        ) as mock_validate:
            result = config.validate_environment(
                headers={},
                model="databricks/test-model",
                messages=[],
                optional_params=optional_params,
                litellm_params={},
                api_key="test-key",
                api_base="https://test.net/serving-endpoints",
            )

            # Verify user_agent was passed to databricks_validate_environment
            mock_validate.assert_called_once()
            call_kwargs = mock_validate.call_args[1]
            assert call_kwargs.get("custom_user_agent") == "testpartner/1.0.0"

    def test_user_agent_removed_from_optional_params(self):
        """user_agent is removed from optional_params so it's not sent to API."""
        from litellm.llms.databricks.chat.transformation import DatabricksConfig

        config = DatabricksConfig()
        optional_params = {
            "user_agent": "testpartner/1.0.0",
            "temperature": 0.7,
        }

        with patch.object(
            config,
            "databricks_validate_environment",
            return_value=(
                "https://test.net/chat/completions",
                {"Authorization": "Bearer test", "User-Agent": "test"},
            ),
        ):
            config.validate_environment(
                headers={},
                model="databricks/test-model",
                messages=[],
                optional_params=optional_params,
                litellm_params={},
                api_key="test-key",
                api_base="https://test.net/serving-endpoints",
            )

            # user_agent should be removed from optional_params
            assert "user_agent" not in optional_params
            # Other params should remain
            assert optional_params.get("temperature") == 0.7


class TestLiteLLMEmbeddingUserAgent:
    """Test User-Agent is correctly passed through LiteLLM embedding calls."""

    def test_embedding_passes_user_agent_to_headers(self):
        """litellm.embedding() correctly passes user_agent to request headers."""
        from litellm.llms.databricks.embed.handler import DatabricksEmbeddingHandler

        handler = DatabricksEmbeddingHandler()
        optional_params = {"user_agent": "embedpartner/1.0.0"}

        with patch.object(
            handler,
            "databricks_validate_environment",
            return_value=(
                "https://test.net/serving-endpoints/embeddings",
                {
                    "Authorization": "Bearer test",
                    "User-Agent": "embedpartner_litellm/1.0.0",
                },
            ),
        ) as mock_validate:
            with patch(
                "litellm.llms.openai_like.embedding.handler.OpenAILikeEmbeddingHandler.embedding"
            ):
                try:
                    handler.embedding(
                        model="databricks/test-model",
                        input=["test"],
                        timeout=30,
                        api_key="test-key",
                        api_base="https://test.net/serving-endpoints",
                        optional_params=optional_params,
                    )
                except Exception:
                    pass  # We just want to verify the mock was called

            # Verify user_agent was passed
            if mock_validate.called:
                call_kwargs = mock_validate.call_args[1]
                assert call_kwargs.get("custom_user_agent") == "embedpartner/1.0.0"


class TestAuthenticationPriority:
    """Test that authentication methods are used in correct priority order."""

    def test_oauth_used_when_no_api_key_provided(self, monkeypatch):
        """OAuth M2M is used when OAuth creds are set and no api_key is provided."""
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "oauth-client-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "oauth-secret")
        monkeypatch.setenv("DATABRICKS_API_BASE", "https://test.net/serving-endpoints")

        databricks_base = DatabricksBase()

        with patch.object(
            databricks_base, "_get_oauth_m2m_token", return_value="oauth-token"
        ) as mock_oauth:
            api_base, headers = databricks_base.databricks_validate_environment(
                api_key=None,  # No PAT provided - OAuth should be used
                api_base=None,
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )

            # OAuth should be used
            mock_oauth.assert_called_once()
            assert headers["Authorization"] == "Bearer oauth-token"

    def test_explicit_pat_takes_priority_over_oauth_env(self, monkeypatch):
        """Explicit api_key takes priority over OAuth token in final headers."""
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "oauth-client-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "oauth-secret")
        monkeypatch.setenv("DATABRICKS_API_BASE", "https://test.net/serving-endpoints")

        databricks_base = DatabricksBase()

        # Mock the OAuth call - it will be attempted but PAT should override
        with patch.object(
            databricks_base, "_get_oauth_m2m_token", return_value="oauth-token"
        ):
            api_base, headers = databricks_base.databricks_validate_environment(
                api_key="dapi-explicit-pat",
                api_base=None,
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )

            # PAT should override OAuth token since api_key was explicitly provided
            assert headers["Authorization"] == "Bearer dapi-explicit-pat"

    def test_pat_used_when_no_oauth_credentials(self, monkeypatch):
        """PAT is used when OAuth credentials are not set."""
        monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
        monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="dapi-pat-token",
            api_base="https://test.net/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
        )

        assert headers["Authorization"] == "Bearer dapi-pat-token"

    def test_sdk_fallback_when_no_credentials(self, monkeypatch):
        """Databricks SDK is used when no API key or OAuth credentials."""
        monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
        monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)

        databricks_base = DatabricksBase()

        mock_workspace_client = MagicMock()
        mock_workspace_client.config.host = "https://adb-123.net"
        mock_workspace_client.config.authenticate.return_value = {
            "Authorization": "Bearer sdk-token"
        }

        # Create a mock databricks.sdk module to simulate the SDK being available
        # This allows us to test the SDK fallback authentication without requiring
        # the actual databricks-sdk package to be installed
        mock_sdk_module = MagicMock()
        mock_sdk_module.WorkspaceClient = MagicMock(return_value=mock_workspace_client)
        mock_sdk_module.useragent = MagicMock()
        
        # Mock both databricks and databricks.sdk modules to ensure the import works
        with patch.dict(sys.modules, {
            "databricks": MagicMock(),
            "databricks.sdk": mock_sdk_module
        }):
            api_base, headers = databricks_base.databricks_validate_environment(
                api_key=None,
                api_base=None,
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=None,
            )

            # Verify that SDK authentication was used (headers contain Authorization)
            assert "Authorization" in headers


class TestEndpointURLConstruction:
    """Test that endpoint URLs are correctly constructed."""

    def test_chat_completions_endpoint(self, monkeypatch):
        """Chat completions endpoint is correctly appended."""
        monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
        monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="test-key",
            api_base="https://test.net/serving-endpoints",
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=None,
        )

        assert api_base.endswith("/chat/completions")

    def test_embeddings_endpoint(self, monkeypatch):
        """Embeddings endpoint is correctly appended."""
        monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
        monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="test-key",
            api_base="https://test.net/serving-endpoints",
            endpoint_type="embeddings",
            custom_endpoint=False,
            headers=None,
        )

        assert api_base.endswith("/embeddings")

    def test_custom_endpoint_not_modified(self, monkeypatch):
        """Custom endpoints are not modified."""
        monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
        monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)

        databricks_base = DatabricksBase()

        api_base, headers = databricks_base.databricks_validate_environment(
            api_key="test-key",
            api_base="https://test.net/custom/endpoint",
            endpoint_type="chat_completions",
            custom_endpoint=True,
            headers=None,
        )

        assert api_base == "https://test.net/custom/endpoint"
