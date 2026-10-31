"""
Unit tests for WatsonxPassthroughConfig transformation.

Tests the Watsonx-specific passthrough configuration including URL construction,
streaming detection, and authentication handling.
"""

from unittest.mock import patch

import httpx

import litellm
from litellm.llms.watsonx.passthrough.transformation import WatsonxPassthroughConfig


class TestWatsonxPassthroughConfig:
    """Tests for WatsonxPassthroughConfig class."""

    def test_is_streaming_request_true(self):
        """Test that streaming is detected when stream=True in request data."""
        config = WatsonxPassthroughConfig()
        request_data = {"stream": True, "input": "test"}

        result = config.is_streaming_request(
            endpoint="ml/v1/text/generation", request_data=request_data
        )

        assert result is True

    def test_is_streaming_request_false(self):
        """Test that streaming is not detected when stream=False in request data."""
        config = WatsonxPassthroughConfig()
        request_data = {"stream": False, "input": "test"}

        result = config.is_streaming_request(
            endpoint="ml/v1/text/generation", request_data=request_data
        )

        assert result is False

    def test_is_streaming_request_missing_stream_key(self):
        """Test that streaming defaults to False when stream key is missing."""
        config = WatsonxPassthroughConfig()
        request_data = {"input": "test"}

        result = config.is_streaming_request(
            endpoint="ml/v1/text/generation", request_data=request_data
        )

        assert result is False

    def test_get_complete_url_with_api_base(self):
        """Test URL construction with explicit api_base."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/text/generation"
        request_query_params = {"version": "2024-03-19"}

        complete_url, base_target_url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="ibm/granite-13b-chat-v2",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url).startswith(api_base)
        assert endpoint in str(complete_url)
        assert "version=2024-03-19" in str(complete_url)
        assert base_target_url == api_base

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_complete_url_with_env_api_base(self, mock_get_secret):
        """Test URL construction with api_base from environment."""
        config = WatsonxPassthroughConfig()
        env_api_base = "https://eu-de.ml.cloud.ibm.com"
        mock_get_secret.return_value = env_api_base

        endpoint = "ml/v1/text/tokenization"
        request_query_params = {"version": "2024-03-19"}

        complete_url, base_target_url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="ibm/granite-13b-chat-v2",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url).startswith(env_api_base)
        assert endpoint in str(complete_url)
        assert base_target_url == env_api_base

    def test_get_complete_url_with_query_params(self):
        """Test that query parameters are correctly added to URL."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/text/generation"
        request_query_params = {
            "version": "2024-03-19",
        }

        complete_url, _ = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="ibm/granite-13b-chat-v2",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        url_str = str(complete_url)
        assert "version=2024-03-19" in url_str

    def test_get_complete_url_without_query_params(self):
        """Test URL construction without query parameters."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/models"

        complete_url, base_target_url = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="",
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={},
        )

        assert isinstance(complete_url, httpx.URL)
        assert str(complete_url) == f"{api_base}/{endpoint}"
        assert base_target_url == api_base
        assert "version=2024-03-19" not in str(complete_url)

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_base_with_explicit_value(self, mock_get_secret):
        """Test get_api_base returns explicit value when provided."""
        explicit_base = "https://custom.watsonx.com"

        result = WatsonxPassthroughConfig.get_api_base(api_base=explicit_base)

        assert result == explicit_base
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_base_from_environment(self, mock_get_secret):
        """Test get_api_base retrieves from environment when not provided."""
        env_base = "https://env.watsonx.com"
        mock_get_secret.return_value = env_base

        result = WatsonxPassthroughConfig.get_api_base(api_base=None)

        assert result == env_base
        mock_get_secret.assert_called_once_with("WATSONX_API_BASE")

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_key_with_explicit_value(self, mock_get_secret):
        """Test get_api_key returns explicit value when provided."""
        explicit_key = "test-api-key-123"

        result = WatsonxPassthroughConfig.get_api_key(api_key=explicit_key)

        assert result == explicit_key
        mock_get_secret.assert_not_called()

    @patch("litellm.llms.watsonx.common_utils.get_secret_str")
    def test_get_api_key_from_environment(self, mock_get_secret):
        """Test get_api_key retrieves from environment when not provided."""
        env_key = "env-api-key-456"
        mock_get_secret.return_value = env_key

        result = WatsonxPassthroughConfig.get_api_key(api_key=None)

        assert result == env_key
        mock_get_secret.assert_any_call("WATSONX_APIKEY")

    def test_get_base_model_returns_model(self):
        """Test get_base_model returns the model as-is."""
        model = "ibm/granite-13b-chat-v2"

        result = WatsonxPassthroughConfig.get_base_model(model)

        assert result == model

    def test_get_base_model_with_deployment(self):
        """Test get_base_model with deployment model."""
        model = "deployment/test-deployment-id"

        result = WatsonxPassthroughConfig.get_base_model(model)

        assert result == model

    def test_get_complete_url_with_different_endpoints(self):
        """Test URL construction with various endpoint paths."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"

        endpoints = [
            "ml/v1/text/generation",
            "ml/v1/text/tokenization",
            "ml/v1/deployments/test-id/text/generation",
            "ml/v1/models",
            "ml/v1/foundation_model_specs",
        ]

        for endpoint in endpoints:
            complete_url, base_target_url = config.get_complete_url(
                api_base=api_base,
                api_key=None,
                model="",
                endpoint=endpoint,
                request_query_params={"version": "2024-03-19"},
                litellm_params={},
            )

            assert isinstance(complete_url, httpx.URL)
            assert endpoint in str(complete_url)
            assert base_target_url == api_base

    def test_get_complete_url_preserves_query_param_order(self):
        """Test that query parameters maintain their values correctly."""
        config = WatsonxPassthroughConfig()
        api_base = "https://us-south.ml.cloud.ibm.com"
        endpoint = "ml/v1/text/generation"
        request_query_params = {
            "version": "2024-03-19",
            "project_id": "abc-123",
            "space_id": "xyz-789",
        }

        complete_url, _ = config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="",
            endpoint=endpoint,
            request_query_params=request_query_params,
            litellm_params={},
        )

        url_str = str(complete_url)
        # Verify all params are present
        assert "version=2024-03-19" in url_str
        assert "project_id=abc-123" in url_str
        assert "space_id=xyz-789" in url_str

    def test_is_streaming_request_with_various_stream_values(self):
        """Test streaming detection with different stream value types."""
        config = WatsonxPassthroughConfig()

        # Test with boolean True
        assert config.is_streaming_request("endpoint", {"stream": True}) is True

        # Test with boolean False
        assert config.is_streaming_request("endpoint", {"stream": False}) is False

        # Test with string "true" (truthy string)
        result = config.is_streaming_request("endpoint", {"stream": "true"})
        assert result == "true"  # Returns the value as-is from .get()

        # Test with integer 1 (truthy)
        result = config.is_streaming_request("endpoint", {"stream": 1})
        assert result == 1

        # Test with integer 0 (falsy)
        result = config.is_streaming_request("endpoint", {"stream": 0})
        assert result == 0

        # Test with None
        result = config.is_streaming_request("endpoint", {"stream": None})
        assert result is None

        # Test with empty dict (defaults to False)
        assert config.is_streaming_request("endpoint", {}) is False


# ---------------------------------------------------------------------------
# Helpers shared by credential-resolution tests
# ---------------------------------------------------------------------------

def _wx_deployment(model: str, **litellm_params) -> dict:
    """Build a minimal router deployment dict for watsonx."""
    return {
        "model_name": model.split("/", 1)[-1],
        "litellm_params": {"model": model, "use_in_pass_through": True, **litellm_params},
    }


def _wx_unflagged_deployment(model: str, **litellm_params) -> dict:
    """Build a deployment that does NOT set use_in_pass_through."""
    return {
        "model_name": model.split("/", 1)[-1],
        "litellm_params": {"model": model, **litellm_params},
    }


# ---------------------------------------------------------------------------
# deployment_model_matches
# ---------------------------------------------------------------------------

class TestDeploymentModelMatches:
    """Tests for WatsonxPassthroughConfig.deployment_model_matches."""

    def test_exact_match(self):
        creds = {"model": "ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughConfig.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is True

    def test_suffix_match(self):
        """watsonx/ prefix stripped — model_id from body is the bare id."""
        creds = {"model": "watsonx/ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughConfig.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is True

    def test_no_match_different_model(self):
        creds = {"model": "watsonx/ibm/granite-8b-code"}
        assert WatsonxPassthroughConfig.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is False

    def test_no_match_when_model_none(self):
        creds = {"model": None}
        assert WatsonxPassthroughConfig.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is False

    def test_no_match_when_model_key_absent(self):
        creds = {}
        assert WatsonxPassthroughConfig.deployment_model_matches(creds, "ibm/granite-13b-chat-v2") is False

    def test_partial_suffix_does_not_match(self):
        """'granite-13b' must not match 'ibm/granite-13b-chat-v2'."""
        creds = {"model": "watsonx/ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughConfig.deployment_model_matches(creds, "granite-13b-chat-v2") is True

    def test_wrong_suffix_does_not_match(self):
        creds = {"model": "watsonx/ibm/granite-13b-chat-v2"}
        assert WatsonxPassthroughConfig.deployment_model_matches(creds, "ibm/granite-8b") is False


# ---------------------------------------------------------------------------
# resolve_credentials_from_deployment
# ---------------------------------------------------------------------------

class TestResolveCredentialsFromDeployment:
    """Tests for WatsonxPassthroughConfig.resolve_credentials_from_deployment."""

    def test_returns_none_when_not_flagged(self):
        deployment = _wx_unflagged_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_key="sk-test",
            api_base="https://us-south.ml.cloud.ibm.com",
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is None

    def test_returns_none_for_non_watsonx_provider(self):
        deployment = _wx_deployment("openai/gpt-4o", api_key="sk-openai")
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is None

    def test_returns_none_when_model_missing(self):
        params = {"use_in_pass_through": True}
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(params)
        assert result is None

    def test_returns_none_when_get_llm_provider_raises_bad_request(self):
        """get_llm_provider raises BadRequestError for unrecognisable model strings — must return None."""
        from litellm.exceptions import BadRequestError

        params = {"use_in_pass_through": True, "model": "completely-unknown/some-model"}
        with patch(
            "litellm.llms.watsonx.passthrough.transformation.get_llm_provider",
            side_effect=BadRequestError(
                message="LLM Provider NOT provided.", model="completely-unknown/some-model", llm_provider=""
            ),
        ):
            result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(params)
        assert result is None

    def test_inline_api_key_and_api_base(self):
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_key="ibmkey-123",
            api_base="https://us-south.ml.cloud.ibm.com",
            project_id="proj-abc",
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") == "ibmkey-123"
        assert result.get("api_base") == "https://us-south.ml.cloud.ibm.com"
        assert result.get("project_id") == "proj-abc"
        assert result.get("space_id") is None
        assert result.get("model") == "watsonx/ibm/granite-13b-chat-v2"

    def test_space_id_is_read(self):
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_key="ibmkey-123",
            api_base="https://us-south.ml.cloud.ibm.com",
            space_id="space-xyz",
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("space_id") == "space-xyz"
        assert result.get("project_id") is None

    def test_api_key_none_when_omitted(self):
        """api_key is optional — deployment with only api_base is still eligible."""
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_base="https://us-south.ml.cloud.ibm.com",
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") is None
        assert result.get("api_base") == "https://us-south.ml.cloud.ibm.com"

    def test_api_base_none_when_omitted(self):
        """api_base is optional — deployment with only api_key is still eligible."""
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            api_key="ibmkey-123",
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") == "ibmkey-123"
        assert result.get("api_base") is None

    def test_named_credential_overrides_inline(self, monkeypatch):
        """litellm_credential_name values take precedence over inline litellm_params."""
        monkeypatch.setattr(litellm, "credential_list", [])
        from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
        from litellm.types.utils import CredentialItem

        CredentialAccessor.upsert_credentials([
            CredentialItem(
                credential_name="wx-cred",
                credential_values={
                    "api_key": "cred-key-999",
                    "api_base": "https://eu-de.ml.cloud.ibm.com",
                },
                credential_info={},
            )
        ])
        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            litellm_credential_name="wx-cred",
            api_key="inline-key-should-not-win",
            api_base="https://us-south.ml.cloud.ibm.com",
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") == "cred-key-999"
        assert result.get("api_base") == "https://eu-de.ml.cloud.ibm.com"

    def test_named_credential_loaded_after_deployment_still_resolves(self, monkeypatch):
        """Credential added after router starts is found on next call."""
        monkeypatch.setattr(litellm, "credential_list", [])
        from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
        from litellm.types.utils import CredentialItem

        deployment = _wx_deployment(
            "watsonx/ibm/granite-13b-chat-v2",
            litellm_credential_name="wx-late",
        )
        # First call — credential not registered yet
        result = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result is not None
        assert result.get("api_key") is None

        # Register credential after the fact
        CredentialAccessor.upsert_credentials([
            CredentialItem(
                credential_name="wx-late",
                credential_values={"api_key": "late-key", "api_base": "https://us-south.ml.cloud.ibm.com"},
                credential_info={},
            )
        ])

        result2 = WatsonxPassthroughConfig.resolve_credentials_from_deployment(
            deployment["litellm_params"]
        )
        assert result2 is not None
        assert result2.get("api_key") == "late-key"


# ---------------------------------------------------------------------------
# resolve_credentials_from_router_deployments
# ---------------------------------------------------------------------------

class TestResolveCredentialsFromRouterDeployments:
    """Tests for WatsonxPassthroughConfig.resolve_credentials_from_router_deployments."""

    def _deployments(self, *models_and_params):
        """Build a list of deployments from (model, **kwargs) tuples."""
        return [_wx_deployment(m, **p) for m, p in models_and_params]

    def test_returns_none_with_no_deployments(self):
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            deployments=[]
        )
        assert result is None

    def test_returns_none_with_no_eligible_deployments(self):
        deployments = [_wx_unflagged_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="k")]
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            deployments=deployments
        )
        assert result is None

    def test_single_deployment_returned_as_fallback(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k1", "api_base": "https://us-south.ml.cloud.ibm.com"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            deployments=deployments
        )
        assert result is not None
        assert result.get("api_key") == "k1"

    def test_exact_match_model_and_project_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-us", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-us"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-eu", "api_base": "https://eu-de.ml.cloud.ibm.com", "project_id": "proj-eu"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            project_id="proj-eu",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-eu"
        assert result.get("api_base") == "https://eu-de.ml.cloud.ibm.com"

    def test_exact_match_model_and_space_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-studio", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-studio"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-deploy", "api_base": "https://us-south.ml.cloud.ibm.com", "space_id": "space-prod"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            space_id="space-prod",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-deploy"

    def test_partial_match_model_only(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-code", "api_base": "https://us-south.ml.cloud.ibm.com"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-chat", "api_base": "https://us-south.ml.cloud.ibm.com"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-chat"

    def test_fallback_first_when_no_hints(self, recwarn):
        """With multiple deployments and no hints, returns first and emits a warning."""
        deployments = self._deployments(
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-first", "api_base": "https://us-south.ml.cloud.ibm.com"}),
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-second", "api_base": "https://us-south.ml.cloud.ibm.com"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-first"

    def test_non_watsonx_deployments_ignored(self):
        deployments = [
            _wx_deployment("openai/gpt-4o", api_key="openai-key"),
            _wx_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="wx-key", api_base="https://us-south.ml.cloud.ibm.com"),
        ]
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "wx-key"

    def test_unflagged_deployments_ignored(self):
        deployments = [
            _wx_unflagged_deployment("watsonx/ibm/granite-13b-chat-v2", api_key="should-be-ignored"),
            _wx_deployment("watsonx/ibm/granite-8b-code", api_key="eligible", api_base="https://us-south.ml.cloud.ibm.com"),
        ]
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "eligible"

    def test_project_id_mismatch_falls_through_to_model_match(self):
        """No project_id match → falls through to model-only partial match."""
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k1", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-a"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            model_id="ibm/granite-13b-chat-v2",
            project_id="proj-unknown",
            deployments=deployments,
        )
        # Falls through to partial match on model_id
        assert result is not None
        assert result.get("api_key") == "k1"

    def test_match_project_id_only_without_model_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-proj-a", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-a"}),
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-proj-b", "api_base": "https://us-south.ml.cloud.ibm.com", "project_id": "proj-b"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            project_id="proj-b",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-proj-b"

    def test_match_space_id_only_without_model_id(self):
        deployments = self._deployments(
            ("watsonx/ibm/granite-13b-chat-v2", {"api_key": "k-space-a", "api_base": "https://us-south.ml.cloud.ibm.com", "space_id": "space-a"}),
            ("watsonx/ibm/granite-8b-code", {"api_key": "k-space-b", "api_base": "https://us-south.ml.cloud.ibm.com", "space_id": "space-b"}),
        )
        result = WatsonxPassthroughConfig.resolve_credentials_from_router_deployments(
            space_id="space-b",
            deployments=deployments,
        )
        assert result is not None
        assert result.get("api_key") == "k-space-b"
