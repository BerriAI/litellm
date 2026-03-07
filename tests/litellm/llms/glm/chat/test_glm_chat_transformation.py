"""
Unit tests for GLM chat completion transformation.

Tests cover authentication header injection, URL construction,
supported parameters, and error class — all without live API calls.
"""

import pytest
from litellm.llms.glm.chat.transformation import GLMChatConfig, GLMError


class TestGLMValidateEnvironment:
    """Test Bearer token injection via validate_environment."""

    def setup_method(self):
        self.config = GLMChatConfig()
        self.model = "glm-4-flash"

    def test_validate_environment_sets_bearer_token(self):
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="test-key-123",
        )
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_validate_environment_sets_content_type(self):
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        assert headers["content-type"] == "application/json"
        assert headers["accept"] == "application/json"

    def test_validate_environment_no_key_omits_auth_header(self, monkeypatch):
        """When no key is available the Authorization header should be absent."""
        monkeypatch.delenv("GLM_API_KEY", raising=False)
        import litellm
        orig = litellm.api_key
        litellm.api_key = None
        try:
            headers = self.config.validate_environment(
                headers={},
                model=self.model,
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )
        finally:
            litellm.api_key = orig
        assert "Authorization" not in headers

    def test_validate_environment_caller_headers_take_precedence(self):
        """Caller-supplied headers should not be overridden by defaults."""
        headers = self.config.validate_environment(
            headers={"X-Custom": "abc"},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="key",
        )
        assert headers["X-Custom"] == "abc"
        assert headers["Authorization"] == "Bearer key"

    def test_validate_environment_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "env-key-xyz")
        headers = self.config.validate_environment(
            headers={},
            model=self.model,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer env-key-xyz"


class TestGLMGetCompleteUrl:
    """Test endpoint URL construction."""

    def setup_method(self):
        self.config = GLMChatConfig()

    def test_default_base_appends_suffix(self):
        url = self.config.get_complete_url(
            api_base=None,
            api_key="k",
            model="glm-4-flash",
            optional_params={},
        )
        assert url.endswith("/chat/completions")
        assert "bigmodel.cn" in url

    def test_custom_base_appends_suffix(self):
        url = self.config.get_complete_url(
            api_base="https://my-glm-proxy.example.com/v1",
            api_key="k",
            model="glm-4",
            optional_params={},
        )
        assert url == "https://my-glm-proxy.example.com/v1/chat/completions"

    def test_no_double_suffix(self):
        url = self.config.get_complete_url(
            api_base="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            api_key="k",
            model="glm-4",
            optional_params={},
        )
        # Must not end with /chat/completions/chat/completions
        assert url.count("/chat/completions") == 1

    def test_trailing_slash_stripped(self):
        url = self.config.get_complete_url(
            api_base="https://custom-base.example.com/api/",
            api_key="k",
            model="glm-4",
            optional_params={},
        )
        assert not url.endswith("//")
        assert url.endswith("/chat/completions")


class TestGLMSupportedParams:
    """Test that required OpenAI-compatible params are advertised."""

    def setup_method(self):
        self.config = GLMChatConfig()

    def test_stream_is_supported(self):
        params = self.config.get_supported_openai_params(model="glm-4-flash")
        assert "stream" in params

    def test_tools_is_supported(self):
        params = self.config.get_supported_openai_params(model="glm-4-flash")
        assert "tools" in params

    def test_temperature_is_supported(self):
        params = self.config.get_supported_openai_params(model="glm-4-flash")
        assert "temperature" in params

    def test_max_tokens_is_supported(self):
        params = self.config.get_supported_openai_params(model="glm-4-flash")
        assert "max_tokens" in params


class TestGLMErrorClass:
    """Test GLMError propagates status code and message correctly."""

    def test_error_carries_status_code(self):
        err = GLMError(status_code=401, message="Unauthorized")
        assert err.status_code == 401

    def test_error_carries_message(self):
        err = GLMError(status_code=429, message="Rate limit exceeded")
        assert err.message == "Rate limit exceeded"

    def test_get_error_class_returns_glm_error(self):
        config = GLMChatConfig()
        err = config.get_error_class("Bad request", 400, {})
        assert isinstance(err, GLMError)
        assert err.status_code == 400


class TestGLMSamplingParams:
    """Test do_sample flag and range clamping via map_openai_params."""

    def setup_method(self):
        self.config = GLMChatConfig()
        self.model = "glm-4-flash"

    def _map(self, **non_default_params) -> dict:
        """Helper to call map_openai_params and return the result."""
        return self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

    # --- do_sample behaviour ---

    def test_do_sample_false_strips_temperature(self):
        result = self._map(do_sample=False, temperature=0.7, top_p=0.9)
        assert "temperature" not in result

    def test_do_sample_false_strips_top_p(self):
        result = self._map(do_sample=False, temperature=0.7, top_p=0.9)
        assert "top_p" not in result

    def test_do_sample_false_never_forwarded_to_api(self):
        result = self._map(do_sample=False, temperature=0.7)
        assert "do_sample" not in result

    def test_do_sample_true_passes_temperature(self):
        result = self._map(do_sample=True, temperature=0.8)
        assert "temperature" in result
        assert result["temperature"] == pytest.approx(0.8)

    def test_do_sample_true_never_forwarded_to_api(self):
        result = self._map(do_sample=True, temperature=0.8)
        assert "do_sample" not in result

    def test_do_sample_not_set_passes_temperature(self):
        """Without do_sample, temperature should pass through normally."""
        result = self._map(temperature=0.5)
        assert "temperature" in result
        assert "do_sample" not in result

    # --- max_tokens clamping ---

    def test_max_tokens_within_limit_unchanged(self):
        result = self._map(max_tokens=1024)
        assert result["max_tokens"] == 1024

    def test_max_tokens_above_limit_clamped(self):
        result = self._map(max_tokens=999_999)
        assert result["max_tokens"] == 131_072

    def test_max_tokens_at_limit(self):
        result = self._map(max_tokens=131_072)
        assert result["max_tokens"] == 131_072

    # --- temperature clamping ---

    def test_temperature_above_max_clamped(self):
        result = self._map(temperature=2.5)
        assert result["temperature"] == pytest.approx(1.0)

    def test_temperature_below_min_clamped(self):
        result = self._map(temperature=-0.5)
        assert result["temperature"] == pytest.approx(0.0)

    def test_temperature_within_range_unchanged(self):
        result = self._map(temperature=0.42)
        assert result["temperature"] == pytest.approx(0.42)

    # --- top_p clamping ---

    def test_top_p_above_max_clamped(self):
        result = self._map(top_p=1.5)
        assert result["top_p"] == pytest.approx(1.0)

    def test_top_p_below_min_clamped(self):
        result = self._map(top_p=0.0)
        assert result["top_p"] == pytest.approx(0.01)

    def test_top_p_within_range_unchanged(self):
        result = self._map(top_p=0.85)
        assert result["top_p"] == pytest.approx(0.85)

    # --- do_sample is in supported params ---

    def test_do_sample_in_supported_params(self):
        params = self.config.get_supported_openai_params(model=self.model)
        assert "do_sample" in params

