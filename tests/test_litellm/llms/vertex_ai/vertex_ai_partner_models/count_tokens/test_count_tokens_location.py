"""
Tests for Vertex AI partner models count_tokens location resolution.

Ref: https://github.com/BerriAI/litellm/issues/23872
"""
import pytest

from litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler import (
    VertexAIPartnerModelsTokenCounter,
)


@pytest.fixture
def counter():
    return VertexAIPartnerModelsTokenCounter()


class TestCountTokensLocationResolution:
    """Verify that vertex_count_tokens_location is respected in handle_count_tokens_request."""

    def _build_litellm_params(
        self,
        vertex_location=None,
        vertex_count_tokens_location=None,
    ):
        params = {}
        if vertex_location is not None:
            params["vertex_location"] = vertex_location
        if vertex_count_tokens_location is not None:
            params["vertex_count_tokens_location"] = vertex_count_tokens_location
        return params

    @pytest.mark.asyncio
    async def test_count_tokens_location_overrides_vertex_location(self, counter, monkeypatch):
        """vertex_count_tokens_location should take precedence over vertex_location."""
        captured = {}

        async def fake_ensure_access_token(self, credentials, project_id, custom_llm_provider):
            return "fake-token", "fake-project"

        def fake_build_endpoint(self, model, project_id, vertex_location, api_base=None):
            captured["vertex_location"] = vertex_location
            return "https://fake-endpoint"

        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_ensure_access_token_async", fake_ensure_access_token
        )
        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_build_count_tokens_endpoint", fake_build_endpoint
        )

        # Mock the HTTP call to avoid real network requests
        class FakeResponse:
            status_code = 200
            def json(self):
                return {"input_tokens": 10}
            def raise_for_status(self):
                pass

        class FakeClient:
            async def post(self, url, headers=None, json=None, **kwargs):
                return FakeResponse()

        import litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler as handler_mod
        monkeypatch.setattr(handler_mod, "get_async_httpx_client", lambda **kwargs: FakeClient())

        litellm_params = self._build_litellm_params(
            vertex_location="us-east5",
            vertex_count_tokens_location="europe-west1",
        )

        await counter.handle_count_tokens_request(
            model="claude-sonnet-4-6",
            request_data={"messages": [{"role": "user", "content": "hi"}]},
            litellm_params=litellm_params,
        )

        assert captured["vertex_location"] == "europe-west1"

    @pytest.mark.asyncio
    async def test_claude_without_count_tokens_location_defaults_to_us_east5(self, counter, monkeypatch):
        """Claude models without any location should default to us-east5."""
        captured = {}

        async def fake_ensure_access_token(self, credentials, project_id, custom_llm_provider):
            return "fake-token", "fake-project"

        def fake_build_endpoint(self, model, project_id, vertex_location, api_base=None):
            captured["vertex_location"] = vertex_location
            return "https://fake-endpoint"

        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_ensure_access_token_async", fake_ensure_access_token
        )
        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_build_count_tokens_endpoint", fake_build_endpoint
        )

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"input_tokens": 10}
            def raise_for_status(self):
                pass

        class FakeClient:
            async def post(self, url, headers=None, json=None, **kwargs):
                return FakeResponse()

        import litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler as handler_mod
        monkeypatch.setattr(handler_mod, "get_async_httpx_client", lambda **kwargs: FakeClient())

        litellm_params = self._build_litellm_params()  # no location at all

        await counter.handle_count_tokens_request(
            model="claude-sonnet-4-6",
            request_data={"messages": [{"role": "user", "content": "hi"}]},
            litellm_params=litellm_params,
        )

        assert captured["vertex_location"] == "us-east5"

    @pytest.mark.asyncio
    async def test_claude_with_vertex_location_uses_it(self, counter, monkeypatch):
        """Claude models with vertex_location but no count_tokens_location should use vertex_location."""
        captured = {}

        async def fake_ensure_access_token(self, credentials, project_id, custom_llm_provider):
            return "fake-token", "fake-project"

        def fake_build_endpoint(self, model, project_id, vertex_location, api_base=None):
            captured["vertex_location"] = vertex_location
            return "https://fake-endpoint"

        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_ensure_access_token_async", fake_ensure_access_token
        )
        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_build_count_tokens_endpoint", fake_build_endpoint
        )

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"input_tokens": 10}
            def raise_for_status(self):
                pass

        class FakeClient:
            async def post(self, url, headers=None, json=None, **kwargs):
                return FakeResponse()

        import litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler as handler_mod
        monkeypatch.setattr(handler_mod, "get_async_httpx_client", lambda **kwargs: FakeClient())

        litellm_params = self._build_litellm_params(vertex_location="asia-southeast1")

        await counter.handle_count_tokens_request(
            model="claude-sonnet-4-6",
            request_data={"messages": [{"role": "user", "content": "hi"}]},
            litellm_params=litellm_params,
        )

        assert captured["vertex_location"] == "asia-southeast1"


class TestCountTokensVersionSuffixStripping:
    """Verify that version suffixes (@default, @20251001, etc.) are stripped
    from model names before sending to the Vertex AI count-tokens endpoint.

    The Vertex AI count-tokens API rejects versioned model names with:
    "claude-sonnet-4-6@default is not supported for token counting"
    while "claude-sonnet-4-6" (without suffix) works correctly.
    """

    def test_strip_version_suffix_at_default(self):
        counter = VertexAIPartnerModelsTokenCounter()
        assert counter._strip_version_suffix("claude-sonnet-4-6@default") == "claude-sonnet-4-6"

    def test_strip_version_suffix_at_date(self):
        counter = VertexAIPartnerModelsTokenCounter()
        assert counter._strip_version_suffix("claude-haiku-4-5@20251001") == "claude-haiku-4-5"

    def test_strip_version_suffix_no_suffix(self):
        counter = VertexAIPartnerModelsTokenCounter()
        assert counter._strip_version_suffix("claude-sonnet-4-6") == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_handle_count_tokens_strips_version_from_request_data(self, monkeypatch):
        """The model name in request_data sent to the API must have @suffix stripped."""
        counter = VertexAIPartnerModelsTokenCounter()
        captured_json = {}

        async def fake_ensure_access_token(self, credentials, project_id, custom_llm_provider):
            return "fake-token", "fake-project"

        def fake_build_endpoint(self, model, project_id, vertex_location, api_base=None):
            return "https://fake-endpoint"

        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_ensure_access_token_async", fake_ensure_access_token
        )
        monkeypatch.setattr(
            VertexAIPartnerModelsTokenCounter, "_build_count_tokens_endpoint", fake_build_endpoint
        )

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"input_tokens": 10}

        class FakeClient:
            async def post(self, url, headers=None, json=None, **kwargs):
                captured_json.update(json or {})
                return FakeResponse()

        import litellm.llms.vertex_ai.vertex_ai_partner_models.count_tokens.handler as handler_mod
        monkeypatch.setattr(handler_mod, "get_async_httpx_client", lambda **kwargs: FakeClient())

        await counter.handle_count_tokens_request(
            model="claude-sonnet-4-6@default",
            request_data={
                "model": "claude-sonnet-4-6@default",
                "messages": [{"role": "user", "content": "hi"}],
            },
            litellm_params={"vertex_location": "us-east5"},
        )

        # The model name sent to the API must NOT have the @default suffix
        assert captured_json["model"] == "claude-sonnet-4-6"
