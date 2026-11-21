"""Credential, endpoint and cost helpers shared by every deAPI config."""

import httpx
import pytest

import litellm
from litellm.llms.deapi.common_utils import authorized_headers, endpoint_url, json_headers, reported_cost

DEAPI_BASE = "https://oai.deapi.ai/v1"


class TestEndpointUrl:
    def test_defaults_to_the_public_gateway(self, deapi_key):
        assert endpoint_url(None, "embeddings") == f"{DEAPI_BASE}/embeddings"

    def test_env_api_base_is_used(self, deapi_key, monkeypatch):
        monkeypatch.setenv("DEAPI_API_BASE", "https://gateway.internal/v1")

        assert endpoint_url(None, "audio/speech") == "https://gateway.internal/v1/audio/speech"

    def test_explicit_api_base_wins_and_loses_its_trailing_slash(self, deapi_key, monkeypatch):
        monkeypatch.setenv("DEAPI_API_BASE", "https://gateway.internal/v1")

        assert endpoint_url("https://proxy.example/v1/", "videos") == "https://proxy.example/v1/videos"


class TestAuthorizedHeaders:
    def test_env_key_becomes_the_bearer_header_and_caller_headers_are_kept(self, deapi_key):
        assert authorized_headers({"X-Trace": "abc"}, None, "Kokoro") == {
            "X-Trace": "abc",
            "Authorization": f"Bearer {deapi_key}",
        }

    def test_explicit_key_wins_over_env(self, deapi_key):
        assert authorized_headers({}, "explicit-key", "Kokoro")["Authorization"] == "Bearer explicit-key"

    def test_json_headers_add_the_content_type(self, deapi_key):
        assert json_headers({}, None, "Kokoro") == {
            "Authorization": f"Bearer {deapi_key}",
            "Content-Type": "application/json",
        }

    def test_missing_key_is_an_authentication_error(self, no_deapi_key):
        with pytest.raises(litellm.AuthenticationError, match="DEAPI_API_KEY"):
            authorized_headers({}, None, "Kokoro")


class TestReportedCost:
    def test_reads_usage_cost_from_a_json_body(self):
        response = httpx.Response(200, json={"data": [], "usage": {"prompt_tokens": 0, "cost": 0.00000017}})

        assert reported_cost(response) == 0.00000017

    def test_reads_the_cost_header_of_a_binary_answer(self):
        response = httpx.Response(200, content=b"ID3", headers={"X-DeAPI-Cost": "0.000041"})

        assert reported_cost(response) == 0.000041

    def test_the_header_and_the_body_agree_so_the_header_is_read_first(self):
        response = httpx.Response(200, json={"usage": {"cost": 0.0042}}, headers={"X-DeAPI-Cost": "0.0042"})

        assert reported_cost(response) == 0.0042

    def test_no_cost_anywhere_is_none(self):
        assert reported_cost(httpx.Response(200, json={"usage": {"prompt_tokens": 3}})) is None
        assert reported_cost(httpx.Response(200, json={"text": "hi"})) is None
        assert reported_cost(httpx.Response(200, content=b"plain transcript")) is None
        assert reported_cost(httpx.Response(200, json=["not", "an", "object"])) is None

    def test_a_malformed_header_falls_back_to_the_body(self):
        response = httpx.Response(200, json={"usage": {"cost": 0.0042}}, headers={"X-DeAPI-Cost": "n/a"})

        assert reported_cost(response) == 0.0042
