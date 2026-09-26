"""Credential, endpoint and cost helpers shared by every Eden AI config."""

import httpx
import pytest

import litellm
from litellm.llms.edenai.common_utils import authorized_headers, endpoint_url, json_headers, reported_cost

EDEN_BASE = "https://api.edenai.run/v3"
EDEN_EU_BASE = "https://api.eu.edenai.run/v3"


class TestEndpointUrl:
    def test_defaults_to_the_global_endpoint(self, eden_key):
        assert endpoint_url(None, "embeddings") == f"{EDEN_BASE}/embeddings"

    def test_env_api_base_moves_to_the_eu_endpoint(self, eden_key, monkeypatch):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)

        assert endpoint_url(None, "audio/speech") == f"{EDEN_EU_BASE}/audio/speech"

    def test_explicit_api_base_wins_and_loses_its_trailing_slash(self, eden_key, monkeypatch):
        monkeypatch.setenv("EDENAI_API_BASE", EDEN_EU_BASE)

        assert (
            endpoint_url("https://proxy.example/v3/", "images/generations")
            == "https://proxy.example/v3/images/generations"
        )


class TestAuthorizedHeaders:
    def test_env_key_becomes_the_bearer_header_and_caller_headers_are_kept(self, eden_key):
        assert authorized_headers({"X-Trace": "abc"}, None, "openai/tts-1") == {
            "X-Trace": "abc",
            "Authorization": f"Bearer {eden_key}",
        }

    def test_explicit_key_wins_over_env(self, eden_key):
        assert authorized_headers({}, "explicit-key", "openai/tts-1")["Authorization"] == "Bearer explicit-key"

    def test_json_headers_add_the_content_type(self, eden_key):
        assert json_headers({}, None, "openai/tts-1") == {
            "Authorization": f"Bearer {eden_key}",
            "Content-Type": "application/json",
        }

    def test_missing_key_is_an_authentication_error(self, no_eden_key):
        with pytest.raises(litellm.AuthenticationError, match="EDENAI_API_KEY"):
            authorized_headers({}, None, "openai/tts-1")


class TestReportedCost:
    def test_reads_the_top_level_cost_of_a_body(self):
        assert reported_cost({"cost": 0.0042, "provider": "openai"}) == 0.0042
        assert reported_cost(b'{"cost": 0.0042, "text": "hi"}') == 0.0042

    def test_reads_the_speech_cost_header(self):
        assert reported_cost(httpx.Headers({"x-edenai-cost": "0.00015", "content-type": "audio/mpeg"})) == 0.00015

    def test_no_cost_anywhere_is_none(self):
        assert reported_cost({"provider": "openai"}) is None
        assert reported_cost(httpx.Headers({"content-type": "audio/mpeg"})) is None
        assert reported_cost(b"not json") is None
