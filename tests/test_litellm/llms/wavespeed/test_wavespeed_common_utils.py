"""Unit tests for the WaveSpeed AI envelope parsing and URL helpers."""

import httpx
import pytest

from litellm.llms.wavespeed.common_utils import (
    DEFAULT_API_BASE,
    WaveSpeedError,
    build_headers,
    build_result_url,
    build_submit_url,
    get_api_base,
    get_outputs,
    get_prediction_id,
    map_status_to_openai,
    optional_entry,
    optional_pair,
    poll_outcome,
    to_request_payload,
    unwrap_envelope,
)


class TestUrls:
    def test_submit_url_defaults_to_the_public_api(self, monkeypatch):
        monkeypatch.delenv("WAVESPEED_API_BASE", raising=False)
        assert build_submit_url(None, "wavespeed-ai/z-image/turbo") == (
            f"{DEFAULT_API_BASE}/api/v3/wavespeed-ai/z-image/turbo"
        )

    def test_api_base_env_override(self, monkeypatch):
        monkeypatch.setenv("WAVESPEED_API_BASE", "https://proxy.internal/")
        assert get_api_base(None) == "https://proxy.internal"
        assert build_result_url(None, "pred-1") == "https://proxy.internal/api/v3/predictions/pred-1/result"

    def test_explicit_api_base_beats_the_env(self, monkeypatch):
        monkeypatch.setenv("WAVESPEED_API_BASE", "https://proxy.internal")
        assert get_api_base("https://other.internal") == "https://other.internal"

    def test_empty_model_is_rejected(self):
        with pytest.raises(WaveSpeedError, match="model is required"):
            build_submit_url(None, "///")

    def test_path_traversal_in_the_model_id_is_rejected(self):
        with pytest.raises(ValueError):
            build_submit_url(None, "wavespeed-ai/../../admin")

    def test_prediction_id_is_percent_encoded(self):
        assert build_result_url("https://api.wavespeed.ai", "a b").endswith("/predictions/a%20b/result")


class TestHeaders:
    def test_headers_carry_auth_and_channel_attribution(self):
        headers = build_headers("sk-test")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["X-Client-Name"] == "litellm"
        assert headers["X-Client-Version"]

    def test_api_key_falls_back_to_the_env(self, monkeypatch):
        monkeypatch.setenv("WAVESPEED_API_KEY", "sk-env")
        assert build_headers(None)["Authorization"] == "Bearer sk-env"

    def test_missing_api_key_raises_401(self, monkeypatch):
        monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
        with pytest.raises(WaveSpeedError) as exc_info:
            build_headers(None)
        assert exc_info.value.status_code == 401


class TestUnwrapEnvelope:
    def test_happy_path(self):
        raw = httpx.Response(200, json={"code": 200, "message": "ok", "data": {"id": "pred-1"}})
        assert unwrap_envelope(raw)["id"] == "pred-1"

    def test_http_error_surfaces_the_status_code(self):
        raw = httpx.Response(503, text="upstream down")
        with pytest.raises(WaveSpeedError) as exc_info:
            unwrap_envelope(raw)
        assert exc_info.value.status_code == 503
        assert "upstream down" in str(exc_info.value)

    def test_non_json_body(self):
        raw = httpx.Response(200, text="<html>gateway</html>")
        with pytest.raises(WaveSpeedError, match="Could not parse"):
            unwrap_envelope(raw)

    def test_non_object_body(self):
        raw = httpx.Response(200, json=["not", "an", "envelope"])
        with pytest.raises(WaveSpeedError, match="Unexpected WaveSpeed response body"):
            unwrap_envelope(raw)

    def test_platform_error_code_uses_the_platform_message(self):
        raw = httpx.Response(200, json={"code": 401, "message": "invalid api key", "data": None})
        with pytest.raises(WaveSpeedError, match="invalid api key"):
            unwrap_envelope(raw)

    def test_platform_error_code_without_a_message(self):
        raw = httpx.Response(200, json={"code": 500, "data": None})
        with pytest.raises(WaveSpeedError, match="WaveSpeed returned code 500"):
            unwrap_envelope(raw)

    def test_missing_data(self):
        raw = httpx.Response(200, json={"code": 200, "message": "ok"})
        with pytest.raises(WaveSpeedError, match="missing `data`"):
            unwrap_envelope(raw)


class TestPredictionHelpers:
    def test_missing_prediction_id_raises(self):
        with pytest.raises(WaveSpeedError, match="missing a prediction id"):
            get_prediction_id({"status": "created"})

    def test_get_outputs_defaults_to_empty(self):
        assert get_outputs({"status": "completed"}) == ()
        assert get_outputs({"status": "completed", "outputs": None}) == ()
        assert get_outputs({"status": "completed", "outputs": ["a"]}) == ["a"]

    @pytest.mark.parametrize(
        "status, expected", [("completed", "done"), ("created", "pending"), ("processing", "pending")]
    )
    def test_poll_outcome_non_terminal_and_success(self, status, expected):
        assert poll_outcome({"status": status}) == expected

    @pytest.mark.parametrize("status", ["failed", "cancelled", "timeout"])
    def test_poll_outcome_terminal_failures(self, status):
        with pytest.raises(WaveSpeedError, match=status):
            poll_outcome({"status": status, "error": "boom"})

    def test_poll_outcome_failure_without_an_error_detail(self):
        with pytest.raises(WaveSpeedError, match="no error detail returned"):
            poll_outcome({"status": "failed"})

    def test_status_mapping(self):
        assert map_status_to_openai("processing") == "in_progress"
        assert map_status_to_openai("cancelled") == "failed"
        assert map_status_to_openai("brand-new-status") == "queued"


class TestPayloadHelpers:
    def test_to_request_payload_accepts_mappings_and_pairs(self):
        assert to_request_payload({"a": 1}) == {"a": 1}
        assert to_request_payload((("a", 1), ("b", 2))) == {"a": 1, "b": 2}

    def test_optional_helpers_drop_none(self):
        assert optional_pair("a", 1) == (("a", 1),)
        assert optional_pair("a", None) == ()
        assert dict(optional_entry("a", 1)) == {"a": 1}
        assert dict(optional_entry("a", None)) == {}
