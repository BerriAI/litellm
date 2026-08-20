"""Unit tests for the WaveSpeed AI envelope parsing and URL helpers."""

import base64

import httpx
import pytest

from litellm.llms.wavespeed.common_utils import (
    CHAT_API_BASE,
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
    to_reference_uri,
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


class TestApiBaseIsolation:
    """Chat and media share the provider slug but not the host."""

    def test_the_chat_base_never_builds_a_prediction_url(self, monkeypatch):
        monkeypatch.delenv("WAVESPEED_API_BASE", raising=False)
        assert get_api_base(CHAT_API_BASE) == DEFAULT_API_BASE
        assert get_api_base(CHAT_API_BASE + "/") == DEFAULT_API_BASE

    def test_the_chat_base_in_the_env_does_not_break_media(self, monkeypatch):
        monkeypatch.setenv("WAVESPEED_API_BASE", CHAT_API_BASE)
        assert build_submit_url(None, "wavespeed-ai/z-image/turbo") == (
            f"{DEFAULT_API_BASE}/api/v3/wavespeed-ai/z-image/turbo"
        )

    def test_a_self_hosted_base_is_still_honored(self, monkeypatch):
        monkeypatch.delenv("WAVESPEED_API_BASE", raising=False)
        assert get_api_base("https://wavespeed.internal.corp") == "https://wavespeed.internal.corp"


class TestReferenceNormalization:
    """WaveSpeed submits JSON, so a reference has to be a URL or a data URI."""

    PNG = b"\x89PNG\r\n\x1a\n" + b"rest-of-the-png"

    def test_urls_and_data_uris_pass_through(self):
        assert to_reference_uri("https://example.com/a.png") == "https://example.com/a.png"
        assert to_reference_uri("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"

    def test_bytes_are_inlined_with_a_sniffed_media_type(self):
        assert to_reference_uri(self.PNG).startswith("data:image/png;base64,")
        assert to_reference_uri(b"\xff\xd8\xffrest").startswith("data:image/jpeg;base64,")
        assert to_reference_uri(b"GIF89arest").startswith("data:image/gif;base64,")
        assert to_reference_uri(b"RIFF1234WEBPrest").startswith("data:image/webp;base64,")

    def test_bytes_round_trip(self):
        encoded = to_reference_uri(self.PNG).split(",", 1)[1]
        assert base64.b64decode(encoded) == self.PNG

    def test_a_path_uses_its_extension_for_the_media_type(self, tmp_path):
        path = tmp_path / "frame.png"
        path.write_bytes(self.PNG)
        assert to_reference_uri(path).startswith("data:image/png;base64,")

    def test_a_binary_file_handle_is_read(self, tmp_path):
        path = tmp_path / "frame.png"
        path.write_bytes(self.PNG)
        with open(path, "rb") as handle:
            assert to_reference_uri(handle).startswith("data:image/png;base64,")

    def test_a_named_tuple_reference_uses_the_filename(self):
        assert to_reference_uri(("frame.jpg", self.PNG)).startswith("data:image/jpeg;base64,")

    def test_unsniffable_bytes_are_rejected_with_an_actionable_message(self):
        with pytest.raises(WaveSpeedError, match="Pass a URL, a data URI, or a named file"):
            to_reference_uri(b"not-a-known-format")

    def test_a_text_mode_handle_is_rejected(self, tmp_path):
        path = tmp_path / "frame.txt"
        path.write_text("hello")
        with open(path) as handle:
            with pytest.raises(WaveSpeedError, match="binary mode"):
                to_reference_uri(handle)

    def test_a_short_tuple_is_rejected(self):
        with pytest.raises(WaveSpeedError, match="missing its content"):
            to_reference_uri(("frame.png",))

    def test_a_tuple_wrapping_a_url_keeps_the_url(self):
        assert to_reference_uri(("frame.png", "https://example.com/a.png")) == "https://example.com/a.png"

    def test_an_unsupported_type_is_rejected(self):
        with pytest.raises(WaveSpeedError, match="Unsupported reference type"):
            to_reference_uri(object())
