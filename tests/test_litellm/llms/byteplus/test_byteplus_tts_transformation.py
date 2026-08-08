import base64
import json

import httpx
import pytest

from litellm.llms.byteplus.common_utils import BytePlusError
from litellm.llms.byteplus.text_to_speech.transformation import BytePlusTextToSpeechConfig


class TestBytePlusTextToSpeechConfig:
    def test_validate_environment_api_key(self, monkeypatch):
        monkeypatch.setenv("BYTEPLUS_API_KEY", "my-api-key")
        config = BytePlusTextToSpeechConfig()
        headers = config.validate_environment(headers={}, model="byteplus/seed-tts-2.0")
        assert headers.get("x-api-key") == "my-api-key"
        assert headers.get("X-Api-Resource-Id") == "seed-tts-2.0"

    def test_validate_environment_legacy_auth(self, monkeypatch):
        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.setenv("BYTEPLUS_TTS_APP_ID", "12345")
        monkeypatch.setenv("BYTEPLUS_TTS_ACCESS_KEY", "access-key-123")
        config = BytePlusTextToSpeechConfig()
        headers = config.validate_environment(headers={}, model="seed-tts-2.0")
        assert headers.get("X-Api-App-Id") == "12345"
        assert headers.get("X-Api-Access-Key") == "access-key-123"
        assert headers.get("X-Api-Resource-Id") == "seed-tts-2.0"

    def test_validate_environment_no_auth_raises(self, monkeypatch):
        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.delenv("BYTEPLUS_TTS_APP_ID", raising=False)
        monkeypatch.delenv("BYTEPLUS_TTS_ACCESS_KEY", raising=False)
        config = BytePlusTextToSpeechConfig()
        with pytest.raises(ValueError, match="BytePlus TTS requires authentication"):
            config.validate_environment(headers={}, model="seed-tts-2.0")

    def test_get_complete_url(self):
        config = BytePlusTextToSpeechConfig()
        url = config.get_complete_url(model="seed-tts-2.0", api_base=None, litellm_params={})
        assert url == "https://voice.ap-southeast-1.bytepluses.com/api/v3/tts/unidirectional"

    def test_transform_text_to_speech_request(self):
        config = BytePlusTextToSpeechConfig()
        data = config.transform_text_to_speech_request(
            model="seed-tts-2.0",
            input="Hello world",
            voice="en_female_skye_emo_v2_mars_bigtts",
            optional_params={"response_format": "mp3"},
            litellm_params={"user": "u-123"},
            headers={},
        )
        payload = data["dict_body"]
        assert payload["user"]["uid"] == "u-123"
        assert payload["req_params"]["text"] == "Hello world"
        assert payload["req_params"]["speaker"] == "en_female_skye_emo_v2_mars_bigtts"
        assert payload["req_params"]["audio_params"]["format"] == "mp3"
        additions = json.loads(payload["req_params"]["additions"])
        assert additions["disable_markdown_filter"] is True

    def test_transform_text_to_speech_request_speed(self):
        config = BytePlusTextToSpeechConfig()
        data = config.transform_text_to_speech_request(
            model="seed-tts-2.0",
            input="Hello world",
            voice="alloy",
            optional_params={"speed": 1.5},
            litellm_params={},
            headers={},
        )
        payload = data["dict_body"]
        assert payload["req_params"]["audio_params"]["speed_ratio"] == 1.5

    def test_transform_text_to_speech_request_no_speed_omitted(self):
        config = BytePlusTextToSpeechConfig()
        data = config.transform_text_to_speech_request(
            model="seed-tts-2.0",
            input="Hello world",
            voice="alloy",
            optional_params={},
            litellm_params={},
            headers={},
        )
        payload = data["dict_body"]
        assert "speed_ratio" not in payload["req_params"]["audio_params"]

    def test_transform_text_to_speech_response_success(self):
        config = BytePlusTextToSpeechConfig()
        chunk1 = base64.b64encode(b"audio-part-1").decode("utf-8")
        chunk2 = base64.b64encode(b"audio-part-2").decode("utf-8")
        raw_lines = f'{{"code":0,"data":"{chunk1}"}}\n{{"code":0,"data":"{chunk2}"}}\n{{"code":20000000,"message":"ok"}}'

        response = httpx.Response(status_code=200, text=raw_lines)
        binary_res = config.transform_text_to_speech_response("seed-tts-2.0", response, None)
        assert binary_res.content == b"audio-part-1audio-part-2"

    def test_transform_text_to_speech_response_content_type_mp3(self):
        config = BytePlusTextToSpeechConfig()
        mp3_header = b"\xff\xfb" + b"\x00" * 10
        chunk = base64.b64encode(mp3_header).decode("utf-8")
        raw_lines = f'{{"code":0,"data":"{chunk}"}}\n{{"code":20000000,"message":"ok"}}'
        response = httpx.Response(status_code=200, text=raw_lines)
        binary_res = config.transform_text_to_speech_response("seed-tts-2.0", response, None)
        assert binary_res.response.headers["content-type"] == "audio/mpeg"

    def test_transform_text_to_speech_response_content_type_ogg(self):
        config = BytePlusTextToSpeechConfig()
        ogg_header = b"OggS" + b"\x00" * 10
        chunk = base64.b64encode(ogg_header).decode("utf-8")
        raw_lines = f'{{"code":0,"data":"{chunk}"}}\n{{"code":20000000,"message":"ok"}}'
        response = httpx.Response(status_code=200, text=raw_lines)
        binary_res = config.transform_text_to_speech_response("seed-tts-2.0", response, None)
        assert binary_res.response.headers["content-type"] == "audio/ogg"

    def test_transform_text_to_speech_response_content_type_pcm_from_request(self):
        config = BytePlusTextToSpeechConfig()
        pcm_data = b"\x00\x01\x02\x03\x04\x05"
        chunk = base64.b64encode(pcm_data).decode("utf-8")
        raw_lines = f'{{"code":0,"data":"{chunk}"}}\n{{"code":20000000,"message":"ok"}}'
        req_payload = json.dumps({"req_params": {"audio_params": {"format": "pcm"}}}).encode("utf-8")
        req = httpx.Request("POST", "https://example.com", content=req_payload)
        response = httpx.Response(status_code=200, text=raw_lines, request=req)
        binary_res = config.transform_text_to_speech_response("seed-tts-2.0", response, None)
        assert binary_res.response.headers["content-type"] == "audio/pcm"

    def test_transform_text_to_speech_response_content_type_pcm_from_logging_obj(self):
        config = BytePlusTextToSpeechConfig()
        pcm_data = b"\x00\x01\x02\x03\x04\x05"
        chunk = base64.b64encode(pcm_data).decode("utf-8")
        raw_lines = f'{{"code":0,"data":"{chunk}"}}\n{{"code":20000000,"message":"ok"}}'
        response = httpx.Response(status_code=200, text=raw_lines)

        class DummyLoggingObj:
            optional_params: dict[str, str] = {"response_format": "pcm"}

        binary_res = config.transform_text_to_speech_response(
            "seed-tts-2.0",
            response,
            DummyLoggingObj(),  # pyright: ignore[reportArgumentType]  # mock logging object for testing
        )
        assert binary_res.response.headers["content-type"] == "audio/pcm"

    def test_transform_text_to_speech_response_error_code(self):
        config = BytePlusTextToSpeechConfig()
        raw_lines = '{"code":40402003,"message":"TTSExceededTextLimit"}'
        response = httpx.Response(status_code=200, text=raw_lines)
        with pytest.raises(BytePlusError, match="TTSExceededTextLimit"):
            config.transform_text_to_speech_response("seed-tts-2.0", response, None)

    def test_dispatch_filters_auth_headers(self, monkeypatch):
        config = BytePlusTextToSpeechConfig()
        captured: dict = {}

        def mock_tts_handler(**kwargs):
            captured["extra_headers"] = kwargs.get("extra_headers")
            return object()

        class FakeHandler:
            def text_to_speech_handler(self, **kwargs):
                return mock_tts_handler(**kwargs)

        config.dispatch_text_to_speech(
            model="seed-tts-2.0",
            input="hello",
            voice="alloy",
            optional_params={},
            litellm_params_dict={},
            logging_obj=None,
            timeout=30.0,
            extra_headers={
                "X-Api-Resource-Id": "attacker-resource",
                "x-api-key": "attacker-key",
                "X-Custom-Header": "keep-this",
            },
            base_llm_http_handler=FakeHandler(),
            aspeech=False,
            api_base=None,
            api_key="real-key",
        )
        headers = captured.get("extra_headers") or {}
        assert "X-Api-Resource-Id" not in headers
        assert "x-api-key" not in headers
        assert headers.get("X-Custom-Header") == "keep-this"

    def test_litellm_speech_dispatch_byteplus(self, monkeypatch):
        import litellm
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        chunk = base64.b64encode(b"mocked-audio").decode("utf-8")
        mock_text = f'{{"code":0,"data":"{chunk}"}}\n{{"code":20000000,"message":"ok"}}'
        mock_resp = httpx.Response(status_code=200, text=mock_text)

        def mock_post(*args, **kwargs):
            return mock_resp

        monkeypatch.setattr(HTTPHandler, "post", mock_post)
        monkeypatch.setattr(httpx.Client, "post", mock_post)
        monkeypatch.setattr(httpx.Client, "send", mock_post)
        res = litellm.speech(
            model="byteplus/seed-tts-2.0",
            input="Hello test",
            api_key="test-api-key",
            voice="id_male_han_uranus_bigtts",
        )
        assert res.content == b"mocked-audio"
