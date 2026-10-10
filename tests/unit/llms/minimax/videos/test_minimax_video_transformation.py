"""
Tests for MiniMax-H3 video generation transformation.
"""

import base64
import io
import json
from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.exceptions import BadRequestError, UnsupportedParamsError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.minimax.videos.transformation import (
    MinimaxVideoConfig,
    MinimaxVideoError,
    _MiniMaxTask,
    _TaskError,
    _video_url_from_task,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)
from litellm.videos.main import avideo_generation

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

API_BASE = "https://api.minimax.io"


def _mock_response(payload: dict) -> Mock:
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = payload
    return mock_response


def _query_response(task: dict) -> Mock:
    return _mock_response({"task": task})


class TestMinimaxVideoCreateRequest:
    def test_text_to_video_defaults(self):
        """A prompt-only request must build the content array and apply
        MiniMax's required resolution/duration/ratio when the caller omits them."""
        data, files, url = MinimaxVideoConfig().transform_video_create_request(
            model="MiniMax-H3",
            prompt="A cinematic shot of a lighthouse at dusk",
            api_base=API_BASE,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{API_BASE}/v2/video_generation"
        assert files == ()
        assert data["model"] == "MiniMax-H3"
        assert data["content"] == [{"type": "text", "text": "A cinematic shot of a lighthouse at dusk"}]
        assert data["resolution"] == "768P"
        assert data["duration"] == 5
        assert data["ratio"] == "16:9"

    def test_explicit_params_beat_defaults(self):
        data, _, _ = MinimaxVideoConfig().transform_video_create_request(
            model="MiniMax-H3-Max",
            prompt="prompt",
            api_base=API_BASE,
            video_create_optional_request_params={"resolution": "480P", "duration": 9, "ratio": "9:16"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["model"] == "MiniMax-H3-Max"
        assert data["resolution"] == "480P"
        assert data["duration"] == 9
        assert data["ratio"] == "9:16"

    def test_map_openai_params_converts_seconds_and_size(self):
        """OpenAI ``seconds``/``size`` must become MiniMax ``duration``/``ratio``
        (1280x720 reduces to 16:9), not be forwarded verbatim."""
        mapped = MinimaxVideoConfig().map_openai_params(
            video_create_optional_params={"seconds": "5", "size": "1280x720", "resolution": "2K"},
            model="MiniMax-H3",
            drop_params=False,
        )

        assert mapped["duration"] == 5
        assert mapped["ratio"] == "16:9"
        assert mapped["resolution"] == "2K"
        assert "seconds" not in mapped
        assert "size" not in mapped

    def test_map_openai_params_drops_fields_minimax_never_takes(self):
        mapped = MinimaxVideoConfig().map_openai_params(
            video_create_optional_params={"user": "u1", "prompt": "p"},
            model="MiniMax-H3",
            drop_params=False,
        )

        assert mapped == {}

    @pytest.mark.parametrize("param,value", [("characters", [{"id": "c"}]), ("image", {"gcsUri": "gs://b/i.png"})])
    def test_unsupported_param_is_rejected_unless_drop_params(self, param, value):
        """Silently dropping a param the caller asked for hides that it had no
        effect; only drop_params opts into that."""
        config = MinimaxVideoConfig()

        with pytest.raises(UnsupportedParamsError, match="drop_params") as raised:
            config.map_openai_params(video_create_optional_params={param: value}, model="MiniMax-H3", drop_params=False)
        assert raised.value.status_code == 400

        assert (
            config.map_openai_params(video_create_optional_params={param: value}, model="MiniMax-H3", drop_params=True)
            == {}
        )

    @pytest.mark.parametrize("size", ["0x0", "1280x0", "0x720", "0:9", "16:0", "wide", "1280*720", "1280x720x3", ""])
    def test_invalid_size_is_a_bad_request(self, size):
        """A zero side would divide by zero in the ratio reduction; anything
        unparseable used to vanish and silently fall back to 16:9."""
        with pytest.raises(BadRequestError, match="size") as raised:
            MinimaxVideoConfig().map_openai_params(
                video_create_optional_params={"size": size}, model="MiniMax-H3", drop_params=False
            )

        assert raised.value.status_code == 400

    def test_ratio_size_passes_through_reduced(self):
        mapped = MinimaxVideoConfig().map_openai_params(
            video_create_optional_params={"size": "32:18"}, model="MiniMax-H3", drop_params=False
        )

        assert mapped["ratio"] == "16:9"

    @pytest.mark.parametrize("seconds", ["0", "-5", "abc", "5.5", "", True, 0, -3, 5.0])
    def test_invalid_seconds_is_a_bad_request(self, seconds):
        """An unparseable duration used to fall back to the 5 s default, so
        the caller got and paid for a video of a length they never asked for."""
        with pytest.raises(BadRequestError, match="seconds") as raised:
            MinimaxVideoConfig().map_openai_params(
                video_create_optional_params={"seconds": seconds}, model="MiniMax-H3", drop_params=False
            )

        assert raised.value.status_code == 400

    @pytest.mark.parametrize("duration", ["ten", 0, -1, True])
    def test_invalid_duration_override_is_a_bad_request(self, duration):
        """duration can also arrive through extra_body or parameters, past the
        seconds mapping; it is what gets billed, so it is checked again."""
        with pytest.raises(BadRequestError, match="seconds"):
            MinimaxVideoConfig().transform_video_create_request(
                model="MiniMax-H3",
                prompt="p",
                api_base=API_BASE,
                video_create_optional_request_params={"duration": duration},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_map_openai_params_explicit_ratio_wins_over_size(self):
        mapped = MinimaxVideoConfig().map_openai_params(
            video_create_optional_params={"size": "1280x720", "ratio": "4:3"},
            model="MiniMax-H3",
            drop_params=False,
        )

        assert mapped["ratio"] == "4:3"

    def test_map_openai_params_merges_parameters_block(self):
        mapped = MinimaxVideoConfig().map_openai_params(
            video_create_optional_params={"parameters": {"callback_url": "https://cb.example/hook", "ratio": "1:1"}},
            model="MiniMax-H3",
            drop_params=False,
        )

        assert mapped == {"callback_url": "https://cb.example/hook", "ratio": "1:1"}

    def test_non_object_parameters_is_a_bad_request(self):
        with pytest.raises(BadRequestError, match="parameters") as raised:
            MinimaxVideoConfig().map_openai_params(
                video_create_optional_params={"parameters": "callback_url=x"},
                model="MiniMax-H3",
                drop_params=False,
            )

        assert raised.value.status_code == 400

    def test_image_reference_file_becomes_first_frame_data_uri(self):
        """A file input_reference must arrive as a first_frame content item
        carrying a base64 data URI, and text-only ratio defaults must not apply."""
        data, _, _ = MinimaxVideoConfig().transform_video_create_request(
            model="MiniMax-H3",
            prompt="Pull focus to the people in the background",
            api_base=API_BASE,
            video_create_optional_request_params={"input_reference": io.BytesIO(PNG_BYTES)},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        image_item = data["content"][1]
        assert image_item["type"] == "image_url"
        assert image_item["role"] == "first_frame"
        assert image_item["image_url"]["url"].startswith("data:image/png;base64,")
        encoded = image_item["image_url"]["url"].split(",", 1)[1]
        assert base64.b64decode(encoded) == PNG_BYTES
        assert "ratio" not in data

    @pytest.mark.parametrize(
        ("file_tuple", "expected_type"),
        [
            (("frame.png", PNG_BYTES), "image/png"),
            (("frame.png", io.BytesIO(PNG_BYTES), "image/png"), "image/png"),
            (("frame.webp", PNG_BYTES, "image/webp", {}), "image/webp"),
        ],
    )
    def test_file_tuple_reference_sends_the_file_bytes(self, file_tuple, expected_type):
        data, _, _ = MinimaxVideoConfig().transform_video_create_request(
            model="MiniMax-H3",
            prompt="Pull focus to the people in the background",
            api_base=API_BASE,
            video_create_optional_request_params={"input_reference": file_tuple},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        url = data["content"][1]["image_url"]["url"]
        assert url.startswith(f"data:{expected_type};base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == PNG_BYTES

    def test_first_frame_reference_is_kept_alongside_explicit_media(self):
        media = [
            {"type": "audio_url", "audio_url": {"url": "https://cdn.example.com/ref.mp3"}, "role": "reference_audio"},
        ]
        data, _, _ = MinimaxVideoConfig().transform_video_create_request(
            model="MiniMax-H3",
            prompt="Character speaking",
            api_base=API_BASE,
            video_create_optional_request_params={
                "content": media,
                "input_reference": "https://cdn.example.com/frame.png",
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["content"] == [
            {"type": "text", "text": "Character speaking"},
            *media,
            {"type": "image_url", "image_url": {"url": "https://cdn.example.com/frame.png"}, "role": "first_frame"},
        ]

    def test_image_reference_url_passthrough(self):
        data, _, _ = MinimaxVideoConfig().transform_video_create_request(
            model="MiniMax-H3",
            prompt="Add more steam to the ramen bowl",
            api_base=API_BASE,
            video_create_optional_request_params={"input_reference": "https://cdn.example.com/frame.png"},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example.com/frame.png"},
            "role": "first_frame",
        }

    def test_explicit_media_content_follows_the_prompt(self):
        """Multimodal-reference (r2va) callers supply the media items; they pass
        through after the prompt and suppress the text-only ratio default."""
        media = [
            {"type": "image_url", "image_url": {"url": "https://cdn.example.com/ref.png"}, "role": "reference_image"},
            {"type": "audio_url", "audio_url": {"url": "https://cdn.example.com/ref.mp3"}, "role": "reference_audio"},
        ]
        data, _, _ = MinimaxVideoConfig().transform_video_create_request(
            model="MiniMax-H3",
            prompt="Character speaking",
            api_base=API_BASE,
            video_create_optional_request_params={"content": media},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["content"] == [{"type": "text", "text": "Character speaking"}, *media]
        assert "ratio" not in data

    @pytest.mark.parametrize(
        "content",
        [
            [{"type": "text", "text": "unscanned text"}, {"type": "image_url"}],
            [{"type": "image_url", "image_url": {"url": "https://x/a.png"}, "text": "unscanned text"}],
            [{"type": "image_url", "image_url": {"url": "https://x/a.png", "text": "unscanned text"}}],
            [{"type": "text", "text": "unscanned text"}, "not an item"],
            "unscanned text",
        ],
    )
    def test_malformed_content_cannot_smuggle_unscanned_text(self, content):
        """A text item hidden behind a malformed sibling, or text on a media
        item, must be rejected rather than let through by a lenient parse."""
        with pytest.raises(UnsupportedParamsError) as raised:
            MinimaxVideoConfig().transform_video_create_request(
                model="MiniMax-H3",
                prompt="a calm lake",
                api_base=API_BASE,
                video_create_optional_request_params={"content": content},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        assert raised.value.status_code == 400

    def test_text_inside_content_is_rejected_so_guardrails_cannot_be_bypassed(self):
        """Guardrails scan prompt; a text item smuggled into content would reach
        MiniMax unscanned while a harmless prompt passed the check."""
        with pytest.raises(UnsupportedParamsError, match="as prompt") as raised:
            MinimaxVideoConfig().transform_video_create_request(
                model="MiniMax-H3",
                prompt="a calm lake",
                api_base=API_BASE,
                video_create_optional_request_params={"content": [{"type": "text", "text": "unscanned text"}]},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        assert raised.value.status_code == 400

    @pytest.mark.parametrize(
        "configured_api_base",
        (
            "https://api.minimax.cn/v1/",
            "https://api.minimax.cn/v1",
            "https://api.minimax.cn/anthropic",
            "https://api.minimax.cn/anthropic/",
            "https://api.minimax.cn",
            "https://api.minimax.cn/",
            "https://api.minimax.cn/v2",
        ),
    )
    def test_any_published_surface_base_reaches_video(self, configured_api_base):
        """A MiniMax key works across that host's API surfaces, so an existing
        chat credential must reach video whichever base it was configured with:
        MiniMax publishes .../v1 (OpenAI-compatible) and .../anthropic
        (Anthropic Messages) alongside the /v2 video API, and users also
        configure the bare host."""
        config = MinimaxVideoConfig()
        api_base = config.get_complete_url(model="MiniMax-H3", api_base=configured_api_base, litellm_params={})

        _, _, url = config.transform_video_create_request(
            model="MiniMax-H3",
            prompt="p",
            api_base=api_base,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://api.minimax.cn/v2/video_generation"

    @pytest.mark.parametrize(
        ("configured_api_base", "expected"),
        (
            ("https://anthropic.example.com/v1", "https://anthropic.example.com"),
            ("https://api.minimax.cn/v1/proxy", "https://api.minimax.cn/v1/proxy"),
            ("https://gateway.internal/anthropic/shim", "https://gateway.internal/anthropic/shim"),
        ),
    )
    def test_surface_suffix_is_only_stripped_from_the_end(self, configured_api_base, expected):
        """Only a trailing surface segment marks the protocol. One appearing
        anywhere else is part of the address: cutting it out of the middle sends
        the request to a different path than the operator configured."""
        config = MinimaxVideoConfig()

        assert config.get_complete_url(model="MiniMax-H3", api_base=configured_api_base, litellm_params={}) == expected

    def test_get_complete_url_defaults_to_international_host(self):
        assert MinimaxVideoConfig().get_complete_url(model="MiniMax-H3", api_base=None, litellm_params={}) == (
            "https://api.minimax.io"
        )


class TestMinimaxVideoCreateResponse:
    def test_task_id_is_encoded_with_provider_and_model(self):
        """The create response only carries task_id; litellm must wrap it so
        later status/content/remix calls can route back to minimax."""
        video_obj = MinimaxVideoConfig().transform_video_create_response(
            model="MiniMax-H3",
            raw_response=_mock_response({"task_id": "424010985738629"}),
            logging_obj=None,
            custom_llm_provider="minimax",
            request_data={"model": "MiniMax-H3", "content": [], "resolution": "2K", "duration": 5, "ratio": "16:9"},
        )

        assert video_obj.status == "queued"
        assert video_obj.model == "MiniMax-H3"
        assert video_obj.seconds == "5"
        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded["custom_llm_provider"] == "minimax"
        assert decoded["model_id"] == "MiniMax-H3"
        assert decoded["video_id"] == "424010985738629"

    def test_usage_carries_cost_inputs(self):
        video_obj = MinimaxVideoConfig().transform_video_create_response(
            model="MiniMax-H3",
            raw_response=_mock_response({"task_id": "t1"}),
            logging_obj=None,
            custom_llm_provider="minimax",
            request_data={"resolution": "768P", "duration": 4},
        )

        assert video_obj.usage["duration_seconds"] == 4.0
        assert video_obj.usage["video_resolution"] == "768p"
        assert "input_image_count" not in video_obj.usage


class TestMinimaxVideoStatus:
    def test_succeeded_task_mapping(self):
        video_obj = MinimaxVideoConfig().transform_video_status_retrieve_response(
            raw_response=_query_response(
                {
                    "id": "424010985738629",
                    "model": "MiniMax-H3",
                    "status": "succeeded",
                    "created_at": 1785125529,
                    "updated_at": 1785125946,
                    "content": {"url": "https://cdn.example.com/output.mp4"},
                    "resolution": "2K",
                    "duration": 5,
                    "usage": {"total_seconds": 5, "input_seconds": 0, "output_seconds": 5, "input_image_count": 1},
                    "ratio": "16:9",
                    "task_type": "generation",
                    "modality": "video",
                }
            ),
            logging_obj=None,
            custom_llm_provider="minimax",
        )

        assert video_obj.status == "completed"
        assert video_obj.created_at == 1785125529
        assert video_obj.completed_at == 1785125946
        assert video_obj.seconds == "5"
        assert video_obj.model == "MiniMax-H3"
        assert video_obj.usage["output_seconds"] == 5
        decoded = decode_video_id_with_provider(video_obj.id)
        assert decoded["custom_llm_provider"] == "minimax"
        assert decoded["model_id"] == "MiniMax-H3"

    def test_running_task_maps_to_in_progress_without_completion(self):
        video_obj = MinimaxVideoConfig().transform_video_status_retrieve_response(
            raw_response=_query_response({"id": "t1", "model": "MiniMax-H3", "status": "running", "created_at": 1}),
            logging_obj=None,
            custom_llm_provider="minimax",
        )

        assert video_obj.status == "in_progress"
        assert video_obj.completed_at is None
        assert video_obj.usage is None

    def test_failed_task_maps_error(self):
        video_obj = MinimaxVideoConfig().transform_video_status_retrieve_response(
            raw_response=_query_response(
                {
                    "id": "t1",
                    "status": "failed",
                    "error": {"code": "1026", "message": "video description contains sensitive content"},
                    "created_at": 1,
                }
            ),
            logging_obj=None,
            custom_llm_provider="minimax",
        )

        assert video_obj.status == "failed"
        assert video_obj.error == {"code": "1026", "message": "video description contains sensitive content"}

    def test_request_decodes_task_id_from_wrapped_video_id(self):
        encoded_video_id = encode_video_id_with_provider("424010985738629", "minimax", "MiniMax-H3")
        url, data = MinimaxVideoConfig().transform_video_status_retrieve_request(
            video_id=encoded_video_id,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{API_BASE}/v2/query/video_generation/424010985738629"
        assert data == {}

    def test_content_request_hits_query_endpoint(self):
        encoded_video_id = encode_video_id_with_provider("424010985738629", "minimax", "MiniMax-H3")
        url, data = MinimaxVideoConfig().transform_video_content_request(
            video_id=encoded_video_id,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{API_BASE}/v2/query/video_generation/424010985738629"

    def test_video_url_from_task_pending_and_failed_are_client_errors(self):
        """A not-ready or failed task is the caller's state, so it must be a
        4xx; a bare ValueError surfaced as a 500 that the router retried."""
        with pytest.raises(MinimaxVideoError, match="still processing") as pending:
            _video_url_from_task(_MiniMaxTask(id="t1", status="running"))
        with pytest.raises(MinimaxVideoError, match="sensitive content") as failed:
            _video_url_from_task(_MiniMaxTask(id="t1", status="failed", error=_TaskError(message="sensitive content")))

        assert pending.value.status_code == 400
        assert failed.value.status_code == 400

    def test_not_ready_message_survives_the_handler_error_path(self):
        """The video handlers rebuild provider errors from response.text, so
        the message must ride on the response or it reaches the caller empty."""
        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        with pytest.raises(MinimaxVideoError) as pending:
            _video_url_from_task(_MiniMaxTask(id="t1", status="queued"))
        with pytest.raises(MinimaxVideoError) as rebuilt:
            BaseLLMHTTPHandler()._handle_error(e=pending.value, provider_config=MinimaxVideoConfig())

        assert rebuilt.value.status_code == 400
        assert "still processing" in rebuilt.value.message

    def test_unknown_status_fails_loud_instead_of_reporting_queued(self):
        """Defaulting an unmapped status to queued would have callers poll a
        task that will never progress."""
        with pytest.raises(MinimaxVideoError, match="expired") as raised:
            MinimaxVideoConfig().transform_video_status_retrieve_response(
                raw_response=_query_response({"id": "t1", "status": "expired"}),
                logging_obj=None,
                custom_llm_provider="minimax",
            )

        assert raised.value.status_code == 502

    @pytest.mark.parametrize(
        "payload",
        [{}, {"task_id": ""}, {"task_id": None}, {"base_resp": {"status_code": 1004, "status_msg": "auth failed"}}],
    )
    def test_create_without_a_task_id_fails_closed(self, payload):
        """Without a task id there is nothing to poll or download, so a
        queued object with an empty id must not be returned (or billed)."""
        with pytest.raises(MinimaxVideoError) as raised:
            MinimaxVideoConfig().transform_video_create_response(
                model="MiniMax-H3",
                raw_response=_mock_response(payload),
                logging_obj=None,
                custom_llm_provider="minimax",
                request_data={"resolution": "768P", "duration": 5},
            )

        assert raised.value.status_code == 502

    @pytest.mark.parametrize("payload", [{}, {"task": None}, {"task": {"status": "running"}}, {"task": {"id": ""}}])
    def test_status_without_a_task_fails_closed(self, payload):
        with pytest.raises(MinimaxVideoError) as raised:
            MinimaxVideoConfig().transform_video_status_retrieve_response(
                raw_response=_query_response(payload["task"]) if "task" in payload else _mock_response(payload),
                logging_obj=None,
                custom_llm_provider="minimax",
            )

        assert raised.value.status_code == 502

    def test_delete_without_a_task_id_fails_closed(self):
        with pytest.raises(MinimaxVideoError):
            MinimaxVideoConfig().transform_video_delete_response(
                raw_response=_mock_response({"status": "cancelled"}),
                logging_obj=None,
            )


class TestMinimaxVideoContentDownload:
    """The download URL comes out of MiniMax's response, so it is fetched
    through safe_get like any other URL litellm did not choose itself."""

    @pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8080/x.mp4"])
    def test_sync_download_refuses_internal_addresses(self, url):
        from litellm.litellm_core_utils.url_utils import SSRFError

        with pytest.raises(SSRFError):
            MinimaxVideoConfig().transform_video_content_response(
                raw_response=_query_response({"id": "t1", "status": "succeeded", "content": {"url": url}}),
                logging_obj=None,
            )

    @pytest.mark.asyncio
    async def test_async_download_refuses_internal_addresses(self):
        from litellm.litellm_core_utils.url_utils import SSRFError

        with pytest.raises(SSRFError):
            await MinimaxVideoConfig().async_transform_video_content_response(
                raw_response=_query_response(
                    {"id": "t1", "status": "succeeded", "content": {"url": "http://169.254.169.254/x.mp4"}}
                ),
                logging_obj=None,
            )

    def test_sync_download_returns_the_bytes_from_the_task_url(self, monkeypatch):
        from litellm.llms.minimax.videos import transformation

        client = Mock()
        client.get.return_value = Mock(content=b"mp4-bytes", raise_for_status=Mock())
        monkeypatch.setattr(litellm, "user_url_validation", False)
        monkeypatch.setattr(transformation, "_get_httpx_client", lambda *args, **kwargs: client)

        content = MinimaxVideoConfig().transform_video_content_response(
            raw_response=_query_response(
                {"id": "t1", "status": "succeeded", "content": {"url": "https://cdn.example.com/v.mp4"}}
            ),
            logging_obj=None,
        )

        assert content == b"mp4-bytes"
        assert client.get.call_args.args[0] == "https://cdn.example.com/v.mp4"

    @pytest.mark.asyncio
    async def test_async_download_returns_the_bytes_from_the_task_url(self, monkeypatch):
        from litellm.llms.minimax.videos import transformation

        response = Mock(content=b"mp4-bytes", raise_for_status=Mock())

        async def get(url, **kwargs):
            return response

        client = Mock()
        client.get = get
        monkeypatch.setattr(litellm, "user_url_validation", False)
        monkeypatch.setattr(transformation, "get_async_httpx_client", lambda **kwargs: client)

        content = await MinimaxVideoConfig().async_transform_video_content_response(
            raw_response=_query_response(
                {"id": "t1", "status": "succeeded", "content": {"url": "https://cdn.example.com/v.mp4"}}
            ),
            logging_obj=None,
        )

        assert content == b"mp4-bytes"

    def test_download_of_a_finished_task_without_a_url_is_a_client_error(self):
        with pytest.raises(MinimaxVideoError, match="no downloadable content") as raised:
            MinimaxVideoConfig().transform_video_content_response(
                raw_response=_query_response({"id": "t1", "status": "succeeded"}),
                logging_obj=None,
            )

        assert raised.value.status_code == 400


class TestMinimaxVideoList:
    def test_request_maps_limit_and_extra_query(self):
        url, params = MinimaxVideoConfig().transform_video_list_request(
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
            limit=4,
            extra_query={"filter.status": "succeeded", "page_num": 2},
        )

        assert url == f"{API_BASE}/v2/query/video_generation"
        assert params == {"page_size": "4", "filter.status": "succeeded", "page_num": 2}

    def test_after_cursor_is_rejected_instead_of_repeating_the_first_page(self):
        with pytest.raises(UnsupportedParamsError, match="page_num") as raised:
            MinimaxVideoConfig().transform_video_list_request(
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
                after="video_abc",
            )

        assert raised.value.status_code == 400

    def test_ascending_order_is_rejected(self):
        with pytest.raises(UnsupportedParamsError, match="newest first") as raised:
            MinimaxVideoConfig().transform_video_list_request(
                api_base=API_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
                order="asc",
            )

        assert raised.value.status_code == 400

    def test_descending_order_is_what_minimax_already_returns(self):
        _, params = MinimaxVideoConfig().transform_video_list_request(
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
            order="desc",
        )

        assert params == {}

    def test_response_adopts_openai_list_shape_with_encoded_ids(self):
        response = MinimaxVideoConfig().transform_video_list_response(
            raw_response=_mock_response(
                {
                    "items": [
                        {"id": "424635601932571", "model": "MiniMax-H3", "status": "succeeded", "duration": 5},
                        {"id": "424635601932588", "model": "MiniMax-H3", "status": "running"},
                    ],
                    "total": 476,
                }
            ),
            logging_obj=None,
            custom_llm_provider="minimax",
        )

        assert response["object"] == "list"
        assert response["total"] == 476
        assert [item["status"] for item in response["data"]] == ["completed", "in_progress"]
        first_decoded = decode_video_id_with_provider(response["first_id"])
        last_decoded = decode_video_id_with_provider(response["last_id"])
        assert first_decoded["video_id"] == "424635601932571"
        assert last_decoded["video_id"] == "424635601932588"
        assert first_decoded["custom_llm_provider"] == "minimax"


class TestMinimaxVideoRemix:
    @pytest.mark.asyncio
    async def test_remix_is_a_client_error_instead_of_a_retryable_500(self, monkeypatch: pytest.MonkeyPatch):
        """MiniMax regeneration only upscales to 2K and ignores any prompt, so
        remix is refused with a 400 the router does not retry."""
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        encoded_video_id: Final = encode_video_id_with_provider("424010985738629", "minimax", "MiniMax-H3")

        with pytest.raises(litellm.BadRequestError, match="remix is not supported by MiniMax") as raised:
            await litellm.avideo_remix(
                video_id=encoded_video_id,
                prompt="a different ending",
                custom_llm_provider="minimax",
            )

        assert raised.value.status_code == 400

    def test_remix_response_is_rejected(self):
        with pytest.raises(UnsupportedParamsError, match="remix is not supported by MiniMax") as raised:
            MinimaxVideoConfig().transform_video_remix_response(
                raw_response=_mock_response({}),
                logging_obj=None,
            )

        assert raised.value.status_code == 400


class TestMinimaxVideoDelete:
    def test_delete_request_and_cancelled_response(self):
        encoded_video_id = encode_video_id_with_provider("424010985738629", "minimax", "MiniMax-H3")
        url, data = MinimaxVideoConfig().transform_video_delete_request(
            video_id=encoded_video_id,
            api_base=API_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{API_BASE}/v2/video_generation/424010985738629"
        assert data == {}

        video_obj = MinimaxVideoConfig().transform_video_delete_response(
            raw_response=_mock_response({"task_id": "424010985738629", "action": "cancelled", "status": "cancelled"}),
            logging_obj=None,
        )
        assert video_obj.status == "cancelled"
        assert video_obj.id == "424010985738629"


class TestMinimaxVideoEnvironment:
    def test_explicit_api_key_wins_over_litellm_params(self):
        headers = MinimaxVideoConfig().validate_environment(
            headers={},
            model="MiniMax-H3",
            api_key="explicit-key",
            litellm_params=GenericLiteLLMParams(api_key="deployment-key"),
        )

        assert headers["Authorization"] == "Bearer explicit-key"
        assert headers["Content-Type"] == "application/json"

    def test_litellm_params_key_used_when_no_explicit_key(self):
        headers = MinimaxVideoConfig().validate_environment(
            headers={},
            model="MiniMax-H3",
            litellm_params=GenericLiteLLMParams(api_key="deployment-key"),
        )

        assert headers["Authorization"] == "Bearer deployment-key"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", None)

        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            MinimaxVideoConfig().validate_environment(
                headers={},
                model="MiniMax-H3",
                litellm_params=GenericLiteLLMParams(),
            )

    def test_env_var_api_key_used(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        monkeypatch.setattr("litellm.api_key", None)

        headers = MinimaxVideoConfig().validate_environment(
            headers={},
            model="MiniMax-H3",
            litellm_params=GenericLiteLLMParams(),
        )

        assert headers["Authorization"] == "Bearer env-key"


class TestMinimaxVideoEndToEndRequest:
    """
    Drive the real avideo_generation() entrypoint rather than the transform.

    map_openai_params runs first and only its output reaches
    transform_video_create_request, so a param the mapper drops never makes
    it into the body even though the transform alone handles it.
    """

    @staticmethod
    async def _wire_body(**kwargs) -> dict:
        sent: Final[list[httpx.Request]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return httpx.Response(200, json={"task_id": "t1"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            handler: Final = AsyncHTTPHandler()
            await handler.close()
            handler.client = http_client
            await avideo_generation(api_key="sk-test", api_base=API_BASE, client=handler, **kwargs)

        (request,) = sent
        return json.loads(request.content)

    @pytest.mark.asyncio
    async def test_input_reference_survives_param_mapping_into_the_body(self):
        """Regression: map_openai_params dropped input_reference, so an
        image-to-video call silently degraded to text-to-video."""
        body = await self._wire_body(
            model="minimax/MiniMax-H3",
            prompt="make it move",
            input_reference="https://cdn.example/first.png",
        )

        assert body["content"] == [
            {"type": "text", "text": "make it move"},
            {"type": "image_url", "image_url": {"url": "https://cdn.example/first.png"}, "role": "first_frame"},
        ]
        assert "input_reference" not in body
        assert "ratio" not in body

    @pytest.mark.asyncio
    async def test_file_input_reference_reaches_the_body_as_a_data_uri(self):
        body = await self._wire_body(
            model="minimax/MiniMax-H3",
            prompt="make it move",
            input_reference=io.BytesIO(PNG_BYTES),
        )

        image_item = body["content"][1]
        assert image_item["role"] == "first_frame"
        assert image_item["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"


@pytest.mark.usefixtures("local_model_cost_map")
class TestMinimaxVideoPricing:
    @pytest.mark.parametrize(
        "model,resolutions",
        [("MiniMax-H3", ("768p", "2k")), ("MiniMax-H3-Max", ("480p", "768p"))],
    )
    def test_every_supported_tier_bills_its_own_rate(self, model, resolutions):
        """A tier with no rate of its own falls back to the base rate, so a 2K
        video would silently bill at the 768P price."""
        from litellm import get_model_info
        from litellm.llms.openai.cost_calculation import video_generation_cost

        info = get_model_info(model=model, custom_llm_provider="minimax")

        for resolution in resolutions:
            tier_rate = info[f"output_cost_per_second_{resolution}"]
            assert tier_rate > 0
            cost = video_generation_cost(
                model=model, duration_seconds=5.0, custom_llm_provider="minimax", video_resolution=resolution
            )
            assert cost == pytest.approx(tier_rate * 5.0)

    def test_higher_resolution_never_bills_less(self):
        from litellm import get_model_info

        h3 = get_model_info(model="MiniMax-H3", custom_llm_provider="minimax")
        h3_max = get_model_info(model="MiniMax-H3-Max", custom_llm_provider="minimax")

        assert h3["output_cost_per_second_2k"] > h3["output_cost_per_second_768p"]
        assert h3_max["output_cost_per_second_768p"] > h3_max["output_cost_per_second_480p"]


@pytest.mark.usefixtures("local_model_cost_map")
class TestMinimaxVideoInputBilling:
    @staticmethod
    def _create_cost(model: str, image_count: int) -> float:
        import litellm

        content = [{"type": "text", "text": "p"}] + [
            {"type": "image_url", "image_url": {"url": f"https://x/{i}.png"}, "role": "reference_image"}
            for i in range(image_count)
        ]
        video_obj = MinimaxVideoConfig().transform_video_create_response(
            model=model,
            raw_response=_mock_response({"task_id": "t1"}),
            logging_obj=None,
            custom_llm_provider="minimax",
            request_data={"model": model, "content": content, "duration": 5, "resolution": "768P"},
        )
        return litellm.completion_cost(
            completion_response=video_obj,
            model=f"minimax/{model}",
            call_type="create_video",
            custom_llm_provider="minimax",
        )

    @pytest.mark.parametrize("model", ["MiniMax-H3", "MiniMax-H3-Max"])
    def test_images_beyond_the_free_allowance_are_billed_per_image(self, model):
        """MiniMax charges for input images past a per-model free allowance;
        billing only output seconds under-recorded those requests."""
        from litellm import get_model_info

        info = get_model_info(model=model, custom_llm_provider="minimax")
        free_images = info["provider_specific_entry"]["minimax_free_input_images"]
        image_rate = info["input_cost_per_image"]
        output_only = self._create_cost(model, image_count=0)

        assert self._create_cost(model, image_count=free_images) == pytest.approx(output_only)
        assert self._create_cost(model, image_count=free_images + 3) == pytest.approx(output_only + 3 * image_rate)
        assert output_only == pytest.approx(info["output_cost_per_second_768p"] * 5)

    @pytest.mark.parametrize(
        "content",
        [
            [{"type": "video_url", "video_url": {"url": "https://x/ref.mp4"}, "role": "reference_video"}],
        ],
    )
    def test_reference_video_is_rejected_because_its_length_cannot_be_billed(self, content):
        """MiniMax bills reference-video seconds, which are only reported after
        the create call that litellm bills."""
        with pytest.raises(UnsupportedParamsError, match="Reference video") as raised:
            MinimaxVideoConfig().transform_video_create_request(
                model="MiniMax-H3",
                prompt="p",
                api_base=API_BASE,
                video_create_optional_request_params={"content": content},
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

        assert raised.value.status_code == 400
