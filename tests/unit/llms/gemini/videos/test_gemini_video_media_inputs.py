"""
Gemini Veo media inputs (image, lastFrame, referenceImages, video) belong in
instances[0]; parameters only carries generation config.

See: https://ai.google.dev/gemini-api/docs/veo#veo-model-parameters
"""

import base64
import io
import json
from typing import Any

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.gemini.videos.transformation import GeminiVideoConfig
from litellm.types.router import GenericLiteLLMParams

VEO_31 = "veo-3.1-generate-preview"
VEO_31_LITE = "veo-3.1-lite-generate-preview"
API_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{VEO_31}:predictLongRunning"

FIRST_FRAME = {"bytesBase64Encoded": "Zmlyc3Q=", "mimeType": "image/png"}
LAST_FRAME = {"bytesBase64Encoded": "bGFzdA==", "mimeType": "image/png"}
DRESS = {"bytesBase64Encoded": "ZHJlc3M=", "mimeType": "image/jpeg"}
VEO_VIDEO = {"uri": "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media"}

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"png-payload"
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"jpeg-payload"


def _request_body(params: dict[str, Any], model: str = VEO_31, prompt: str = "a prompt") -> dict:
    data, files, _ = GeminiVideoConfig().transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base=API_BASE,
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
    return data


def _file(content: bytes, name: str | None = None) -> io.BytesIO:
    file = io.BytesIO(content)
    if name is not None:
        file.name = name
    return file


def _b64(content: bytes) -> str:
    return base64.b64encode(content).decode("utf-8")


@pytest.mark.parametrize(
    ("params", "expected_body"),
    [
        pytest.param(
            {"aspectRatio": "16:9", "durationSeconds": 4},
            {"instances": [{"prompt": "a prompt"}], "parameters": {"aspectRatio": "16:9", "durationSeconds": 4}},
            id="text-to-video",
        ),
        pytest.param(
            {"image": FIRST_FRAME, "durationSeconds": 6, "personGeneration": "allow_adult"},
            {
                "instances": [{"prompt": "a prompt", "image": FIRST_FRAME}],
                "parameters": {"durationSeconds": 6, "personGeneration": "allow_adult"},
            },
            id="image-to-video",
        ),
        pytest.param(
            {"image": FIRST_FRAME, "lastFrame": LAST_FRAME, "aspectRatio": "9:16"},
            {
                "instances": [{"prompt": "a prompt", "image": FIRST_FRAME, "lastFrame": LAST_FRAME}],
                "parameters": {"aspectRatio": "9:16"},
            },
            id="interpolation",
        ),
        pytest.param(
            {"referenceImages": [{"image": DRESS, "referenceType": "asset"}], "durationSeconds": 8},
            {
                "instances": [{"prompt": "a prompt", "referenceImages": [{"image": DRESS, "referenceType": "asset"}]}],
                "parameters": {"durationSeconds": 8},
            },
            id="reference-images",
        ),
        pytest.param(
            {"video": VEO_VIDEO, "resolution": "720p"},
            {"instances": [{"prompt": "a prompt", "video": VEO_VIDEO}], "parameters": {"resolution": "720p"}},
            id="video-extension",
        ),
    ],
)
def test_media_inputs_go_to_instance_and_config_stays_in_parameters(params, expected_body):
    assert _request_body(params) == expected_body


def test_file_like_last_frame_keeps_its_bytes_and_mime_type():
    body = _request_body({"image": _file(JPEG_BYTES), "lastFrame": _file(PNG_BYTES)})

    assert body["instances"][0] == {
        "prompt": "a prompt",
        "image": {"bytesBase64Encoded": _b64(JPEG_BYTES), "mimeType": "image/jpeg"},
        "lastFrame": {"bytesBase64Encoded": _b64(PNG_BYTES), "mimeType": "image/png"},
    }


def test_reference_images_accept_file_likes_bare_or_wrapped():
    body = _request_body(
        {
            "referenceImages": [
                _file(PNG_BYTES),
                {"image": _file(JPEG_BYTES), "referenceType": "ASSET"},
                DRESS,
            ]
        }
    )

    assert body["instances"][0]["referenceImages"] == [
        {"image": {"bytesBase64Encoded": _b64(PNG_BYTES), "mimeType": "image/png"}, "referenceType": "asset"},
        {"image": {"bytesBase64Encoded": _b64(JPEG_BYTES), "mimeType": "image/jpeg"}, "referenceType": "ASSET"},
        {"image": DRESS, "referenceType": "asset"},
    ]


@pytest.mark.parametrize(
    ("video_file", "expected_mime_type"),
    [
        pytest.param(_file(b"mp4-bytes", name="clip.mp4"), "video/mp4", id="named-mp4"),
        pytest.param(_file(b"webm-bytes", name="clip.webm"), "video/webm", id="named-webm"),
        pytest.param(_file(b"unnamed-bytes"), "video/mp4", id="unnamed-defaults-to-mp4"),
    ],
)
def test_file_like_video_keeps_its_bytes_and_mime_type(video_file, expected_mime_type):
    expected_bytes = video_file.getvalue()

    body = _request_body({"video": video_file})

    assert body["instances"][0]["video"] == {
        "bytesBase64Encoded": _b64(expected_bytes),
        "mimeType": expected_mime_type,
    }


class _NonIOBaseSeekableReader:
    def __init__(self, content: bytes) -> None:
        self._file = io.BytesIO(content)

    def read(self) -> bytes:
        return self._file.read()

    def seek(self, offset: int) -> int:
        return self._file.seek(offset)


def test_partly_read_video_wrapper_is_sent_from_the_start():
    video_file = _NonIOBaseSeekableReader(b"full-video-bytes")
    video_file.read()

    body = _request_body({"video": video_file})

    assert body["instances"][0]["video"] == {"bytesBase64Encoded": _b64(b"full-video-bytes"), "mimeType": "video/mp4"}


@pytest.mark.parametrize(
    ("model", "params"),
    [
        pytest.param(
            VEO_31, {"image": FIRST_FRAME, "lastFrame": LAST_FRAME, "durationSeconds": 4}, id="interpolation-4s"
        ),
        pytest.param(VEO_31, {"personGeneration": "allow_adult"}, id="text-to-video-allow-adult-eu"),
        pytest.param(VEO_31, {"resolution": "1080p", "durationSeconds": "8"}, id="1080p-8s-as-string"),
        pytest.param(VEO_31, {"resolution": "4k", "durationSeconds": 8}, id="4k-8s"),
        pytest.param(VEO_31, {"video": VEO_VIDEO, "resolution": "720p", "durationSeconds": 8}, id="extension-720p-8s"),
        pytest.param(VEO_31_LITE, {"image": FIRST_FRAME, "lastFrame": LAST_FRAME}, id="lite-interpolation"),
        pytest.param(VEO_31_LITE, {"resolution": "1080p", "durationSeconds": 8}, id="lite-1080p"),
        pytest.param(f"gemini/{VEO_31}", {"referenceImages": [DRESS] * 3}, id="three-reference-images"),
    ],
)
def test_documented_valid_combinations_are_not_rejected(model, params):
    body = _request_body(params, model=model)

    assert body["instances"][0]["prompt"] == "a prompt"


@pytest.mark.parametrize(
    ("model", "params", "expected_message"),
    [
        pytest.param(VEO_31, {"lastFrame": LAST_FRAME}, "lastFrame requires image", id="last-frame-without-image"),
        pytest.param(
            VEO_31,
            {"image": FIRST_FRAME, "referenceImages": [DRESS]},
            "referenceImages cannot be combined with image",
            id="reference-images-with-image",
        ),
        pytest.param(
            VEO_31,
            {"image": FIRST_FRAME, "video": VEO_VIDEO},
            "video (extension) cannot be combined with image",
            id="video-with-image",
        ),
        pytest.param(
            VEO_31,
            {"referenceImages": [DRESS] * 4},
            "at most 3 referenceImages are allowed, got 4",
            id="four-reference-images",
        ),
        pytest.param(
            VEO_31,
            {"referenceImages": [{"image": DRESS, "referenceType": "style"}]},
            "referenceType 'style' is not supported",
            id="style-reference",
        ),
        pytest.param(
            VEO_31, {"referenceImages": DRESS}, "referenceImages must be a list", id="reference-images-not-list"
        ),
        pytest.param(
            VEO_31,
            {"referenceImages": [{"referenceType": "asset", "img": DRESS}]},
            "invalid media input",
            id="reference-without-image-key",
        ),
        pytest.param(
            VEO_31,
            {"referenceImages": [DRESS], "durationSeconds": 4},
            "durationSeconds must be 8 when using referenceImages, got 4",
            id="reference-images-4s",
        ),
        pytest.param(
            VEO_31,
            {"video": VEO_VIDEO, "durationSeconds": 6},
            "durationSeconds must be 8 when using video extension, got 6",
            id="extension-6s",
        ),
        pytest.param(
            VEO_31,
            {"resolution": "1080p", "durationSeconds": 4},
            "durationSeconds must be 8 when using 1080p resolution, got 4",
            id="1080p-4s",
        ),
        pytest.param(
            VEO_31,
            {"resolution": "4K", "durationSeconds": "6"},
            "durationSeconds must be 8 when using 4k resolution, got 6",
            id="4k-6s",
        ),
        pytest.param(
            VEO_31,
            {"referenceImages": [DRESS], "durationSeconds": "eight"},
            "durationSeconds must be 8 when using referenceImages, got eight",
            id="reference-images-non-numeric-duration",
        ),
        pytest.param(
            VEO_31,
            {"resolution": "1080p", "durationSeconds": [8]},
            "durationSeconds must be 8 when using 1080p resolution, got [8]",
            id="1080p-list-duration",
        ),
        pytest.param(
            VEO_31,
            {"video": VEO_VIDEO, "resolution": "1080p", "durationSeconds": 8},
            "video extension only supports 720p resolution, got '1080p'",
            id="extension-1080p",
        ),
        pytest.param(
            VEO_31,
            {"image": FIRST_FRAME, "personGeneration": "allow_all"},
            "personGeneration must be 'allow_adult'",
            id="image-to-video-allow-all",
        ),
        pytest.param(
            VEO_31,
            {"referenceImages": [DRESS], "personGeneration": "allow_all"},
            "personGeneration must be 'allow_adult'",
            id="reference-images-allow-all",
        ),
        pytest.param(
            VEO_31,
            {"image": FIRST_FRAME, "lastFrame": "/tmp/last.png"},
            "invalid media input",
            id="last-frame-as-path-string",
        ),
    ],
)
def test_invalid_combinations_raise_bad_request(model, params, expected_message):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        _request_body(params, model=model)

    assert exc_info.value.status_code == 400
    assert expected_message in str(exc_info.value), str(exc_info.value)


def _recording_client(requests: list[dict]) -> HTTPHandler:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"name": "models/veo/operations/abc", "done": False})

    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_video_generation_sends_last_frame_in_instance():
    requests: list[dict] = []

    litellm.video_generation(
        model=f"gemini/{VEO_31}",
        prompt="a prompt",
        input_reference=FIRST_FRAME,
        lastFrame=LAST_FRAME,
        api_key="test-key",
        client=_recording_client(requests),
    )

    assert requests == [
        {"instances": [{"prompt": "a prompt", "image": FIRST_FRAME, "lastFrame": LAST_FRAME}], "parameters": {}}
    ]


def test_video_generation_rejects_invalid_request_before_calling_google():
    requests: list[dict] = []

    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.video_generation(
            model=f"gemini/{VEO_31}",
            prompt="a prompt",
            lastFrame=LAST_FRAME,
            api_key="test-key",
            client=_recording_client(requests),
        )

    assert exc_info.value.status_code == 400
    assert requests == [], "invalid request must not reach the Gemini API"
