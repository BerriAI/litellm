import io
import os

import httpx

from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import encode_character_id_with_provider


def test_video_content_request_encodes_video_id_path_segment():
    config = OpenAIVideoConfig()

    url, params = config.transform_video_content_request(
        video_id="../../responses?x=1#frag",
        api_base="https://api.openai.com/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        url
        == "https://api.openai.com/v1/videos/..%2F..%2Fresponses%3Fx%3D1%23frag/content"
    )
    assert params == {}


def test_video_content_request_encodes_variant_query_param():
    """``variant`` is user-controlled and was previously interpolated raw
    into the query string.  A value like ``thumbnail&extra=1`` would
    inject additional query parameters into the upstream request."""
    config = OpenAIVideoConfig()

    url, _ = config.transform_video_content_request(
        video_id="vid_123",
        api_base="https://api.openai.com/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
        variant="thumbnail&extra=1",
    )

    # ``&`` and ``=`` must be percent-encoded so they cannot terminate
    # the ``variant`` value or open a new query parameter.
    assert "?variant=thumbnail%26extra%3D1" in url
    # Sanity: the legitimate "thumbnail" value still round-trips cleanly.
    url2, _ = config.transform_video_content_request(
        video_id="vid_123",
        api_base="https://api.openai.com/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
        variant="thumbnail",
    )
    assert url2.endswith("?variant=thumbnail")


def test_wrapped_character_id_is_decoded_then_encoded_as_path_segment():
    config = OpenAIVideoConfig()
    character_id = encode_character_id_with_provider(
        "../../characters?x=1#frag",
        provider="openai",
        model_id="sora",
    )

    url, params = config.transform_video_get_character_request(
        character_id=character_id,
        api_base="https://api.openai.com/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert (
        url
        == "https://api.openai.com/v1/videos/characters/..%2F..%2Fcharacters%3Fx%3D1%23frag"
    )
    assert params == {}


def test_create_character_request_falls_back_when_video_name_is_not_a_string():
    """A file opened from a raw descriptor exposes an ``int`` ``name``.  That value
    used to be forwarded straight into ``mimetypes.guess_type()``, which raises,
    so the multipart filename now falls back to the default whenever ``name`` is
    not a usable string."""
    config = OpenAIVideoConfig()
    fd = os.open(__file__, os.O_RDONLY)
    video = io.open(fd, "rb")

    try:
        assert not isinstance(video.name, str)
        _, files_list = config.transform_video_create_character_request(
            name="hero",
            video=video,
            api_base="https://api.openai.com/v1/videos",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
    finally:
        video.close()

    video_entry = next(entry for entry in files_list if entry[0] == "video")
    filename, payload, content_type = video_entry[1]
    assert filename == "input_video.mp4"
    assert payload is video
    assert content_type == "video/mp4"


def test_create_character_request_uses_file_name_when_available(tmp_path):
    config = OpenAIVideoConfig()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    with open(video_path, "rb") as video:
        _, files_list = config.transform_video_create_character_request(
            name="hero",
            video=video,
            api_base="https://api.openai.com/v1/videos",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    video_entry = next(entry for entry in files_list if entry[0] == "video")
    assert video_entry[1][0] == str(video_path)
    assert video_entry[1][2] == "video/mp4"


def test_video_create_response_parses_payload_with_unknown_fields():
    config = OpenAIVideoConfig()
    raw_response = httpx.Response(
        200,
        json={
            "id": "video_123",
            "object": "video",
            "status": "completed",
            "seconds": "8",
            "unexpected_upstream_field": {"nested": True},
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/videos"),
    )

    video_obj = config.transform_video_create_response(
        model="sora-2",
        raw_response=raw_response,
        logging_obj=None,
    )

    assert video_obj.id == "video_123"
    assert video_obj.seconds == "8"
    assert video_obj.usage == {"duration_seconds": 8.0}
