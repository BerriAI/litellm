import io
import os

import httpx
import pytest

from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    extract_original_video_id,
)


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


def _video_payload(video_id: str = "video_123", **overrides: object) -> dict:
    payload = {
        "id": video_id,
        "object": "video",
        "status": "completed",
        "seconds": "8",
        "size": "1280x720",
        "unexpected_upstream_field": {"nested": True},
    }
    payload.update(overrides)
    return payload


def _json_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "https://api.openai.com/v1/videos"),
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "transform_video_remix_response",
        "transform_video_status_retrieve_response",
        "transform_video_edit_response",
        "transform_video_extension_response",
    ],
)
def test_video_response_transforms_parse_payload_and_wrap_id(method_name):
    """Every /videos response transform parses the upstream body into a VideoObject
    and swaps in a LiteLLM-managed id that decodes back to the provider's id."""
    config = OpenAIVideoConfig()
    transform = getattr(config, method_name)

    video_obj = transform(
        raw_response=_json_response(_video_payload()),
        logging_obj=None,
        custom_llm_provider="openai",
    )

    assert video_obj.status == "completed"
    assert video_obj.seconds == "8"
    assert video_obj.size == "1280x720"
    assert video_obj.id != "video_123"
    assert extract_original_video_id(video_obj.id) == "video_123"
    assert decode_video_id_with_provider(video_obj.id)["custom_llm_provider"] == "openai"


@pytest.mark.parametrize(
    "method_name",
    [
        "transform_video_remix_response",
        "transform_video_status_retrieve_response",
        "transform_video_edit_response",
        "transform_video_extension_response",
    ],
)
def test_video_response_transforms_leave_id_raw_without_provider(method_name):
    config = OpenAIVideoConfig()
    transform = getattr(config, method_name)

    video_obj = transform(
        raw_response=_json_response(_video_payload()),
        logging_obj=None,
    )

    assert video_obj.id == "video_123"


def test_video_delete_response_parses_payload_without_wrapping_id():
    """Delete returns the deleted object as-is; it must still parse, and it must not
    re-encode the id (the caller already holds the managed one)."""
    config = OpenAIVideoConfig()

    video_obj = config.transform_video_delete_response(
        raw_response=_json_response(_video_payload(status="deleted")),
        logging_obj=None,
    )

    assert video_obj.id == "video_123"
    assert video_obj.status == "deleted"


@pytest.mark.parametrize(
    "method_name",
    ["transform_video_create_character_response", "transform_video_get_character_response"],
)
def test_character_response_transforms_parse_payload(method_name):
    config = OpenAIVideoConfig()
    transform = getattr(config, method_name)

    character = transform(
        raw_response=_json_response(
            {
                "id": "char_123",
                "object": "character",
                "created_at": 1700000000,
                "name": "hero",
                "unexpected_upstream_field": "ignored",
            }
        ),
        logging_obj=None,
    )

    assert character.id == "char_123"
    assert character.name == "hero"
    assert character.created_at == 1700000000
