import io

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


def test_video_edit_request_forwards_uploaded_file_as_multipart():
    """An uploaded source video must leave as a multipart ``video`` file part,
    not be dropped in favor of a JSON id reference."""
    config = OpenAIVideoConfig()
    source = io.BytesIO(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isomBODY")
    source.name = "clip.mp4"

    url, data, files = config.transform_video_edit_request(
        prompt="make it nighttime",
        video_id="",
        api_base="https://api.openai.com/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
        video_file=source,
    )

    assert url == "https://api.openai.com/v1/videos/edits"
    assert data == {"prompt": "make it nighttime"}
    assert files is not None
    field_names = [field for field, _ in files]
    assert field_names == ["video"]
    _, (filename, content, content_type) = files[0]
    assert filename == "clip.mp4"
    assert content is source
    assert content_type == "video/mp4"


def test_video_edit_request_without_file_sends_json_id_reference():
    """The id-reference path must stay JSON (files is None) so existing
    remix/edit-by-id callers keep working."""
    config = OpenAIVideoConfig()

    url, data, files = config.transform_video_edit_request(
        prompt="brighter",
        video_id="video_abc123",
        api_base="https://api.openai.com/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://api.openai.com/v1/videos/edits"
    assert data == {"prompt": "brighter", "video": {"id": "video_abc123"}}
    assert files is None
