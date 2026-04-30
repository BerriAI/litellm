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
