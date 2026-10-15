import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_xai_image_edit_config():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    config = ProviderConfigManager.get_provider_image_edit_config(
        model="grok-imagine-image",
        provider=LlmProviders.XAI,
    )
    assert isinstance(config, XAIImageEditConfig)


def test_get_complete_url_default():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    url = XAIImageEditConfig().get_complete_url(
        model="grok-imagine-image",
        api_base=None,
        litellm_params={},
    )
    assert url.endswith("/v1/images/edits")


def test_uses_json_not_multipart():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    assert XAIImageEditConfig().use_multipart_form_data() is False


def test_map_size_to_aspect_ratio():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    mapped = XAIImageEditConfig().map_openai_params(
        image_edit_optional_params={"size": "1024x1792", "n": 1},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["aspect_ratio"] == "9:16"
    assert mapped["n"] == 1
    assert "size" not in mapped


def test_validate_environment_requires_credentials():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    with pytest.raises(Exception, match="Missing xAI credentials"):
        XAIImageEditConfig().validate_environment(
            headers={},
            model="grok-imagine-image",
            api_key=None,
            litellm_params={},
        )


def test_validate_environment_oauth_injects_bearer():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    with patch(
        "litellm.llms.xai.oauth.XAIOAuthAuthenticator.get_access_token",
        return_value="oauth-token",
    ):
        headers = XAIImageEditConfig().validate_environment(
            headers={},
            model="grok-imagine-image",
            api_key=None,
            litellm_params={"use_xai_oauth": True},
        )
    assert headers["Authorization"] == "Bearer oauth-token"


def test_map_string_n_to_int():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    mapped = XAIImageEditConfig().map_openai_params(
        image_edit_optional_params={"n": "1"},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["n"] == 1
    assert isinstance(mapped["n"], int)


def test_transform_string_n_to_int():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    data, _ = XAIImageEditConfig().transform_image_edit_request(
        model="xai/grok-imagine-image",
        prompt="make the cube red",
        image="https://imgen.x.ai/source.jpeg",
        image_edit_optional_request_params={"n": "1"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert data["n"] == 1
    assert isinstance(data["n"], int)


def test_transform_bytes_to_data_uri_and_response():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    config = XAIImageEditConfig()
    data, files = config.transform_image_edit_request(
        model="xai/grok-imagine-image",
        prompt="make the cube red",
        image=b"\xff\xd8\xfffakejpeg",
        image_edit_optional_request_params={"aspect_ratio": "1:1"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
    assert data["model"] == "grok-imagine-image"
    assert data["prompt"] == "make the cube red"
    assert data["aspect_ratio"] == "1:1"
    assert data["image"]["url"].startswith("data:image/jpeg;base64,")

    raw = httpx.Response(
        200,
        json={"data": [{"url": "https://imgen.x.ai/edited.jpeg", "mime_type": "image/jpeg"}]},
    )
    response = config.transform_image_edit_response(
        model="grok-imagine-image",
        raw_response=raw,
        logging_obj=MagicMock(),
    )
    assert response.data is not None
    assert response.data[0].url == "https://imgen.x.ai/edited.jpeg"


def test_transform_duck_typed_reader_to_data_uri():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    class Reader:
        def read(self) -> bytes:
            return b"\x89PNG\r\n\x1a\nfakepng"

    data, _ = XAIImageEditConfig().transform_image_edit_request(
        model="grok-imagine-image",
        prompt="make it night",
        image=Reader(),
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert data["image"]["url"] == "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode()


def test_transform_http_url_passthrough():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    data, files = XAIImageEditConfig().transform_image_edit_request(
        model="grok-imagine-image",
        prompt="make it night",
        image="https://imgen.x.ai/source.jpeg",
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
    assert data["image"] == {"url": "https://imgen.x.ai/source.jpeg"}


def test_transform_multiple_images_uses_images_array():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    data, _ = XAIImageEditConfig().transform_image_edit_request(
        model="grok-imagine-image",
        prompt="combine styles",
        image=["https://imgen.x.ai/a.jpeg", "https://imgen.x.ai/b.jpeg"],
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "image" not in data
    assert data["images"] == [
        {"url": "https://imgen.x.ai/a.jpeg"},
        {"url": "https://imgen.x.ai/b.jpeg"},
    ]


def _edit_request(image, optional_params=None):
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    data, _ = XAIImageEditConfig().transform_image_edit_request(
        model="grok-imagine-image",
        prompt="make it night",
        image=image,
        image_edit_optional_request_params=optional_params or {},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    return data


def test_validate_environment_api_key_sets_bearer_and_keeps_caller_content_type():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    headers = XAIImageEditConfig().validate_environment(
        headers={"Content-Type": "application/custom"},
        model="grok-imagine-image",
        api_key="fake-key",
        litellm_params={},
    )

    assert headers == {"Content-Type": "application/custom", "Authorization": "Bearer fake-key"}


def test_map_rejects_unknown_param_unless_dropped():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    config = XAIImageEditConfig()
    with pytest.raises(ValueError, match="Parameter quality is not supported"):
        config.map_openai_params(
            image_edit_optional_params={"quality": "hd", "n": 2},
            model="grok-imagine-image",
            drop_params=False,
        )

    assert config.map_openai_params(
        image_edit_optional_params={"quality": "hd", "n": 2, "resolution": "2k"},
        model="grok-imagine-image",
        drop_params=True,
    ) == {"n": 2, "resolution": "2k"}


def test_get_complete_url_appends_v1_to_a_bare_custom_base():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    url = XAIImageEditConfig().get_complete_url(
        model="grok-imagine-image",
        api_base="https://xai-gateway.example.com/",
        litellm_params={"api_key": "fake-key"},
    )

    assert url == "https://xai-gateway.example.com/v1/images/edits"


@pytest.mark.parametrize("image", [None, [], [None]])
def test_transform_rejects_missing_reference_image(image):
    with pytest.raises(ValueError, match="requires at least one reference image"):
        _edit_request(image)


@pytest.mark.parametrize(
    "image,expected",
    [
        ({"url": "https://imgen.x.ai/a.jpeg"}, {"url": "https://imgen.x.ai/a.jpeg"}),
        ({"file_id": "file-123"}, {"file_id": "file-123"}),
    ],
)
def test_transform_passes_url_and_file_id_references_through(image, expected):
    assert _edit_request(image)["image"] == expected


@pytest.mark.parametrize(
    "image",
    [
        bytearray(b"\x89PNG\r\n\x1a\nfakepng"),
        BytesIO(b"\x89PNG\r\n\x1a\nfakepng"),
    ],
)
def test_transform_encodes_binary_inputs_as_data_uri(image):
    expected = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode()

    assert _edit_request(image)["image"]["url"] == expected


def test_transform_reads_a_partially_consumed_stream_from_the_start_and_restores_position():
    stream = BytesIO(b"\x89PNG\r\n\x1a\nfakepng")
    stream.seek(4)

    url = _edit_request(stream)["image"]["url"]

    assert url.endswith(base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode())
    assert stream.tell() == 4


def test_transform_encodes_text_reader_as_utf8():
    class TextReader:
        def read(self) -> str:
            return "plain-text-bytes"

    url = _edit_request(TextReader())["image"]["url"]

    assert url.endswith(base64.b64encode(b"plain-text-bytes").decode())


def test_transform_rejects_unsupported_image_type():
    with pytest.raises(ValueError, match="Unsupported image input type"):
        _edit_request(12345)


@pytest.mark.parametrize(
    "raw",
    [
        httpx.Response(502, text="<html>bad gateway</html>"),
        httpx.Response(200, json={"data": []}),
    ],
)
def test_transform_response_raises_provider_error_without_images(raw):
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    with pytest.raises(BaseLLMException) as exc_info:
        XAIImageEditConfig().transform_image_edit_response(
            model="grok-imagine-image",
            raw_response=raw,
            logging_obj=MagicMock(),
        )

    assert exc_info.value.status_code == raw.status_code
