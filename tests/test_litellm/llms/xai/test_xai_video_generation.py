from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.litellm_core_utils.url_utils import SSRFError
from litellm.llms.xai.videos.transformation import XAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_xai_video_config():
    config = ProviderConfigManager.get_provider_video_config(
        model="grok-imagine-video",
        provider=LlmProviders.XAI,
    )
    assert isinstance(config, XAIVideoConfig)


def test_map_seconds_and_size():
    mapped = XAIVideoConfig().map_openai_params(
        video_create_optional_params={"seconds": "10", "size": "1280x720"},
        model="grok-imagine-video",
        drop_params=True,
    )
    assert mapped["duration"] == 10
    assert mapped["aspect_ratio"] == "16:9"


def test_map_seconds_nine_is_not_clamped_to_six():
    mapped = XAIVideoConfig().map_openai_params(
        video_create_optional_params={"seconds": "9"},
        model="grok-imagine-video-1.5",
        drop_params=True,
    )
    assert mapped["duration"] == 9
    assert "seconds" not in mapped


def test_get_complete_url_create_and_status_root():
    config = XAIVideoConfig()
    create_url = config.get_complete_url(
        model="grok-imagine-video",
        api_base="https://api.x.ai/v1",
        litellm_params={},
    )
    assert create_url == "https://api.x.ai/v1/videos/generations"

    status_root = config.get_complete_url(
        model="",
        api_base="https://api.x.ai/v1",
        litellm_params={},
    )
    assert status_root == "https://api.x.ai/v1"


def test_validate_environment_oauth_injects_bearer():
    with patch(
        "litellm.llms.xai.oauth.XAIOAuthAuthenticator.get_access_token",
        return_value="oauth-token",
    ):
        headers = XAIVideoConfig().validate_environment(
            headers={},
            model="grok-imagine-video",
            api_key=None,
            litellm_params=GenericLiteLLMParams(use_xai_oauth=True),
        )
    assert headers["Authorization"] == "Bearer oauth-token"


def test_transform_create_and_status_response():
    config = XAIVideoConfig()
    data, files, api_base = config.transform_video_create_request(
        model="xai/grok-imagine-video",
        prompt="a cat walking",
        api_base="https://api.x.ai/v1/videos/generations",
        video_create_optional_request_params={"duration": 6},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert data["model"] == "grok-imagine-video"
    assert data["prompt"] == "a cat walking"
    assert data["duration"] == 6
    assert files == []

    created = config.transform_video_create_response(
        model="grok-imagine-video",
        raw_response=httpx.Response(200, json={"request_id": "req-123"}),
        logging_obj=MagicMock(),
        custom_llm_provider="xai",
    )
    assert created.status == "processing"
    assert created.id

    status_url, params = config.transform_video_status_retrieve_request(
        video_id=created.id,
        api_base="https://api.x.ai/v1",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert status_url.endswith("/videos/req-123")
    assert params == {}

    status = config.transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={
                "status": "done",
                "request_id": "req-123",
                "video": {"url": "https://vidgen.x.ai/x.mp4", "duration": 6},
                "progress": 100,
                "model": "grok-imagine-video",
            },
        ),
        logging_obj=MagicMock(),
        custom_llm_provider="xai",
    )
    assert status.status == "completed"
    assert status.seconds == "6"
    assert status._hidden_params.get("video_url") == "https://vidgen.x.ai/x.mp4"


def test_validate_environment_requires_credentials():
    with pytest.raises(Exception, match="Missing xAI credentials"):
        XAIVideoConfig().validate_environment(
            headers={},
            model="grok-imagine-video",
            api_key=None,
            litellm_params=GenericLiteLLMParams(),
        )


def test_content_request_is_get_status():
    url, params = XAIVideoConfig().transform_video_content_request(
        video_id="req-123",
        api_base="https://api.x.ai/v1",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert url == "https://api.x.ai/v1/videos/req-123"
    assert params == {}


@pytest.mark.parametrize("video_id", ["..", ""])
@pytest.mark.parametrize(
    "transform_name",
    ["transform_video_status_retrieve_request", "transform_video_content_request"],
)
def test_video_id_rejects_empty_and_dot_path_segments(video_id, transform_name):
    transform = getattr(XAIVideoConfig(), transform_name)
    with pytest.raises(ValueError, match="video_id"):
        transform(
            video_id=video_id,
            api_base="https://api.x.ai/v1",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


@pytest.mark.parametrize(
    "transform_name",
    ["transform_video_status_retrieve_request", "transform_video_content_request"],
)
def test_video_id_parent_path_is_one_encoded_segment(transform_name):
    transform = getattr(XAIVideoConfig(), transform_name)
    url, params = transform(
        video_id="../models",
        api_base="https://api.x.ai/v1",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert url == "https://api.x.ai/v1/videos/..%2Fmodels"
    assert "/videos/../" not in url
    assert params == {}


def test_video_id_is_percent_encoded_as_one_segment():
    url, params = XAIVideoConfig().transform_video_status_retrieve_request(
        video_id="req-123?x=1#frag",
        api_base="https://api.x.ai/v1",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert url == "https://api.x.ai/v1/videos/req-123%3Fx%3D1%23frag"
    assert params == {}


def test_content_response_fetches_cdn_via_shared_client():
    status = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"status": "done", "video": {"url": "https://vidgen.x.ai/x.mp4"}},
    )
    cdn = MagicMock()
    video_resp = MagicMock()
    video_resp.content = b"mp4-bytes"
    video_resp.raise_for_status.return_value = None
    with patch(
        "litellm.llms.xai.videos.transformation._get_httpx_client",
        return_value=cdn,
    ), patch(
        "litellm.llms.xai.videos.transformation.safe_get",
        return_value=video_resp,
    ) as safe_get:
        body = XAIVideoConfig().transform_video_content_response(status, logging_obj=MagicMock())
    assert body == b"mp4-bytes"
    safe_get.assert_called_once_with(cdn, "https://vidgen.x.ai/x.mp4")


@pytest.mark.asyncio
async def test_async_content_response_does_not_use_sync_client():
    status = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"status": "done", "video": {"url": "https://vidgen.x.ai/x.mp4"}},
    )
    async_client = MagicMock()
    video_resp = MagicMock()
    video_resp.content = b"async-mp4"
    video_resp.raise_for_status.return_value = None
    with patch(
        "litellm.llms.xai.videos.transformation.get_async_httpx_client",
        return_value=async_client,
    ), patch(
        "litellm.llms.xai.videos.transformation._get_httpx_client",
    ) as sync_client, patch(
        "litellm.llms.xai.videos.transformation.async_safe_get",
        new=AsyncMock(return_value=video_resp),
    ) as async_safe_get:
        body = await XAIVideoConfig().async_transform_video_content_response(
            status, logging_obj=MagicMock()
        )
    assert body == b"async-mp4"
    sync_client.assert_not_called()
    async_safe_get.assert_awaited_once_with(async_client, "https://vidgen.x.ai/x.mp4")


def test_content_response_raises_when_status_has_no_url():
    status = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"status": "pending"},
    )
    with pytest.raises(ValueError, match="not ready"):
        XAIVideoConfig().transform_video_content_response(status, logging_obj=MagicMock())


def test_content_response_returns_raw_bytes_when_not_json():
    raw = httpx.Response(
        200,
        headers={"content-type": "video/mp4"},
        content=b"already-mp4",
    )
    assert XAIVideoConfig().transform_video_content_response(raw, logging_obj=MagicMock()) == b"already-mp4"


def test_content_response_rejects_internal_cdn_host():
    status = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"status": "done", "video": {"url": "http://127.0.0.1/secret.mp4"}},
    )

    def boom(*args, **kwargs):
        raise AssertionError("unsafe CDN fetch must not run")

    with patch(
        "litellm.llms.xai.videos.transformation._get_httpx_client",
        return_value=MagicMock(get=boom),
    ):
        with pytest.raises((SSRFError, ValueError)):
            XAIVideoConfig().transform_video_content_response(status, logging_obj=MagicMock())


def test_content_response_fetches_public_cdn_via_safe_get():
    status = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"status": "done", "video": {"url": "https://vidgen.x.ai/x.mp4"}},
    )
    video_resp = MagicMock()
    video_resp.content = b"safe-mp4"
    video_resp.raise_for_status.return_value = None
    with patch(
        "litellm.llms.xai.videos.transformation.safe_get",
        return_value=video_resp,
        create=True,
    ) as safe_get:
        body = XAIVideoConfig().transform_video_content_response(status, logging_obj=MagicMock())
    assert body == b"safe-mp4"
    assert safe_get.call_count == 1
    assert safe_get.call_args.args[1] == "https://vidgen.x.ai/x.mp4"


@pytest.mark.asyncio
async def test_async_content_response_rejects_internal_cdn_host():
    status = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"status": "done", "video": {"url": "http://127.0.0.1/secret.mp4"}},
    )

    async def boom(*args, **kwargs):
        raise AssertionError("unsafe async CDN fetch must not run")

    with patch(
        "litellm.llms.xai.videos.transformation.get_async_httpx_client",
        return_value=MagicMock(get=boom),
    ):
        with pytest.raises((SSRFError, ValueError)):
            await XAIVideoConfig().async_transform_video_content_response(
                status, logging_obj=MagicMock()
            )


@pytest.mark.asyncio
async def test_async_content_response_fetches_public_cdn_via_async_safe_get():
    status = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"status": "done", "video": {"url": "https://vidgen.x.ai/x.mp4"}},
    )
    video_resp = MagicMock()
    video_resp.content = b"async-safe-mp4"
    video_resp.raise_for_status.return_value = None
    with patch(
        "litellm.llms.xai.videos.transformation.async_safe_get",
        new=AsyncMock(return_value=video_resp),
        create=True,
    ) as async_safe_get:
        body = await XAIVideoConfig().async_transform_video_content_response(
            status, logging_obj=MagicMock()
        )
    assert body == b"async-safe-mp4"
    async_safe_get.assert_awaited()
    assert async_safe_get.call_args.args[1] == "https://vidgen.x.ai/x.mp4"
