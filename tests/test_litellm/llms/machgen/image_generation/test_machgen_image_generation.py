"""
Unit tests for the MachGen image generation provider (submit -> poll -> asset).
"""

import base64
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.machgen.common_utils import MachGenError
from litellm.llms.machgen.image_generation.handler import MachGenImageGeneration
from litellm.llms.machgen.image_generation.transformation import (
    MachGenImageGenerationConfig,
)
from litellm.types.utils import ImageResponse

MODEL = "FLUX.2-dev"
PROMPT = "an isometric reading nook"
ASSET_URL = "https://api.machgen.ai/api/v0/assets/t-abc123"


def _response(payload=None, status_code: int = 200, content: bytes = b"") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        content=None if payload is not None else content,
        request=httpx.Request("GET", "https://api.machgen.ai"),
    )


def _submit_response(task_id: str = "t-abc123") -> httpx.Response:
    return _response({"task_id": task_id})


def _task_response(status: str, **extra) -> httpx.Response:
    return _response({"task_id": "t-abc123", "status": status, **extra})


def _completed_response() -> httpx.Response:
    return _task_response("COMPLETED", task_output={"image": ASSET_URL})


def _sync_client(get_side_effect) -> MagicMock:
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = _submit_response()
    client.get.side_effect = get_side_effect
    return client


def _generate(client, optional_params=None, api_key: str = "MGA_key:secret") -> ImageResponse:
    return MachGenImageGeneration(polling_interval=0).image_generation(
        model=MODEL,
        prompt=PROMPT,
        model_response=ImageResponse(),
        optional_params=optional_params or {},
        litellm_params={},
        logging_obj=MagicMock(),
        timeout=60,
        api_key=api_key,
        client=client,
    )


class TestMachGenTransformation:
    def setup_method(self):
        self.config = MachGenImageGenerationConfig()

    def test_request_body_is_a_text_to_image_task(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt=PROMPT,
            optional_params={"width": 1024, "height": 768, "seed": 7},
            litellm_params={},
            headers={},
        )

        assert body == {
            "prompt": PROMPT,
            "model": MODEL,
            "task_type": "T2I",
            "image_config": {"width": 1024, "height": 768},
            "seed": 7,
        }

    def test_height_defaults_because_machgen_rejects_a_missing_height(self):
        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt=PROMPT,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert body["image_config"] == {"height": 1024}

    def test_size_maps_to_width_and_height(self):
        params = self.config.map_openai_params(
            non_default_params={"size": "1536x1024"},
            optional_params={},
            model=MODEL,
            drop_params=False,
        )

        assert params == {"width": 1536, "height": 1024}

    def test_invalid_size_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid size format"):
            self.config.map_openai_params(
                non_default_params={"size": "big"},
                optional_params={},
                model=MODEL,
                drop_params=False,
            )

    def test_machgen_specific_params_pass_through(self):
        params = self.config.map_openai_params(
            non_default_params={"infer_steps": 30, "guidance_scale": [5.0], "enhance_prompt": True},
            optional_params={},
            model=MODEL,
            drop_params=False,
        )

        assert params == {"infer_steps": 30, "guidance_scale": [5.0], "enhance_prompt": True}

        body = self.config.transform_image_generation_request(
            model=MODEL,
            prompt=PROMPT,
            optional_params=params,
            litellm_params={},
            headers={},
        )
        assert body["image_config"]["infer_steps"] == 30
        assert body["image_config"]["guidance_scale"] == [5.0]
        assert body["enhance_prompt"] is True

    def test_unsupported_param_raises_unless_dropped(self):
        with pytest.raises(ValueError, match="style"):
            self.config.map_openai_params(
                non_default_params={"style": "vivid"},
                optional_params={},
                model=MODEL,
                drop_params=False,
            )

        assert (
            self.config.map_openai_params(
                non_default_params={"style": "vivid"},
                optional_params={},
                model=MODEL,
                drop_params=True,
            )
            == {}
        )

    def test_validate_environment_sets_bearer_auth(self):
        headers = self.config.validate_environment(
            headers={"x-trace": "1"},
            model=MODEL,
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="MGA_key:secret",
        )

        assert headers["Authorization"] == "Bearer MGA_key:secret"
        assert headers["x-trace"] == "1"

    def test_validate_environment_requires_a_key(self, monkeypatch):
        monkeypatch.delenv("MACHGEN_API_KEY", raising=False)
        with pytest.raises(MachGenError, match="MACHGEN_API_KEY"):
            self.config.validate_environment(
                headers={},
                model=MODEL,
                messages=[],
                optional_params={},
                litellm_params={},
            )

    def test_api_base_is_configurable(self):
        assert (
            self.config.get_complete_url(
                api_base="https://machgen.internal/",
                api_key="MGA_key:secret",
                model=MODEL,
                optional_params={},
                litellm_params={},
            )
            == "https://machgen.internal/api/v0/generate"
        )


class TestMachGenHandler:
    def test_polls_until_the_task_completes(self):
        client = _sync_client([_task_response("PENDING"), _task_response("RUNNING"), _completed_response()])

        response = _generate(client, optional_params={"height": 1024})

        assert [image.url for image in response.data] == [ASSET_URL]
        assert client.post.call_args.kwargs["url"] == "https://api.machgen.ai/api/v0/generate"
        assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer MGA_key:secret"
        assert client.get.call_count == 3
        assert all(
            call.kwargs["url"] == "https://api.machgen.ai/api/v0/tasks/t-abc123" for call in client.get.call_args_list
        )

    def test_task_id_is_url_escaped(self):
        client = _sync_client([_completed_response()])
        client.post.return_value = _submit_response("../../admin/keys")

        _generate(client)

        assert client.get.call_args.kwargs["url"] == "https://api.machgen.ai/api/v0/tasks/..%2F..%2Fadmin%2Fkeys"

    def test_failed_task_surfaces_the_provider_error(self):
        client = _sync_client([_task_response("FAILED", error_msg="prompt blocked by moderation")])

        with pytest.raises(MachGenError, match="prompt blocked by moderation"):
            _generate(client)

    def test_completed_task_without_an_asset_raises(self):
        client = _sync_client([_task_response("COMPLETED", task_output={})])

        with pytest.raises(MachGenError, match="without an image asset"):
            _generate(client)

    def test_missing_task_id_raises(self):
        client = _sync_client([_completed_response()])
        client.post.return_value = _response({})

        with pytest.raises(MachGenError, match="No task_id"):
            _generate(client)

    def test_b64_json_downloads_the_authenticated_asset(self):
        asset = _response(content=b"image-bytes")
        client = _sync_client([_completed_response(), asset])

        response = _generate(client, optional_params={"response_format": "b64_json"})

        assert response.data[0].b64_json == base64.b64encode(b"image-bytes").decode("utf-8")
        assert response.data[0].url is None
        download_call = client.get.call_args_list[-1]
        assert download_call.kwargs["url"] == ASSET_URL
        assert download_call.kwargs["headers"]["Authorization"] == "Bearer MGA_key:secret"

    def test_litellm_image_generation_routes_machgen_models(self):
        import litellm

        client = _sync_client([_completed_response()])

        response = litellm.image_generation(
            model=f"machgen/{MODEL}",
            prompt=PROMPT,
            size="1024x1024",
            api_key="MGA_key:secret",
            client=client,
        )

        assert [image.url for image in response.data] == [ASSET_URL]
        assert client.post.call_args.kwargs["json"]["model"] == MODEL
        assert client.post.call_args.kwargs["json"]["image_config"] == {"width": 1024, "height": 1024}

    @pytest.mark.asyncio
    async def test_async_generation_polls_and_returns_the_asset(self):
        client = MagicMock(spec=AsyncHTTPHandler)
        client.post = AsyncMock(return_value=_submit_response())
        client.get = AsyncMock(side_effect=[_task_response("RUNNING"), _completed_response()])

        response = await MachGenImageGeneration(polling_interval=0).async_image_generation(
            model=MODEL,
            prompt=PROMPT,
            model_response=ImageResponse(),
            optional_params={},
            litellm_params={},
            logging_obj=MagicMock(),
            timeout=60,
            api_key="MGA_key:secret",
            client=client,
        )

        assert [image.url for image in response.data] == [ASSET_URL]
        assert client.get.await_count == 2
