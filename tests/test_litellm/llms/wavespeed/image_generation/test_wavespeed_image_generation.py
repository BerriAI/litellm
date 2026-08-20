"""Unit tests for the WaveSpeed AI image generation submit/poll flow."""

from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm

from litellm.llms.wavespeed.common_utils import WaveSpeedError
from litellm.llms.wavespeed.image_generation.handler import WaveSpeedImageGeneration
from litellm.llms.wavespeed.image_generation.transformation import (
    WaveSpeedImageGenerationConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

MODEL = "wavespeed-ai/z-image/turbo"
SUBMIT_URL = f"https://api.wavespeed.ai/api/v3/{MODEL}"
RESULT_URL = "https://api.wavespeed.ai/api/v3/predictions/pred-123/result"
OUTPUT_URL = "https://cdn.wavespeed.ai/pred-123.png"


def envelope(data):
    return {"code": 200, "message": "success", "data": data}


def prediction(status, **extra):
    return envelope({"id": "pred-123", "model": MODEL, "status": status, **extra})


@pytest.fixture(autouse=True)
def mocked_transport(monkeypatch):
    """No test here may reach the network: respx only intercepts httpx, so pin httpx transport."""
    monkeypatch.setattr("litellm.llms.wavespeed.image_generation.handler.DEFAULT_POLLING_INTERVAL", 0)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


@pytest.fixture
def generate():
    handler = WaveSpeedImageGeneration()

    def run():
        return handler.image_generation(
            model=MODEL,
            prompt="a red panda",
            model_response=ImageResponse(),
            optional_params={},
            litellm_params={"api_key": "sk-test", "api_base": None},
            logging_obj=MagicMock(),
            timeout=None,
        )

    return run


@respx.mock
def test_submit_then_poll_until_completed(generate):
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(
        side_effect=[
            httpx.Response(200, json=prediction("processing")),
            httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
        ]
    )

    response = generate()

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert submit.call_count == 1
    assert poll.call_count == 2
    assert submit.calls[0].request.headers["authorization"] == "Bearer sk-test"
    assert submit.calls[0].request.headers["x-client-name"] == "litellm"


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout"])
@respx.mock
def test_terminal_failure_status_raises(generate, status):
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction(status, error="nsfw content")))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert status in str(exc_info.value)
    assert "nsfw content" in str(exc_info.value)


@respx.mock
def test_submit_is_issued_exactly_once_when_polling_fails(generate):
    """A submission is a billable task, so a poll failure must never re-submit it."""
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(side_effect=httpx.ConnectError("connection reset"))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert submit.call_count == 1
    assert poll.call_count == 5
    assert "5 times in a row" in str(exc_info.value)


@respx.mock
def test_transient_poll_failures_are_tolerated(generate):
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(
        side_effect=[
            httpx.ConnectError("connection reset"),
            httpx.ConnectError("connection reset"),
            httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
        ]
    )

    response = generate()

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert submit.call_count == 1


@respx.mock
def test_non_200_envelope_code_raises(generate):
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json={"code": 401, "message": "invalid api key"}))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert "invalid api key" in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_async_submit_then_poll_until_completed():
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(
        side_effect=[
            httpx.Response(200, json=prediction("processing")),
            httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
        ]
    )

    response = await WaveSpeedImageGeneration().async_image_generation(
        model=MODEL,
        prompt="a red panda",
        model_response=ImageResponse(),
        optional_params={},
        litellm_params={"api_key": "sk-test", "api_base": None},
        logging_obj=MagicMock(),
        timeout=None,
    )

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert submit.call_count == 1


class TestWaveSpeedImageGenerationConfig:
    def setup_method(self):
        self.config = WaveSpeedImageGenerationConfig()

    def test_size_is_mapped_to_wavespeed_format(self):
        assert self.config.map_openai_params({"size": "1024x1536"}, {}, MODEL, False) == {"size": "1024*1536"}

    def test_invalid_size_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid size format"):
            self.config.map_openai_params({"size": "huge"}, {}, MODEL, False)

    def test_n_greater_than_one_maps_to_num_images(self):
        assert self.config.map_openai_params({"n": 4}, {}, MODEL, False) == {"num_images": 4}
        assert self.config.map_openai_params({"n": 1}, {}, MODEL, False) == {}

    def test_b64_response_format_is_rejected_unless_dropped(self):
        with pytest.raises(ValueError, match="response_format"):
            self.config.map_openai_params({"response_format": "b64_json"}, {}, MODEL, False)
        assert self.config.map_openai_params({"response_format": "b64_json"}, {}, MODEL, True) == {}

    def test_model_specific_params_pass_through(self):
        assert self.config.map_openai_params({"guidance_scale": 3.5}, {}, MODEL, False) == {"guidance_scale": 3.5}

    def test_submit_url_keeps_multi_segment_model_ids(self):
        assert (
            self.config.get_complete_url(None, "sk-test", "bytedance/seedance-2.5/text-to-video", {}, {})
            == "https://api.wavespeed.ai/api/v3/bytedance/seedance-2.5/text-to-video"
        )

    def test_api_base_override(self):
        assert (
            self.config.get_complete_url("https://proxy.internal/", "sk-test", MODEL, {}, {})
            == f"https://proxy.internal/api/v3/{MODEL}"
        )

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
        with pytest.raises(WaveSpeedError, match="WAVESPEED_API_KEY"):
            self.config.validate_environment({}, MODEL, [], {}, {})

    def test_completed_prediction_without_outputs_raises(self):
        raw = httpx.Response(200, json=prediction("completed", outputs=[]))
        with pytest.raises(WaveSpeedError, match="without any outputs"):
            self.config.transform_image_generation_response(
                model=MODEL,
                raw_response=raw,
                model_response=ImageResponse(),
                logging_obj=MagicMock(),
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )


@pytest.fixture
def zero_poll_budget(monkeypatch):
    """Make the polling deadline expire immediately so the timeout path is reachable."""
    monkeypatch.setattr("litellm.llms.wavespeed.image_generation.handler.DEFAULT_MAX_POLLING_TIME", 0)


@respx.mock
def test_sync_poll_timeout(generate, zero_poll_budget):
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction("processing")))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert exc_info.value.status_code == 408
    assert "did not finish within" in str(exc_info.value)
    assert submit.call_count == 1
    assert poll.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_async_poll_timeout(zero_poll_budget):
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))

    with pytest.raises(WaveSpeedError) as exc_info:
        await WaveSpeedImageGeneration().async_image_generation(
            model=MODEL,
            prompt="a red panda",
            model_response=ImageResponse(),
            optional_params={},
            litellm_params={"api_key": "sk-test", "api_base": None},
            logging_obj=MagicMock(),
            timeout=None,
        )

    assert exc_info.value.status_code == 408


@pytest.mark.asyncio
@respx.mock
async def test_async_submit_is_issued_exactly_once_when_polling_fails():
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(side_effect=httpx.ConnectError("connection reset"))

    with pytest.raises(WaveSpeedError) as exc_info:
        await WaveSpeedImageGeneration().async_image_generation(
            model=MODEL,
            prompt="a red panda",
            model_response=ImageResponse(),
            optional_params={},
            litellm_params={"api_key": "sk-test", "api_base": None},
            logging_obj=MagicMock(),
            timeout=None,
        )

    assert submit.call_count == 1
    assert poll.call_count == 5
    assert "5 times in a row" in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_aimg_generation_flag_dispatches_to_the_async_path():
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])))

    pending = WaveSpeedImageGeneration().image_generation(
        model=MODEL,
        prompt="a red panda",
        model_response=ImageResponse(),
        optional_params={},
        litellm_params={"api_key": "sk-test", "api_base": None},
        logging_obj=MagicMock(),
        timeout=None,
        aimg_generation=True,
    )

    response = await pending
    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert submit.call_count == 1


@respx.mock
def test_litellm_params_object_is_accepted(monkeypatch):
    """images/main.py can hand the handler a GenericLiteLLMParams rather than a dict."""
    monkeypatch.delenv("WAVESPEED_API_BASE", raising=False)
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])))

    response = WaveSpeedImageGeneration().image_generation(
        model=MODEL,
        prompt="a red panda",
        model_response=ImageResponse(),
        optional_params={},
        litellm_params=GenericLiteLLMParams(api_key="sk-test"),
        logging_obj=MagicMock(),
        timeout=None,
    )

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert submit.calls[0].request.headers["authorization"] == "Bearer sk-test"


@respx.mock
def test_extra_headers_are_merged_and_cannot_be_dropped(generate):
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])))

    WaveSpeedImageGeneration().image_generation(
        model=MODEL,
        prompt="a red panda",
        model_response=ImageResponse(),
        optional_params={},
        litellm_params={"api_key": "sk-test", "api_base": None},
        logging_obj=MagicMock(),
        timeout=None,
        extra_headers={"X-Trace-Id": "abc123"},
    )

    assert submit.calls[0].request.headers["x-trace-id"] == "abc123"
    assert submit.calls[0].request.headers["x-client-name"] == "litellm"


def test_supported_openai_params_and_error_class():
    config = WaveSpeedImageGenerationConfig()
    assert config.get_supported_openai_params(MODEL) == ["n", "size", "response_format"]

    error = config.get_error_class("boom", 503, {})
    assert isinstance(error, WaveSpeedError)
    assert error.status_code == 503
