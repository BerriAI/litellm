"""Unit tests for the WaveSpeed AI image generation submit/poll flow."""

from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm

from litellm.llms.wavespeed.common_utils import (
    POST_SUBMIT_ERROR_STATUS_CODE,
    WaveSpeedError,
)
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
    assert poll.call_count == 10
    assert "10 times in a row" in str(exc_info.value)


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


HOURS_OF_PROCESSING = 700  # one poll per virtual second: well past the old 600s cap


@pytest.fixture
def virtual_clock(monkeypatch):
    """Sleeping advances a fake wall clock, so 'ten minutes later' costs no real time."""
    import time as real_time

    start = real_time.time()
    elapsed = [0.0]

    def fake_sleep(seconds):
        elapsed[0] += seconds

    async def fake_async_sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr("litellm.llms.wavespeed.image_generation.handler.DEFAULT_POLLING_INTERVAL", 1.0)
    monkeypatch.setattr("litellm.llms.wavespeed.image_generation.handler.time.sleep", fake_sleep)
    monkeypatch.setattr("litellm.llms.wavespeed.image_generation.handler.asyncio.sleep", fake_async_sleep)
    monkeypatch.setattr("litellm.llms.wavespeed.image_generation.handler.time.time", lambda: start + elapsed[0])
    return elapsed


def _long_running_polls():
    return [httpx.Response(200, json=prediction("processing"))] * HOURS_OF_PROCESSING + [
        httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL]))
    ]


@respx.mock
def test_sync_prediction_processing_past_ten_minutes_still_completes(generate, virtual_clock):
    """There is no default overall polling cap: only a terminal status ends the wait."""
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(side_effect=_long_running_polls())

    response = generate()

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert virtual_clock[0] > 600
    assert submit.call_count == 1
    assert poll.call_count == HOURS_OF_PROCESSING + 1


@pytest.mark.asyncio
@respx.mock
async def test_async_prediction_processing_past_ten_minutes_still_completes(virtual_clock):
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(side_effect=_long_running_polls())

    response = await run_async()

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert virtual_clock[0] > 600
    assert submit.call_count == 1
    assert poll.call_count == HOURS_OF_PROCESSING + 1


@pytest.mark.asyncio
@respx.mock
async def test_caller_cancellation_still_bounds_a_long_wait():
    """An unset cap must not make the call unstoppable: the caller can bound it with its own deadline."""
    import asyncio

    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction("processing")))

    async def never_ending_wait():
        await asyncio.sleep(0)  # let the loop breathe under the zero poll interval
        return await run_async()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(never_ending_wait(), timeout=0.3)


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
    assert poll.call_count == 10
    assert "10 times in a row" in str(exc_info.value)


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


@respx.mock
def test_poll_requests_honour_the_caller_timeout(generate):
    """A short caller timeout must not be replaced by the client's multi-minute default."""
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(
        return_value=httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL]))
    )

    WaveSpeedImageGeneration().image_generation(
        model=MODEL,
        prompt="a red panda",
        model_response=ImageResponse(),
        optional_params={},
        litellm_params={"api_key": "sk-test", "api_base": None},
        logging_obj=MagicMock(),
        timeout=2.5,
    )

    assert poll.call_count == 1
    assert poll.calls[0].request.extensions.get("timeout") == {
        "connect": 2.5,
        "read": 2.5,
        "write": 2.5,
        "pool": 2.5,
    }


async def run_async(**overrides):
    return await WaveSpeedImageGeneration().async_image_generation(
        model=MODEL,
        prompt="a red panda",
        model_response=ImageResponse(),
        optional_params={},
        litellm_params={"api_key": "sk-test", "api_base": None},
        logging_obj=MagicMock(),
        timeout=None,
        **overrides,
    )


@pytest.mark.asyncio
@respx.mock
async def test_async_submit_reset_after_send_is_not_replayed():
    """A reset after the request was written may have created a paid task: send once, fail non-retryably."""
    submit = respx.post(SUBMIT_URL).mock(side_effect=httpx.RemoteProtocolError("server disconnected"))
    poll = respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction("completed")))

    with pytest.raises(WaveSpeedError) as exc_info:
        await run_async()

    assert submit.call_count == 1
    assert poll.call_count == 0
    assert exc_info.value.status_code == POST_SUBMIT_ERROR_STATUS_CODE
    assert not litellm._should_retry(exc_info.value.status_code)


@respx.mock
def test_sync_submit_reset_after_send_is_not_retried_and_not_retryable(generate):
    submit = respx.post(SUBMIT_URL).mock(side_effect=httpx.RemoteProtocolError("server disconnected"))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert submit.call_count == 1
    assert not litellm._should_retry(exc_info.value.status_code)


@respx.mock
def test_poll_gateway_error_then_success_is_tolerated(generate):
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(
        side_effect=[
            httpx.Response(502, text="bad gateway"),
            httpx.Response(429, text="slow down"),
            httpx.Response(503, text="unavailable"),
            httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
        ]
    )

    response = generate()

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert submit.call_count == 1
    assert poll.call_count == 4


@pytest.mark.asyncio
@respx.mock
async def test_async_poll_gateway_error_then_success_is_tolerated():
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(
        side_effect=[
            httpx.Response(504, text="timeout"),
            httpx.Response(200, json=prediction("completed", outputs=[OUTPUT_URL])),
        ]
    )

    response = await run_async()

    assert [image.url for image in response.data] == [OUTPUT_URL]
    assert poll.call_count == 2


@respx.mock
def test_persistent_poll_502_fails_with_prediction_id_and_is_not_retryable(generate):
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(return_value=httpx.Response(502, text="bad gateway"))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert submit.call_count == 1
    assert poll.call_count == 10
    assert "pred-123" in str(exc_info.value)
    assert "502" in str(exc_info.value)
    assert not litellm._should_retry(exc_info.value.status_code)


@respx.mock
def test_unusable_poll_response_names_the_prediction(generate):
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(return_value=httpx.Response(404, text="not found"))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert "pred-123" in str(exc_info.value)
    assert exc_info.value.status_code == 404


@respx.mock
def test_deleted_prediction_is_terminal(generate):
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    poll = respx.get(RESULT_URL).mock(return_value=httpx.Response(200, json=prediction("deleted")))

    with pytest.raises(WaveSpeedError):
        generate()

    assert poll.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_router_retries_never_resubmit_after_the_task_was_accepted():
    """Router num_retries re-runs the whole call; after the submit it must not issue a second paid POST."""
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(200, json=prediction("created")))
    respx.get(RESULT_URL).mock(return_value=httpx.Response(502, text="bad gateway"))
    router = litellm.Router(
        model_list=[
            {
                "model_name": "ws-image",
                "litellm_params": {"model": f"wavespeed/{MODEL}", "api_key": "sk-test"},
            }
        ],
        num_retries=2,
    )

    with pytest.raises(Exception, match="pred-123"):
        await router.aimage_generation(model="ws-image", prompt="a red panda")

    assert submit.call_count == 1


def _router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "ws-image",
                "litellm_params": {"model": f"wavespeed/{MODEL}", "api_key": "sk-test"},
            }
        ],
        num_retries=2,
    )


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "outcome",
    [
        httpx.Response(503, text="unavailable"),
        httpx.Response(500, text="boom"),
        httpx.ReadTimeout("read timed out"),
        httpx.ReadError("connection lost"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
    ids=["503", "500", "read-timeout", "read-error", "reset"],
)
async def test_router_sends_one_submit_when_the_outcome_is_unknown(outcome):
    """5xx / read timeout / reset after send: the task may exist, so Router retries must not re-POST."""
    if isinstance(outcome, httpx.Response):
        submit = respx.post(SUBMIT_URL).mock(return_value=outcome)
    else:
        submit = respx.post(SUBMIT_URL).mock(side_effect=outcome)

    with pytest.raises(Exception, match="may already exist"):
        await _router().aimage_generation(model="ws-image", prompt="a red panda")

    assert submit.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_sync_submit_5xx_is_non_retryable(generate):
    respx.post(SUBMIT_URL).mock(return_value=httpx.Response(502, text="bad gateway"))

    with pytest.raises(WaveSpeedError) as exc_info:
        generate()

    assert exc_info.value.status_code == POST_SUBMIT_ERROR_STATUS_CODE
    assert "may already exist" in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    ("status", "expected_posts"),
    [(429, 3), (400, 1), (401, 1), (422, 1)],
)
async def test_definite_submit_rejections_keep_their_retry_semantics(status, expected_posts):
    """A 4xx is a definite rejection: 429 is retried by the Router as before, the rest are not."""
    submit = respx.post(SUBMIT_URL).mock(return_value=httpx.Response(status, text="rejected"))

    with pytest.raises(Exception, match="rejected") as exc_info:
        await _router().aimage_generation(model="ws-image", prompt="a red panda")

    assert submit.call_count == expected_posts
    assert getattr(exc_info.value, "status_code", None) == status


@pytest.mark.asyncio
@respx.mock
async def test_connection_never_established_stays_retryable():
    submit = respx.post(SUBMIT_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(Exception, match="refused"):
        await _router().aimage_generation(model="ws-image", prompt="a red panda")

    assert submit.call_count > 1


@pytest.mark.asyncio
@respx.mock
async def test_connect_timeout_stays_retryable():
    submit = respx.post(SUBMIT_URL).mock(side_effect=httpx.ConnectTimeout("connect timed out"))

    with pytest.raises(Exception, match="timed out"):
        await _router().aimage_generation(model="ws-image", prompt="a red panda")

    assert submit.call_count > 1
