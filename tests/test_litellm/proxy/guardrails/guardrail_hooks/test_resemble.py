import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from httpx import Request, Response

sys.path.insert(0, os.path.abspath("../.."))

import litellm  # noqa: E402
from litellm import DualCache  # noqa: E402
from litellm.proxy._types import UserAPIKeyAuth  # noqa: E402
from litellm.proxy.guardrails.guardrail_hooks.resemble.resemble import (  # noqa: E402
    RESEMBLE_DEFAULT_API_BASE,
    ResembleGuardrail,
    ResembleGuardrailMissingSecrets,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2  # noqa: E402


def _make_guardrail(**overrides):
    defaults = dict(
        api_key="test-key",
        guardrail_name="resemble-test",
        event_hook="pre_call",
        default_on=True,
        poll_interval_seconds=0.001,
        poll_timeout_seconds=1.0,
    )
    defaults.update(overrides)
    return ResembleGuardrail(**defaults)


def _fake_post_response(body, status_code=200, request=None):
    return Response(
        status_code=status_code,
        json=body,
        request=request
        or Request(method="POST", url="https://app.resemble.ai/api/v2/detect"),
    )


def _fake_get_response(body, status_code=200, uuid="abc-123"):
    return Response(
        status_code=status_code,
        json=body,
        request=Request(
            method="GET", url=f"https://app.resemble.ai/api/v2/detect/{uuid}"
        ),
    )


# ---------------------------------------------------------------------------
# Init / config tests
# ---------------------------------------------------------------------------


def test_resemble_guard_registered_via_init_guardrails_v2(monkeypatch):
    """`resemble` is accepted by init_guardrails_v2 and loads the class."""
    litellm.set_verbose = True
    litellm.guardrail_name_config_map = {}
    monkeypatch.setenv("RESEMBLE_API_KEY", "test-key")

    init_guardrails_v2(
        all_guardrails=[
            {
                "guardrail_name": "deepfake-detect",
                "litellm_params": {
                    "guardrail": "resemble",
                    "mode": "pre_call",
                    "default_on": True,
                },
            }
        ],
        config_file_path="",
    )


def test_missing_api_key_raises():
    with pytest.raises(ResembleGuardrailMissingSecrets):
        ResembleGuardrail(guardrail_name="r")


def test_api_base_strips_trailing_slash(monkeypatch):
    guard = _make_guardrail(api_base="https://custom.example/api/v2/")
    assert guard.api_base == "https://custom.example/api/v2"


def test_default_api_base_fallback(monkeypatch):
    monkeypatch.delenv("RESEMBLE_API_BASE", raising=False)
    guard = _make_guardrail()
    assert guard.api_base == RESEMBLE_DEFAULT_API_BASE


# ---------------------------------------------------------------------------
# URL extraction tests
# ---------------------------------------------------------------------------


class TestExtractMediaUrls:
    def setup_method(self):
        self.guard = _make_guardrail()

    def test_plain_text_audio_url(self):
        data = {
            "messages": [
                {"role": "user", "content": "Check https://cdn.example.com/c.mp3 pls"}
            ]
        }
        assert self.guard._extract_media_urls(data) == ["https://cdn.example.com/c.mp3"]

    def test_openai_image_url_part(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this real?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/face.png"},
                        },
                    ],
                }
            ]
        }
        assert self.guard._extract_media_urls(data) == [
            "https://cdn.example.com/face.png"
        ]

    def test_openai_input_audio_part(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"url": "https://cdn.example.com/a.wav"},
                        }
                    ],
                }
            ]
        }
        assert self.guard._extract_media_urls(data) == ["https://cdn.example.com/a.wav"]

    def test_anthropic_source_url(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://cdn.example.com/x.jpg",
                            },
                        }
                    ],
                }
            ]
        }
        assert self.guard._extract_media_urls(data) == ["https://cdn.example.com/x.jpg"]

    def test_metadata_fallback_when_no_content_url(self):
        data = {
            "messages": [{"role": "user", "content": "no url here"}],
            "metadata": {"mediaUrl": "https://cdn.example.com/clip.mp4"},
        }
        assert self.guard._extract_media_urls(data) == [
            "https://cdn.example.com/clip.mp4"
        ]

    def test_custom_metadata_key(self):
        guard = _make_guardrail(metadata_key="audio_src")
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"audio_src": "https://cdn.example.com/x.m4a"},
        }
        assert guard._extract_media_urls(data) == ["https://cdn.example.com/x.m4a"]

    def test_metadata_list_accepted(self):
        guard = _make_guardrail()
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {
                "mediaUrl": [
                    "https://cdn.example.com/a.mp3",
                    "https://cdn.example.com/b.mp3",
                ]
            },
        }
        assert guard._extract_media_urls(data) == [
            "https://cdn.example.com/a.mp3",
            "https://cdn.example.com/b.mp3",
        ]

    def test_returns_empty_list_when_no_url(self):
        data = {"messages": [{"role": "user", "content": "nothing to see"}]}
        assert self.guard._extract_media_urls(data) == []

    def test_multiple_content_part_urls_preserved_in_order(self):
        """
        P1 regression test: the extractor must return every URL referenced in
        a multimodal content array, in order. Returning only the first URL
        let callers smuggle a synthetic URL past the guardrail by placing a
        benign one earlier in the array.
        """
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/real.jpg"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/fake.jpg"},
                        },
                    ],
                }
            ]
        }
        assert self.guard._extract_media_urls(data) == [
            "https://cdn.example.com/real.jpg",
            "https://cdn.example.com/fake.jpg",
        ]

    def test_duplicate_urls_are_deduped_in_order(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/a.jpg"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/a.jpg"},
                        },
                    ],
                }
            ]
        }
        assert self.guard._extract_media_urls(data) == ["https://cdn.example.com/a.jpg"]

    def test_urls_across_multiple_messages(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/one.jpg"},
                        },
                    ],
                },
                {"role": "assistant", "content": "ok"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://cdn.example.com/two.jpg"},
                        },
                    ],
                },
            ]
        }
        assert self.guard._extract_media_urls(data) == [
            "https://cdn.example.com/one.jpg",
            "https://cdn.example.com/two.jpg",
        ]


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateDetection:
    def setup_method(self):
        self.guard = _make_guardrail(threshold=0.5)

    def test_fake_label_always_fails(self):
        item = {
            "metrics": {
                "label": "fake",
                "aggregated_score": "0.2",
                "score": ["0.1"],
            }
        }
        result = self.guard._evaluate_detection(item)
        assert result["verdict"] is False
        assert result["label"] == "fake"

    def test_real_low_score_passes(self):
        item = {
            "metrics": {
                "label": "real",
                "aggregated_score": "0.1",
                "score": ["0.1"],
            }
        }
        result = self.guard._evaluate_detection(item)
        assert result["verdict"] is True
        assert result["score"] == 0.1

    def test_real_high_score_fails_on_threshold(self):
        item = {
            "metrics": {
                "label": "real",
                "aggregated_score": "0.7",
                "score": ["0.7"],
            }
        }
        result = self.guard._evaluate_detection(item)
        assert result["verdict"] is False

    def test_image_metrics_shape(self):
        item = {"image_metrics": {"label": "fake", "score": 0.9}}
        result = self.guard._evaluate_detection(item)
        assert result["verdict"] is False
        assert result["score"] == 0.9

    def test_video_metrics_shape(self):
        item = {"video_metrics": {"label": "real", "score": 0.2}}
        result = self.guard._evaluate_detection(item)
        assert result["verdict"] is True

    def test_zero_score_is_preserved(self):
        item = {"metrics": {"label": "real", "aggregated_score": 0.0}}
        result = self.guard._evaluate_detection(item)
        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# Hook behaviour tests (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_passes_without_media_url():
    guard = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "plain text, no media"}]}

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
    ) as post_mock:
        result = await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    assert result == data
    post_mock.assert_not_called()


@pytest.mark.asyncio
async def test_apply_guardrail_scans_generic_image_inputs():
    guard = _make_guardrail()
    inputs = {"images": ["https://cdn.example.com/image.jpg"]}

    create_response = _fake_post_response(
        {
            "success": True,
            "item": {
                "uuid": "img-2",
                "status": "completed",
                "media_type": "image",
                "image_metrics": {"label": "real", "score": 0.1},
            },
        }
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=create_response,
    ) as post_mock:
        result = await guard.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="request",
        )

    assert result == inputs
    assert (
        post_mock.call_args.kwargs["json"]["url"]
        == "https://cdn.example.com/image.jpg"
    )


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_generic_text_media_url():
    guard = _make_guardrail()
    inputs = {"texts": ["check https://cdn.example.com/cloned.wav"]}

    create_response = _fake_post_response(
        {
            "success": True,
            "item": {
                "uuid": "audio-2",
                "status": "completed",
                "media_type": "audio",
                "metrics": {"label": "fake", "aggregated_score": "0.9"},
            },
        }
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=create_response,
    ):
        with pytest.raises(HTTPException):
            await guard.apply_guardrail(
                inputs=inputs,
                request_data={},
                input_type="request",
            )


@pytest.mark.asyncio
async def test_pre_call_passes_when_audio_is_real():
    guard = _make_guardrail()
    data = {
        "messages": [
            {"role": "user", "content": "check https://cdn.example.com/clip.mp3"}
        ]
    }

    create_response = _fake_post_response(
        {
            "success": True,
            "item": {"uuid": "u-1", "status": "processing"},
        }
    )
    poll_response = _fake_get_response(
        {
            "success": True,
            "item": {
                "uuid": "u-1",
                "media_type": "audio",
                "status": "completed",
                "metrics": {
                    "label": "real",
                    "score": ["0.1", "0.2"],
                    "aggregated_score": "0.15",
                    "consistency": "0.9",
                },
            },
        },
        uuid="u-1",
    )

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=create_response,
        ),
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            return_value=poll_response,
        ),
    ):
        result = await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    assert result == data


@pytest.mark.asyncio
async def test_pre_call_blocks_fake_audio():
    guard = _make_guardrail()
    data = {
        "messages": [{"role": "user", "content": "https://cdn.example.com/cloned.wav"}]
    }

    create_response = _fake_post_response(
        {"success": True, "item": {"uuid": "u-2", "status": "processing"}}
    )
    poll_response = _fake_get_response(
        {
            "success": True,
            "item": {
                "uuid": "u-2",
                "media_type": "audio",
                "status": "completed",
                "metrics": {
                    "label": "fake",
                    "score": ["0.9"],
                    "aggregated_score": "0.95",
                },
                "audio_source_tracing": {
                    "label": "elevenlabs",
                    "error_message": None,
                },
            },
        },
        uuid="u-2",
    )

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=create_response,
        ),
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            return_value=poll_response,
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="completion",
            )

    assert exc_info.value.status_code == 400
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail["resemble"]["label"] == "fake"
    assert detail["resemble"]["media_url"] == "https://cdn.example.com/cloned.wav"
    assert detail["resemble"]["audio_source_tracing"]["label"] == "elevenlabs"


@pytest.mark.asyncio
async def test_pre_call_blocks_image_threshold_exceeded():
    guard = _make_guardrail(threshold=0.5)
    data = {
        "messages": [{"role": "user", "content": "https://cdn.example.com/photo.jpg"}]
    }

    create_response = _fake_post_response(
        {
            "success": True,
            "item": {
                "uuid": "img-1",
                "status": "completed",
                "media_type": "image",
                "image_metrics": {"label": "real", "score": 0.85, "type": "facial"},
            },
        }
    )

    # Synchronous completion — no GET is expected.
    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=create_response,
        ) as post_mock,
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
        ) as get_mock,
    ):
        with pytest.raises(HTTPException):
            await guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="completion",
            )

    post_mock.assert_called_once()
    get_mock.assert_not_called()


@pytest.mark.asyncio
async def test_pre_call_fails_open_on_api_error():
    guard = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "https://cdn.example.com/x.mp3"}]}

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        side_effect=Exception("network down"),
    ):
        result = await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    assert result == data  # fail-open: untouched


@pytest.mark.asyncio
async def test_pre_call_fails_closed_on_api_error_when_configured():
    guard = _make_guardrail(fail_closed=True)
    data = {"messages": [{"role": "user", "content": "https://cdn.example.com/x.mp3"}]}

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        side_effect=Exception("network down"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="completion",
            )
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_pre_call_times_out_and_fails_open():
    guard = _make_guardrail(poll_interval_seconds=0.001, poll_timeout_seconds=0.02)
    data = {
        "messages": [{"role": "user", "content": "https://cdn.example.com/slow.mp3"}]
    }

    create_response = _fake_post_response(
        {"success": True, "item": {"uuid": "slow-1", "status": "processing"}}
    )
    polling_response = _fake_get_response(
        {"success": True, "item": {"uuid": "slow-1", "status": "processing"}},
        uuid="slow-1",
    )

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=create_response,
        ),
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            return_value=polling_response,
        ),
    ):
        result = await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )
    assert result == data


@pytest.mark.asyncio
async def test_create_payload_includes_flags():
    guard = _make_guardrail(
        media_type="audio",
        audio_source_tracing=True,
        use_reverse_search=True,
        zero_retention_mode=True,
    )
    data = {
        "messages": [{"role": "user", "content": "https://cdn.example.com/clip.mp3"}]
    }

    create_response = _fake_post_response(
        {
            "success": True,
            "item": {
                "uuid": "c-1",
                "status": "completed",
                "media_type": "audio",
                "metrics": {
                    "label": "real",
                    "score": ["0.1"],
                    "aggregated_score": "0.1",
                },
            },
        }
    )

    post_mock = AsyncMock(return_value=create_response)
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new=post_mock,
    ):
        await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    call_kwargs = post_mock.call_args.kwargs
    assert call_kwargs["url"].endswith("/detect")
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
    body = call_kwargs["json"]
    assert body["url"] == "https://cdn.example.com/clip.mp3"
    assert body["media_type"] == "audio"
    assert body["audio_source_tracing"] is True
    assert body["use_reverse_search"] is True
    assert body["zero_retention_mode"] is True


@pytest.mark.asyncio
async def test_moderation_hook_also_scans():
    guard = _make_guardrail(event_hook="during_call")
    data = {"messages": [{"role": "user", "content": "https://cdn.example.com/x.mp3"}]}

    create_response = _fake_post_response(
        {
            "success": True,
            "item": {
                "uuid": "m-1",
                "status": "completed",
                "media_type": "audio",
                "metrics": {
                    "label": "fake",
                    "score": ["0.9"],
                    "aggregated_score": "0.95",
                },
            },
        }
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=create_response,
    ):
        with pytest.raises(HTTPException):
            await guard.async_moderation_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="completion",
            )


@pytest.mark.asyncio
async def test_pre_call_blocks_when_second_of_two_urls_is_fake():
    """
    P1 regression (end-to-end): a request with [real, fake] image URLs must be
    blocked. Previously we returned the first URL from the extractor, so the
    fake one was never sent to Resemble.
    """
    guard = _make_guardrail()
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://cdn.example.com/real.jpg"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://cdn.example.com/fake.jpg"},
                    },
                ],
            }
        ]
    }

    async def _post_side_effect(*args, **kwargs):
        body = kwargs.get("json") or {}
        url = body.get("url")
        if url == "https://cdn.example.com/real.jpg":
            return _fake_post_response(
                {
                    "success": True,
                    "item": {
                        "uuid": "real-1",
                        "status": "completed",
                        "media_type": "image",
                        "image_metrics": {"label": "real", "score": 0.05},
                    },
                }
            )
        if url == "https://cdn.example.com/fake.jpg":
            return _fake_post_response(
                {
                    "success": True,
                    "item": {
                        "uuid": "fake-1",
                        "status": "completed",
                        "media_type": "image",
                        "image_metrics": {"label": "fake", "score": 0.95},
                    },
                }
            )
        raise AssertionError(f"Unexpected POST url={url}")

    post_mock = AsyncMock(side_effect=_post_side_effect)
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new=post_mock,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="completion",
            )

    # Both URLs must have been submitted for scanning.
    assert post_mock.call_count == 2
    submitted = [call.kwargs["json"]["url"] for call in post_mock.call_args_list]
    assert submitted == [
        "https://cdn.example.com/real.jpg",
        "https://cdn.example.com/fake.jpg",
    ]
    # And the fake URL must be what surfaces in the error.
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert detail["resemble"]["media_url"] == "https://cdn.example.com/fake.jpg"
    assert detail["resemble"]["label"] == "fake"


@pytest.mark.asyncio
async def test_poll_treats_metric_less_completed_as_still_processing():
    """
    P2 regression: if the API reports ``completed`` but has no metrics yet,
    the poll loop must not return that item — otherwise _evaluate_detection
    falls through to ``unknown / 0.0`` and silently passes the request.
    """
    guard = _make_guardrail(poll_interval_seconds=0.001, poll_timeout_seconds=0.05)
    data = {"messages": [{"role": "user", "content": "https://cdn.example.com/x.mp3"}]}

    create_response = _fake_post_response(
        {"success": True, "item": {"uuid": "mless", "status": "processing"}}
    )
    # "completed" without any of metrics/image_metrics/video_metrics — must
    # NOT be treated as terminal. The poll should keep looping until the
    # deadline fires.
    metric_less_response = _fake_get_response(
        {"success": True, "item": {"uuid": "mless", "status": "completed"}},
        uuid="mless",
    )

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=create_response,
        ),
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            return_value=metric_less_response,
        ) as get_mock,
    ):
        # With fail_closed=False (the default), a timeout fails open — the
        # request passes through untouched. The important thing is that the
        # poll loop kept polling instead of short-circuiting on the empty
        # completed item.
        result = await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    assert result == data
    assert get_mock.call_count >= 2  # polled more than once, did not return early


@pytest.mark.asyncio
async def test_poll_returns_as_soon_as_metrics_arrive():
    """Companion to the P2 test: once metrics land, polling terminates."""
    guard = _make_guardrail(poll_interval_seconds=0.001, poll_timeout_seconds=1.0)
    data = {"messages": [{"role": "user", "content": "https://cdn.example.com/x.mp3"}]}

    create_response = _fake_post_response(
        {"success": True, "item": {"uuid": "late-metrics", "status": "processing"}}
    )
    responses = [
        # First poll: completed but no metrics — keep polling.
        _fake_get_response(
            {
                "success": True,
                "item": {"uuid": "late-metrics", "status": "completed"},
            },
            uuid="late-metrics",
        ),
        # Second poll: metrics land, label = real → pass.
        _fake_get_response(
            {
                "success": True,
                "item": {
                    "uuid": "late-metrics",
                    "status": "completed",
                    "media_type": "audio",
                    "metrics": {
                        "label": "real",
                        "score": ["0.1"],
                        "aggregated_score": "0.1",
                    },
                },
            },
            uuid="late-metrics",
        ),
    ]

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            return_value=create_response,
        ),
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get",
            side_effect=responses,
        ) as get_mock,
    ):
        result = await guard.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="completion",
        )

    assert result == data
    assert get_mock.call_count == 2
