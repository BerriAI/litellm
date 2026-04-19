"""
Unit tests for the openai/ image_generation config dispatcher.

Covers :func:`litellm.llms.openai.image_generation.get_openai_image_generation_config`
and the :class:`OpenAICompatibleImageGenerationConfig` fallback used for
community OpenAI-compatible image endpoints (e.g. third-party aggregators
and ark-style services).
"""

import os
import sys
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.openai.image_generation import (
    DallE2ImageGenerationConfig,
    DallE3ImageGenerationConfig,
    GPTImageGenerationConfig,
    OpenAICompatibleImageGenerationConfig,
    get_openai_image_generation_config,
)
from litellm.types.utils import ImageResponse
from litellm.utils import get_optional_params_image_gen


@pytest.mark.parametrize(
    "model, expected_config",
    [
        ("dall-e-2", DallE2ImageGenerationConfig),
        ("dall-e-2-vtest", DallE2ImageGenerationConfig),
        ("", DallE2ImageGenerationConfig),  # empty string defaults to dall-e-2
        ("dall-e-3", DallE3ImageGenerationConfig),
        ("dall-e-3-preview", DallE3ImageGenerationConfig),
        ("gpt-image-1", GPTImageGenerationConfig),
        ("gpt-image-1-preview", GPTImageGenerationConfig),
        # unknown / community models fall through to the generic config
        ("doubao-seedream-4-5-251128", OpenAICompatibleImageGenerationConfig),
        ("doubao-seedream-5-0-260128", OpenAICompatibleImageGenerationConfig),
        ("some-self-hosted-image-model", OpenAICompatibleImageGenerationConfig),
    ],
)
def test_get_openai_image_generation_config(model, expected_config):
    """Dispatcher returns the right transformer for each model family."""
    assert isinstance(get_openai_image_generation_config(model), expected_config)


def test_openai_compatible_supported_params_superset():
    """
    The generic config accepts the union of standard OpenAI image params so
    it works with both dall-e-style and gpt-image-style upstreams.
    """
    config = OpenAICompatibleImageGenerationConfig()
    supported = config.get_supported_openai_params(model="doubao-seedream-4-5-251128")

    # dall-e-3 style
    for k in ["n", "response_format", "quality", "size", "user", "style"]:
        assert k in supported

    # gpt-image-1 style
    for k in ["background", "moderation", "output_compression", "output_format"]:
        assert k in supported


def test_openai_compatible_accepts_response_format():
    """
    Regression: before this config existed, passing ``response_format="url"``
    to an ``openai/<non-dall-e>`` model raised UnsupportedParamsError because
    it routed to GPTImageGenerationConfig (which doesn't list response_format).
    """
    config = OpenAICompatibleImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={"response_format": "url"},
        optional_params={},
        model="doubao-seedream-4-5-251128",
        drop_params=False,
    )
    assert mapped["response_format"] == "url"


def test_openai_compatible_rejects_truly_unknown_param_without_drop_params():
    """
    Guards against accidentally flagging every extension as supported: a
    parameter that's not in the union (nor a valid OpenAI image param)
    should still raise unless ``drop_params=True``.
    """
    config = OpenAICompatibleImageGenerationConfig()
    with pytest.raises(ValueError):
        config.map_openai_params(
            non_default_params={"definitely_not_an_openai_param": True},
            optional_params={},
            model="doubao-seedream-4-5-251128",
            drop_params=False,
        )


def test_openai_compatible_forwards_vendor_params_via_extra_body():
    """
    End-to-end: params outside OpenAI's standard image-generation set (e.g.
    Volcengine ark's ``watermark`` / ``seed``) should be transparently
    forwarded to the upstream via ``extra_body`` (LiteLLM's existing
    mechanism for openai-compatible providers). The generic config does
    not need to know about these params by name.
    """
    optional_params = get_optional_params_image_gen(
        model="doubao-seedream-4-5-251128",
        response_format="url",
        size="2048x2048",
        custom_llm_provider="openai",
        watermark=False,  # volcengine-specific
        seed=42,  # volcengine-specific
    )
    # Standard params land on the top level
    assert optional_params.get("response_format") == "url"
    assert optional_params.get("size") == "2048x2048"
    # Vendor extras are preserved inside extra_body
    extra_body = optional_params.get("extra_body", {})
    assert extra_body.get("watermark") is False
    assert extra_body.get("seed") == 42


def test_openai_compatible_transform_response_passthrough():
    """transform_image_generation_response leaves the response shape intact."""
    config = OpenAICompatibleImageGenerationConfig()
    raw = MagicMock(spec=httpx.Response)
    raw.json.return_value = {
        "created": 1,
        "data": [{"url": "https://example.com/x.jpeg"}],
    }
    logging_obj = MagicMock()
    resp: ImageResponse = config.transform_image_generation_response(
        model="doubao-seedream-4-5-251128",
        raw_response=raw,
        model_response=ImageResponse(),
        logging_obj=logging_obj,
        request_data={"prompt": "hi"},
        optional_params={"size": "2048x2048", "response_format": "url"},
        litellm_params={},
        encoding=None,
    )
    assert resp.data[0].url == "https://example.com/x.jpeg"
    assert resp.size == "2048x2048"
    assert resp.output_format == "url"
