"""
Tests for ``litellm.cost_calculator.default_image_cost_calculator``,
focused on the token-based fallback path used when the (quality, size)
lookup chain cannot resolve a per-image / per-pixel entry.

Motivation: newer image-gen models (e.g. ``gpt-image-2``) accept
"thousands of valid resolutions" but only publish per-token pricing,
so the size-aliased lookup misses on non-standard sizes.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

import litellm
from litellm.cost_calculator import default_image_cost_calculator
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    ImageObject,
    ImageResponse,
    PromptTokensDetailsWrapper,
    Usage,
)


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


def _image_response(
    text_in: int = 0,
    image_in: int = 0,
    image_out: int = 0,
    cached_in: int = 0,
) -> ImageResponse:
    usage = Usage(
        prompt_tokens=text_in + image_in,
        completion_tokens=image_out,
        total_tokens=text_in + image_in + image_out,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=text_in,
            image_tokens=image_in,
            cached_tokens=cached_in,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            text_tokens=0,
            image_tokens=image_out,
            reasoning_tokens=0,
        ),
    )
    response = ImageResponse(
        created=1234567890,
        data=[ImageObject(url="http://example.com/image.jpg")],
    )
    response.usage = usage
    return response


class TestDefaultImageCostCalculator:
    def test_per_image_entry_hits_existing_path(self):
        """Regression: ``high/1024-x-1024/gpt-image-1`` resolves via the
        size-aliased lookup chain and returns its ``input_cost_per_image``
        unchanged — the new fallback must not interfere.
        """
        cost = default_image_cost_calculator(
            model="openai/gpt-image-1",
            custom_llm_provider="openai",
            quality="high",
            n=1,
            size="1024x1024",
            image_response=_image_response(text_in=10, image_out=4000),
        )
        expected = litellm.model_cost["high/1024-x-1024/gpt-image-1"][
            "input_cost_per_image"
        ]
        assert cost == expected

    def test_per_image_entry_takes_precedence_over_token_fallback(self, monkeypatch):
        """Regression: when both a (quality, size) per-image entry and a
        plain token-cost entry exist for the same model, the per-image
        entry wins.
        """
        monkeypatch.setitem(
            litellm.model_cost,
            "high/1024-x-1024/synthetic-image-model",
            {
                "input_cost_per_image": 0.5,
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )
        monkeypatch.setitem(
            litellm.model_cost,
            "synthetic-image-model",
            {
                "input_cost_per_token": 5e-6,
                "output_cost_per_image_token": 3e-5,
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )

        cost = default_image_cost_calculator(
            model="openai/synthetic-image-model",
            custom_llm_provider="openai",
            quality="high",
            n=1,
            size="1024x1024",
            image_response=_image_response(text_in=100, image_out=4000),
        )
        assert cost == 0.5

    def test_token_fallback_for_non_standard_size(self, monkeypatch):
        """Token-based fallback triggers when only a plain ``model`` entry
        with token-cost keys is registered and the (quality, size) chain
        misses.
        """
        monkeypatch.setitem(
            litellm.model_cost,
            "synthetic-token-only-model",
            {
                "input_cost_per_token": 5e-6,
                "output_cost_per_image_token": 3e-5,
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )

        cost = default_image_cost_calculator(
            model="openai/synthetic-token-only-model",
            custom_llm_provider="openai",
            quality="low",
            n=1,
            size="2048x768",
            image_response=_image_response(text_in=25, image_out=772),
        )
        expected = 25 * 5e-6 + 772 * 3e-5
        assert abs(cost - expected) < 1e-9

    def test_token_fallback_includes_image_input_tokens(self, monkeypatch):
        """For image-edit responses both ``image_tokens`` (input) and
        ``image_tokens`` (output) must contribute to cost when their
        per-token rates are registered.
        """
        monkeypatch.setitem(
            litellm.model_cost,
            "synthetic-edit-model",
            {
                "input_cost_per_token": 5e-6,
                "input_cost_per_image_token": 8e-6,
                "output_cost_per_image_token": 3e-5,
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )

        cost = default_image_cost_calculator(
            model="openai/synthetic-edit-model",
            custom_llm_provider="openai",
            quality="medium",
            n=1,
            size="1280x720",
            image_response=_image_response(text_in=510, image_in=1452, image_out=5488),
        )
        expected = 510 * 5e-6 + 1452 * 8e-6 + 5488 * 3e-5
        assert abs(cost - expected) < 1e-9

    def test_token_fallback_subtracts_cached_input_tokens(self, monkeypatch):
        """Cached input tokens are billed at ``cache_read_input_token_cost``;
        the uncached remainder uses the standard ``input_cost_per_token``
        rate so caching is reflected in the final cost.
        """
        monkeypatch.setitem(
            litellm.model_cost,
            "synthetic-cached-model",
            {
                "input_cost_per_token": 5e-6,
                "cache_read_input_token_cost": 1.25e-6,
                "output_cost_per_image_token": 3e-5,
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )

        cost = default_image_cost_calculator(
            model="openai/synthetic-cached-model",
            custom_llm_provider="openai",
            quality="low",
            n=1,
            size="2048x768",
            image_response=_image_response(text_in=200, cached_in=160, image_out=600),
        )
        # 40 uncached text + 160 cached text + 600 image_out
        expected = 40 * 5e-6 + 160 * 1.25e-6 + 600 * 3e-5
        assert abs(cost - expected) < 1e-9

    def test_token_fallback_splits_cached_tokens_between_text_and_image(
        self, monkeypatch
    ):
        """Image-edit responses report a single ``cached_tokens`` count.
        Charge text input first against the standard cache rate; the
        remainder is billed at the dedicated image cache rate. Avoids
        double-billing the text portion at the image cache rate.
        """
        monkeypatch.setitem(
            litellm.model_cost,
            "synthetic-image-cache-model",
            {
                "input_cost_per_token": 5e-6,
                "cache_read_input_token_cost": 1.25e-6,
                "input_cost_per_image_token": 8e-6,
                "cache_read_input_image_token_cost": 2e-6,
                "output_cost_per_image_token": 3e-5,
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )

        # cached_in (300) > text_in (10) so 10 are charged at the text cache
        # rate and the remaining 290 at the image cache rate.
        cost = default_image_cost_calculator(
            model="openai/synthetic-image-cache-model",
            custom_llm_provider="openai",
            quality="high",
            n=1,
            size="1280x720",
            image_response=_image_response(
                text_in=10, image_in=1452, cached_in=300, image_out=5488
            ),
        )
        # text uncached: 0, text cache: 10, image uncached: 1162, image cache: 290
        expected = 0 * 5e-6 + 10 * 1.25e-6 + 1162 * 8e-6 + 290 * 2e-6 + 5488 * 3e-5
        assert abs(cost - expected) < 1e-9

    def test_token_fallback_returns_zero_for_free_model(self, monkeypatch):
        """A model that declares zero per-token rates is genuinely free —
        the fallback must return ``0.0`` rather than raising.
        """
        monkeypatch.setitem(
            litellm.model_cost,
            "synthetic-free-image-model",
            {
                "input_cost_per_token": 0.0,
                "output_cost_per_image_token": 0.0,
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )

        cost = default_image_cost_calculator(
            model="openai/synthetic-free-image-model",
            custom_llm_provider="openai",
            quality="low",
            n=1,
            size="2048x768",
            image_response=_image_response(text_in=25, image_out=772),
        )
        assert cost == 0.0

    def test_unmapped_model_without_image_response_raises(self):
        """Negative: cost map miss + no ``image_response`` to fall back on
        — preserve the original behaviour of raising rather than silently
        returning 0.
        """
        with pytest.raises(Exception, match="Model not found in cost map"):
            default_image_cost_calculator(
                model="openai/totally-unmapped-image-model",
                custom_llm_provider="openai",
                quality="high",
                n=1,
                size="1024x1024",
            )

    def test_entry_without_pricing_keys_raises(self, monkeypatch):
        """Negative: cost map entry resolves but carries no per-image,
        per-pixel, or token cost keys — ``raise`` the original
        ``No pricing information found`` error.
        """
        monkeypatch.setitem(
            litellm.model_cost,
            "high/1024-x-1024/synthetic-no-pricing-model",
            {
                "litellm_provider": "openai",
                "mode": "image_generation",
            },
        )

        with pytest.raises(Exception, match="No pricing information found"):
            default_image_cost_calculator(
                model="openai/synthetic-no-pricing-model",
                custom_llm_provider="openai",
                quality="high",
                n=1,
                size="1024x1024",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
