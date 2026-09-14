"""
Regression test for https://github.com/BerriAI/litellm/issues/13048

When caching is enabled, a cached completion response must preserve
`provider_specific_fields` (e.g. Anthropic web_search citations) after the
store -> load -> reconstruct round trip.
"""

import pytest

import litellm
from litellm import Cache, ModelResponse
from litellm.types.utils import Choices, Message


def _make_response_with_citations() -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        model="claude-sonnet-4-20250514",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="Pears are fruits.",
                    role="assistant",
                    provider_specific_fields={
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "cited_text": "Pears are fruits.",
                                "url": "https://en.wikipedia.org/wiki/Pear",
                            }
                        ]
                    },
                ),
            )
        ],
    )


@pytest.mark.parametrize("cache_type", ["local", "disk"])
def test_cache_round_trip_preserves_provider_specific_fields(cache_type, tmp_path):
    litellm.cache = Cache(type=cache_type, disk_cache_dir=str(tmp_path / "litellm_cache"))
    try:
        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "Search the web and tell me about pears."}],
        }
        original = _make_response_with_citations()

        litellm.cache.add_cache(result=original, **kwargs)

        cached = litellm.cache.get_cache(**kwargs)
        assert cached is not None, "expected a cache hit"

        # Reconstruct the same way `_convert_cached_result_to_model_response` does
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_model_response_object,
        )

        reconstructed = convert_to_model_response_object(
            response_object=cached,
            model_response_object=ModelResponse(),
        )

        psf = reconstructed.choices[0].message.provider_specific_fields
        assert psf is not None, "provider_specific_fields were dropped by the cache round trip"
        assert "citations" in psf, f"citations missing from provider_specific_fields: {psf}"
    finally:
        litellm.cache = None
