"""
CircleCI local_testing runs these with --cov=litellm (Codecov patch coverage).

tests/test_litellm/llms/xai/ is not part of that split; keep credential tests here
so image_credentials.py is exercised in the combined upload.
"""

from litellm.constants import XAI_API_BASE
from litellm.llms.xai.image_credentials import resolve_for_openai_sdk_image_generation


def test_resolve_passthrough_when_provider_has_no_resolver():
    out = resolve_for_openai_sdk_image_generation(
        "openai",
        "https://api.openai.com/v1",
        "sk-primary",
        "sk-dynamic",
    )
    assert out == ("https://api.openai.com/v1", "sk-primary", "sk-dynamic")


def test_resolve_xai_uses_dynamic_api_key_when_api_key_missing():
    api_base, key, dynamic_key = resolve_for_openai_sdk_image_generation(
        "xai",
        None,
        None,
        "sk-from-dynamic",
    )
    assert api_base == XAI_API_BASE
    assert key == "sk-from-dynamic"
    assert dynamic_key is None
