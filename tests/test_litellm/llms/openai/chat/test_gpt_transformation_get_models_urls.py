"""
Tests for the discovery URL built by OpenAIGPTConfig.get_models
(litellm/llms/openai/chat/gpt_transformation.py)

Regression coverage: api_base values carrying a path (e.g. merge's
https://api-gateway.merge.dev/v1/openai) must probe {api_base}/models
instead of being stripped to {host}/v1/models.
"""

import httpx
import pytest

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


@pytest.mark.parametrize(
    "api_base,expected_url",
    [
        (None, "https://api.openai.com/v1/models"),
        ("https://api.openai.com", "https://api.openai.com/v1/models"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1/models"),
        (
            "https://api-gateway.merge.dev/v1/openai",
            "https://api-gateway.merge.dev/v1/openai/models",
        ),
        ("http://localhost:8080", "http://localhost:8080/v1/models"),
        ("http://localhost:8080/v1", "http://localhost:8080/v1/models"),
    ],
)
def test_get_models_probes_path_preserving_url(respx_mock, api_base, expected_url):
    config = OpenAIGPTConfig()
    route = respx_mock.get(expected_url).mock(
        return_value=httpx.Response(200, json={"data": [{"id": "m1"}]})
    )

    models = config.get_models(api_key="sk-test", api_base=api_base)

    assert str(route.calls[0].request.url) == expected_url
    assert models == ["m1"]


def test_get_models_omits_auth_header_without_a_key(respx_mock, monkeypatch):
    """An unset key must not leak OPENAI_API_KEY to a third-party gateway."""
    import litellm

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)
    route = respx_mock.get("https://gateway.example.com/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    OpenAIGPTConfig().get_models(api_key=None, api_base="https://gateway.example.com/v1")

    assert "authorization" not in route.calls[0].request.headers
