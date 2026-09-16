import pytest

from litellm.llms.sap.chat.direct_transformation import (
    SapDeploymentAnthropicChatConfig,
    _canonical_anthropic_model,
)
from litellm.llms.sap.direct_connect import reset_discovery_memo

_MODEL = "anthropic--claude-4.8-opus"
_BASE_URL = "https://api.example.com/v2"
_RESOURCE_GROUP = "default"
_DEPLOYMENT_URL = "https://api.example.com/v2/inference/deployments/d38af17dc133a768"

_DISCOVERY_PAYLOAD = {
    "count": 1,
    "resources": [
        {
            "id": "d38af17dc133a768",
            "createdAt": "2026-07-14T02:44:50Z",
            "status": "RUNNING",
            "scenarioId": "foundation-models",
            "deploymentUrl": _DEPLOYMENT_URL,
            "details": {"resources": {"backendDetails": {"model": {"name": _MODEL, "version": "1"}}}},
        }
    ],
}


@pytest.fixture(autouse=True)
def _reset_memo():
    reset_discovery_memo()
    yield
    reset_discovery_memo()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHTTPClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return _FakeResponse(self._payload)


def _fake_token_factory():
    def factory(service_key, *, resource_group=None):
        return (lambda: "Bearer test-token", _BASE_URL, resource_group or _RESOURCE_GROUP)

    return factory


def _config(payload=_DISCOVERY_PAYLOAD):
    client = _FakeHTTPClient(payload)
    cfg = SapDeploymentAnthropicChatConfig(http_client=client, token_creator_factory=_fake_token_factory())
    return cfg, client


def test_provider_tag_is_sap():
    cfg, _ = _config()
    assert cfg.custom_llm_provider == "sap"


def test_validate_environment_injects_sap_headers_without_discovery():
    cfg, client = _config()
    headers = cfg.validate_environment(
        headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        model=f"deployment/{_MODEL}",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="svc-key",
    )
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["AI-Resource-Group"] == _RESOURCE_GROUP
    assert headers["Content-Type"] == "application/json"
    assert headers["AI-Client-Type"] == "LiteLLM"
    assert headers["anthropic-beta"] == "prompt-caching-2024-07-31"
    assert client.calls == []


def test_litellm_params_resource_group_reaches_ai_resource_group_header():
    cfg, _ = _config()
    headers = cfg.validate_environment(
        headers={},
        model=f"deployment/{_MODEL}",
        messages=[],
        optional_params={},
        litellm_params={"resource_group": "team-a"},
        api_key="svc-key",
    )
    assert headers["AI-Resource-Group"] == "team-a"


def test_get_complete_url_discovery_forwards_resource_group_header():
    cfg, client = _config()
    cfg.get_complete_url(
        api_base=None,
        api_key="svc-key",
        model=f"deployment/{_MODEL}",
        optional_params={},
        litellm_params={"resource_group": "team-a"},
        stream=False,
    )
    assert client.calls[0]["headers"]["AI-Resource-Group"] == "team-a"


def test_get_complete_url_pinned_api_base_skips_discovery():
    cfg, client = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(
        api_base=pinned, api_key=None, model=f"deployment/{_MODEL}", optional_params={}, litellm_params={}, stream=False
    )
    assert url == f"{pinned}/invoke"
    assert client.calls == []


def test_get_complete_url_stream_suffix_when_pinned():
    cfg, _ = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(
        api_base=pinned, api_key=None, model=_MODEL, optional_params={}, litellm_params={}, stream=True
    )
    assert url == f"{pinned}/invoke-with-response-stream"


def test_get_complete_url_discovers_and_strips_deployment_prefix():
    cfg, client = _config()
    url = cfg.get_complete_url(
        api_base=None, api_key="svc-key", model=f"deployment/{_MODEL}", optional_params={}, litellm_params={}, stream=False
    )
    assert url == f"{_DEPLOYMENT_URL}/invoke"
    call = client.calls[0]
    assert call["url"] == f"{_BASE_URL}/lm/deployments"
    assert call["params"] == {"scenarioId": "foundation-models", "status": "RUNNING"}
    assert call["headers"]["Authorization"] == "Bearer test-token"


def test_get_complete_url_discovery_is_cached_across_calls():
    cfg, client = _config()
    first = cfg.get_complete_url(
        api_base=None, api_key="svc-key", model=_MODEL, optional_params={}, litellm_params={}, stream=False
    )
    second = cfg.get_complete_url(
        api_base=None, api_key="svc-key", model=_MODEL, optional_params={}, litellm_params={}, stream=True
    )
    assert first == f"{_DEPLOYMENT_URL}/invoke"
    assert second == f"{_DEPLOYMENT_URL}/invoke-with-response-stream"
    assert len(client.calls) == 1


def test_transform_request_drops_resource_group_from_invoke_body():
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 16, "resource_group": "development-sandbox"},
        litellm_params={"resource_group": "development-sandbox"},
        headers={},
    )
    assert "resource_group" not in body
    assert body["max_tokens"] == 16


def test_canonical_anthropic_model_reorders_known_and_passes_through_unknown():
    assert _canonical_anthropic_model(f"deployment/{_MODEL}") == "claude-opus-4-8"
    assert _canonical_anthropic_model(_MODEL) == "claude-opus-4-8"
    unknown = "deployment/anthropic--claude-3.7-sonnet"
    assert _canonical_anthropic_model(unknown) == unknown


def test_supported_params_include_thinking_and_reasoning_effort():
    cfg, _ = _config()
    params = cfg.get_supported_openai_params(f"deployment/{_MODEL}")
    assert "thinking" in params
    assert "reasoning_effort" in params


def test_map_openai_params_preserves_adaptive_thinking_and_maps_effort():
    cfg, _ = _config()
    mapped = cfg.map_openai_params(
        non_default_params={"thinking": {"type": "adaptive"}, "reasoning_effort": "high"},
        optional_params={},
        model=f"deployment/{_MODEL}",
        drop_params=False,
    )
    assert mapped["thinking"]["type"] == "adaptive"
    assert mapped["output_config"] == {"effort": "high"}


def test_transform_request_preserves_output_config_via_canonical_name():
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"max_tokens": 16, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        litellm_params={},
        headers={},
    )
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}


def test_transform_request_drops_unknown_fields_that_sap_rejects():
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "max_tokens": 16,
            "temperature": 1,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "enable_thinking": True,
            "beta": True,
            "enable_prompt_caching": False,
        },
        litellm_params={},
        headers={},
    )
    assert "enable_thinking" not in body
    assert "beta" not in body
    assert "enable_prompt_caching" not in body
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"}
    assert body["max_tokens"] == 16
    assert body["temperature"] == 1
    assert body["anthropic_version"]


def test_transform_request_keeps_full_anthropic_invoke_body_shape():
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        optional_params={
            "max_tokens": 16,
            "temperature": 1,
            "top_p": 0.9,
            "stop_sequences": ["x"],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "tools": [{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "metadata": {"user_id": "u"},
        },
        litellm_params={},
        headers={},
    )
    for kept in ("system", "top_p", "stop_sequences", "tools", "tool_choice", "metadata"):
        assert kept in body


def test_sign_request_is_noop_when_not_streaming():
    cfg, _ = _config()
    headers, body = cfg.sign_request(
        headers={"Authorization": "Bearer test-token"},
        optional_params={},
        request_data={"max_tokens": 1},
        api_base=_DEPLOYMENT_URL,
        stream=False,
    )
    assert body is None
    assert "accept" not in headers


def test_sign_request_sets_event_stream_accept_when_streaming():
    cfg, _ = _config()
    headers, body = cfg.sign_request(
        headers={"Authorization": "Bearer test-token"},
        optional_params={},
        request_data={"max_tokens": 1},
        api_base=_DEPLOYMENT_URL,
        stream=True,
    )
    assert body is None
    assert headers["accept"] == "application/vnd.amazon.eventstream"
