import pytest

from litellm.llms.sap.chat.gemini_direct_transformation import SapDeploymentGeminiChatConfig
from litellm.llms.sap.direct_connect import reset_discovery_memo
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

_MODEL = "gemini-2.5-flash-lite"
_BASE_URL = "https://api.example.com/v2"
_RESOURCE_GROUP = "feedback-ai"
_DEPLOYMENT_URL = "https://api.example.com/v2/inference/deployments/df3358aa1f7a057f"

_DISCOVERY_PAYLOAD = {
    "count": 1,
    "resources": [
        {
            "id": "df3358aa1f7a057f",
            "createdAt": "2026-09-14T02:44:50Z",
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
        return (lambda: "Bearer test-token", _BASE_URL, resource_group or "default")

    return factory


def _config(payload=_DISCOVERY_PAYLOAD):
    client = _FakeHTTPClient(payload)
    cfg = SapDeploymentGeminiChatConfig(http_client=client, token_creator_factory=_fake_token_factory())
    return cfg, client


def test_provider_tag_is_sap():
    cfg, _ = _config()
    assert cfg.custom_llm_provider == "sap"


def test_stream_param_is_not_added_to_request_body():
    cfg, _ = _config()
    assert cfg.supports_stream_param_in_request_body is False


def test_validate_environment_injects_sap_headers_without_discovery():
    cfg, client = _config()
    headers = cfg.validate_environment(
        headers={"x-caller": "keep-me"},
        model=f"deployment/{_MODEL}",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="svc-key",
    )
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["AI-Resource-Group"] == "default"
    assert headers["Content-Type"] == "application/json"
    assert headers["AI-Client-Type"] == "LiteLLM"
    assert headers["x-caller"] == "keep-me"
    assert client.calls == []


def test_litellm_params_resource_group_reaches_ai_resource_group_header():
    cfg, _ = _config()
    headers = cfg.validate_environment(
        headers={},
        model=f"deployment/{_MODEL}",
        messages=[],
        optional_params={},
        litellm_params={"resource_group": _RESOURCE_GROUP},
        api_key="svc-key",
    )
    assert headers["AI-Resource-Group"] == _RESOURCE_GROUP


def test_get_complete_url_pinned_api_base_builds_generate_content():
    cfg, client = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(
        api_base=pinned, api_key=None, model=f"deployment/{_MODEL}", optional_params={}, litellm_params={}, stream=False
    )
    assert url == f"{pinned}/models/{_MODEL}:generateContent"
    assert client.calls == []


def test_get_complete_url_stream_uses_stream_generate_content_sse():
    cfg, _ = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(
        api_base=pinned, api_key=None, model=f"deployment/{_MODEL}", optional_params={}, litellm_params={}, stream=True
    )
    assert url == f"{pinned}/models/{_MODEL}:streamGenerateContent?alt=sse"


def test_get_complete_url_discovers_and_strips_deployment_prefix():
    cfg, client = _config()
    url = cfg.get_complete_url(
        api_base=None,
        api_key="svc-key",
        model=f"deployment/{_MODEL}",
        optional_params={},
        litellm_params={"resource_group": _RESOURCE_GROUP},
        stream=False,
    )
    assert url == f"{_DEPLOYMENT_URL}/models/{_MODEL}:generateContent"
    call = client.calls[0]
    assert call["url"] == f"{_BASE_URL}/lm/deployments"
    assert call["headers"]["Authorization"] == "Bearer test-token"
    assert call["headers"]["AI-Resource-Group"] == _RESOURCE_GROUP


def test_supported_params_are_broad_and_prefix_insensitive():
    cfg, _ = _config()
    bare = cfg.get_supported_openai_params(_MODEL)
    prefixed = cfg.get_supported_openai_params(f"deployment/{_MODEL}")
    assert bare == prefixed
    assert bare == VertexGeminiConfig().get_supported_openai_params(_MODEL)
    for native in ("temperature", "top_p", "max_tokens", "tools", "tool_choice", "response_format", "stream"):
        assert native in bare


def test_transform_request_strips_deployment_prefix_and_builds_gemini_body():
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.5},
        litellm_params={},
        headers={},
    )
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["generationConfig"]["temperature"] == 0.5
    assert "model" not in body


def test_transform_request_drops_resource_group_from_gemini_body():
    """resource_group routes as the AI-Resource-Group header; it must never reach the generateContent
    body, where SAP's Vertex executable would reject it as an unknown request argument."""
    import json

    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.5, "resource_group": "development-sandbox"},
        litellm_params={"resource_group": "development-sandbox"},
        headers={},
    )
    assert "resource_group" not in json.dumps(body)
    assert body["generationConfig"]["temperature"] == 0.5


def test_model_response_iterator_parses_sse_text_frames():
    cfg, _ = _config()
    frames = [
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}',
        'data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP"}]}',
    ]
    iterator = cfg.get_model_response_iterator(streaming_response=iter(frames), sync_stream=True)

    def _drain(it):
        while True:
            try:
                yield next(it)
            except StopIteration:
                return

    text = "".join(
        chunk.choices[0].delta.content
        for chunk in _drain(iterator)
        if chunk is not None and chunk.choices and chunk.choices[0].delta.content
    )
    assert text == "Hello world"
