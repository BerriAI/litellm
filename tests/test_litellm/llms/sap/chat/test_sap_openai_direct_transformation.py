import pytest

from litellm.llms.openai.chat.gpt_5_transformation import OpenAIGPT5Config
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.sap.chat.openai_direct_transformation import SapDeploymentOpenAIChatConfig
from litellm.llms.sap.direct_connect import reset_discovery_memo

_MODEL = "gpt-4o"
_REASONING_MODEL = "gpt-5.6-sol"
_BASE_URL = "https://api.example.com/v2"
_RESOURCE_GROUP = "default"
_DEPLOYMENT_URL = "https://api.example.com/v2/inference/deployments/dd09d6a6a1ac6b0c"
_DEFAULT_API_VERSION = "2024-12-01-preview"

_DISCOVERY_PAYLOAD = {
    "count": 1,
    "resources": [
        {
            "id": "dd09d6a6a1ac6b0c",
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
    cfg = SapDeploymentOpenAIChatConfig(http_client=client, token_creator_factory=_fake_token_factory())
    return cfg, client


def test_provider_tag_is_sap():
    cfg, _ = _config()
    assert cfg.custom_llm_provider == "sap"


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
    assert headers["AI-Resource-Group"] == _RESOURCE_GROUP
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
        litellm_params={"resource_group": "team-a"},
        api_key="svc-key",
    )
    assert headers["AI-Resource-Group"] == "team-a"


def test_get_complete_url_pinned_api_base_skips_discovery():
    cfg, client = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(
        api_base=pinned, api_key=None, model=f"deployment/{_MODEL}", optional_params={}, litellm_params={}, stream=False
    )
    assert url == f"{pinned}/chat/completions?api-version={_DEFAULT_API_VERSION}"
    assert client.calls == []


def test_get_complete_url_honors_explicit_api_version():
    cfg, _ = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(
        api_base=pinned,
        api_key=None,
        model=f"deployment/{_MODEL}",
        optional_params={},
        litellm_params={"api_version": "2023-05-15"},
        stream=False,
    )
    assert url == f"{pinned}/chat/completions?api-version=2023-05-15"


def test_get_complete_url_discovers_and_strips_deployment_prefix():
    cfg, client = _config()
    url = cfg.get_complete_url(
        api_base=None,
        api_key="svc-key",
        model=f"deployment/{_MODEL}",
        optional_params={},
        litellm_params={"resource_group": "team-a"},
        stream=False,
    )
    assert url == f"{_DEPLOYMENT_URL}/chat/completions?api-version={_DEFAULT_API_VERSION}"
    call = client.calls[0]
    assert call["url"] == f"{_BASE_URL}/lm/deployments"
    assert call["params"] == {"scenarioId": "foundation-models", "status": "RUNNING"}
    assert call["headers"]["Authorization"] == "Bearer test-token"
    assert call["headers"]["AI-Resource-Group"] == "team-a"


def test_supported_params_are_broad_and_prefix_insensitive():
    cfg, _ = _config()
    bare = cfg.get_supported_openai_params(_MODEL)
    prefixed = cfg.get_supported_openai_params(f"deployment/{_MODEL}")
    assert bare == prefixed
    assert bare == OpenAIGPTConfig().get_supported_openai_params(_MODEL)
    for native in ("temperature", "top_p", "tools", "tool_choice", "response_format", "stream"):
        assert native in bare


def test_reasoning_series_supported_params_include_reasoning_effort_and_verbosity():
    cfg, _ = _config()
    params = cfg.get_supported_openai_params(f"deployment/{_REASONING_MODEL}")
    assert "reasoning_effort" in params
    assert "verbosity" in params


def test_non_reasoning_model_supported_params_exclude_reasoning_effort():
    cfg, _ = _config()
    params = cfg.get_supported_openai_params(f"deployment/{_MODEL}")
    assert "reasoning_effort" not in params
    assert "verbosity" not in params


def test_reasoning_series_map_converts_max_tokens_and_keeps_reasoning_effort():
    cfg, _ = _config()
    mapped = cfg.map_openai_params(
        non_default_params={"reasoning_effort": "high", "max_tokens": 2000},
        optional_params={},
        model=f"deployment/{_REASONING_MODEL}",
        drop_params=False,
    )
    assert mapped["reasoning_effort"] == "high"
    assert mapped["max_completion_tokens"] == 2000
    assert "max_tokens" not in mapped


def test_non_reasoning_model_map_keeps_temperature_and_max_tokens():
    cfg, _ = _config()
    mapped = cfg.map_openai_params(
        non_default_params={"temperature": 0.3, "max_tokens": 500},
        optional_params={},
        model=f"deployment/{_MODEL}",
        drop_params=False,
    )
    assert mapped["temperature"] == 0.3
    assert mapped["max_tokens"] == 500
    assert "max_completion_tokens" not in mapped


def test_gpt5_config_dependency_is_injectable():
    class _Sentinel(OpenAIGPT5Config):
        def get_supported_openai_params(self, model: str) -> list:
            return ["injected"]

    cfg = SapDeploymentOpenAIChatConfig(gpt5_config=_Sentinel())
    assert cfg.get_supported_openai_params(f"deployment/{_REASONING_MODEL}") == ["injected"]


def test_transform_request_strips_deployment_prefix_and_keeps_native_params():
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"temperature": 0.5, "tools": [{"type": "function", "function": {"name": "f"}}]},
        litellm_params={},
        headers={},
    )
    assert body["model"] == _MODEL
    assert body["temperature"] == 0.5
    assert body["tools"] == [{"type": "function", "function": {"name": "f"}}]


def test_transform_request_drops_resource_group_from_chat_body():
    """resource_group routes as the AI-Resource-Group header; leaking it into the OpenAI chat body makes
    SAP reject the /invoke call with 400 'Unknown parameter: resource_group'."""
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


def test_transform_request_drops_client_sent_beta_field():
    """Claude Code sends a top-level `beta` field on gpt-5.x reasoning requests; SAP direct-connect is
    fail-closed and rejects it with 400 'Unknown parameter: beta'. The body allowlist must strip it while
    keeping the reasoning params SAP does accept."""
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_REASONING_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"beta": True, "reasoning_effort": "high", "max_completion_tokens": 2000},
        litellm_params={},
        headers={},
    )
    assert "beta" not in body
    assert body["reasoning_effort"] == "high"
    assert body["max_completion_tokens"] == 2000


def test_transform_request_drops_arbitrary_unknown_field():
    """The allowlist is not a per-field blocklist: any field outside the model's supported params is
    dropped, so a client sending some future unknown key cannot crash the SAP call."""
    cfg, _ = _config()
    body = cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={"some_future_unknown": "x", "temperature": 0.4, "tools": [{"type": "function"}]},
        litellm_params={},
        headers={},
    )
    assert "some_future_unknown" not in body
    assert body["temperature"] == 0.4
    assert body["tools"] == [{"type": "function"}]
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_transform_request_does_not_mutate_caller_optional_params():
    cfg, _ = _config()
    optional_params = {"beta": True, "temperature": 0.2}
    cfg.transform_request(
        model=f"deployment/{_MODEL}",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )
    assert optional_params == {"beta": True, "temperature": 0.2}
