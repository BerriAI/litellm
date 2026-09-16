import pytest

from litellm.llms.sap.direct_connect import reset_discovery_memo
from litellm.llms.sap.responses.transformation import SapDeploymentOpenAIResponsesConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

_MODEL = "gpt-5.5"
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


def _config(payload=_DISCOVERY_PAYLOAD, model=_MODEL):
    client = _FakeHTTPClient(payload)
    cfg = SapDeploymentOpenAIResponsesConfig(
        model=model, http_client=client, token_creator_factory=_fake_token_factory()
    )
    return cfg, client


def test_provider_tag_is_sap():
    cfg, _ = _config()
    assert cfg.custom_llm_provider == LlmProviders.SAP_GENERATIVE_AI_HUB


def test_validate_environment_injects_sap_headers_and_runs_discovery():
    cfg, client = _config()
    headers = cfg.validate_environment(
        headers={"x-caller": "keep-me"},
        model=f"deployment/{_MODEL}",
        litellm_params=GenericLiteLLMParams(api_key="svc-key"),
    )
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["AI-Resource-Group"] == _RESOURCE_GROUP
    assert headers["Content-Type"] == "application/json"
    assert headers["AI-Client-Type"] == "LiteLLM"
    assert headers["x-caller"] == "keep-me"
    call = client.calls[0]
    assert call["url"] == f"{_BASE_URL}/lm/deployments"
    assert call["params"] == {"scenarioId": "foundation-models", "status": "RUNNING"}


def test_validate_environment_then_get_complete_url_reuses_memoized_discovery():
    cfg, client = _config()
    params = GenericLiteLLMParams(api_key="svc-key", resource_group="team-a")
    cfg.validate_environment(headers={}, model=f"deployment/{_MODEL}", litellm_params=params)
    url = cfg.get_complete_url(api_base=None, litellm_params=params.model_dump())
    assert url == f"{_DEPLOYMENT_URL}/responses?api-version={_DEFAULT_API_VERSION}"
    assert len(client.calls) == 1


def test_validate_environment_resource_group_reaches_ai_resource_group_header():
    cfg, _ = _config()
    headers = cfg.validate_environment(
        headers={},
        model=f"deployment/{_MODEL}",
        litellm_params=GenericLiteLLMParams(api_key="svc-key", resource_group="team-a"),
    )
    assert headers["AI-Resource-Group"] == "team-a"


def test_get_complete_url_pinned_api_base_skips_discovery():
    cfg, client = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(api_base=pinned, litellm_params={})
    assert url == f"{pinned}/responses?api-version={_DEFAULT_API_VERSION}"
    assert client.calls == []


def test_get_complete_url_honors_explicit_api_version():
    cfg, _ = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    url = cfg.get_complete_url(api_base=pinned, litellm_params={"api_version": "2023-05-15"})
    assert url == f"{pinned}/responses?api-version=2023-05-15"


def test_get_complete_url_discovers_when_unpinned():
    cfg, client = _config()
    url = cfg.get_complete_url(api_base=None, litellm_params={"api_key": "svc-key", "resource_group": "team-a"})
    assert url == f"{_DEPLOYMENT_URL}/responses?api-version={_DEFAULT_API_VERSION}"
    call = client.calls[0]
    assert call["url"] == f"{_BASE_URL}/lm/deployments"
    assert call["params"] == {"scenarioId": "foundation-models", "status": "RUNNING"}


def _transform(cfg, optional_params, model=f"deployment/{_MODEL}"):
    return cfg.transform_responses_api_request(
        model=model,
        input="hi",
        response_api_optional_request_params=dict(optional_params),
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def test_transform_request_strips_deployment_prefix_from_model():
    cfg, _ = _config()
    body = _transform(cfg, {"max_output_tokens": 200})
    assert body["model"] == _MODEL
    assert body["input"] == "hi"
    assert body["max_output_tokens"] == 200


def test_transform_request_drops_client_sent_beta_and_enable_thinking():
    """SAP direct-connect /responses is fail-closed on the body: a client-injected top-level `beta` or
    `enable_thinking` survives the OpenAI responses transform (ResponsesAPIRequestParams is a plain dict
    at runtime) and 400s the call. The allowlist must strip both while keeping legitimate responses params."""
    cfg, _ = _config()
    body = _transform(
        cfg,
        {"beta": True, "enable_thinking": True, "reasoning": {"effort": "low"}, "max_output_tokens": 200},
    )
    assert "beta" not in body
    assert "enable_thinking" not in body
    assert body["reasoning"] == {"effort": "low"}
    assert body["max_output_tokens"] == 200


def test_transform_request_drops_arbitrary_unknown_field():
    """The allowlist is not a per-field blocklist: any field outside the model's supported responses params
    is dropped, so a future unknown key a client sends cannot crash the SAP call."""
    cfg, _ = _config()
    body = _transform(cfg, {"some_future_unknown": "x", "temperature": 0.4})
    assert "some_future_unknown" not in body
    assert body["temperature"] == 0.4
    assert body["model"] == _MODEL
    assert body["input"] == "hi"


class TestSapResponsesRegistry:
    """`_get_python_responses_api_config` gates the SAP responses config to GPT deployments only: Claude
    runs on a Bedrock invoke executable with no native Responses surface, so it must return None and keep
    flowing through the chat-completions bridge, and Gemini has no direct-connect responses path either."""

    @pytest.mark.parametrize("model", ["sap/deployment/gpt-5.5", "deployment/gpt-5.5", "sap/gpt-4o"])
    def test_registry_returns_config_for_gpt(self, model):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(provider="sap", model=model)
        assert isinstance(cfg, SapDeploymentOpenAIResponsesConfig)

    def test_registry_config_carries_bare_model(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(provider="sap", model="sap/deployment/gpt-5.5")
        assert isinstance(cfg, SapDeploymentOpenAIResponsesConfig)
        assert cfg._model == "gpt-5.5"

    @pytest.mark.parametrize(
        "model",
        ["sap/deployment/anthropic--claude-4.8-opus", "sap/deployment/gemini-3.5-flash"],
    )
    def test_registry_returns_none_for_non_gpt(self, model):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(provider="sap", model=model)
        assert cfg is None

    def test_registry_returns_none_for_missing_model(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(provider="sap", model=None)
        assert cfg is None
