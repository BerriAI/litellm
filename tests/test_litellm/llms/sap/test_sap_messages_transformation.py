import pytest

from litellm.llms.sap.chat.handler import GenAIHubOrchestrationError
from litellm.llms.sap.direct_connect import reset_discovery_memo
from litellm.llms.sap.messages.transformation import SapDeploymentAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams

_MODEL = "anthropic--claude-4.8-opus"
_BASE_URL = "https://api.example.com/v2"
_RESOURCE_GROUP = "default"
_DEPLOYMENT_URL = "https://api.example.com/v2/inference/deployments/d38af17dc133a768"

_DISCOVERY_PAYLOAD = {
    "count": 2,
    "resources": [
        {
            "id": "gpt",
            "createdAt": "2026-09-10T07:31:33Z",
            "status": "RUNNING",
            "scenarioId": "foundation-models",
            "deploymentUrl": "https://api.example.com/v2/inference/deployments/gpt",
            "details": {"resources": {"backendDetails": {"model": {"name": "gpt-4o", "version": "latest"}}}},
        },
        {
            "id": "d38af17dc133a768",
            "createdAt": "2026-07-14T02:44:50Z",
            "status": "RUNNING",
            "scenarioId": "foundation-models",
            "deploymentUrl": _DEPLOYMENT_URL,
            "details": {"resources": {"backendDetails": {"model": {"name": _MODEL, "version": "1"}}}},
        },
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


def _fake_token_factory(calls):
    def factory(service_key, *, resource_group=None):
        calls.append(service_key)
        return (lambda: "Bearer test-token", _BASE_URL, resource_group or _RESOURCE_GROUP)

    return factory


def _config(payload=_DISCOVERY_PAYLOAD):
    client = _FakeHTTPClient(payload)
    cfg = SapDeploymentAnthropicMessagesConfig(
        http_client=client,
        token_creator_factory=_fake_token_factory([]),
    )
    return cfg, client


def _validate(cfg, *, api_base=None):
    return cfg.validate_anthropic_messages_environment(
        headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        model=_MODEL,
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="svc-key",
        api_base=api_base,
    )


def test_provider_tag_is_sap():
    cfg, _ = _config()
    assert cfg.custom_llm_provider == "sap"


def test_get_complete_url_builds_invoke_suffix_when_not_streaming():
    cfg, _ = _config()
    url = cfg.get_complete_url(
        api_base=_DEPLOYMENT_URL, api_key=None, model=_MODEL, optional_params={}, litellm_params={}, stream=False
    )
    assert url == f"{_DEPLOYMENT_URL}/invoke"


def test_get_complete_url_builds_stream_suffix_when_streaming():
    cfg, _ = _config()
    url = cfg.get_complete_url(
        api_base=_DEPLOYMENT_URL, api_key=None, model=_MODEL, optional_params={}, litellm_params={}, stream=True
    )
    assert url == f"{_DEPLOYMENT_URL}/invoke-with-response-stream"


def test_get_complete_url_raises_when_no_deployment_url():
    cfg, _ = _config()
    with pytest.raises(GenAIHubOrchestrationError) as exc:
        cfg.get_complete_url(
            api_base=None, api_key=None, model=_MODEL, optional_params={}, litellm_params={}, stream=False
        )
    assert exc.value.status_code == 500


def test_validate_injects_sap_headers_and_resolves_deployment_url():
    cfg, client = _config()
    headers, api_base = _validate(cfg)

    assert api_base == _DEPLOYMENT_URL
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["AI-Resource-Group"] == _RESOURCE_GROUP
    assert headers["Content-Type"] == "application/json"
    assert headers["AI-Client-Type"] == "LiteLLM"
    # incoming headers are preserved, not dropped
    assert headers["anthropic-beta"] == "prompt-caching-2024-07-31"

    call = client.calls[0]
    assert call["url"] == f"{_BASE_URL}/lm/deployments"
    assert call["params"] == {"scenarioId": "foundation-models", "status": "RUNNING"}
    assert call["headers"]["Authorization"] == "Bearer test-token"
    assert call["headers"]["AI-Resource-Group"] == _RESOURCE_GROUP


def test_litellm_params_resource_group_reaches_ai_resource_group_header():
    cfg, client = _config()
    headers, _ = cfg.validate_anthropic_messages_environment(
        headers={},
        model=_MODEL,
        messages=[],
        optional_params={},
        litellm_params={"resource_group": "team-a"},
        api_key="svc-key",
        api_base=None,
    )
    assert headers["AI-Resource-Group"] == "team-a"
    assert client.calls[0]["headers"]["AI-Resource-Group"] == "team-a"


def test_explicit_api_base_pin_skips_discovery():
    cfg, client = _config()
    pinned = "https://api.example.com/v2/inference/deployments/pinned"
    headers, api_base = _validate(cfg, api_base=pinned)

    assert api_base == pinned
    assert headers["Authorization"] == "Bearer test-token"
    assert client.calls == []


def test_discovery_result_is_cached_across_calls():
    cfg, client = _config()
    first = _validate(cfg)[1]
    second = _validate(cfg)[1]

    assert first == second == _DEPLOYMENT_URL
    assert len(client.calls) == 1


def test_deployment_prefix_is_stripped_before_backend_match():
    cfg, client = _config()
    headers, api_base = cfg.validate_anthropic_messages_environment(
        headers={},
        model=f"deployment/{_MODEL}",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="svc-key",
        api_base=None,
    )
    assert api_base == _DEPLOYMENT_URL
    assert client.calls[0]["params"] == {"scenarioId": "foundation-models", "status": "RUNNING"}


def test_validate_raises_400_when_credentials_missing():
    def failing_factory(service_key, *, resource_group=None):
        raise ValueError("No SAP AI Core service key found")

    cfg = SapDeploymentAnthropicMessagesConfig(
        http_client=_FakeHTTPClient(_DISCOVERY_PAYLOAD),
        token_creator_factory=failing_factory,
    )
    with pytest.raises(GenAIHubOrchestrationError) as exc:
        _validate(cfg)
    assert exc.value.status_code == 400


def test_sign_request_is_noop_when_not_streaming():
    cfg, _ = _config()
    incoming = {"Authorization": "Bearer test-token"}
    headers, body = cfg.sign_request(
        headers=incoming, optional_params={}, request_data={"max_tokens": 1}, api_base=_DEPLOYMENT_URL, stream=False
    )
    assert body is None
    assert "accept" not in headers
    assert headers["Authorization"] == "Bearer test-token"


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


def test_request_transform_strips_cache_control_scope_but_keeps_ephemeral():
    cfg, _ = _config()
    request = cfg.transform_anthropic_messages_request(
        model=_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "scope": "global"}}
                ],
            }
        ],
        anthropic_messages_optional_request_params={
            "max_tokens": 16,
            "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "scope": "global"}}],
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    system_cc = request["system"][0]["cache_control"]
    message_cc = request["messages"][0]["content"][0]["cache_control"]
    assert system_cc == {"type": "ephemeral"}
    assert message_cc == {"type": "ephemeral"}
    assert request["anthropic_version"] == "bedrock-2023-05-31"
    assert "stream" not in request
    assert "model" not in request


def test_transform_anthropic_messages_request_drops_unknown_fields_that_sap_rejects():
    cfg, _ = _config()
    request = cfg.transform_anthropic_messages_request(
        model=_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "max_tokens": 4096,
            "temperature": 1,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "enable_thinking": True,
            "beta": True,
            "enable_prompt_caching": False,
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "enable_thinking" not in request
    assert "beta" not in request
    assert "enable_prompt_caching" not in request
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "low"}
    assert request["max_tokens"] == 4096
    assert request["temperature"] == 1
    assert request["anthropic_version"] == "bedrock-2023-05-31"


def test_transform_anthropic_messages_request_keeps_full_anthropic_invoke_body_shape():
    cfg, _ = _config()
    request = cfg.transform_anthropic_messages_request(
        model=_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        anthropic_messages_optional_request_params={
            "system": [{"type": "text", "text": "sys"}],
            "max_tokens": 4096,
            "temperature": 1,
            "top_p": 0.9,
            "stop_sequences": ["x"],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "tools": [{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "metadata": {"user_id": "u"},
        },
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    for kept in ("system", "top_p", "stop_sequences", "tools", "tool_choice", "metadata"):
        assert kept in request
