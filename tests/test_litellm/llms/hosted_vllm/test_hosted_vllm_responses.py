import litellm
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_responses_config_for_hosted_vllm():
    cfg = ProviderConfigManager.get_provider_responses_api_config(
        provider=litellm.LlmProviders.HOSTED_VLLM,
        model="gpt-oss-120b",
    )
    assert cfg is not None


def test_hosted_vllm_responses_uses_native_http_handler(monkeypatch):
    from litellm import responses as litellm_responses
    from litellm.responses import main as responses_main

    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://localhost:8000")

    called = {"native": False, "fallback": False}

    def fake_native(*args, **kwargs):
        called["native"] = True
        return responses_main.mock_responses_api_response(mock_response="ok")

    def fake_fallback(*args, **kwargs):
        called["fallback"] = True
        raise AssertionError(
            "responses() should not fall back to completion translation for hosted_vllm"
        )

    monkeypatch.setattr(
        responses_main.base_llm_http_handler, "response_api_handler", fake_native
    )
    monkeypatch.setattr(
        responses_main.litellm_completion_transformation_handler,
        "response_api_handler",
        fake_fallback,
    )

    resp = litellm_responses(model="hosted_vllm/gpt-oss-120b", input="hi")

    assert called["native"] is True
    assert called["fallback"] is False
    assert resp is not None


def test_hosted_vllm_responses_falls_back_when_endpoint_missing(monkeypatch):
    """
    Backwards-compatible behavior: if a hosted_vllm deployment doesn't expose `/responses`
    yet (404/405/501), LiteLLM should fall back to the chat-completions bridge.
    """
    from litellm import responses as litellm_responses
    from litellm.responses import main as responses_main

    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://localhost:8000")

    called = {"native": False, "fallback": False}

    class _NotFound(Exception):
        status_code = 404

    def fake_native(*args, **kwargs):
        called["native"] = True
        raise _NotFound()

    def fake_fallback(*args, **kwargs):
        called["fallback"] = True
        return responses_main.mock_responses_api_response(mock_response="ok")

    monkeypatch.setattr(
        responses_main.base_llm_http_handler, "response_api_handler", fake_native
    )
    monkeypatch.setattr(
        responses_main.litellm_completion_transformation_handler,
        "response_api_handler",
        fake_fallback,
    )

    resp = litellm_responses(model="hosted_vllm/gpt-oss-120b", input="hi")

    assert called["native"] is True
    assert called["fallback"] is True
    assert resp is not None

