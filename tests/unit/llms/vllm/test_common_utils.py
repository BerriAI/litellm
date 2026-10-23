import httpx
import pytest
import respx

from litellm.llms.vllm.common_utils import VLLMModelInfo

MODELS_URL = "http://vllm.example/v1/models"
MODEL_RESPONSE = {"data": [{"id": "Qwen/Qwen3-8B"}]}


def test_get_models_without_api_key_sends_no_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)

    with respx.mock:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=MODEL_RESPONSE))
        models = VLLMModelInfo().get_models(api_base="http://vllm.example/v1")

    assert models == ["Qwen/Qwen3-8B"]
    assert "authorization" not in route.calls[0].request.headers


def test_get_models_sends_api_key_as_bearer_token() -> None:
    with respx.mock:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=MODEL_RESPONSE))
        models = VLLMModelInfo().get_models(
            api_key="vllm-test-key",
            api_base="http://vllm.example/v1",
        )

    assert models == ["Qwen/Qwen3-8B"]
    assert route.calls[0].request.headers["authorization"] == "Bearer vllm-test-key"


def test_get_models_requires_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    monkeypatch.delenv("VLLM_API_BASE", raising=False)

    with pytest.raises(ValueError, match="HOSTED_VLLM_API_BASE"):
        VLLMModelInfo().get_models(api_key="vllm-test-key")


def test_get_api_key_falls_back_to_hosted_vllm_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "hosted-vllm-test-key")

    assert VLLMModelInfo.get_api_key() == "hosted-vllm-test-key"


def test_get_api_base_prefers_hosted_vllm_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "http://hosted-vllm.example/v1")
    monkeypatch.setenv("VLLM_API_BASE", "http://vllm.example/v1")

    assert VLLMModelInfo.get_api_base() == "http://hosted-vllm.example/v1"
