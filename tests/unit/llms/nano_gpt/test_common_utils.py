from typing import Final

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.nano_gpt.common_utils import NanoGPTModelInfo


@pytest.mark.parametrize("api_base", ["https://catalog.example/api/v1", "https://catalog.example/api/v1/"])
def test_catalog_preserves_publisher_names_and_uses_deployment_credentials(api_base: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://catalog.example/api/v1/models"
        assert request.headers["Authorization"] == "Bearer deployment-key"
        return httpx.Response(
            200,
            json={"data": [{"id": "openai/example"}, {"id": "nano/model"}, {"id": "openai/example"}]},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        provider: Final = NanoGPTModelInfo(client=HTTPHandler(client=client))
        assert provider.get_models(api_key="deployment-key", api_base=api_base) == [
            "nano-gpt/openai/example",
            "nano-gpt/nano/model",
        ]


def test_catalog_uses_nanogpt_environment_without_openai_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NANOGPT_API_BASE", "https://catalog.example/subscription/v1")
    monkeypatch.setenv("NANOGPT_API_KEY", "nanogpt-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://catalog.example/subscription/v1/models"
        assert request.headers["Authorization"] == "Bearer nanogpt-key"
        return httpx.Response(200, json={"data": [{"id": "publisher/subscription-model"}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        provider: Final = NanoGPTModelInfo(client=HTTPHandler(client=client))
        assert provider.get_models() == ["nano-gpt/publisher/subscription-model"]


def test_public_catalog_does_not_send_an_unrelated_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NANOGPT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")

    def respond(request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"data": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert NanoGPTModelInfo(client=HTTPHandler(client=client)).get_models() == []


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_catalog_errors_and_redirects_do_not_produce_models(status: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "catalog.example"
        return httpx.Response(status, headers={"Location": "https://other.example/models"})

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True) as client:
        provider: Final = NanoGPTModelInfo(client=HTTPHandler(client=client))
        with pytest.raises(httpx.HTTPStatusError):
            provider.get_models(api_key="deployment-key", api_base="https://catalog.example/api/v1")


@pytest.mark.parametrize("payload", [{}, {"data": [{"id": ""}]}, {"data": [{"id": None}]}])
def test_malformed_catalog_is_rejected(payload: object) -> None:
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        with pytest.raises(ValidationError):
            NanoGPTModelInfo(client=HTTPHandler(client=client)).get_models()


def test_base_model_removes_only_the_litellm_provider_prefix() -> None:
    assert NanoGPTModelInfo.get_base_model("nano-gpt/openai/example") == "openai/example"
    assert NanoGPTModelInfo.get_base_model("publisher/nano-gpt/example") == "publisher/nano-gpt/example"
