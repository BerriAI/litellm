import platform
from typing import Final

import httpx
import openai
import pytest
import respx

import litellm
from litellm._version import version as LITELLM_VERSION
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig

SARVAM_USER_AGENT: Final = f"litellm/{LITELLM_VERSION} Python/{platform.python_version()}"


def _sarvam_config() -> SimpleProviderConfig:
    config: Final = JSONProviderRegistry.get("sarvam")
    assert config is not None
    return config


def _sarvam_chat_url() -> str:
    return f"{_sarvam_config().base_url}/chat/completions"


def _chat_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "sarvam-105b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_sarvam_attribution_headers_report_running_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)

    assert dict(_sarvam_config().get_attribution_headers()) == {"User-Agent": SARVAM_USER_AGENT}


@pytest.mark.parametrize("override", ["my-gateway/2.0", ""])
def test_litellm_user_agent_env_replaces_the_template(monkeypatch: pytest.MonkeyPatch, override: str) -> None:
    monkeypatch.setenv("LITELLM_USER_AGENT", override)

    assert dict(_sarvam_config().get_attribution_headers()) == {"User-Agent": override}


def test_generated_config_class_forwards_attribution_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    instance: Final = create_config_class(_sarvam_config())()

    assert dict(instance.get_attribution_headers()) == {"User-Agent": SARVAM_USER_AGENT}


def test_json_provider_without_user_agent_sends_no_attribution_headers() -> None:
    config: Final = JSONProviderRegistry.get("aiand")
    assert config is not None

    assert dict(config.get_attribution_headers()) == {}


def test_sarvam_chat_completion_sends_the_litellm_user_agent(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    route: Final = respx_mock.post(_sarvam_chat_url()).mock(return_value=httpx.Response(200, json=_chat_response()))

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == SARVAM_USER_AGENT


def test_sarvam_caller_user_agent_wins(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_sarvam_chat_url()).mock(return_value=httpx.Response(200, json=_chat_response()))

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
        headers={"User-Agent": "caller-app/1.0"},
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == "caller-app/1.0"


class _BrandedClient(openai.OpenAI):
    @property
    def user_agent(self) -> str:
        return "branded-app/3.0"


class _UnbrandedSubclass(openai.OpenAI):
    pass


@pytest.mark.parametrize(
    ("client_class", "client_default_headers", "expected_user_agent"),
    [
        pytest.param(
            openai.OpenAI, {"User-Agent": "client-default/1.0"}, "client-default/1.0", id="client_user_agent_wins"
        ),
        pytest.param(_BrandedClient, {}, "branded-app/3.0", id="subclass_user_agent_wins"),
        pytest.param(openai.OpenAI, {}, SARVAM_USER_AGENT, id="sdk_default_user_agent_replaced"),
        pytest.param(_UnbrandedSubclass, {}, SARVAM_USER_AGENT, id="subclass_sdk_default_user_agent_replaced"),
    ],
)
def test_sarvam_user_agent_with_a_caller_client(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    client_class: type[openai.OpenAI],
    client_default_headers: dict[str, str],
    expected_user_agent: str,
) -> None:
    monkeypatch.delenv("LITELLM_USER_AGENT", raising=False)
    route: Final = respx_mock.post(_sarvam_chat_url()).mock(return_value=httpx.Response(200, json=_chat_response()))
    client: Final = client_class(
        api_key="test-key", base_url=_sarvam_config().base_url, default_headers=client_default_headers
    )

    litellm.completion(
        model="sarvam/sarvam-105b",
        api_key="test-key",
        messages=[{"role": "user", "content": "Hello"}],
        client=client,
    )

    assert route.called
    assert route.calls.last.request.headers["User-Agent"] == expected_user_agent
