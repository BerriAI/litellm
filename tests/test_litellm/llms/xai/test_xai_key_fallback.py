import asyncio
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.realtime_api import main as realtime_main
from litellm.types.router import GenericLiteLLMParams


class FakeLogging:
    def update_from_kwargs(self, **kwargs):
        pass


def test_get_api_key_prefers_xai_key_over_environment_and_generic_key(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    assert XAIModelInfo.get_api_key(None) == "xai_key_value"


def test_get_api_key_prefers_explicit_key_for_both_orderings(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    assert XAIModelInfo.get_api_key("param_api_key") == "param_api_key"
    assert (
        XAIModelInfo.get_api_key("param_api_key", legacy_generic_before_env=True)
        == "param_api_key"
    )


def test_get_api_key_prefers_environment_over_generic_key_by_default(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    assert XAIModelInfo.get_api_key(None) == "env_api_key"


def test_get_api_key_does_not_use_generic_key_by_default(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    assert XAIModelInfo.get_api_key(None) is None


def test_get_api_key_legacy_order_prefers_generic_key_over_env(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    assert (
        XAIModelInfo.get_api_key(None, legacy_generic_before_env=True)
        == "common_api_key"
    )


def test_get_api_key_legacy_order_prefers_xai_key_over_generic_key(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    assert (
        XAIModelInfo.get_api_key(None, legacy_generic_before_env=True)
        == "xai_key_value"
    )


def test_get_api_key_returns_none_when_no_key_is_available(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    assert XAIModelInfo.get_api_key(None) is None


def test_chat_config_uses_xai_key_fallback(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    _, api_key = XAIChatConfig()._get_openai_compatible_provider_info(None, None)

    assert api_key == "xai_key_value"


def test_chat_config_uses_environment_key_fallback(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    _, api_key = XAIChatConfig()._get_openai_compatible_provider_info(None, None)

    assert api_key == "env_api_key"


def test_chat_config_does_not_use_generic_key_fallback(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    _, api_key = XAIChatConfig()._get_openai_compatible_provider_info(None, None)

    assert api_key is None


def test_chat_config_prefers_explicit_api_key(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    _, api_key = XAIChatConfig()._get_openai_compatible_provider_info(
        None, "param_api_key"
    )

    assert api_key == "param_api_key"


def test_responses_config_preserves_generic_key_precedence(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    headers = XAIResponsesAPIConfig().validate_environment({}, "xai/grok-3-mini", None)

    assert headers["Authorization"] == "Bearer common_api_key"


def test_responses_config_prefers_litellm_params_api_key(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    headers = XAIResponsesAPIConfig().validate_environment(
        {},
        "xai/grok-3-mini",
        GenericLiteLLMParams(api_key="param_api_key"),
    )

    assert headers["Authorization"] == "Bearer param_api_key"


def test_responses_config_uses_environment_key_fallback(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    headers = XAIResponsesAPIConfig().validate_environment({}, "xai/grok-3-mini", None)

    assert headers["Authorization"] == "Bearer env_api_key"


def test_responses_config_raises_when_no_key_is_available(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(ValueError) as exc_info:
        XAIResponsesAPIConfig().validate_environment({}, "xai/grok-3-mini", None)

    error_message = str(exc_info.value)
    assert "api_key" in error_message
    assert "litellm.xai_key" in error_message
    assert "litellm.api_key" in error_message
    assert "XAI_API_KEY" in error_message


def test_responses_config_prefers_xai_key_over_generic_key(monkeypatch):
    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")

    headers = XAIResponsesAPIConfig().validate_environment({}, "xai/grok-3-mini", None)

    assert headers["Authorization"] == "Bearer xai_key_value"


def test_realtime_config_uses_xai_key_through_provider_resolution(monkeypatch):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")
    monkeypatch.setattr(
        realtime_main.xai_realtime, "async_realtime", mock_async_realtime
    )

    asyncio.run(
        realtime_main._arealtime(
            model="xai/grok-4-1-fast-non-reasoning",
            websocket=object(),
            litellm_logging_obj=FakeLogging(),
        )
    )

    assert captured_kwargs["api_key"] == "xai_key_value"


def test_realtime_config_uses_xai_key_when_provider_does_not_resolve_key(monkeypatch):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    def mock_get_llm_provider(model, api_base, api_key):
        return model, "xai", None, api_base

    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.setenv("XAI_API_KEY", "env_api_key")
    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(
        realtime_main.xai_realtime, "async_realtime", mock_async_realtime
    )

    asyncio.run(
        realtime_main._arealtime(
            model="xai/grok-4-1-fast-non-reasoning",
            websocket=object(),
            litellm_logging_obj=FakeLogging(),
        )
    )

    assert captured_kwargs["api_key"] == "xai_key_value"


def test_realtime_config_uses_generic_key_when_provider_does_not_resolve_key(
    monkeypatch,
):
    captured_kwargs = {}

    async def mock_async_realtime(**kwargs):
        captured_kwargs.update(kwargs)

    def mock_get_llm_provider(model, api_base, api_key):
        return model, "xai", None, api_base

    monkeypatch.setattr(litellm, "xai_key", None)
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(realtime_main, "get_llm_provider", mock_get_llm_provider)
    monkeypatch.setattr(
        realtime_main.xai_realtime, "async_realtime", mock_async_realtime
    )

    asyncio.run(
        realtime_main._arealtime(
            model="xai/grok-4-1-fast-non-reasoning",
            websocket=object(),
            litellm_logging_obj=FakeLogging(),
        )
    )

    assert captured_kwargs["api_key"] == "common_api_key"


def test_get_models_uses_xai_key_fallback(monkeypatch):
    captured_kwargs = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "grok-test"}]}

    def mock_get(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(litellm, "xai_key", "xai_key_value")
    monkeypatch.setattr(litellm, "api_key", "common_api_key")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm.module_level_client, "get", mock_get)

    assert XAIModelInfo().get_models() == ["xai/grok-test"]
    assert captured_kwargs["headers"]["Authorization"] == "Bearer xai_key_value"
