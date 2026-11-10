"""
Tests for Crusoe provider integration
"""

import asyncio
import importlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from unittest import mock

import pytest
import respx

import litellm
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome

CRUSOE_API_BASE = "https://api.inference.crusoecloud.com/v1"


def test_crusoe_json_registry():
    """Test CrusoeChatConfig is loaded from JSON provider registry"""
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("crusoe")
    config = JSONProviderRegistry.get("crusoe")
    assert config is not None
    assert config.base_url == CRUSOE_API_BASE
    assert config.api_key_env == "CRUSOE_API_KEY"
    assert config.api_base_env == "CRUSOE_API_BASE"


def test_crusoe_get_openai_compatible_provider_info():
    """Test Crusoe provider info retrieval"""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("crusoe"))()

    # Test with default values (no env vars set)
    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == CRUSOE_API_BASE
        assert api_key is None

    # Test with environment variables
    with mock.patch.dict(
        os.environ,
        {
            "CRUSOE_API_KEY": "test-key",
            "CRUSOE_API_BASE": "https://custom.crusoecloud.com/v1",
        },
    ):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.crusoecloud.com/v1"
        assert api_key == "test-key"

    # Test with explicit parameters (should override env vars)
    with mock.patch.dict(
        os.environ,
        {
            "CRUSOE_API_KEY": "env-key",
            "CRUSOE_API_BASE": "https://env.crusoecloud.com/v1",
        },
    ):
        api_base, api_key = config.get_openai_compatible_provider_info("https://param.crusoecloud.com/v1", "param-key")
        assert api_base == "https://param.crusoecloud.com/v1"
        assert api_key == "param-key"


def test_get_llm_provider_crusoe():
    """Test that get_llm_provider correctly identifies Crusoe"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with crusoe/model-name format
    model, provider, api_key, api_base = get_llm_provider("crusoe/meta-llama/Llama-3.3-70B-Instruct")
    assert model == "meta-llama/Llama-3.3-70B-Instruct"
    assert provider == "crusoe"


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))


_EXTERNAL_SLUG: Final = "external_openai_like"
_EXTERNAL_BASE: Final = "https://llm.example.test/v1"
_EXTERNAL_KEY_ENV: Final = "EXTERNAL_OPENAI_LIKE_API_KEY"


def _providers_file(path: Path, providers: Mapping[str, Mapping[str, object]]) -> Path:
    path.write_text(json.dumps(providers))
    return path


def _external_provider(base_url: str = _EXTERNAL_BASE) -> Mapping[str, Mapping[str, object]]:
    entry: Final = {
        "base_url": base_url,
        "api_key_env": _EXTERNAL_KEY_ENV,
        "constraints": {"temperature_max": 0.2},
        "param_mappings": {"max_completion_tokens": "max_tokens"},
    }
    return {_EXTERNAL_SLUG: entry}


@contextmanager
def _registry_snapshot() -> Iterator[None]:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    before: Final = dict(JSONProviderRegistry._providers)
    try:
        yield
    finally:
        JSONProviderRegistry._providers.clear()
        JSONProviderRegistry._providers.update(before)


def test_external_providers_file_resolves_base_url_and_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    path: Final = _providers_file(tmp_path / "providers.json", _external_provider())
    monkeypatch.setenv(_EXTERNAL_KEY_ENV, "sk-external")
    assert JSONProviderRegistry.exists(_EXTERNAL_SLUG) is False
    with _registry_snapshot():
        JSONProviderRegistry.load_file(str(path))
        assert JSONProviderRegistry.exists("crusoe") is True
        model, provider, api_key, api_base = get_llm_provider(f"{_EXTERNAL_SLUG}/demo-model")
    assert model == "demo-model"
    assert provider == _EXTERNAL_SLUG
    assert api_base == _EXTERNAL_BASE
    assert api_key == "sk-external"


def test_later_providers_file_replaces_the_same_slug(tmp_path: Path) -> None:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    first: Final = _providers_file(tmp_path / "first.json", _external_provider())
    second: Final = _providers_file(
        tmp_path / "second.json",
        _external_provider(base_url="https://two.example.test/v1"),
    )
    with _registry_snapshot():
        JSONProviderRegistry.load_file(str(first))
        JSONProviderRegistry.load_file(str(second))
        loaded: Final = JSONProviderRegistry.get(_EXTERNAL_SLUG)
        assert loaded is not None
        assert loaded.base_url == "https://two.example.test/v1"
        assert JSONProviderRegistry.exists("crusoe") is True


def test_relative_providers_path_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    monkeypatch.chdir(tmp_path)
    _providers_file(Path("providers.json"), _external_provider())
    with _registry_snapshot():
        JSONProviderRegistry.load_file("providers.json")
        loaded: Final = JSONProviderRegistry.get(_EXTERNAL_SLUG)
        assert loaded is not None
        assert loaded.base_url == _EXTERNAL_BASE


def test_missing_providers_file_raises_not_found(tmp_path: Path) -> None:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    missing: Final = tmp_path / "missing.json"
    with pytest.raises(Exception, match=f"Providers file not found: {missing}"):
        JSONProviderRegistry.load_file(str(missing))


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "[]",
        json.dumps({"external_openai_like": {"base_url": "https://llm.example.test/v1"}}),
    ],
)
def test_bad_providers_file_names_the_path(tmp_path: Path, contents: str) -> None:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    path: Final = tmp_path / "providers.json"
    path.write_text(contents)
    with pytest.raises(Exception, match=f"Error loading providers file {path}"):
        JSONProviderRegistry.load_file(str(path))


def test_empty_providers_file_keeps_bundled_providers(tmp_path: Path) -> None:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    path: Final = _providers_file(tmp_path / "providers.json", {})
    with _registry_snapshot():
        JSONProviderRegistry.load_file(str(path))
        assert JSONProviderRegistry.exists("crusoe") is True
        assert JSONProviderRegistry.exists(_EXTERNAL_SLUG) is False


def test_external_provider_completion_uses_file_base_url_and_temperature_cap(tmp_path: Path) -> None:
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    path: Final = _providers_file(tmp_path / "providers.json", _external_provider())
    with _registry_snapshot():
        JSONProviderRegistry.load_file(str(path))
        with respx.mock() as upstream:
            route: Final = upstream.post(f"{_EXTERNAL_BASE}/chat/completions").respond(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "demo-model",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
            response: Final = litellm.completion(
                model=f"{_EXTERNAL_SLUG}/demo-model",
                api_key="sk-external",
                messages=[{"role": "user", "content": "Say hello"}],
                temperature=1.5,
                max_completion_tokens=7,
            )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert str(request.url) == f"{_EXTERNAL_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-external"
    assert body["model"] == "demo-model"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 7
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "hello"


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}
