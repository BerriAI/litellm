import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.types.utils import ModelResponse, Usage

STREAMLAKE_BASE_URL = "https://vanchin.streamlake.ai/api/gateway/v1/endpoints"
STREAMLAKE_CHAT_URL = f"{STREAMLAKE_BASE_URL}/chat/completions"

REPO_ROOT: Final = Path(__file__).parents[4]


class TestStreamlakeProviderConfig:
    def test_streamlake_provider_resolution(self):
        model, provider, api_key, api_base = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "ep-test-endpoint"
        assert provider == "streamlake"
        assert api_base == STREAMLAKE_BASE_URL
        assert api_key == "sk-test"

    def test_streamlake_api_base_override(self):
        model, provider, api_key, api_base = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base="https://custom.streamlake.example/v1",
            api_key="sk-test",
        )

        assert model == "ep-test-endpoint"
        assert provider == "streamlake"
        assert api_base == "https://custom.streamlake.example/v1"
        assert api_key == "sk-test"

    def test_streamlake_url_autodetection(self):
        model, provider, api_key, api_base = get_llm_provider(
            model="some-public-model",
            custom_llm_provider=None,
            api_base=STREAMLAKE_BASE_URL,
            api_key=None,
        )
        assert provider == "streamlake"
        assert api_base == STREAMLAKE_BASE_URL

    def test_streamlake_api_key_resolved_from_env(self, monkeypatch):
        monkeypatch.setenv("STREAMLAKE_API_KEY", "sk-env-key")

        _, provider, api_key, _ = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert provider == "streamlake"
        assert api_key == "sk-env-key"

    def test_streamlake_validate_environment_sets_bearer(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        headers = config.validate_environment(
            headers={},
            model="ep-test-endpoint",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-resolved-key",
            api_base=None,
        )
        assert headers["Authorization"] == "Bearer sk-resolved-key"

    def test_streamlake_complete_url_appends_chat_completions(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        url = config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="ep-test-endpoint",
            optional_params={},
            litellm_params={},
        )
        assert url == STREAMLAKE_CHAT_URL

    def test_streamlake_max_completion_tokens_mapped_to_max_tokens(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 1024},
            optional_params={},
            model="ep-test-endpoint",
            drop_params=False,
        )
        assert optional_params == {"max_tokens": 1024}
        assert "max_completion_tokens" not in optional_params

    def test_streamlake_chat_params_pass_through(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"temperature": 0.2, "top_p": 0.9, "stream": True},
            optional_params={},
            model="ep-test-endpoint",
            drop_params=False,
        )
        assert optional_params == {"temperature": 0.2, "top_p": 0.9, "stream": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async, api_base",
    (
        (False, None),
        (False, "https://custom.streamlake.example/v1"),
        (False, "https://custom.streamlake.example/v1/"),
        (True, None),
        (True, "https://custom.streamlake.example/v1"),
        (True, "https://custom.streamlake.example/v1/"),
    ),
    ids=(
        "sync-default",
        "sync-override",
        "sync-override-trailing-slash",
        "async-default",
        "async-override",
        "async-override-trailing-slash",
    ),
)
async def test_streamlake_completion_uses_openai_compatible_http(
    is_async: bool, api_base: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    response_body: Final[dict[str, object]] = {
        "id": "chatcmpl-streamlake-test",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "ep-test-endpoint",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "StreamLake response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    expected_url: Final[str] = f"{(api_base or STREAMLAKE_BASE_URL).rstrip('/')}/chat/completions"
    expected_request_body: Final[dict[str, object]] = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "ep-test-endpoint",
        "max_tokens": 512,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == expected_url
        assert request.headers["authorization"] == "Bearer sk-streamlake-test"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == expected_request_body
        return httpx.Response(200, json=response_body, request=request)

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(expected_url).mock(side_effect=handler)
        if is_async:
            response: Final = await litellm.acompletion(
                model="streamlake/ep-test-endpoint",
                messages=[{"role": "user", "content": "hello"}],
                api_key="sk-streamlake-test",
                api_base=api_base,
                max_completion_tokens=512,
            )
        else:
            response: Final = litellm.completion(
                model="streamlake/ep-test-endpoint",
                messages=[{"role": "user", "content": "hello"}],
                api_key="sk-streamlake-test",
                api_base=api_base,
                max_completion_tokens=512,
            )

    assert route.call_count == 1
    assert len(response.choices) == 1
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "StreamLake response"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 5


def _streamlake_keys(cost_map: dict[str, object]) -> tuple[str, ...]:
    return tuple(key for key in cost_map if key.startswith("streamlake/"))


def test_root_and_backup_cost_maps_carry_the_same_streamlake_rows() -> None:
    root: Final = json.loads((REPO_ROOT / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())

    assert _streamlake_keys(root)
    assert {key: root[key] for key in _streamlake_keys(root)} == {key: backup[key] for key in _streamlake_keys(backup)}


@pytest.mark.usefixtures("local_model_cost_map")
def test_endpoint_id_call_is_priced_from_the_lowercase_model_streamlake_returns() -> None:
    keys: Final = _streamlake_keys(litellm.model_cost)
    assert keys

    for key in keys:
        info = litellm.get_model_info(model=key)
        response = ModelResponse(
            model=key.split("/", 1)[1].lower(),
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        cost = litellm.completion_cost(
            completion_response=response,
            model="streamlake/ep-test-endpoint",
            custom_llm_provider="streamlake",
        )
        expected = 1000 * info["input_cost_per_token"] + 500 * info["output_cost_per_token"]
        assert cost == pytest.approx(expected), key


@pytest.mark.usefixtures("local_model_cost_map")
def test_tool_params_follow_the_function_calling_flag_of_the_lowercase_model_row() -> None:
    provider: Final = JSONProviderRegistry.get("streamlake")
    assert provider is not None
    config: Final = create_config_class(provider)()
    keys: Final = _streamlake_keys(litellm.model_cost)
    assert keys

    for key in keys:
        supported = config.get_supported_openai_params(model=key.split("/", 1)[1].lower())
        supports_function_calling = litellm.get_model_info(model=key).get("supports_function_calling") is True
        assert ("tools" in supported) is supports_function_calling, key
