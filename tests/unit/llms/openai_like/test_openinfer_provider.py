import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.proxy.auth.model_checks import get_provider_models

_REPO_ROOT: Final = Path(__file__).resolve().parents[4]
_PRIMARY_PRICES: Final = json.loads((_REPO_ROOT / "model_prices_and_context_window.json").read_text())
_BACKUP_PRICES: Final = json.loads((_REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())
_CATALOG_MODELS: Final = tuple(
    key for key, row in _PRIMARY_PRICES.items() if isinstance(row, dict) and row.get("litellm_provider") == "openinfer"
)
_REASONING_MODELS: Final = tuple(key for key in _CATALOG_MODELS if _PRIMARY_PRICES[key].get("supports_reasoning"))
_CHAT_URL: Final = "https://api.openinfer.ai/v1/chat/completions"
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_CHAT_RESPONSE: Final = {
    "id": "chatcmpl_openinfer",
    "object": "chat.completion",
    "created": 1_792_000_000,
    "model": "@oi/Qwen3.5-27B",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello from OpenInfer"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def _sent_chat_request(model: str, **params: object) -> tuple[str, str, dict[str, object]]:
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_RESPONSE)
        litellm.completion(
            model=f"openinfer/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sk-test",
            **params,
        )
    request: Final = route.calls.last.request
    return str(request.url), request.headers["authorization"], json.loads(request.content)


def test_provider_prefixed_model_routes_to_openinfer():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="openinfer/@oi/Qwen3.5-27B",
        custom_llm_provider=None,
        api_base=None,
        api_key="sk-test",
    )

    assert model == "@oi/Qwen3.5-27B"
    assert provider == "openinfer"
    assert api_key == "sk-test"
    assert api_base == "https://api.openinfer.ai/v1"


def test_api_key_and_base_resolved_from_env(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPENINFER_API_KEY", "sk-env-key")
    monkeypatch.setenv("OPENINFER_API_BASE", "https://proxy.internal/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="openinfer/@oi/Qwen3.5-9B",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "openinfer"
    assert api_key == "sk-env-key"
    assert api_base == "https://proxy.internal/v1"


def test_url_autodetection_from_api_base(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPENINFER_API_KEY", "sk-env-key")

    _, provider, api_key, _ = get_llm_provider(
        model="@oi/Qwen3.5-27B",
        custom_llm_provider=None,
        api_base="https://api.openinfer.ai/v1",
        api_key=None,
    )

    assert provider == "openinfer"
    assert api_key == "sk-env-key"


def test_chat_request_reaches_openinfer_with_the_bare_model_id():
    url, authorization, body = _sent_chat_request("@oi/Qwen3.5-27B")

    assert url == _CHAT_URL
    assert authorization == "Bearer sk-test"
    assert body["model"] == "@oi/Qwen3.5-27B"


def test_tools_reach_openinfer_for_a_model_without_a_cost_map_row():
    _, _, body = _sent_chat_request("@oi/model-without-a-catalog-row", tools=[_WEATHER_TOOL])

    assert body["tools"] == [_WEATHER_TOOL]


def _register_openinfer_row(model: str, **capabilities: bool) -> None:
    litellm.register_model(
        {f"openinfer/{model}": {"litellm_provider": "openinfer", "mode": "chat", **capabilities}},
        persist_across_reloads=False,
        add_to_provider_model_sets=False,
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_model_info_keeps_a_row_that_turns_tools_off():
    model: Final = "@oi/row-without-tools"
    _register_openinfer_row(model, supports_function_calling=False)

    assert litellm.get_model_info(model, custom_llm_provider="openinfer")["supports_function_calling"] is False
    assert litellm.supports_function_calling(model, custom_llm_provider="openinfer") is False


@pytest.mark.usefixtures("local_model_cost_map")
def test_model_info_fills_tool_support_from_the_openinfer_default():
    model: Final = "@oi/row-silent-on-tools"
    _register_openinfer_row(model)

    assert litellm.get_model_info(model, custom_llm_provider="openinfer")["supports_function_calling"] is True
    assert litellm.supports_function_calling(model, custom_llm_provider="openinfer") is True


def test_max_tokens_and_max_completion_tokens_both_reach_openinfer():
    _, _, body = _sent_chat_request("@oi/Qwen3.5-27B", max_tokens=64, max_completion_tokens=128)

    assert body["max_tokens"] == 64
    assert body["max_completion_tokens"] == 128


@pytest.mark.parametrize("model", _REASONING_MODELS)
def test_reasoning_effort_reaches_openinfer_for_reasoning_rows(model: str):
    _, _, body = _sent_chat_request(model.removeprefix("openinfer/"), reasoning_effort="low")

    assert body["reasoning_effort"] == "low"


def test_responses_request_goes_to_openinfer_natively():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.openinfer.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_openinfer",
                "object": "response",
                "created_at": 1_792_000_000,
                "model": "@oi/Qwen3.5-27B",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_openinfer",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from OpenInfer", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(model="openinfer/@oi/Qwen3.5-27B", input="Say hello", api_key="sk-test")

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "@oi/Qwen3.5-27B"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from OpenInfer"


@pytest.mark.asyncio
async def test_anthropic_messages_request_goes_to_openinfer_natively(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.openinfer.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_openinfer",
                "type": "message",
                "role": "assistant",
                "model": "@oi/Qwen3.5-27B",
                "content": [{"type": "text", "text": "Hello from OpenInfer"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="openinfer/@oi/Qwen3.5-27B",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sk-test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "@oi/Qwen3.5-27B"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from OpenInfer"


def test_wildcard_lists_every_catalog_model():
    assert sorted(get_provider_models("openinfer") or []) == sorted(_CATALOG_MODELS)


@pytest.mark.parametrize("model", _CATALOG_MODELS)
def test_catalog_row_matches_its_backup_and_fits_its_context(model: str):
    row: Final = _PRIMARY_PRICES[model]

    assert row == _BACKUP_PRICES[model]
    assert row["mode"] == "chat"
    assert 0 < row["max_output_tokens"] <= row["max_input_tokens"]
    assert row["input_cost_per_token"] > 0
    assert row["output_cost_per_token"] > 0


@pytest.mark.parametrize("model", _CATALOG_MODELS)
def test_chat_response_is_priced_at_its_catalog_row(model: str):
    row: Final = litellm.model_cost[model]
    response: Final = litellm.ModelResponse(
        model=model.removeprefix("openinfer/"),
        usage=litellm.Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
    )

    cost: Final = litellm.completion_cost(completion_response=response, model=model)

    assert cost == pytest.approx(1000 * row["input_cost_per_token"] + 200 * row["output_cost_per_token"])
