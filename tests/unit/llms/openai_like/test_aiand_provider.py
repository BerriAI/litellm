import datetime
import json
import time
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.health_check_helpers import wildcard_health_check_models
from litellm.litellm_core_utils.model_retirement import is_past_deprecation_date
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
from litellm.utils import ProviderConfigManager

AIAND_API_BASE = "https://api.aiand.com/v1"


def test_aiand_chat_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIAND_API_KEY", "test-key")

    provider_config = JSONProviderRegistry.get("aiand")
    assert provider_config is not None
    assert litellm.LlmProviders.AIAND.value == "aiand"
    assert "aiand" in litellm.openai_compatible_providers
    assert provider_config.base_url == AIAND_API_BASE
    assert provider_config.api_key_env == "AIAND_API_KEY"
    assert provider_config.api_base_env == "AIAND_API_BASE"

    model, provider, api_key, api_base = get_llm_provider("aiand/openai/gpt-oss-120b")
    assert model == "openai/gpt-oss-120b"
    assert provider == "aiand"
    assert api_key == "test-key"
    assert api_base == AIAND_API_BASE


def test_aiand_completion_request(respx_mock: respx.MockRouter):
    messages = [{"role": "user", "content": "Hello"}]
    route = respx_mock.post(f"{AIAND_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/gpt-oss-120b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
    )

    litellm.completion(
        model="aiand/openai/gpt-oss-120b",
        api_key="test-key",
        messages=messages,
    )

    assert route.called
    request = route.calls.last.request
    assert str(request.url) == f"{AIAND_API_BASE}/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    request_body = json.loads(request.read())
    assert request_body["model"] == "openai/gpt-oss-120b"
    assert request_body["messages"] == messages


def test_aiand_responses_provider():
    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="aiand",
        model="aiand/openai/gpt-oss-120b",
    )

    assert config is not None
    assert config.custom_llm_provider == "aiand"
    assert config.get_complete_url(api_base=None, litellm_params={}) == f"{AIAND_API_BASE}/responses"


def _aiand_catalog() -> frozenset[str]:
    return frozenset(name for name, info in litellm.model_cost.items() if info.get("litellm_provider") == "aiand")


def _utc_today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


def test_aiand_wildcard_lists_only_models_not_yet_retired():
    catalog: Final = _aiand_catalog()
    retired: Final = frozenset(
        name
        for name in catalog
        if (retired_on := litellm.model_cost[name].get("deprecation_date")) is not None
        and datetime.date.fromisoformat(retired_on) <= _utc_today()
    )

    assert catalog - retired
    assert frozenset(get_known_models_from_wildcard("aiand/*")) == catalog - retired


def test_aiand_wildcard_drops_a_model_whose_retirement_date_passes_while_running(monkeypatch: pytest.MonkeyPatch):
    model: Final = "aiand/acme/retires-while-running"
    monkeypatch.setitem(litellm.model_cost, model, {"litellm_provider": "aiand", "deprecation_date": "2999-01-01"})
    try:
        litellm.add_known_models({model: litellm.model_cost[model]})
        assert model in get_known_models_from_wildcard("aiand/*")

        monkeypatch.setitem(litellm.model_cost, model, {"litellm_provider": "aiand", "deprecation_date": "2000-01-01"})

        assert model not in get_known_models_from_wildcard("aiand/*")
    finally:
        litellm.aiand_models.discard(model)
        litellm.add_known_models({})


def test_aiand_models_by_provider_leaves_out_models_retired_at_load():
    retired: Final = frozenset(
        name
        for name, row in litellm.model_cost.items()
        if row.get("litellm_provider") == "aiand" and is_past_deprecation_date(row)
    )

    assert retired
    assert retired.isdisjoint(litellm.models_by_provider["aiand"])


def test_aiand_wildcard_health_check_skips_a_model_whose_retirement_date_passes_while_running(
    monkeypatch: pytest.MonkeyPatch,
):
    model: Final = "aiand/acme/free-and-retiring"
    live_row: Final = {
        "litellm_provider": "aiand",
        "mode": "chat",
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "deprecation_date": "2999-01-01",
    }
    monkeypatch.setitem(litellm.model_cost, model, live_row)
    try:
        litellm.add_known_models({model: live_row})
        assert model in wildcard_health_check_models(wildcard="aiand/*", custom_llm_provider="aiand")

        monkeypatch.setitem(litellm.model_cost, model, {**live_row, "deprecation_date": "2000-01-01"})

        assert model not in wildcard_health_check_models(wildcard="aiand/*", custom_llm_provider="aiand")
    finally:
        litellm.aiand_models.discard(model)
        litellm.add_known_models({})


@pytest.mark.parametrize("host_zone", ["Etc/GMT+12", "Etc/GMT-14"])
def test_aiand_models_retire_on_the_utc_date_whatever_the_host_zone(monkeypatch: pytest.MonkeyPatch, host_zone: str):
    today: Final = _utc_today()
    rows: Final = {
        "aiand/acme/retires-today": {"litellm_provider": "aiand", "deprecation_date": today.isoformat()},
        "aiand/acme/retires-tomorrow": {
            "litellm_provider": "aiand",
            "deprecation_date": (today + datetime.timedelta(days=1)).isoformat(),
        },
    }
    for name, row in rows.items():
        monkeypatch.setitem(litellm.model_cost, name, row)
    monkeypatch.setenv("TZ", host_zone)
    time.tzset()
    try:
        litellm.add_known_models(rows)
        listed: Final = frozenset(get_known_models_from_wildcard("aiand/*"))

        assert "aiand/acme/retires-today" not in listed
        assert "aiand/acme/retires-tomorrow" in listed
    finally:
        monkeypatch.undo()
        time.tzset()
        for name in rows:
            litellm.aiand_models.discard(name)
        litellm.add_known_models({})


def test_aiand_calls_are_billed_from_their_cost_rows():
    catalog: Final = _aiand_catalog()
    assert catalog

    for model in sorted(catalog):
        row: Final = litellm.model_cost[model]
        prompt_cost, completion_cost = litellm.cost_per_token(model=model, prompt_tokens=1000, completion_tokens=500)

        assert prompt_cost == pytest.approx(1000 * row["input_cost_per_token"])
        assert completion_cost == pytest.approx(500 * row["output_cost_per_token"])
        assert prompt_cost > 0 and completion_cost > 0


@pytest.mark.asyncio
async def test_aiand_anthropic_messages_reach_ai_and_with_the_thinking_budget(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    thinking: Final = {"type": "enabled", "budget_tokens": 1024}
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{AIAND_API_BASE}/messages").respond(
            200,
            json={
                "id": "msg_aiand",
                "type": "message",
                "role": "assistant",
                "model": "zai-org/glm-5.3",
                "content": [{"type": "thinking", "thinking": "A greeting", "signature": ""}, {"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="aiand/zai-org/glm-5.3",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=2048,
            thinking=thinking,
            api_key="aiand-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert body["model"] == "zai-org/glm-5.3"
    assert body["thinking"] == thinking
    assert "reasoning_effort" not in body
    assert [block["type"] for block in response["content"]] == ["thinking", "text"]
