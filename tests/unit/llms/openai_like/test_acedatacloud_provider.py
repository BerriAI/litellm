import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm import completion, cost_per_token, get_llm_provider, model_cost, responses
from litellm.llms.openai_like.acedatacloud import CATALOG_URL, AceDataCloudChatConfig
from litellm.llms.openai_like.dynamic_config import create_config_class, create_responses_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.types.utils import PromptTokensDetails, Usage
from litellm.utils import ProviderConfigManager


def test_acedatacloud_prefix_resolves_key_and_preserves_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-test-key")
    assert get_llm_provider("acedatacloud/gpt-6-luna") == (
        "gpt-6-luna",
        "acedatacloud",
        "ace-test-key",
        "https://api.acedata.cloud/v1",
    )


def test_acedatacloud_explicit_credentials_override_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "environment-key")
    assert get_llm_provider(
        "acedatacloud/gpt-6-luna", api_key="explicit-key", api_base="https://example.test/openai"
    ) == ("gpt-6-luna", "acedatacloud", "explicit-key", "https://example.test/openai")


def test_acedatacloud_chat_url_and_reasoning_tools_are_preserved() -> None:
    provider: Final = JSONProviderRegistry.get("acedatacloud")
    assert provider is not None
    config: Final = create_config_class(provider)()
    assert config.get_complete_url(None, "test-key", "gpt-6-luna", {}, {}) == (
        "https://api.acedata.cloud/v1/chat/completions"
    )
    supported: Final = config.get_supported_openai_params("gpt-6-luna")
    assert {"tools", "tool_choice", "reasoning_effort", "stream_options"}.issubset(supported)


def test_acedatacloud_responses_uses_standard_base_path() -> None:
    provider: Final = JSONProviderRegistry.get("acedatacloud")
    assert provider is not None
    config: Final = create_responses_config_class(provider)()
    assert config.get_complete_url(None, {}) == "https://api.acedata.cloud/v1/responses"


def test_acedatacloud_add_model_fields() -> None:
    fields: Final = json.loads(
        (Path(litellm.__file__).parent / "proxy/public_endpoints/provider_create_fields.json").read_text()
    )
    matches: Final = tuple(field for field in fields if field["litellm_provider"] == "acedatacloud")
    assert len(matches) == 1
    provider: Final = matches[0]
    assert provider["provider"] == "ACEDATACLOUD"
    assert provider["provider_display_name"] == "Ace Data Cloud"
    assert provider["default_model_placeholder"] == "acedatacloud/gpt-6.2-astra"
    assert tuple((field["key"], field["required"], field["field_type"]) for field in provider["credential_fields"]) == (
        ("api_key", True, "password"),
    )


def test_acedatacloud_chat_request_and_credit_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-test-key")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.acedata.cloud/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-ace",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-6-luna",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 5,
                    "total_tokens": 16,
                    "cost": {"currency": "credit", "amount": 7},
                },
            },
        )
        response: Final = completion(
            model="acedatacloud/gpt-6-luna",
            messages=[{"role": "user", "content": "Reply with exactly OK."}],
            reasoning_effort="none",
            max_completion_tokens=64,
            num_retries=0,
        )
    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer ace-test-key"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "reasoning_effort": "none",
        "max_completion_tokens": 64,
    }
    assert response.choices[0].message.content == "OK"
    assert "cost" not in response.model_dump(warnings="error")["usage"]
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (11, 5)


def test_stream_usage_keeps_tokens_and_excludes_credit_cost() -> None:
    choice = {"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}
    meta = {"id": "chatcmpl-fixture", "object": "chat.completion.chunk", "created": 1, "model": "gpt-6-luna"}
    events = [
        {**meta, "choices": [choice]},
        {**meta, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **meta,
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "total_tokens": 16,
                "cost": {"amount": 7, "currency": "credit"},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
    with respx.mock() as upstream:
        route = upstream.post("https://api.acedata.cloud/v1/chat/completions").respond(
            200, content=content, headers={"content-type": "text/event-stream"}
        )
        chunks = list(
            completion(
                model="acedatacloud/gpt-6-luna",
                api_key="fixture",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
                stream_options={"include_usage": True},
                num_retries=0,
            )
        )
    assert route.call_count == 1
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "OK"
    usage = [c.model_dump(warnings="error")["usage"] for c in chunks if getattr(c, "usage", None)]
    assert usage and usage[-1]["total_tokens"] == 16
    assert not isinstance(usage[-1].get("cost"), dict)


def test_acedatacloud_responses_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-test-key")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.acedata.cloud/v1/responses").respond(
            200,
            json={
                "id": "resp_ace",
                "object": "response",
                "created_at": 1,
                "model": "gpt-6-luna",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_ace",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 5,
                    "total_tokens": 16,
                    "cost": {"amount": 7, "currency": "credit"},
                },
            },
        )
        response: Final = responses(
            model="acedatacloud/gpt-6-luna",
            input="Reply with exactly OK.",
            reasoning={"effort": "none"},
            max_output_tokens=64,
            num_retries=0,
        )
    assert route.call_count == 1
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer ace-test-key"
    assert json.loads(request.content) == {
        "model": "gpt-6-luna",
        "input": "Reply with exactly OK.",
        "reasoning": {"effort": "none"},
        "max_output_tokens": 64,
    }
    assert response.output[0].content[0].text == "OK"
    assert response.model_dump(warnings="error")["usage"]["total_tokens"] == 16


@pytest.mark.parametrize(
    ("prompt_tokens", "suffix"),
    [(272000, ""), (272001, "_above_272k_tokens")],
)
def test_acedatacloud_cost_uses_its_catalog_and_long_prompt_rates(prompt_tokens: int, suffix: str) -> None:
    """272K boundary: public Ace Data Cloud model catalog, verified 2026-10-06.

    https://platform.acedata.cloud/api/v1/models/catalog/
    """
    rates: Final = model_cost["acedatacloud/gpt-6-luna"]
    usage: Final = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
        total_tokens=prompt_tokens + 5,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=10),
    )
    input_cost, output_cost = cost_per_token(
        model="acedatacloud/gpt-6-luna",
        custom_llm_provider="acedatacloud",
        usage_object=usage,
    )
    expected_input: Final = (prompt_tokens - 10) * rates[f"input_cost_per_token{suffix}"] + 10 * rates[
        f"cache_read_input_token_cost{suffix}"
    ]
    assert input_cost == pytest.approx(expected_input)
    assert output_cost == pytest.approx(5 * rates[f"output_cost_per_token{suffix}"])


def test_public_discovery_includes_new_chat_ids_without_sending_the_key() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.get(CATALOG_URL).respond(
            200,
            json={
                "items": [
                    {"id": "new-chat", "modality": "chat"},
                    {"id": "gpt-4o-image", "modality": "chat"},
                    {"id": "image-model", "modality": "image"},
                    {"id": "embedding-model", "modality": "embedding"},
                ]
            },
        )
        assert AceDataCloudChatConfig().get_models(api_key="private-key") == [
            "acedatacloud/new-chat",
            "acedatacloud/gpt-4o-image",
        ]
    assert "authorization" not in route.calls.last.request.headers


def test_native_endpoints_follow_each_model_contract() -> None:
    provider: Final = JSONProviderRegistry.get("acedatacloud")
    assert provider is not None
    for model, endpoints in provider.model_supported_endpoints.items():
        assert set(endpoints) == set(model_cost[f"acedatacloud/{model}"]["supported_endpoints"])
        config: Final = ProviderConfigManager.get_provider_responses_api_config("acedatacloud", model)
        assert (config is not None) == ("/v1/responses" in endpoints)


def test_custom_discovery_does_not_use_another_providers_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    monkeypatch.setenv("ACEDATACLOUD_API_KEY", "ace-secret")
    with respx.mock() as upstream:
        route: Final = upstream.get("https://custom.test/v1/models").respond(200, json={"data": [{"id": "custom"}]})
        assert AceDataCloudChatConfig().get_models(api_base="https://custom.test/v1") == ["custom"]
    assert route.calls.last.request.headers["authorization"] == "Bearer ace-secret"


@pytest.mark.asyncio
async def test_messages_only_id_uses_native_bearer_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock() as upstream:
        route = upstream.post("https://api.acedata.cloud/v1/messages").respond(
            200,
            json={
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )
        result = await litellm.anthropic_messages(
            model="acedatacloud/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
            api_key="fixture-key",
        )
    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer fixture-key"
    assert "x-api-key" not in route.calls.last.request.headers
    assert json.loads(route.calls.last.request.content)["model"] == "claude-sonnet-5-5"
    assert result["content"][0]["text"] == "OK"


def test_request_priced_models_do_not_turn_the_fee_into_token_pricing() -> None:
    for model, entry in model_cost.items():
        if model.startswith("acedatacloud/") and "input_cost_per_request" in entry:
            assert "input_cost_per_token" not in entry and "output_cost_per_token" not in entry
            response: Final = {
                "model": model,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            }
            assert litellm.completion_cost(
                completion_response=response, custom_llm_provider="acedatacloud"
            ) == pytest.approx(entry["input_cost_per_request"])


def test_explicit_token_price_override_takes_precedence_over_request_fee() -> None:
    assert cost_per_token(
        model="acedatacloud/gpt-4o-all",
        custom_llm_provider="acedatacloud",
        prompt_tokens=100,
        completion_tokens=10,
        custom_cost_per_token={"input_cost_per_token": 0.1, "output_cost_per_token": 0.2},
    ) == pytest.approx((10, 2))


@pytest.mark.parametrize("prompt_tokens", [199999, 200000, 200001])
def test_inclusive_prompt_band_keeps_its_first_billable_token(prompt_tokens: int) -> None:
    # Inclusive boundary in the API contract, 2026-11-23: https://platform.acedata.cloud/api/v1/apis/aff80c74-b190-445b-9124-6422ed763c50/
    row: Final = model_cost["acedatacloud/grok-4.7"]
    suffix: Final = "_above_199999_tokens" if prompt_tokens >= 200000 else ""
    input_cost, output_cost = cost_per_token(
        model="acedatacloud/grok-4.7",
        custom_llm_provider="acedatacloud",
        prompt_tokens=prompt_tokens,
        completion_tokens=10,
    )
    assert input_cost == pytest.approx(prompt_tokens * row[f"input_cost_per_token{suffix}"])
    assert output_cost == pytest.approx(10 * row[f"output_cost_per_token{suffix}"])
