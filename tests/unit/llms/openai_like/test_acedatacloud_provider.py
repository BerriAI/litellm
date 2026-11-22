import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm import completion, cost_per_token, get_llm_provider, model_cost, responses
from litellm.llms.openai_like.acedatacloud import CATALOG_URL, AceDataCloudModelInfo
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
                "usage": {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
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
        assert AceDataCloudModelInfo().get_models(api_key="private-key") == [
            "acedatacloud/new-chat",
            "acedatacloud/gpt-4o-image",
        ]
    assert "authorization" not in route.calls.last.request.headers


@pytest.mark.parametrize("api_base", [None, "https://api.acedata.cloud/v1", "https://api.acedata.cloud/openai"])
def test_messages_model_uses_native_transport_and_bearer_auth(api_base: str | None) -> None:
    # Published contract, 2026-11-23: https://platform.acedata.cloud/api/v1/apis/a43a98fc-f833-4cfb-8596-8066faad2c42/
    model: Final = "claude-sonnet-5-5"
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.acedata.cloud/v1/messages").respond(
            200,
            json={
                "id": "msg_ace",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )
        response: Final = completion(
            model=f"acedatacloud/{model}",
            api_key="private-key",
            api_base=api_base,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            num_retries=0,
        )
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer private-key"
    assert "x-api-key" not in request.headers
    assert json.loads(request.content)["model"] == model
    assert response.choices[0].message.content == "OK"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (3, 1)


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
        assert AceDataCloudModelInfo().get_models(api_base="https://custom.test/v1") == ["custom"]
    assert route.calls.last.request.headers["authorization"] == "Bearer ace-secret"


def test_messages_model_streams_through_the_same_native_transport() -> None:
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )
    body: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.acedata.cloud/v1/messages").respond(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
        stream: Final = completion(
            model="acedatacloud/claude-sonnet-5-5",
            api_key="private-key",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            stream=True,
            num_retries=0,
        )
        assert "".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices) == "Hello"
    assert route.calls.last.request.headers["authorization"] == "Bearer private-key"


def test_messages_authentication_error_does_not_fall_back_to_chat() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.acedata.cloud/v1/messages").respond(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid key"}}
        )
        with pytest.raises(litellm.AuthenticationError):
            completion(
                model="acedatacloud/claude-sonnet-5-5",
                api_key="bad-key",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=16,
                num_retries=0,
            )
    assert route.call_count == 1


def test_messages_does_not_send_another_providers_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACEDATACLOUD_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    with respx.mock() as upstream:
        with pytest.raises(litellm.AuthenticationError, match="ACEDATACLOUD_API_KEY"):
            completion(
                model="acedatacloud/claude-sonnet-5-5",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=16,
                num_retries=0,
            )
        assert not upstream.calls


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
