import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.clf_ai_gateway.chat.transformation import ClfAiGatewayConfig
from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts

DEFAULT_API_BASE: Final = "https://api.clfaigateway.dev/v1"
CHAT_URL: Final = f"{DEFAULT_API_BASE}/chat/completions"
REPO_ROOT: Final = Path(litellm.__file__).parent.parent


def _chat_response(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "glm-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": usage,
    }


def _clf_rows(path: Path) -> dict[str, dict[str, object]]:
    rows: Final = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in rows.items() if value.get("litellm_provider") == "clf_ai_gateway"}


def test_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "fake-environment-key")
    monkeypatch.delenv("CLF_AI_GATEWAY_API_BASE", raising=False)
    assert get_llm_provider(model="clf_ai_gateway/glm-5.3") == (
        "glm-5.3",
        "clf_ai_gateway",
        "fake-environment-key",
        DEFAULT_API_BASE,
    )
    assert get_llm_provider(
        model="clf_ai_gateway/glm-5.3", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("glm-5.3", "clf_ai_gateway", "fake-explicit-key", "https://gateway.example/v1")

    monkeypatch.setenv("CLF_AI_GATEWAY_API_BASE", "https://env-gateway.example/v1")
    assert get_llm_provider(model="clf_ai_gateway/glm-5.3")[3] == "https://env-gateway.example/v1"


def test_api_base_alone_selects_the_provider_and_keeps_an_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "fake-environment-key")
    _, provider, key, _ = get_llm_provider(model="glm-5.3", api_base=DEFAULT_API_BASE)
    assert (provider, key) == ("clf_ai_gateway", "fake-environment-key")

    _, provider_with_key, explicit_key, _ = get_llm_provider(
        model="glm-5.3", api_base=DEFAULT_API_BASE, api_key="fake-explicit-key"
    )
    assert (provider_with_key, explicit_key) == ("clf_ai_gateway", "fake-explicit-key")


def test_chat_forwards_reasoning_effort_and_bills_cached_tokens(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(CHAT_URL).respond(
            200,
            json=_chat_response(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": 80},
                }
            ),
        )
        response: Final = litellm.completion(
            model="clf_ai_gateway/glm-5.3",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "glm-5.3"
    assert body["reasoning_effort"] == "low"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost["clf_ai_gateway/glm-5.3"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("functions", [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}]),
        ("function_call", "auto"),
    ],
)
def test_legacy_fields_the_gateway_rejects_are_refused_or_dropped(
    local_model_cost_map: None, param: str, value: object
) -> None:
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(
            model="clf_ai_gateway/glm-5.3",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "hi"}],
            **{param: value},
        )

    with respx.mock() as upstream:
        route: Final = upstream.post(CHAT_URL).respond(
            200, json=_chat_response({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        )
        litellm.completion(
            model="clf_ai_gateway/glm-5.3",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "hi"}],
            drop_params=True,
            **{param: value},
        )
    assert param not in json.loads(route.calls.last.request.content)


@pytest.mark.parametrize(
    "params",
    [
        {"logit_bias": {"50256": -100}},
        {"logprobs": True, "top_logprobs": 2},
        {"store": False},
        {"service_tier": "auto"},
    ],
)
def test_params_the_gateway_accepts_are_forwarded_without_drop_params(
    local_model_cost_map: None, params: dict[str, object]
) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(CHAT_URL).respond(
            200, json=_chat_response({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        )
        litellm.completion(
            model="clf_ai_gateway/glm-5.3",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "hi"}],
            **params,
        )
    body: Final = json.loads(route.calls.last.request.content)
    assert {key: body.get(key) for key in params} == params


@pytest.mark.parametrize(
    ("effort", "drop_params", "forwarded"),
    [
        ("high", True, None),
        ("high", False, "high"),
        ("xhigh", True, "xhigh"),
        ({"effort": "low", "summary": "auto"}, True, "low"),
        ({"effort": "low", "summary": "auto"}, False, "low"),
        ({"effort": "high", "summary": "auto"}, True, None),
    ],
)
def test_drop_params_drops_a_reasoning_effort_level_the_model_does_not_declare(
    local_model_cost_map: None, effort: str | dict[str, str], drop_params: bool, forwarded: str | None
) -> None:
    assert "high" not in litellm.model_cost["clf_ai_gateway/qwen3.8-27b"]["reasoning_effort_levels"]
    with respx.mock() as upstream:
        route: Final = upstream.post(CHAT_URL).respond(
            200, json=_chat_response({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        )
        litellm.completion(
            model="clf_ai_gateway/qwen3.8-27b",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort=effort,
            drop_params=drop_params,
        )
    assert json.loads(route.calls.last.request.content).get("reasoning_effort") == forwarded


def test_a_reasoning_effort_dict_without_a_level_is_refused_unless_dropped(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(CHAT_URL).respond(
            200, json=_chat_response({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        )
        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.completion(
                model="clf_ai_gateway/glm-5.3",
                api_key="fake-test-key",
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort={"summary": "auto"},
            )
        litellm.completion(
            model="clf_ai_gateway/glm-5.3",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort={"summary": "auto"},
            drop_params=True,
        )
    assert route.call_count == 1
    assert "reasoning_effort" not in json.loads(route.calls.last.request.content)


def test_reasoning_effort_is_advertised_only_for_reasoning_models(local_model_cost_map: None) -> None:
    config: Final = ClfAiGatewayConfig()
    assert "reasoning_effort" in config.get_supported_openai_params(model="glm-5.3")
    assert "reasoning_effort" not in config.get_supported_openai_params(model="not-in-the-cost-map")
    assert litellm.get_supported_openai_params(
        model="glm-5.3", custom_llm_provider="clf_ai_gateway"
    ) == config.get_supported_openai_params(model="glm-5.3")


def test_cost_map_rows_match_the_backup_and_resolve_their_effort_levels() -> None:
    rows: Final = _clf_rows(REPO_ROOT / "model_prices_and_context_window.json")
    assert rows
    assert rows == _clf_rows(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json")
    for key, row in rows.items():
        assert row["input_cost_per_token"] > 0, key
        assert row["output_cost_per_token"] > 0, key
        assert row["max_output_tokens"] <= row["max_input_tokens"], key
        assert resolve_supported_reasoning_efforts({**row, "key": key}, deployment_is_mapped=True) == tuple(
            row["reasoning_effort_levels"]
        ), key


def test_validate_environment_reports_the_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLF_AI_GATEWAY_API_KEY", raising=False)
    missing: Final = litellm.validate_environment(model="clf_ai_gateway/glm-5.3")
    assert missing["keys_in_environment"] is False
    assert missing["missing_keys"] == ["CLF_AI_GATEWAY_API_KEY"]

    monkeypatch.setenv("CLF_AI_GATEWAY_API_KEY", "fake-environment-key")
    present: Final = litellm.validate_environment(model="clf_ai_gateway/glm-5.3")
    assert present == {"keys_in_environment": True, "missing_keys": []}


def test_add_model_form_entry_points_at_a_mapped_model_and_the_default_base(local_model_cost_map: None) -> None:
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    entries: Final = [
        entry
        for entry in json.loads(fields_path.read_text(encoding="utf-8"))
        if entry["litellm_provider"] == "clf_ai_gateway"
    ]
    assert len(entries) == 1
    assert entries[0]["default_model_placeholder"] in litellm.model_cost
    fields: Final = {field["key"]: field for field in entries[0]["credential_fields"]}
    assert fields["api_key"]["required"] is True
    assert fields["api_base"]["required"] is False
    assert fields["api_base"]["placeholder"] == DEFAULT_API_BASE
