import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.apistore.cloud/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_REPO_ROOT: Final = Path(__file__).resolve().parents[4]
_PRIMARY_PRICES: Final = json.loads((_REPO_ROOT / "model_prices_and_context_window.json").read_text())
_BACKUP_PRICES: Final = json.loads((_REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())
_CATALOG_MODELS: Final = tuple(
    sorted(
        key
        for key, row in _PRIMARY_PRICES.items()
        if isinstance(row, dict) and row.get("litellm_provider") == "apistore"
    )
)
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_apistore",
    "object": "chat.completion",
    "created": 1_794_000_000,
    "model": "deepseek-v4-1-flash",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello from API Store"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_apistore_is_a_registered_provider():
    assert litellm.LlmProviders.APISTORE.value == "apistore"
    assert "apistore" in litellm.provider_list


def test_apistore_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APISTORE_API_KEY", "apistore-test-key")
    monkeypatch.delenv("APISTORE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="apistore/deepseek-v4-1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-1-flash"
    assert provider == "apistore"
    assert api_key == "apistore-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_apistore_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APISTORE_API_KEY", "apistore-env-key")
    monkeypatch.setenv("APISTORE_API_BASE", "https://apistore.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="apistore/deepseek-v4-1-flash",
        custom_llm_provider=None,
        api_base="https://apistore.internal.example/v1",
        api_key="apistore-explicit-key",
    )

    assert provider == "apistore"
    assert api_key == "apistore-explicit-key"
    assert api_base == "https://apistore.internal.example/v1"


def test_apistore_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APISTORE_API_KEY", "apistore-env-key")
    monkeypatch.setenv("APISTORE_API_BASE", "https://apistore.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="apistore/deepseek-v4-1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "apistore"
    assert api_base == "https://apistore.env.example/v1"


def test_apistore_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="apistore/deepseek-v4-1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="apistore-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer apistore-test-key"
    assert body["model"] == "deepseek-v4-1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from API Store"


@pytest.mark.parametrize("model", _CATALOG_MODELS)
def test_catalog_row_matches_its_backup_and_fits_its_context(model: str):
    row: Final = _PRIMARY_PRICES[model]

    assert row == _BACKUP_PRICES[model]
    assert row["mode"] == "chat"
    assert 0 < row["max_output_tokens"] <= row["max_input_tokens"]
    assert 0 < row["cache_read_input_token_cost"] < row["input_cost_per_token"] < row["output_cost_per_token"]


# Wednesday 10:00 UTC+8, inside API Store's peak hours, when every model bills its standard rates
_WEEKDAY_PEAK: Final = datetime(2026, 11, 18, 2, 0, tzinfo=timezone.utc)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", _CATALOG_MODELS)
def test_chat_response_is_priced_at_its_catalog_row(model: str):
    row: Final = _PRIMARY_PRICES[model]
    response: Final = litellm.ModelResponse(
        model=model.removeprefix("apistore/"),
        usage=litellm.Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
    )

    with pinned_billing_time(_WEEKDAY_PEAK):
        cost: Final = litellm.completion_cost(completion_response=response, model=model)

    assert cost == pytest.approx(1000 * row["input_cost_per_token"] + 200 * row["output_cost_per_token"])


_LONG_PROMPT_MODELS: Final = tuple(
    (model, int(key.removeprefix("input_cost_per_token_above_").removesuffix("k_tokens")) * 1000)
    for model in _CATALOG_MODELS
    for key in _PRIMARY_PRICES[model]
    if key.startswith("input_cost_per_token_above_")
)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(("model", "threshold"), _LONG_PROMPT_MODELS)
def test_long_prompt_bills_the_whole_request_at_the_long_prompt_rates(model: str, threshold: int):
    row: Final = _PRIMARY_PRICES[model]
    suffix: Final = f"_above_{threshold // 1000}k_tokens"

    def cost(prompt_tokens: int) -> float:
        response = litellm.ModelResponse(
            model=model.removeprefix("apistore/"),
            usage=litellm.Usage(prompt_tokens=prompt_tokens, completion_tokens=1000, total_tokens=prompt_tokens + 1000),
        )
        return litellm.completion_cost(completion_response=response, model=model)

    assert cost(threshold) == pytest.approx(
        threshold * row["input_cost_per_token"] + 1000 * row["output_cost_per_token"]
    )
    assert cost(threshold + 1) == pytest.approx(
        (threshold + 1) * row[f"input_cost_per_token{suffix}"] + 1000 * row[f"output_cost_per_token{suffix}"]
    )


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    ("moment", "off_peak"),
    (
        (datetime(2026, 11, 18, 2, 0, tzinfo=timezone.utc), False),  # Wednesday 10:00 UTC+8, peak
        (datetime(2026, 11, 18, 5, 0, tzinfo=timezone.utc), True),  # Wednesday 13:00 UTC+8, lunch break
        (datetime(2026, 11, 18, 7, 0, tzinfo=timezone.utc), False),  # Wednesday 15:00 UTC+8, peak
        (datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc), True),  # Wednesday 20:00 UTC+8
        (datetime(2026, 11, 21, 2, 0, tzinfo=timezone.utc), True),  # Saturday 10:00 UTC+8
        (datetime(2026, 11, 23, 1, 30, tzinfo=timezone.utc), False),  # Monday 09:30 UTC+8, peak
    ),
)
def test_deepseek_is_billed_at_off_peak_rates_outside_weekday_peak_hours(moment: datetime, off_peak: bool):
    for model in ("apistore/deepseek-v4-1-flash", "apistore/deepseek-v4-1-flash-economy"):
        row = _PRIMARY_PRICES[model]
        rates = row["off_peak_pricing"] if off_peak else row
        response = litellm.ModelResponse(
            model=model.removeprefix("apistore/"),
            usage=litellm.Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
        )
        with pinned_billing_time(moment):
            cost = litellm.completion_cost(completion_response=response, model=model)
        assert cost == pytest.approx(1000 * rates["input_cost_per_token"] + 200 * rates["output_cost_per_token"])


def test_apistore_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling

    assert supports_function_calling("apistore/a-model-not-in-the-cost-map") is True


@pytest.mark.usefixtures("local_model_cost_map")
def test_apistore_wildcard_expands_to_every_apistore_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("apistore/*")) == set(_CATALOG_MODELS)


def test_apistore_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    apistore: Final = next(provider for provider in providers if provider["litellm_provider"] == "apistore")

    assert apistore["provider"] == "APISTORE"
    assert apistore["default_model_placeholder"] in _CATALOG_MODELS
    assert {field["key"]: field["required"] for field in apistore["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_apistore_supported_endpoints_match_their_backup():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = _REPO_ROOT / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["apistore"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["apistore"]["endpoints"]

    assert backup == root
    assert root["chat_completions"] is True
    assert root["image_generations"] is False
    assert root["video_generations"] is False


@pytest.mark.parametrize("text_completion", (False, True))
def test_apistore_missing_key_rejects_before_openai_fallback(monkeypatch: pytest.MonkeyPatch, text_completion: bool):
    monkeypatch.delenv("APISTORE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(url__regex=r"https://api\.apistore\.cloud/v1/.*").respond(
            200, json=_CHAT_COMPLETION
        )
        with pytest.raises(litellm.AuthenticationError, match="APISTORE_API_KEY"):
            if text_completion:
                litellm.text_completion(model="apistore/deepseek-v4-1-flash", prompt="Say hello")
            else:
                litellm.completion(
                    model="apistore/deepseek-v4-1-flash", messages=[{"role": "user", "content": "Say hello"}]
                )
    assert route.call_count == 0


def test_apistore_chat_completion_uses_authenticated_client(monkeypatch: pytest.MonkeyPatch):
    from openai import OpenAI

    monkeypatch.delenv("APISTORE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")
    with OpenAI(api_key="apistore-client-key", base_url=_DEFAULT_API_BASE) as client:
        with respx.mock() as upstream:
            route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
            response: Final = litellm.completion(
                model="apistore/deepseek-v4-1-flash", messages=[{"role": "user", "content": "Say hello"}], client=client
            )
    assert route.calls.last.request.headers["authorization"] == "Bearer apistore-client-key"
    assert response.choices[0].message.content == "Hello from API Store"


@pytest.mark.parametrize("model", [m for m in _CATALOG_MODELS if m.startswith("apistore/claude-")])
def test_claude_cache_writes_cost_more_than_plain_input(model: str):
    row: Final = _PRIMARY_PRICES[model]

    assert row["cache_creation_input_token_cost"] == pytest.approx(1.25 * row["input_cost_per_token"])
    assert row["cache_creation_input_token_cost_above_1hr"] == pytest.approx(2 * row["input_cost_per_token"])
