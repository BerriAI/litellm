import json
import re
from itertools import chain
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_REPO_ROOT: Final = Path(__file__).resolve().parents[4]
_PRIMARY_PRICES: Final = json.loads((_REPO_ROOT / "model_prices_and_context_window.json").read_text())
_BACKUP_PRICES: Final = json.loads((_REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())
_CATALOG_MODELS: Final = tuple(
    key for key, row in _PRIMARY_PRICES.items() if isinstance(row, dict) and row.get("litellm_provider") == "kyma"
)
_TIER_KEY: Final = re.compile(r"^input_cost_per_token_above_(\d+)k_tokens$")


def _long_prompt_tiers(model: str) -> tuple[tuple[str, int], ...]:
    matches: Final = map(_TIER_KEY.match, _PRIMARY_PRICES[model])
    return tuple((model, int(match.group(1))) for match in matches if match is not None)


_LONG_PROMPT_TIERS: Final = tuple(chain.from_iterable(map(_long_prompt_tiers, _CATALOG_MODELS)))
_CHAT_URL: Final = "https://kymaapi.com/v1/chat/completions"


def test_kyma_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KYMA_API_KEY", "kyma-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="kyma/qwen3.7-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen3.7-flash"
    assert provider == "kyma"
    assert api_key == "kyma-test-key"
    assert api_base == "https://kymaapi.com/v1"


def test_kyma_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KYMA_API_KEY", "kyma-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="kyma/qwen3.7-flash",
        custom_llm_provider=None,
        api_base="https://kyma.internal.example/v1",
        api_key="kyma-explicit-key",
    )

    assert provider == "kyma"
    assert api_key == "kyma-explicit-key"
    assert api_base == "https://kyma.internal.example/v1"


def test_bare_model_on_kyma_base_url_keeps_openai_routing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-6-luna",
        custom_llm_provider=None,
        api_base="https://kymaapi.com/v1",
        api_key=None,
    )

    assert (model, provider, api_base) == ("gpt-6-luna", "openai", "https://kymaapi.com/v1")
    assert api_key is None


def test_kyma_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    kyma: Final = next(provider for provider in providers if provider["litellm_provider"] == "kyma")

    assert kyma["default_model_placeholder"] in _CATALOG_MODELS
    assert {field["key"]: field["required"] for field in kyma["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_kyma_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200,
            json={
                "id": "chatcmpl_kyma",
                "object": "chat.completion",
                "created": 1_792_000_000,
                "model": "qwen3.7-flash",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello from Kyma"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="kyma/qwen3.7-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="kyma-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_URL
    assert request.headers["authorization"] == "Bearer kyma-test-key"
    assert body["model"] == "qwen3.7-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Kyma"


@pytest.mark.parametrize("model", _CATALOG_MODELS)
def test_catalog_row_matches_its_backup_and_fits_its_context(model: str):
    row: Final = _PRIMARY_PRICES[model]

    assert row == _BACKUP_PRICES[model]
    assert row["mode"] == "chat"
    assert 0 < row["max_output_tokens"] <= row["max_input_tokens"]
    assert row["input_cost_per_token"] > 0
    assert row["output_cost_per_token"] > 0


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", _CATALOG_MODELS)
def test_chat_response_is_priced_at_its_catalog_row(model: str):
    row: Final = _PRIMARY_PRICES[model]
    response: Final = litellm.ModelResponse(
        model=model.removeprefix("kyma/"),
        usage=litellm.Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
    )

    cost: Final = litellm.completion_cost(completion_response=response, model=model)

    assert cost == pytest.approx(1000 * row["input_cost_per_token"] + 200 * row["output_cost_per_token"])


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(("model", "threshold_k"), _LONG_PROMPT_TIERS)
def test_long_prompt_is_priced_at_its_tier(model: str, threshold_k: int):
    row: Final = _PRIMARY_PRICES[model]
    prompt_tokens: Final = threshold_k * 1000 + 1000
    response: Final = litellm.ModelResponse(
        model=model.removeprefix("kyma/"),
        usage=litellm.Usage(prompt_tokens=prompt_tokens, completion_tokens=200, total_tokens=prompt_tokens + 200),
    )

    cost: Final = litellm.completion_cost(completion_response=response, model=model)

    assert cost == pytest.approx(
        prompt_tokens * row[f"input_cost_per_token_above_{threshold_k}k_tokens"]
        + 200 * row[f"output_cost_per_token_above_{threshold_k}k_tokens"]
    )
