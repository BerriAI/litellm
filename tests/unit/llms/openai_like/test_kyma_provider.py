import json
import re
from collections.abc import Callable
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
from litellm.proxy.common_request_processing import cost_headers
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

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
_UNLISTED_MODEL: Final = "kyma/model-added-after-this-release"


def _kyma_chat_body(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl_kyma",
        "object": "chat.completion",
        "created": 1_792_000_000,
        "model": "sonar",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
        "usage": usage,
    }


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


@pytest.mark.parametrize("model", ("kyma/sonar", _UNLISTED_MODEL))
def test_kyma_response_is_billed_at_the_cost_kyma_reports(model: str):
    reported_cost: Final = 0.00721
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": reported_cost})
        )
        response: Final = litellm.completion(model=model, messages=[{"role": "user", "content": "ping"}], api_key="k")

    assert response._hidden_params["response_cost"] == pytest.approx(reported_cost)


def test_deployment_pricing_overrides_the_cost_kyma_reports():
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": 0.00721})
        )
        response: Final = litellm.completion(
            model="kyma/sonar",
            messages=[{"role": "user", "content": "ping"}],
            api_key="k",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )

    assert response._hidden_params["response_cost"] == pytest.approx(4 * 0.001 + 3 * 0.002)



def _catalog_price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    row: Final = _PRIMARY_PRICES[model]
    return prompt_tokens * row["input_cost_per_token"] + completion_tokens * row["output_cost_per_token"]


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("reported_cost", ("-0.5", "1e999"))
def test_unusable_cost_from_kyma_falls_back_to_the_catalog_price(reported_cost: str):
    usage: Final = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": "REPORTED_COST"}
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200,
            content=json.dumps(_kyma_chat_body(usage)).replace('"REPORTED_COST"', reported_cost),
            headers={"content-type": "application/json"},
        )
        response: Final = litellm.completion(model="kyma/sonar", messages=[{"role": "user", "content": "ping"}], api_key="k")

    assert response._hidden_params["response_cost"] == pytest.approx(_catalog_price("kyma/sonar", 4, 3))


@pytest.mark.usefixtures("local_model_cost_map")
def test_deployment_base_model_overrides_the_cost_kyma_reports():
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": 0.00721})
        )
        response: Final = litellm.completion(
            model="kyma/sonar",
            messages=[{"role": "user", "content": "ping"}],
            api_key="k",
            base_model="kyma/qwen3.7-flash",
        )

    assert response._hidden_params["response_cost"] == pytest.approx(_catalog_price("kyma/qwen3.7-flash", 4, 3))


def test_interaction_on_kyma_is_billed_at_the_cost_kyma_reports():
    reported_cost: Final = 0.00721
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": reported_cost})
        )
        interaction: Final = litellm.interactions.create(model="kyma/qwen3.7-flash", input="ping", api_key="k")

    assert interaction._hidden_params["response_cost"] == pytest.approx(reported_cost)


@pytest.mark.parametrize("model", ("openrouter/openai/gpt-4o-mini", "deepseek/deepseek-chat"))
def test_interaction_on_another_provider_keeps_its_upstream_cost_out_of_the_body(model: str):
    chat_usage: Final = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": 0.00721}
    responses_body: Final = {
        "id": "resp_upstream",
        "object": "response",
        "created_at": 1_792_000_000,
        "status": "completed",
        "model": "x",
        "output": [
            {
                "type": "message",
                "id": "msg_upstream",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "pong", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7, "cost": 0.00721},
    }
    with respx.mock(assert_all_called=False) as upstream:
        upstream.route(path__regex=r".*/chat/completions$").respond(200, json=_kyma_chat_body(chat_usage))
        upstream.route(path__regex=r".*/responses$").respond(200, json=responses_body)
        interaction: Final = litellm.interactions.create(model=model, input="ping", api_key="k")

    assert "cost" not in interaction.model_dump(exclude_none=True)["usage"]


def test_sdk_completion_cost_prices_kyma_at_the_cost_kyma_reports():
    reported_cost: Final = 0.00721
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": reported_cost})
        )
        response: Final = litellm.completion(model="kyma/sonar", messages=[{"role": "user", "content": "ping"}], api_key="k")

    assert litellm.completion_cost(completion_response=response) == pytest.approx(reported_cost)


def _kyma_sdk_response(usage: dict[str, float]) -> litellm.ModelResponse:
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(200, json=_kyma_chat_body(usage))
        response: Final = litellm.completion(model="kyma/sonar", messages=[{"role": "user", "content": "ping"}], api_key="k")
    assert isinstance(response, litellm.ModelResponse)
    return response


_KYMA_TOKEN_USAGE: Final = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
_KYMA_REPORTED_COST: Final = 0.00721


def test_sdk_completion_cost_uses_the_callers_token_prices_over_the_cost_kyma_reports():
    caller_prices: Final = {"input_cost_per_token": 0.5, "output_cost_per_token": 2.0}
    with_reported_cost: Final = _kyma_sdk_response({**_KYMA_TOKEN_USAGE, "cost": _KYMA_REPORTED_COST})
    without_reported_cost: Final = _kyma_sdk_response(_KYMA_TOKEN_USAGE)

    assert litellm.completion_cost(
        completion_response=with_reported_cost, custom_cost_per_token=caller_prices
    ) == pytest.approx(litellm.completion_cost(completion_response=without_reported_cost, custom_cost_per_token=caller_prices))
    assert litellm.completion_cost(
        completion_response=with_reported_cost, custom_cost_per_token=caller_prices
    ) != pytest.approx(_KYMA_REPORTED_COST)


def test_sdk_completion_cost_uses_the_callers_per_second_price_over_the_cost_kyma_reports():
    response: Final = _kyma_sdk_response({**_KYMA_TOKEN_USAGE, "cost": _KYMA_REPORTED_COST})

    assert litellm.completion_cost(completion_response=response, custom_cost_per_second=50.0) == pytest.approx(
        50.0 * response._response_ms / 1000
    )


def _with_kyma_discount_and_margin(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(litellm, "cost_discount_config", {"kyma": 0.5})
    monkeypatch.setattr(litellm, "cost_margin_config", {"kyma": 0.1})
    return 0.5 * 1.1


def test_cost_kyma_reports_takes_the_configured_discount_and_margin(monkeypatch: pytest.MonkeyPatch):
    adjustment: Final = _with_kyma_discount_and_margin(monkeypatch)
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7, "cost": 0.008})
        )
        response: Final = litellm.completion(model="kyma/sonar", messages=[{"role": "user", "content": "ping"}], api_key="k")

    assert response._hidden_params["response_cost"] == pytest.approx(0.008 * adjustment)


def _kyma_stream_chunk(content: str, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl_kyma",
        created=1_792_000_000,
        model="sonar",
        object="chat.completion.chunk",
        choices=[StreamingChoices(finish_reason=finish_reason, index=0, delta=Delta(content=content, role="assistant"))],
    )


def test_streamed_cost_kyma_reports_takes_the_configured_discount_and_margin(monkeypatch: pytest.MonkeyPatch):
    adjustment: Final = _with_kyma_discount_and_margin(monkeypatch)
    usage_chunk: Final = _kyma_stream_chunk("")
    usage_chunk.usage = Usage(prompt_tokens=4, completion_tokens=3, total_tokens=7, cost=0.008)
    logging_obj: Final = LiteLLMLogging(
        model="sonar",
        messages=[{"role": "user", "content": "ping"}],
        stream=True,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="kyma-call",
        function_id="kyma-function",
    )
    logging_obj.update_environment_variables(
        model="sonar",
        user=None,
        optional_params={},
        litellm_params={"custom_llm_provider": "kyma"},
        custom_llm_provider="kyma",
    )

    response: Final = litellm.stream_chunk_builder(
        chunks=[_kyma_stream_chunk("pong", finish_reason="stop"), usage_chunk],
        messages=[{"role": "user", "content": "ping"}],
        logging_obj=logging_obj,
    )

    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(0.008 * adjustment)


def test_cost_headers_show_the_discount_and_margin_taken_from_the_cost_kyma_reports(monkeypatch: pytest.MonkeyPatch):
    _with_kyma_discount_and_margin(monkeypatch)
    logging_obj: Final = LiteLLMLogging(
        model="sonar",
        messages=[{"role": "user", "content": "ping"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="kyma-call",
        function_id="kyma-function",
    )
    logging_obj.update_environment_variables(
        model="sonar",
        user=None,
        optional_params={},
        litellm_params={"custom_llm_provider": "kyma"},
        custom_llm_provider="kyma",
    )
    response: Final = litellm.ModelResponse(
        model="sonar", usage=Usage(prompt_tokens=4, completion_tokens=3, total_tokens=7, cost=0.008)
    )

    headers: Final = cost_headers(logging_obj.response_cost_calculator(result=response), logging_obj)

    assert float(headers["x-litellm-response-cost-original"] or "nan") == pytest.approx(0.008)
    assert float(headers["x-litellm-response-cost-discount-amount"] or "nan") == pytest.approx(0.004)
    assert float(headers["x-litellm-response-cost-margin-amount"] or "nan") == pytest.approx(0.0004)
    assert float(headers["x-litellm-response-cost"] or "nan") == pytest.approx(0.0044)
    assert headers["x-litellm-response-cost-input"] is None
    assert headers["x-litellm-response-cost-output"] is None


@pytest.mark.parametrize("thinking", ({"type": "enabled", "budget_tokens": 1024}, {"type": "adaptive"}))
def test_messages_thinking_on_a_kyma_claude_model_is_sent_as_reasoning_effort(thinking: dict[str, object]):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        litellm.anthropic.messages.create(
            model="kyma/claude-haiku-4-5",
            max_tokens=2048,
            messages=[{"role": "user", "content": "ping"}],
            thinking=thinking,
            api_key="k",
        )

    sent: Final = json.loads(route.calls.last.request.content)
    assert "thinking" not in sent
    assert sent["reasoning_effort"] in ("low", "medium", "high")

def test_kyma_model_missing_from_the_catalog_keeps_its_tools():
    tools: Final = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200, json=_kyma_chat_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        litellm.completion(model=_UNLISTED_MODEL, messages=[{"role": "user", "content": "ping"}], api_key="k", tools=tools)

    assert json.loads(route.calls.last.request.content)["tools"] == tools


def test_kyma_wildcard_lists_every_catalog_model():
    assert sorted(get_known_models_from_wildcard("kyma/*")) == sorted(_CATALOG_MODELS)


def _generate_image() -> object:
    return litellm.image_generation(model="kyma/flux-2", prompt="a cat", api_key="k")


def _transcribe() -> object:
    return litellm.transcription(model="kyma/whisper-1", file=("a.wav", b"RIFF0000WAVE", "audio/wav"), api_key="k")


@pytest.mark.parametrize("call", (_generate_image, _transcribe))
def test_unpriced_kyma_media_calls_are_refused_without_calling_kyma(call: Callable[[], object]):
    with respx.mock(assert_all_called=False) as upstream:
        kyma: Final = upstream.route(host="kymaapi.com").respond(202, json={"id": "job_kyma"})
        with pytest.raises(litellm.BadRequestError):
            call()

    assert kyma.call_count == 0
