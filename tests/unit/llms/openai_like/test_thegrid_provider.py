import json
from datetime import datetime
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

_REPO_ROOT: Final = Path(__file__).resolve().parents[4]
_PRIMARY_PRICES: Final = json.loads((_REPO_ROOT / "model_prices_and_context_window.json").read_text())
_BACKUP_PRICES: Final = json.loads((_REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())
_PROVIDERS_JSON: Final = json.loads(
    (Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json").read_text()
)
_CATALOG_MODELS: Final = tuple(
    key for key, row in _PRIMARY_PRICES.items() if isinstance(row, dict) and row.get("litellm_provider") == "thegrid"
)
_MODEL: Final = "thegrid/code-standard"
_MODEL_ID: Final = "code-standard"
_CHAT_URL: Final = "https://api.thegrid.ai/v1/chat/completions"
_RESPONSES_URL: Final = "https://api.thegrid.ai/v1/responses"
_MESSAGES_URL: Final = "https://api.thegrid.ai/v1/messages"
# The Grid AI's usage.cost is several times what it charges, so it must never be the bill.
_REPORTED_COST: Final = 3.8e-06


def _chat_body(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl_thegrid",
        "object": "chat.completion",
        "created": 1_792_000_000,
        "model": _MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
        "usage": usage,
    }


def _responses_body(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "resp_thegrid",
        "object": "response",
        "created_at": 1_792_000_000,
        "model": _MODEL_ID,
        "status": "completed",
        "output": [
            {
                "id": "msg_thegrid",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "pong", "annotations": []}],
            }
        ],
        "usage": usage,
    }


def _catalog_price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    row: Final = litellm.model_cost[model]
    return prompt_tokens * row["input_cost_per_token"] + completion_tokens * row["output_cost_per_token"]


def test_thegrid_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200, json=_chat_body({"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5})
        )
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_completion_tokens=256,
            api_key="grid-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer grid-test-key"
    assert body["model"] == _MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "ping"}]
    assert body["max_completion_tokens"] == 256
    assert response.choices[0].message.content == "pong"


def test_thegrid_reads_its_key_and_base_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THEGRID_API_KEY", "grid-env-key")
    monkeypatch.setenv("THEGRID_API_BASE", "https://grid.internal.example/v1")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://grid.internal.example/v1/chat/completions").respond(
            200, json=_chat_body({"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5})
        )
        litellm.completion(model=_MODEL, messages=[{"role": "user", "content": "ping"}])

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer grid-env-key"


def test_thegrid_without_its_own_key_fails_before_sending_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("THEGRID_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setattr(litellm, "openai_key", "openai-secret")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200, json=_chat_body({"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5})
        )
        with pytest.raises(litellm.AuthenticationError, match="THEGRID_API_KEY"):
            litellm.completion(model=_MODEL, messages=[{"role": "user", "content": "ping"}])

    assert route.call_count == 0


def test_thegrid_does_not_forward_the_openai_organization(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-openai-secret")
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200, json=_chat_body({"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5})
        )
        litellm.completion(model=_MODEL, messages=[{"role": "user", "content": "ping"}], api_key="grid-test-key")

    assert "openai-organization" not in route.calls.last.request.headers


def test_thegrid_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_RESPONSES_URL).respond(
            200, json=_responses_body({"input_tokens": 4, "output_tokens": 1, "total_tokens": 5})
        )
        response: Final = litellm.responses(model=_MODEL, input="ping", api_key="grid-test-key")

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer grid-test-key"
    assert body["model"] == _MODEL_ID
    assert body["input"] == "ping"
    assert response.output[0].content[0].text == "pong"


@pytest.mark.asyncio
async def test_thegrid_anthropic_messages_are_sent_natively_with_anthropic_fields(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_MESSAGES_URL).respond(
            200,
            json={
                "id": "msg_thegrid",
                "type": "message",
                "role": "assistant",
                "model": _MODEL_ID,
                "content": [{"type": "text", "text": "pong"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
        )
        bridged: Final = upstream.post(_CHAT_URL)
        response: Final = await litellm.anthropic.messages.acreate(
            model=_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=2048,
            top_k=5,
            thinking={"type": "enabled", "budget_tokens": 1024},
            api_key="grid-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert bridged.call_count == 0
    assert request.headers["authorization"] == "Bearer grid-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == _MODEL_ID
    assert body["top_k"] == 5
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert response["content"][0]["text"] == "pong"


@pytest.mark.usefixtures("local_model_cost_map")
def test_thegrid_chat_is_billed_at_its_catalog_row_not_its_reported_usage_cost():
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(
            200,
            json=_chat_body({"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18, "cost": _REPORTED_COST}),
        )
        response: Final = litellm.completion(
            model=_MODEL, messages=[{"role": "user", "content": "ping"}], api_key="grid-test-key"
        )

    expected: Final = _catalog_price(_MODEL, 13, 5)
    assert response._hidden_params["response_cost"] == pytest.approx(expected)
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


@pytest.mark.usefixtures("local_model_cost_map")
def test_thegrid_responses_are_billed_at_their_catalog_row_not_their_reported_usage_cost():
    with respx.mock() as upstream:
        upstream.post(_RESPONSES_URL).respond(
            200,
            json=_responses_body({"input_tokens": 13, "output_tokens": 5, "total_tokens": 18, "cost": _REPORTED_COST}),
        )
        response: Final = litellm.responses(model=_MODEL, input="ping", api_key="grid-test-key")

    assert response._hidden_params["response_cost"] == pytest.approx(_catalog_price(_MODEL, 13, 5))


def _stream_chunk(content: str, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl_thegrid",
        created=1_792_000_000,
        model=_MODEL_ID,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(finish_reason=finish_reason, index=0, delta=Delta(content=content, role="assistant"))
        ],
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_thegrid_stream_is_billed_at_its_catalog_row_not_its_reported_usage_cost():
    usage_chunk: Final = _stream_chunk("")
    usage_chunk.usage = Usage(prompt_tokens=13, completion_tokens=5, total_tokens=18, cost=_REPORTED_COST)
    logging_obj: Final = LiteLLMLogging(
        model=_MODEL_ID,
        messages=[{"role": "user", "content": "ping"}],
        stream=True,
        call_type="completion",
        start_time=datetime(2026, 11, 9, 12, 0, 0),
        litellm_call_id="thegrid-call",
        function_id="thegrid-function",
    )
    logging_obj.update_environment_variables(
        model=_MODEL_ID,
        user=None,
        optional_params={},
        litellm_params={"custom_llm_provider": "thegrid"},
        custom_llm_provider="thegrid",
    )

    response: Final = litellm.stream_chunk_builder(
        chunks=[_stream_chunk("pong", finish_reason="stop"), usage_chunk],
        messages=[{"role": "user", "content": "ping"}],
        logging_obj=logging_obj,
    )

    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(_catalog_price(_MODEL, 13, 5))


def test_thegrid_wildcard_lists_every_catalog_model():
    assert _CATALOG_MODELS
    assert sorted(get_known_models_from_wildcard("thegrid/*")) == sorted(_CATALOG_MODELS)
    assert set(litellm.get_valid_models(custom_llm_provider="thegrid")) == set(_CATALOG_MODELS)


@pytest.mark.parametrize("model", _CATALOG_MODELS)
def test_thegrid_catalog_row_matches_its_backup_and_declares_the_provider_endpoints(model: str):
    row: Final = _PRIMARY_PRICES[model]
    assert _BACKUP_PRICES[model] == row
    assert row["supported_endpoints"] == _PROVIDERS_JSON["thegrid"]["supported_endpoints"]
    assert row["max_output_tokens"] == row["max_tokens"]
    assert row["input_cost_per_token"] > 0
    assert row["output_cost_per_token"] > 0


def test_thegrid_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    thegrid: Final = next(provider for provider in providers if provider["litellm_provider"] == "thegrid")

    assert thegrid["default_model_placeholder"] in _CATALOG_MODELS
    assert {field["key"]: field["required"] for field in thegrid["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
