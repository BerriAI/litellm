import json
from typing import Final

import httpx
import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig


class Ticket(BaseModel):
    urgent: bool


def _hesperan() -> SimpleProviderConfig:
    provider: Final = JSONProviderRegistry.get("hesperan")
    assert provider is not None
    return provider


def test_hesperan_route_uses_registered_base_url_and_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_hesperan().api_key_env, "hsp_test")
    monkeypatch.delenv("HESPERAN_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(model="hesperan/hesperan-1")

    assert (model, provider, api_key, api_base) == ("hesperan-1", "hesperan", "hsp_test", _hesperan().base_url)


def test_hesperan_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _hesperan().api_base_env == "HESPERAN_API_BASE"
    monkeypatch.setenv("HESPERAN_API_BASE", "http://localhost:8788/v1")

    _, _, _, api_base = get_llm_provider(model="hesperan/hesperan-1")

    assert api_base == "http://localhost:8788/v1"


@pytest.mark.usefixtures("local_model_cost_map")
def test_hesperan_model_supports_response_schema() -> None:
    assert litellm.supports_response_schema(model="hesperan/hesperan-1") is True


@pytest.mark.usefixtures("local_model_cost_map")
@respx.mock
def test_hesperan_completion_forwards_json_schema_and_bills_input_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_hesperan().api_key_env, "hsp_test")
    monkeypatch.delenv("HESPERAN_API_BASE", raising=False)
    route: Final = respx.post(f"{_hesperan().base_url}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "hesperan-1",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"urgent":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12},
                "hesperan": {"answers": {"urgent": {"type": "noul", "value": True, "probability": 0.88}}},
            },
        )
    )

    response: Final = litellm.completion(
        model="hesperan/hesperan-1",
        messages=[{"role": "user", "content": "I was charged twice."}],
        response_format=Ticket,
    )

    sent: Final = route.calls.last.request
    body: Final = json.loads(sent.content)
    assert sent.headers["authorization"] == "Bearer hsp_test"
    assert body["model"] == "hesperan-1"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["properties"]["urgent"]["type"] == "boolean"
    assert response.choices[0].message.content == '{"urgent":true}'
    assert getattr(response, "hesperan")["answers"]["urgent"]["probability"] == 0.88
    input_cost: Final = litellm.get_model_info("hesperan/hesperan-1")["input_cost_per_token"]
    assert input_cost is not None
    assert response._hidden_params["response_cost"] == pytest.approx(12 * input_cost)
