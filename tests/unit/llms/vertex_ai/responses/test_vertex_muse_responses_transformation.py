import json
from pathlib import Path
from typing import Final
from unittest.mock import Mock

import httpx
import pytest
import respx
from google.oauth2.credentials import Credentials
from pydantic import TypeAdapter
from pytest_mock import MockerFixture

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.utils import ProviderConfigManager

MODELS: Final = ("muse-wonder-1.0", "muse-ember-1.0", "muse-spark-1.3")
ENDPOINT: Final = (
    "https://aiplatform.googleapis.com/v1/projects/unit-project/locations/global/publishers/meta/v1/responses"
)
ROOT: Final = Path(__file__).parents[5]
CATALOG: Final = TypeAdapter(dict[str, dict[str, object]])
FIELDS: Final = TypeAdapter(dict[str, object])
pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


def _response(model: str) -> dict[str, object]:
    return {
        "id": "resp_muse",
        "object": "response",
        "created_at": 1_700_000_000,
        "model": model,
        "status": "completed",
        "output": [
            {
                "id": "msg_muse",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
        "parallel_tool_calls": True,
        "tools": [],
        "tool_choice": "auto",
    }


@pytest.fixture(autouse=True)
def google_credentials(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.delenv("VERTEXAI_CREDENTIALS", raising=False)
    credentials: Final = mocker.Mock(
        spec=Credentials, token="unit-token", expired=False, quota_project_id=None, requires_scopes=False
    )
    mocker.patch("google.auth.default", return_value=(credentials, "unit-project"))
    return credentials


@pytest.mark.parametrize("model", MODELS)
def test_muse_responses_select_a_native_vertex_adapter(model: str) -> None:
    assert ProviderConfigManager.get_provider_responses_api_config("vertex_ai", f"meta/{model}") is not None
    assert ProviderConfigManager.get_provider_responses_api_config("vertex_ai", "gemini-4-argon") is None


@pytest.mark.parametrize("model", MODELS)
def test_muse_metadata_resolves_the_vertex_entry_and_bundled_fallback(model: str) -> None:
    key: Final = f"vertex_ai/meta/{model}"
    main: Final = CATALOG.validate_json((ROOT / "model_prices_and_context_window.json").read_bytes())[key]
    backup: Final = CATALOG.validate_json((ROOT / "litellm/model_prices_and_context_window_backup.json").read_bytes())[
        key
    ]
    info: Final = litellm.get_model_info(key)
    assert main == backup
    assert info["key"] == key
    assert info["mode"] == "responses"
    assert info["max_input_tokens"] == main["max_input_tokens"]
    assert litellm.supports_reasoning(key) is True
    assert litellm.supports_function_calling(key) is True


@pytest.mark.parametrize("model", MODELS)
def test_muse_responses_send_google_auth_and_a_bare_model_to_the_global_endpoint(
    model: str, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    response: Final = litellm.responses(
        model=f"vertex_ai/meta/{model}",
        input="Say hello",
        vertex_project="unit-project",
        max_output_tokens=32,
        reasoning={"effort": "high"},
        store=False,
        num_retries=0,
    )
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer unit-token"
    assert FIELDS.validate_json(request.content) == {
        "model": model,
        "input": "Say hello",
        "max_output_tokens": 32,
        "reasoning": {"effort": "high"},
        "store": False,
    }
    assert response.output_text == "Hello"
    assert response.usage is not None
    assert response.usage.total_tokens == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
async def test_muse_async_streaming_preserves_output_and_completed_usage(
    model: str, respx_mock: respx.MockRouter
) -> None:
    events: Final = (
        {"type": "response.created", "response": {**_response(model), "status": "in_progress", "output": []}},
        {
            "type": "response.output_text.delta",
            "item_id": "msg_muse",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        },
        {"type": "response.completed", "response": _response(model)},
    )
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    route: Final = respx_mock.post(ENDPOINT).mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )
    client: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport())
    try:
        stream: Final = await litellm.aresponses(
            model=f"vertex_ai/meta/{model}",
            input="Say hello",
            vertex_project="unit-project",
            stream=True,
            store=False,
            num_retries=0,
            client=client,
        )
        chunks: Final = tuple([chunk async for chunk in stream])
    finally:
        await client.close()
    deltas: Final = tuple(chunk.delta for chunk in chunks if chunk.type == "response.output_text.delta")
    completed: Final = tuple(chunk for chunk in chunks if chunk.type == "response.completed")
    assert deltas == ("Hello",)
    assert len(completed) == 1
    assert completed[0].response.usage.total_tokens == 9
    assert FIELDS.validate_json(route.calls.last.request.content)["model"] == model


def test_muse_defaults_to_stateless_requests_and_honors_custom_token_prices(respx_mock: respx.MockRouter) -> None:
    model: Final = "vertex_ai/meta/muse-wonder-1.0"
    input_rate: Final = 0.001
    output_rate: Final = 0.002
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    response: Final = litellm.responses(
        model=model,
        input="Say hello",
        vertex_project="unit-project",
        input_cost_per_token=input_rate,
        output_cost_per_token=output_rate,
        num_retries=0,
    )
    assert FIELDS.validate_json(route.calls.last.request.content)["store"] is False
    assert response.hidden_params["response_cost"] == pytest.approx(7 * input_rate + 2 * output_rate)


def test_muse_refuses_server_side_response_storage() -> None:
    with pytest.raises(litellm.UnsupportedParamsError, match="stateless"):
        litellm.responses(
            model="vertex_ai/meta/muse-wonder-1.0",
            input="Say hello",
            vertex_project="unit-project",
            store=True,
            num_retries=0,
        )


def test_muse_chat_completions_bridge_uses_the_native_responses_endpoint(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    response: Final = litellm.completion(
        model="vertex_ai/meta/muse-wonder-1.0",
        messages=[{"role": "user", "content": "Say hello"}],
        vertex_project="unit-project",
        max_tokens=32,
        num_retries=0,
    )
    assert route.called
    assert response.choices[0].message.content == "Hello"
    assert response.usage.total_tokens == 9


def test_muse_accepts_nested_workload_identity_credentials(
    respx_mock: respx.MockRouter, mocker: MockerFixture, google_credentials: Mock
) -> None:
    credential_type: Final = mocker.patch("google.auth.identity_pool.Credentials")
    credential_type.from_info.return_value = google_credentials
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    response: Final = litellm.responses(
        model="vertex_ai/meta/muse-wonder-1.0",
        input="Say hello",
        vertex_project="unit-project",
        vertex_credentials={
            "type": "external_account",
            "audience": "unit-audience",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": "https://sts.googleapis.com/v1/token",
            "credential_source": {"file": "/unit/subject.token"},
        },
        num_retries=0,
    )
    assert route.calls.last.request.headers["authorization"] == "Bearer unit-token"
    assert response.output_text == "Hello"
