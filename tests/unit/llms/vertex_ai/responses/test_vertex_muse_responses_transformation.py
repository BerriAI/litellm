import json
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlparse

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


def _event_stream(model: str) -> str:
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
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


@pytest.fixture(autouse=True)
def google_credentials(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.delenv("VERTEXAI_CREDENTIALS", raising=False)
    credentials: Final = mocker.Mock(
        spec=Credentials, token="unit-token", expired=False, quota_project_id=None, requires_scopes=False
    )
    mocker.patch("google.auth.default", return_value=(credentials, "unit-project"))
    return credentials


@pytest.mark.parametrize("model", MODELS)
def test_muse_responses_select_a_native_vertex_adapter_only_under_the_meta_publisher(model: str) -> None:
    assert ProviderConfigManager.get_provider_responses_api_config("vertex_ai", f"meta/{model}") is not None
    assert ProviderConfigManager.get_provider_responses_api_config("vertex_ai", model) is None
    assert ProviderConfigManager.get_provider_responses_api_config("vertex_ai", "gemini-4-argon") is None


@pytest.mark.asyncio
async def test_muse_responses_websocket_streams_a_turn_over_the_vertex_http_endpoint(
    respx_mock: respx.MockRouter,
) -> None:
    model: Final = "muse-wonder-1.0"
    route: Final = respx_mock.post(ENDPOINT).mock(
        return_value=httpx.Response(200, text=_event_stream(model), headers={"content-type": "text/event-stream"})
    )
    websocket: Final = Mock(send_text=AsyncMock(), receive_text=AsyncMock(side_effect=ConnectionError))
    client: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport())
    try:
        await litellm._aresponses_websocket(
            model=f"vertex_ai/meta/{model}",
            websocket=websocket,
            vertex_project="unit-project",
            first_message=json.dumps({"type": "response.create", "input": "Say hello", "store": False}),
            client=client,
        )
    finally:
        await client.close()
    sent: Final = tuple(FIELDS.validate_json(call.args[0]) for call in websocket.send_text.await_args_list)
    assert FIELDS.validate_json(route.calls.last.request.content)["model"] == model
    assert [event["type"] for event in sent] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]


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
    route: Final = respx_mock.post(ENDPOINT).mock(
        return_value=httpx.Response(200, text=_event_stream(model), headers={"content-type": "text/event-stream"})
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


def test_muse_drop_params_sends_store_false_instead_of_refusing(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    litellm.responses(
        model="vertex_ai/meta/muse-wonder-1.0",
        input="Say hello",
        vertex_project="unit-project",
        store=True,
        drop_params=True,
        num_retries=0,
    )
    assert FIELDS.validate_json(route.calls.last.request.content)["store"] is False


@pytest.mark.parametrize(
    ("model", "input_tokens", "price_suffix"),
    tuple((model, 47, "") for model in MODELS)
    + (("muse-ember-1.0", 200_000, ""), ("muse-ember-1.0", 200_001, "_above_200k_tokens")),
)
def test_muse_reports_catalog_cost_for_uncached_cached_and_output_tokens(
    model: str, input_tokens: int, price_suffix: str, respx_mock: respx.MockRouter
) -> None:
    key: Final = f"vertex_ai/meta/{model}"
    entry: Final = FIELDS.validate_python(litellm.model_cost[key])
    cached_tokens: Final = 11
    output_tokens: Final = 13
    usage: Final = {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json={**_response(model), "usage": usage}))
    response: Final = litellm.responses(model=key, input="Say hello", vertex_project="unit-project", num_retries=0)
    input_rate: Final = TypeAdapter(float).validate_python(entry[f"input_cost_per_token{price_suffix}"])
    cached_rate: Final = TypeAdapter(float).validate_python(entry[f"cache_read_input_token_cost{price_suffix}"])
    output_rate: Final = TypeAdapter(float).validate_python(entry[f"output_cost_per_token{price_suffix}"])
    expected: Final = (
        (input_tokens - cached_tokens) * input_rate + cached_tokens * cached_rate + output_tokens * output_rate
    )
    assert response.hidden_params["response_cost"] == pytest.approx(expected)


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


@pytest.mark.parametrize("model", ["muse-spark-9.9", "muse-nova-1.0"])
def test_muse_chat_completions_bridge_covers_models_missing_from_the_cost_map(
    model: str, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    response: Final = litellm.completion(
        model=f"vertex_ai/meta/{model}",
        messages=[{"role": "user", "content": "Say hello"}],
        vertex_project="unit-project",
        num_retries=0,
    )
    assert FIELDS.validate_json(route.calls.last.request.content)["model"] == model
    assert response.choices[0].message.content == "Hello"


@pytest.mark.parametrize("surface", ["responses", "chat"])
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("choice", ["auto", "none", "required", {"type": "function", "name": "report_number"}])
def test_muse_drop_params_preserves_disabled_tools_and_downgrades_forced_choices(
    model: str, surface: str, choice: str | dict[str, str], respx_mock: respx.MockRouter
) -> None:
    tool: Final = {
        "type": "function",
        "name": "report_number",
        "description": "Report the fixture number.",
        "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]},
    }
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    if surface == "responses":
        litellm.responses(
            model=f"vertex_ai/meta/{model}",
            input="Report number 7.",
            tools=[tool],
            tool_choice=choice,
            vertex_project="unit-project",
            drop_params=True,
            num_retries=0,
        )
    else:
        chat_tool: Final = {
            "type": "function",
            "function": {key: value for key, value in tool.items() if key != "type"},
        }
        chat_choice: Final = (
            {"type": "function", "function": {"name": choice["name"]}} if isinstance(choice, dict) else choice
        )
        litellm.completion(
            model=f"vertex_ai/meta/{model}",
            messages=[{"role": "user", "content": "Report number 7."}],
            tools=[chat_tool],
            tool_choice=chat_choice,
            vertex_project="unit-project",
            drop_params=True,
            num_retries=0,
        )
    body: Final = FIELDS.validate_json(route.calls.last.request.content)
    wire_tool: Final = tool if surface == "responses" else {**tool, "strict": None}
    assert body.get("tool_choice") == "auto"
    assert body.get("tools", []) == ([] if choice == "none" else [wire_tool])


@pytest.mark.parametrize("choice", ["none", "required", {"type": "function", "name": "report_number"}])
@pytest.mark.parametrize("model", MODELS)
def test_muse_rejects_unsupported_tool_choices_before_contacting_vertex(
    model: str, choice: str | dict[str, str], respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    with pytest.raises(litellm.UnsupportedParamsError, match=r"tool_choice.*auto"):
        litellm.responses(
            model=f"vertex_ai/meta/{model}",
            input="Report number 7.",
            tool_choice=choice,
            vertex_project="unit-project",
            drop_params=False,
            num_retries=0,
        )
    assert not route.called


def test_muse_accepts_nested_workload_identity_credentials(respx_mock: respx.MockRouter, mocker: MockerFixture) -> None:
    credential_type: Final = mocker.patch("google.auth.identity_pool.Credentials")
    credential_type.from_info.return_value = mocker.Mock(
        spec=Credentials, token="wif-token", expired=False, quota_project_id=None, requires_scopes=False
    )
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
    assert route.calls.last.request.headers["authorization"] == "Bearer wif-token"
    assert response.output_text == "Hello"


@pytest.mark.parametrize(("field", "value"), [("background", True), ("previous_response_id", "resp_unit")])
def test_muse_drop_params_omits_server_side_state_fields(
    field: str, value: object, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    litellm.responses(
        model="vertex_ai/meta/muse-wonder-1.0",
        input="Say hello",
        vertex_project="unit-project",
        drop_params=True,
        num_retries=0,
        **{field: value},
    )
    assert field not in FIELDS.validate_json(route.calls.last.request.content)


@pytest.mark.parametrize("location", ["us-central1", "europe-west4"])
def test_muse_rejects_a_regional_location_before_contacting_vertex(
    location: str, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    with pytest.raises(litellm.UnsupportedParamsError, match="global location"):
        litellm.responses(
            model="vertex_ai/meta/muse-wonder-1.0",
            input="Say hello",
            vertex_project="unit-project",
            vertex_location=location,
            num_retries=0,
        )
    assert not route.called


@pytest.mark.parametrize("location", [None, "global"])
def test_muse_accepts_the_global_location(location: str | None, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    litellm.responses(
        model="vertex_ai/meta/muse-wonder-1.0",
        input="Say hello",
        vertex_project="unit-project",
        vertex_location=location,
        num_retries=0,
    )
    assert route.called


@pytest.mark.parametrize(
    ("api_base", "gateway"),
    [
        ("https://vertex-gateway.example", "https://vertex-gateway.example"),
        ("https://vertex-gateway.example/v1/", "https://vertex-gateway.example"),
        ("https://vertex-gateway.example/google-vertex/v1", "https://vertex-gateway.example/google-vertex"),
        ("https://vertex-gateway.example/google-vertex?tenant=a", "https://vertex-gateway.example/google-vertex"),
        (ENDPOINT.removesuffix("/responses"), "https://aiplatform.googleapis.com"),
        (ENDPOINT, "https://aiplatform.googleapis.com"),
    ],
)
def test_muse_sends_a_custom_api_base_to_the_meta_responses_path(
    api_base: str, gateway: str, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT.replace("https://aiplatform.googleapis.com", gateway)).mock(
        return_value=httpx.Response(200, json=_response("muse-wonder-1.0"))
    )
    response: Final = litellm.responses(
        model="vertex_ai/meta/muse-wonder-1.0",
        input="Say hello",
        vertex_project="unit-project",
        api_base=api_base,
        num_retries=0,
    )
    assert route.calls.last.request.url.query.decode() == urlparse(api_base).query
    assert response.output_text == "Hello"


def test_muse_drop_params_sends_a_regional_location_to_the_global_endpoint(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-wonder-1.0")))
    litellm.responses(
        model="vertex_ai/meta/muse-wonder-1.0",
        input="Say hello",
        vertex_project="unit-project",
        vertex_location="us-central1",
        drop_params=True,
        num_retries=0,
    )
    assert route.called


@pytest.mark.parametrize("prefix", ["vertex", "vertex_ai"])
def test_muse_chat_completions_bridge_authenticates_with_the_deployment_credentials(
    prefix: str, respx_mock: respx.MockRouter, mocker: MockerFixture
) -> None:
    deployment_credentials: Final = mocker.Mock(
        spec=Credentials, token=f"{prefix}-token", expired=False, quota_project_id=None, requires_scopes=False
    )
    credential_type: Final = mocker.patch("google.auth.identity_pool.Credentials")
    credential_type.from_info.return_value = deployment_credentials
    route: Final = respx_mock.post(ENDPOINT.replace("unit-project", f"{prefix}-project")).mock(
        return_value=httpx.Response(200, json=_response("muse-ember-1.0"))
    )
    response: Final = litellm.completion(
        model="vertex_ai/meta/muse-ember-1.0",
        messages=[{"role": "user", "content": "Say hello"}],
        num_retries=0,
        **{
            f"{prefix}_project": f"{prefix}-project",
            f"{prefix}_credentials": {
                "type": "external_account",
                "audience": f"{prefix}-audience",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": "https://sts.googleapis.com/v1/token",
                "credential_source": {"file": "/unit/subject.token"},
            },
        },
    )
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == f"Bearer {prefix}-token"
    assert credential_type.from_info.call_args.args[0]["audience"] == f"{prefix}-audience"
    assert not {"vertex_project", "vertex_credentials", "vertex_location"} & FIELDS.validate_json(request.content).keys()
    assert response.choices[0].message.content == "Hello"


def test_muse_chat_completions_bridge_rejects_a_regional_deployment_location(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-ember-1.0")))
    with pytest.raises(litellm.UnsupportedParamsError, match="global location"):
        litellm.completion(
            model="vertex_ai/meta/muse-ember-1.0",
            messages=[{"role": "user", "content": "Say hello"}],
            vertex_project="unit-project",
            vertex_location="us-central1",
            num_retries=0,
        )
    assert not route.called


@pytest.mark.parametrize("param", [{"stop": ["END"]}, {"seed": 7}, {"n": 2}, {"logprobs": True}])
def test_muse_chat_completions_bridge_refuses_params_it_cannot_forward(
    param: dict[str, object], respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-ember-1.0")))
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(
            model="vertex_ai/meta/muse-ember-1.0",
            messages=[{"role": "user", "content": "Say hello"}],
            vertex_project="unit-project",
            num_retries=0,
            **param,
        )
    assert not route.called
    litellm.completion(
        model="vertex_ai/meta/muse-ember-1.0",
        messages=[{"role": "user", "content": "Say hello"}],
        vertex_project="unit-project",
        drop_params=True,
        num_retries=0,
        **param,
    )
    assert route.called


def test_muse_chat_completions_bridge_forwards_user_and_reasoning_effort(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-ember-1.0")))
    litellm.completion(
        model="vertex_ai/meta/muse-ember-1.0",
        messages=[{"role": "user", "content": "Say hello"}],
        vertex_project="unit-project",
        user="unit-user",
        reasoning_effort="low",
        num_retries=0,
    )
    body: Final = FIELDS.validate_json(route.calls.last.request.content)
    assert body["user"] == "unit-user"
    assert body["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("surface", ["responses", "chat"])
def test_muse_refuses_automatic_truncation_unless_dropped(surface: str, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-ember-1.0")))

    def call(**kwargs: object) -> None:
        if surface == "responses":
            _ = litellm.responses(model="vertex_ai/meta/muse-ember-1.0", input="Say hello", **kwargs)
        else:
            _ = litellm.completion(
                model="vertex_ai/meta/muse-ember-1.0", messages=[{"role": "user", "content": "Say hello"}], **kwargs
            )

    with pytest.raises(litellm.UnsupportedParamsError, match="truncation"):
        call(vertex_project="unit-project", truncation="auto", num_retries=0)
    assert not route.called
    call(vertex_project="unit-project", truncation="auto", drop_params=True, num_retries=0)
    assert "truncation" not in FIELDS.validate_json(route.calls.last.request.content)
    call(vertex_project="unit-project", truncation="disabled", num_retries=0)
    assert FIELDS.validate_json(route.calls.last.request.content)["truncation"] == "disabled"


def test_muse_keeps_the_routed_model_when_extra_body_names_another(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-spark-1.3")))
    _ = litellm.responses(
        model="vertex_ai/meta/muse-spark-1.3",
        input="Say hello",
        vertex_project="unit-project",
        extra_body={"model": "muse-wonder-1.0", "metadata": {"k": "v"}},
        num_retries=0,
    )
    body: Final = FIELDS.validate_json(route.calls.last.request.content)
    assert body["model"] == "muse-spark-1.3"
    assert body["metadata"] == {"k": "v"}


@pytest.mark.parametrize("model", MODELS)
def test_muse_chat_completions_bridge_forwards_a_pdf_file_part(model: str, respx_mock: respx.MockRouter) -> None:
    pdf: Final = {"file_data": "data:application/pdf;base64,JVBERi0xLjQK", "filename": "unit.pdf"}
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    response: Final = litellm.completion(
        model=f"vertex_ai/meta/{model}",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Summarize"}, {"type": "file", "file": pdf}]}],
        vertex_project="unit-project",
        num_retries=0,
    )
    assert FIELDS.validate_json(route.calls.last.request.content)["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Summarize"}, {"type": "input_file", **pdf}],
        }
    ]
    assert response.choices[0].message.content == "Hello"
    assert litellm.get_model_info(f"vertex_ai/meta/{model}")["supports_pdf_input"] is True


def test_muse_chat_completions_bridge_falls_back_to_the_alias_when_the_project_is_blank(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(ENDPOINT.replace("unit-project", "alias-project")).mock(
        return_value=httpx.Response(200, json=_response("muse-ember-1.0"))
    )
    response: Final = litellm.completion(
        model="vertex_ai/meta/muse-ember-1.0",
        messages=[{"role": "user", "content": "Say hello"}],
        vertex_project="",
        vertex_ai_project="alias-project",
        num_retries=0,
    )
    assert route.called
    assert response.choices[0].message.content == "Hello"


def _row_levels(model: str, supported: bool) -> tuple[str, ...]:
    row: Final = CATALOG.validate_json((ROOT / "model_prices_and_context_window.json").read_bytes())[
        f"vertex_ai/meta/{model}"
    ]
    return tuple(
        level
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max")
        if row.get(f"supports_{level}_reasoning_effort", True) is supported
    )


def _effort_cases(supported: bool) -> tuple[tuple[str, str], ...]:
    return tuple((model, level) for model in MODELS for level in _row_levels(model, supported))


def _call_with_effort(surface: str, model: str, level: str, **kwargs: object) -> None:
    if surface == "responses":
        _ = litellm.responses(
            model=f"vertex_ai/meta/{model}", input="Say hello", reasoning={"effort": level, "summary": "auto"}, **kwargs
        )
        return
    _ = litellm.completion(
        model=f"vertex_ai/meta/{model}", messages=[{"role": "user", "content": "Say hello"}], reasoning_effort=level, **kwargs
    )


@pytest.mark.parametrize("surface", ["responses", "chat"])
@pytest.mark.parametrize(("model", "level"), _effort_cases(supported=False))
def test_muse_drops_a_reasoning_effort_its_row_refuses_only_under_drop_params(
    surface: str, model: str, level: str, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    with pytest.raises(litellm.UnsupportedParamsError, match=level):
        _call_with_effort(surface, model, level, vertex_project="unit-project", num_retries=0)
    assert not route.called
    _call_with_effort(surface, model, level, vertex_project="unit-project", drop_params=True, num_retries=0)
    assert "effort" not in FIELDS.validate_python(
        FIELDS.validate_json(route.calls.last.request.content).get("reasoning") or {}
    )


@pytest.mark.parametrize(("model", "level"), _effort_cases(supported=True))
def test_muse_forwards_a_reasoning_effort_its_row_accepts(model: str, level: str, respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    _call_with_effort("responses", model, level, vertex_project="unit-project", num_retries=0)
    assert FIELDS.validate_json(route.calls.last.request.content)["reasoning"] == {"effort": level, "summary": "auto"}


def test_muse_messages_drop_disabled_thinking_under_drop_params(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response("muse-ember-1.0")))
    _ = litellm.anthropic.messages.create(
        model="vertex_ai/meta/muse-ember-1.0",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=64,
        thinking={"type": "disabled"},
        vertex_project="unit-project",
        drop_params=True,
    )
    assert route.called
    assert "effort" not in FIELDS.validate_python(
        FIELDS.validate_json(route.calls.last.request.content).get("reasoning") or {}
    )


@pytest.mark.parametrize("model", MODELS)
def test_muse_reaches_its_responses_endpoint_when_the_chat_completions_bridge_is_requested(
    model: str, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(ENDPOINT).mock(return_value=httpx.Response(200, json=_response(model)))
    response: Final = litellm.responses(
        model=f"vertex_ai/meta/{model}",
        input="Say hello",
        vertex_project="unit-project",
        use_chat_completions_api=True,
        num_retries=0,
    )
    assert route.call_count == 1
    assert response.output_text == "Hello"
