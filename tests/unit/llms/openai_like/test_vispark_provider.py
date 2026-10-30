import json
from typing import Final

import pytest
import respx

import litellm

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_vispark",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "vision-medium",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Vispark"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}

_TEXT_COMPLETION: Final = {
    "id": "cmpl_vispark",
    "object": "text_completion",
    "created": 1_790_000_000,
    "model": "vision-small",
    "choices": [{"index": 0, "text": "Hello from Vispark", "finish_reason": "stop", "logprobs": None}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_vispark_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")
    monkeypatch.delenv("VISPARK_LAB_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="vispark/vision-medium",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "vision-medium"
    assert provider == "vispark"
    assert api_key == "vispark-env-key"
    assert api_base == "https://api.lab.vispark.in/v1"


def test_vispark_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")
    monkeypatch.setenv("VISPARK_LAB_API_BASE", "https://vispark.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="vispark/vision-medium",
        custom_llm_provider=None,
        api_base="https://vispark.internal.example/v1",
        api_key="vispark-explicit-key",
    )

    assert provider == "vispark"
    assert api_key == "vispark-explicit-key"
    assert api_base == "https://vispark.internal.example/v1"


def test_vispark_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")
    monkeypatch.setenv("VISPARK_LAB_API_BASE", "https://vispark.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="vispark/vision-medium",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "vispark"
    assert api_base == "https://vispark.env.example/v1"


def test_vispark_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="vision-medium",
        custom_llm_provider=None,
        api_base="https://api.lab.vispark.in/v1",
        api_key=None,
    )

    assert model == "vision-medium"
    assert provider == "vispark"
    assert api_key == "vispark-env-key"
    assert api_base == "https://api.lab.vispark.in/v1"


def test_vispark_chat_completion_request_maps_max_completion_tokens():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="vispark/vision-medium",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="vispark-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer vispark-test-key"
    assert body["model"] == "vision-medium"
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "Hello from Vispark"


@pytest.mark.parametrize("additional_properties", [False, True])
def test_vispark_strips_additional_properties_from_a_json_schema_response_format(additional_properties: bool):
    city_schema: Final = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "additionalProperties": additional_properties,
    }
    schema: Final = {
        "type": "object",
        "properties": {"city": city_schema},
        "required": ["city"],
        "additionalProperties": False,
    }
    response_format: Final = {"type": "json_schema", "json_schema": {"name": "City", "schema": schema, "strict": True}}
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        litellm.completion(
            model="vispark/vision-medium",
            messages=[{"role": "user", "content": "Capital of France as JSON"}],
            response_format=response_format,
            api_key="vispark-test-key",
        )

    sent_format: Final = json.loads(route.calls.last.request.content)["response_format"]
    assert "additionalProperties" not in json.dumps(sent_format)
    assert sent_format["json_schema"]["strict"] is True
    assert sent_format["json_schema"]["schema"]["properties"]["city"]["properties"] == {"name": {"type": "string"}}
    assert city_schema["additionalProperties"] == additional_properties


def _map_field_response_format() -> dict[str, object]:
    schema: Final = {
        "type": "object",
        "properties": {"facts": {"type": "object", "additionalProperties": {"type": "string"}}},
        "required": ["facts"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "Facts", "schema": schema}}


def test_vispark_rejects_a_map_field_response_format_without_drop_params():
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        with pytest.raises(litellm.UnsupportedParamsError) as error:
            litellm.completion(
                model="vispark/vision-medium",
                messages=[{"role": "user", "content": "Facts about Paris as JSON"}],
                response_format=_map_field_response_format(),
                api_key="vispark-test-key",
            )

    assert error.value.status_code == 400
    assert route.call_count == 0


def test_vispark_drops_a_map_field_schema_with_drop_params():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        litellm.completion(
            model="vispark/vision-medium",
            messages=[{"role": "user", "content": "Facts about Paris as JSON"}],
            response_format=_map_field_response_format(),
            drop_params=True,
            api_key="vispark-test-key",
        )

    sent_schema: Final = json.loads(route.calls.last.request.content)["response_format"]["json_schema"]["schema"]
    assert sent_schema == {"type": "object", "properties": {"facts": {"type": "object"}}, "required": ["facts"]}


def test_vispark_keeps_a_schema_property_named_additional_properties():
    schema: Final = {
        "type": "object",
        "properties": {"additionalProperties": {"type": "string", "additionalProperties": False}},
        "required": ["additionalProperties"],
        "additionalProperties": False,
    }
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        litellm.completion(
            model="vispark/vision-medium",
            messages=[{"role": "user", "content": "Describe Paris as JSON"}],
            response_format={"type": "json_schema", "json_schema": {"name": "Notes", "schema": schema}},
            api_key="vispark-test-key",
        )

    sent_schema: Final = json.loads(route.calls.last.request.content)["response_format"]["json_schema"]["schema"]
    assert sent_schema == {
        "type": "object",
        "properties": {"additionalProperties": {"type": "string"}},
        "required": ["additionalProperties"],
    }


def _file_id_messages() -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this file"},
                {"type": "file", "file": {"file_id": "file-abc123"}},
            ],
        }
    ]


def test_vispark_rejects_a_file_part_sent_by_file_id_without_drop_params():
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        with pytest.raises(litellm.BadRequestError, match="file_id") as error:
            litellm.completion(model="vispark/vision-medium", messages=_file_id_messages(), api_key="vispark-test-key")

    assert error.value.status_code == 400
    assert route.call_count == 0


@pytest.mark.asyncio
async def test_vispark_drops_a_file_part_sent_by_file_id_with_drop_params(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        await litellm.acompletion(
            model="vispark/vision-medium",
            messages=_file_id_messages(),
            drop_params=True,
            api_key="vispark-test-key",
        )

    sent_content: Final = json.loads(route.calls.last.request.content)["messages"][0]["content"]
    assert sent_content == [{"type": "text", "text": "Summarize this file"}]


def test_other_json_providers_keep_additional_properties_and_file_parts():
    file_part: Final = {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0xLjQK"}}
    response_format: Final = {
        "type": "json_schema",
        "json_schema": {
            "name": "City",
            "schema": {"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False},
        },
    }
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.scaleway.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="scaleway/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": [{"type": "text", "text": "Read this"}, file_part]}],
            response_format=response_format,
            api_key="scaleway-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["response_format"] == response_format
    assert body["messages"][0]["content"][1] == file_part


_PDF_DATA_URL: Final = "data:application/pdf;base64,JVBERi0xLjQK"


@pytest.mark.parametrize(
    "file",
    [{"file_data": _PDF_DATA_URL}, {"file_id": "file-abc123", "file_data": _PDF_DATA_URL}],
    ids=["file_data", "file_id_and_file_data"],
)
def test_vispark_sends_a_pdf_file_part_as_an_image_url(file: dict[str, str]):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        litellm.completion(
            model="vispark/vision-medium",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What code is in this PDF?"},
                        {"type": "file", "file": file},
                    ],
                }
            ],
            api_key="vispark-test-key",
        )

    sent_content: Final = json.loads(route.calls.last.request.content)["messages"][0]["content"]
    assert sent_content[0] == {"type": "text", "text": "What code is in this PDF?"}
    assert sent_content[1]["type"] == "image_url"
    assert sent_content[1]["image_url"]["url"] == _PDF_DATA_URL
    assert all(part["type"] != "file" for part in sent_content)


def test_vispark_text_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/completions").respond(200, json=_TEXT_COMPLETION)
        response: Final = litellm.text_completion(
            model="vispark/vision-small",
            prompt="Say hello",
            api_key="vispark-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "vision-small"
    assert body["prompt"] == "Say hello"
    assert response.choices[0].text == "Hello from Vispark"


def test_vispark_cost_follows_the_cost_map():
    cost_entry: Final = litellm.model_cost["vispark/vision-large"]
    prompt_tokens: Final = cost_entry["min_billed_input_tokens"] + 1_000

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="vision-large",
        custom_llm_provider="vispark",
        prompt_tokens=prompt_tokens,
        completion_tokens=500,
    )

    assert prompt_cost == pytest.approx(prompt_tokens * cost_entry["input_cost_per_token"])
    assert completion_cost == pytest.approx(500 * cost_entry["output_cost_per_token"])


@pytest.mark.parametrize("model", ["vision-small", "vision-medium", "vision-large"])
def test_vispark_bills_a_short_prompt_at_its_minimum_input_tokens(model: str):
    cost_entry: Final = litellm.model_cost[f"vispark/{model}"]
    minimum: Final = cost_entry["min_billed_input_tokens"]

    prompt_cost, _ = litellm.cost_per_token(
        model=model,
        custom_llm_provider="vispark",
        prompt_tokens=minimum // 2,
        completion_tokens=1,
    )

    assert prompt_cost == pytest.approx(minimum * cost_entry["input_cost_per_token"])


@pytest.mark.parametrize(
    "custom_pricing",
    [
        {"input_cost_per_token": 2e-6, "output_cost_per_token": 5e-6},
        {"tiered_pricing": [{"range": [0, 1_000_000], "input_cost_per_token": 2e-6, "output_cost_per_token": 5e-6}]},
    ],
    ids=["flat", "tiered"],
)
def test_vispark_router_deployment_with_custom_pricing_still_bills_the_minimum_input_tokens(
    custom_pricing: dict[str, object],
):
    minimum: Final = litellm.model_cost["vispark/vision-small"]["min_billed_input_tokens"]
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "vispark-custom-priced",
                "litellm_params": {
                    "model": "vispark/vision-small",
                    "api_key": "vispark-test-key",
                    **custom_pricing,
                },
            }
        ]
    )
    with respx.mock() as upstream:
        upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = router.completion(
            model="vispark-custom-priced", messages=[{"role": "user", "content": "Say hello"}]
        )

    completion_tokens: Final = _CHAT_COMPLETION["usage"]["completion_tokens"]
    assert response._hidden_params["response_cost"] == pytest.approx(minimum * 2e-6 + completion_tokens * 5e-6)


def test_vispark_deployment_minimum_override_stays_on_its_own_deployment():
    builtin: Final = litellm.model_cost["vispark/vision-small"]
    minimum: Final = builtin["min_billed_input_tokens"]
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "vispark-no-minimum",
                "litellm_params": {"model": "vispark/vision-small", "api_key": "vispark-test-key"},
                "model_info": {"min_billed_input_tokens": 0},
            },
            {
                "model_name": "vispark-default",
                "litellm_params": {"model": "vispark/vision-small", "api_key": "vispark-test-key"},
            },
        ]
    )
    with respx.mock() as upstream:
        upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json={**_CHAT_COMPLETION, "model": "vision-small"}
        )
        no_minimum: Final = router.completion(
            model="vispark-no-minimum", messages=[{"role": "user", "content": "Say hello"}]
        )
        default: Final = router.completion(model="vispark-default", messages=[{"role": "user", "content": "Say hello"}])

    usage: Final = _CHAT_COMPLETION["usage"]
    completion_cost: Final = usage["completion_tokens"] * builtin["output_cost_per_token"]
    assert no_minimum._hidden_params["response_cost"] == pytest.approx(
        usage["prompt_tokens"] * builtin["input_cost_per_token"] + completion_cost
    )
    assert default._hidden_params["response_cost"] == pytest.approx(
        minimum * builtin["input_cost_per_token"] + completion_cost
    )
    assert litellm.model_cost["vispark/vision-small"]["min_billed_input_tokens"] == minimum


def test_vispark_explicit_custom_cost_per_token_still_bills_the_minimum_input_tokens():
    minimum: Final = litellm.model_cost["vispark/vision-small"]["min_billed_input_tokens"]

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="vision-small",
        custom_llm_provider="vispark",
        prompt_tokens=minimum // 2,
        completion_tokens=3,
        custom_cost_per_token={"input_cost_per_token": 2e-6, "output_cost_per_token": 5e-6},
    )

    assert prompt_cost == pytest.approx(minimum * 2e-6)
    assert completion_cost == pytest.approx(3 * 5e-6)


def test_vispark_explicit_custom_cost_per_token_bills_no_minimum_without_a_counted_prompt():
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="vision-small",
        custom_llm_provider="vispark",
        prompt_tokens=0,
        completion_tokens=3,
        custom_cost_per_token={"input_cost_per_token": 2e-6, "output_cost_per_token": 5e-6},
    )

    assert prompt_cost == 0.0
    assert completion_cost == pytest.approx(3 * 5e-6)


def test_vispark_models_are_listed_for_the_provider():
    from litellm.utils import get_valid_models

    expected: Final = {name for name, entry in litellm.model_cost.items() if entry.get("litellm_provider") == "vispark"}

    assert expected
    assert set(litellm.models_by_provider["vispark"]) == expected
    assert expected <= set(get_valid_models(custom_llm_provider="vispark"))
