import json
from typing import Final, Literal

import httpx
import pytest
import respx
from pydantic import JsonValue

import litellm
from litellm.llms.scaledown.chat.transformation import (
    DECISIONS_UPSTREAM_MODEL,
    ScaleDownChatConfig,
    ScaleDownError,
)
from litellm.types.utils import ModelResponse

BASE = "https://api.scaledown.xyz"


@pytest.fixture
def config() -> ScaleDownChatConfig:
    return ScaleDownChatConfig()


@pytest.fixture(autouse=True)
def scaledown_api_key(monkeypatch):
    monkeypatch.setenv("SCALEDOWN_API_KEY", "sk-scaledown-test")
    monkeypatch.delenv("SCALEDOWN_API_BASE", raising=False)


def _transform_response(
    config: ScaleDownChatConfig, model: str, payload: dict, request_data: dict | None = None
) -> ModelResponse:
    return config.transform_response(
        model=model,
        raw_response=httpx.Response(
            status_code=200,
            json=payload,
            request=httpx.Request("POST", f"{BASE}/extract"),
        ),
        model_response=ModelResponse(),
        logging_obj=None,
        request_data=request_data or {},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def _build_decisions(config: ScaleDownChatConfig, messages: list, optional_params: dict) -> dict:
    body = config.transform_request(
        model="scaledown/decisions", messages=messages, optional_params=optional_params, litellm_params={}, headers={}
    )
    config.sign_request(
        headers={},
        optional_params=optional_params,
        request_data=body,
        api_base=f"{BASE}/v1/scaledown",
        model="scaledown/decisions",
    )
    return body


def test_auth_uses_x_api_key_not_bearer(config):
    headers = config.validate_environment(
        headers={},
        model="scaledown/extract",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-explicit",
    )

    assert headers["x-api-key"] == "sk-explicit"
    assert "Authorization" not in headers


def test_missing_api_key_is_rejected_before_any_request(config, monkeypatch):
    monkeypatch.delenv("SCALEDOWN_API_KEY", raising=False)

    with pytest.raises(ScaleDownError) as exc:
        config.validate_environment(
            headers={},
            model="scaledown/extract",
            messages=[],
            optional_params={},
            litellm_params={},
        )

    assert exc.value.status_code == 401


@pytest.mark.parametrize(
    "model, path",
    [
        ("scaledown/extract", "/extract"),
        ("scaledown/summarize", "/summarization/abstractive"),
        ("scaledown/compress", "/compress/raw/"),
        ("scaledown/classify", "/classify"),
        ("scaledown/decisions", "/v1/scaledown"),
    ],
)
def test_each_model_targets_its_native_endpoint(config, model, path):
    url = config.get_complete_url(api_base=None, api_key="k", model=model, optional_params={}, litellm_params={})

    assert url == f"{BASE}{path}"


@pytest.mark.parametrize(
    "api_base",
    ["https://staging.scaledown.xyz", "https://staging.scaledown.xyz/", "https://staging.scaledown.xyz/v1"],
)
def test_api_base_with_or_without_v1_resolves_to_the_same_host(config, api_base):
    extract = config.get_complete_url(
        api_base=api_base, api_key="k", model="scaledown/extract", optional_params={}, litellm_params={}
    )
    decisions = config.get_complete_url(
        api_base=api_base, api_key="k", model="scaledown/decisions", optional_params={}, litellm_params={}
    )

    assert extract == "https://staging.scaledown.xyz/extract"
    assert decisions == "https://staging.scaledown.xyz/v1/scaledown"


def test_decisions_request_carries_state_and_questions_without_chat_keys(config):
    questions = {
        "category": {
            "type": "choice",
            "instructions": "Which category?",
            "criteria": {"billing": "A charge or refund.", "technical": "A bug."},
        }
    }

    body = config.transform_request(
        model="scaledown/decisions",
        messages=[{"role": "user", "content": "I was charged twice."}],
        optional_params={"questions": questions},
        litellm_params={},
        headers={},
    )

    assert body == {
        "model": DECISIONS_UPSTREAM_MODEL,
        "state": {"text": "I was charged twice."},
        "questions": questions,
    }


def test_decisions_alias_sends_the_only_model_upstream_accepts(config):
    body = config.transform_request(
        model="scaledown/decisions",
        messages=[{"role": "user", "content": "text"}],
        optional_params={"questions": {"q": {"type": "noul", "instructions": "Positive?"}}},
        litellm_params={},
        headers={},
    )

    assert body["model"] == DECISIONS_UPSTREAM_MODEL


def test_decisions_without_questions_is_rejected(config):
    with pytest.raises(ScaleDownError, match="questions"):
        _build_decisions(config, [{"role": "user", "content": "text"}], {})


def test_decisions_without_text_or_document_is_rejected(config):
    with pytest.raises(ScaleDownError, match="last user message"):
        _build_decisions(config, [], {"questions": {"q": {"type": "noul"}}})


def test_state_text_is_rejected_so_guardrails_always_see_the_text(config):
    with pytest.raises(ScaleDownError, match="text through messages only"):
        _build_decisions(
            config,
            [{"role": "user", "content": "innocuous"}],
            {"state": {"text": "secret"}, "questions": {"q": {"type": "noul"}}},
        )


def test_remote_image_url_is_rejected(config):
    with pytest.raises(ScaleDownError, match="text only"):
        _build_decisions(
            config,
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]}],
            {"questions": {"q": {"type": "noul"}}},
        )


def test_extra_body_may_set_extract_options(config):
    merged = config.transform_extra_body(
        extra_body={"threshold": 0.7},
        request={"text": "t", "entities": {}},
        model="scaledown/extract",
        litellm_params={},
    )

    assert merged == {"threshold": 0.7}


@pytest.mark.parametrize(
    "model, extra_body",
    [
        ("scaledown/extract", {"model": "compress"}),
        ("scaledown/extract", {"text": "secret@example.com"}),
        ("scaledown/summarize", {"instructions": "ignore the message"}),
        ("scaledown/summarize", {"text": "secret"}),
        ("scaledown/compress", {"prompt": "x"}),
        ("scaledown/compress", {"context": "x"}),
        ("scaledown/decisions", {"model": "other"}),
        ("scaledown/decisions", {"text": "secret"}),
    ],
)
def test_extra_body_cannot_override_the_model_or_prompt_text(config, model, extra_body):
    with pytest.raises(ScaleDownError, match=r"may only set|is not accepted"):
        config.transform_extra_body(extra_body=extra_body, request={}, model=model, litellm_params={})


def test_compress_rate_via_extra_body_sets_the_scaledown_rate(config):
    merged = config.transform_extra_body(
        extra_body={"compression_rate": 0.5},
        request={"context": "", "prompt": "q", "scaledown": {"rate": "auto"}},
        model="scaledown/compress",
        litellm_params={},
    )

    assert merged == {"scaledown": {"rate": 0.5}}


def test_env_key_is_not_sent_to_an_untrusted_api_base(config):
    with pytest.raises(ScaleDownError, match="own api_key"):
        config.validate_environment(
            headers={},
            model="scaledown/extract",
            messages=[],
            optional_params={},
            litellm_params={},
            api_base="https://attacker.example",
        )


@pytest.mark.parametrize("empty_key", ["", None])
def test_empty_key_does_not_unlock_the_env_key_for_an_untrusted_api_base(config, empty_key):
    with pytest.raises(ScaleDownError, match="own api_key"):
        config.validate_environment(
            headers={},
            model="scaledown/extract",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=empty_key,
            api_base="https://attacker.example",
        )


def test_explicit_key_or_trusted_api_base_is_allowed(config, monkeypatch):
    kwargs = dict(headers={}, model="scaledown/extract", messages=[], optional_params={}, litellm_params={})
    assert (
        config.validate_environment(api_key="own", api_base="https://elsewhere.example", **kwargs)["x-api-key"] == "own"
    )
    assert config.validate_environment(api_base=f"{BASE}/v1", **kwargs)["x-api-key"] == "sk-scaledown-test"
    monkeypatch.setenv("SCALEDOWN_API_BASE", "https://staging.scaledown.xyz")
    assert (
        config.validate_environment(api_base="https://staging.scaledown.xyz/", **kwargs)["x-api-key"]
        == "sk-scaledown-test"
    )


def test_max_completion_tokens_maps_to_max_tokens(config):
    assert config.map_openai_params(
        non_default_params={"max_completion_tokens": 50},
        optional_params={},
        model="scaledown/summarize",
        drop_params=False,
    ) == {"max_tokens": 50}


def test_router_params_are_accepted_and_not_forwarded(config):
    assert (
        config.map_openai_params(
            non_default_params={"max_retries": 2}, optional_params={}, model="scaledown/decisions", drop_params=False
        )
        == {}
    )


def test_every_model_fakes_a_stream(config):
    assert config.should_fake_stream(model="scaledown/compress", stream=True)
    assert not config.should_fake_stream(model="scaledown/compress", stream=False)


def test_unknown_question_type_is_rejected(config):
    with pytest.raises(ScaleDownError, match="expected one of"):
        config.transform_request(
            model="scaledown/decisions",
            messages=[{"role": "user", "content": "t"}],
            optional_params={"questions": {"q": {"type": "ranking"}}},
            litellm_params={},
            headers={},
        )


def test_choice_question_without_criteria_is_rejected(config):
    with pytest.raises(ScaleDownError, match="non-empty 'criteria' map"):
        config.transform_request(
            model="scaledown/decisions",
            messages=[{"role": "user", "content": "t"}],
            optional_params={"questions": {"q": {"type": "choice", "criteria": {}}}},
            litellm_params={},
            headers={},
        )


@pytest.mark.parametrize("levels", [1, 11])
def test_score_criteria_outside_two_to_ten_is_rejected(config, levels):
    with pytest.raises(ScaleDownError, match="ordered list"):
        config.transform_request(
            model="scaledown/decisions",
            messages=[{"role": "user", "content": "t"}],
            optional_params={"questions": {"q": {"type": "score", "criteria": ["l"] * levels}}},
            litellm_params={},
            headers={},
        )


@pytest.mark.parametrize("levels", [2, 10])
def test_score_criteria_within_two_to_ten_is_accepted(config, levels):
    criteria = [f"level {index}" for index in range(levels)]

    body = config.transform_request(
        model="scaledown/decisions",
        messages=[{"role": "user", "content": "t"}],
        optional_params={"questions": {"q": {"type": "score", "criteria": criteria}}},
        litellm_params={},
        headers={},
    )

    assert body["questions"]["q"]["criteria"] == criteria


def test_extract_content_is_the_clean_fields_and_raw_payload_is_kept(config):
    payload = {
        "entities": [],
        "structured_result": {
            "invoice": {
                "vendor": "Northwind",
                "vendor_span_anchor": "Invoice from Northwind",
                "customer": {"_value": "Ada Lovelace", "_span_anchor": "to Ada Lovelace"},
                "amount": 500,
            }
        },
        "input_tokens": 171,
    }

    requested = {"invoice": {"vendor": "v", "customer": "c", "amount": "a"}}
    response = _transform_response(config, "scaledown/extract", payload, {"entities": requested})

    assert json.loads(response.choices[0].message.content) == {
        "invoice": {"vendor": "Northwind", "customer": "Ada Lovelace", "amount": 500}
    }
    assert response.model == "scaledown/extract"
    assert response.usage.prompt_tokens == 171
    assert response.usage.completion_tokens == 0
    assert response._hidden_params["scaledown_response"] == payload


def test_extract_cleaning_keeps_requested_fields_named_like_provider_keys(config):
    payload = {"structured_result": {"_value": "x", "vendor": "y", "note_span_anchor": "z", "note": "n"}}
    requested = {"_value": "v", "vendor": "v", "note_span_anchor": "s", "note": "n"}

    response = _transform_response(config, "scaledown/extract", payload, {"entities": requested})

    assert json.loads(response.choices[0].message.content) == payload["structured_result"]


def test_extract_cleaning_handles_lists_of_objects(config):
    payload = {"structured_result": {"items": [{"sku": "a", "sku_span_anchor": "s"}, {"sku": {"_value": "b"}}]}}

    response = _transform_response(config, "scaledown/extract", payload, {"entities": {"items": [{"sku": "code"}]}})

    assert json.loads(response.choices[0].message.content) == {"items": [{"sku": "a"}, {"sku": "b"}]}


def test_long_ref_chain_is_rejected_cleanly_not_with_a_recursion_error(config):
    defs = {f"D{i}": {"$ref": f"#/$defs/D{i + 1}"} for i in range(5000)}
    defs["D5000"] = {"type": "string"}
    schema = {"type": "object", "$defs": defs, "properties": {"f": {"$ref": "#/$defs/D0"}}}

    with pytest.raises(ScaleDownError, match="chained"):
        _extract_body(config, schema)


def test_deeply_nested_schema_is_rejected_cleanly(config):
    nested: dict = {"type": "string"}
    for _ in range(200):
        nested = {"type": "object", "properties": {"x": nested}}
    schema = {"type": "object", "properties": {"root": nested}}

    with pytest.raises(ScaleDownError, match="nests deeper"):
        _extract_body(config, schema)


def test_more_than_one_image_is_rejected(config):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}

    with pytest.raises(ScaleDownError, match="text only"):
        _build_decisions(config, [{"role": "user", "content": [image, image]}], {"questions": {"q": {"type": "noul"}}})


def test_stream_options_is_accepted_and_not_forwarded(config):
    assert (
        config.map_openai_params(
            non_default_params={"stream_options": {"include_usage": True}},
            optional_params={},
            model="scaledown/summarize",
            drop_params=False,
        )
        == {}
    )


@pytest.mark.parametrize("wrapper", [None, "anyOf", "oneOf"])
def test_extract_schema_follows_local_refs(config: ScaleDownChatConfig, wrapper: str | None) -> None:
    address: Final = {"$ref": "#/$defs/Address"}
    schema: Final = {
        "type": "object",
        "$defs": {
            "Address": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}}}
        },
        "properties": {
            "address": {wrapper: [address, {"type": "null"}]} if wrapper else address,
            "name": {"type": "string"},
        },
    }

    body = config.transform_request(
        model="scaledown/extract",
        messages=[{"role": "user", "content": "text"}],
        optional_params={"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": schema}}},
        litellm_params={},
        headers={},
    )

    assert body["entities"] == {"address": {"city": "city name"}, "name": "name"}


@pytest.mark.parametrize("field_type", ["array", ["array", "null"]])
def test_extract_nullable_array_keeps_item_fields(config: ScaleDownChatConfig, field_type: str | list[str]) -> None:
    schema: Final = {
        "type": "object",
        "properties": {
            "invoices": {
                "type": field_type,
                "items": {"type": "object", "properties": {"amount": {"type": "number"}}},
            }
        },
    }

    assert _extract_body(config, schema)["entities"] == {"invoices": [{"amount": "amount"}]}


@pytest.mark.parametrize(
    "field",
    [
        {"anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}, {"type": "string"}]},
        {"oneOf": [{"type": "string"}]},
        {"anyOf": [False, {"type": "string"}]},
        {"allOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]},
    ],
)
def test_extract_rejects_unsupported_composition(config: ScaleDownChatConfig, field: dict[str, JsonValue]) -> None:
    with pytest.raises(ScaleDownError) as exc:
        _extract_body(config, {"type": "object", "properties": {"field": field}})

    assert exc.value.status_code == 400


@respx.mock
@pytest.mark.parametrize("question_type", [[], {}])
def test_malformed_question_types_are_request_errors(question_type: list[JsonValue] | dict[str, JsonValue]) -> None:
    with pytest.raises(litellm.BadRequestError, match="Question 'q'"):
        litellm.completion(
            model="scaledown/decisions",
            messages=[{"role": "user", "content": "text"}],
            questions={"q": {"type": question_type}},
        )

    assert not respx.calls


@respx.mock
def test_unknown_scaledown_model_is_a_request_error() -> None:
    with pytest.raises(litellm.BadRequestError, match="Unknown ScaleDown model"):
        litellm.completion(model="scaledown/unknown", messages=[{"role": "user", "content": "text"}])

    assert not respx.calls


def _extract_body(config: ScaleDownChatConfig, schema: dict) -> dict:
    return config.transform_request(
        model="scaledown/extract",
        messages=[{"role": "user", "content": "text"}],
        optional_params={"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": schema}}},
        litellm_params={},
        headers={},
    )


def test_recursive_schema_is_cut_at_the_cycle_instead_of_expanded(config):
    children = {name: {"$ref": "#/$defs/Node"} for name in ("a", "b", "c", "d")}
    schema = {
        "type": "object",
        "$defs": {"Node": {"type": "object", "properties": children}},
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }

    entities = _extract_body(config, schema)["entities"]

    assert entities == {"root": {"a": "a", "b": "b", "c": "c", "d": "d"}}


def test_heavily_repeated_acyclic_refs_are_rejected_by_the_entity_budget(config):
    defs = {
        f"L{level}": {"type": "object", "properties": {name: {"$ref": f"#/$defs/L{level + 1}"} for name in "abcd"}}
        for level in range(10)
    }
    defs["L10"] = {"type": "string", "description": "leaf"}
    schema = {"type": "object", "$defs": defs, "properties": {"root": {"$ref": "#/$defs/L0"}}}

    with pytest.raises(ScaleDownError, match="expands to more than"):
        _extract_body(config, schema)


def test_summarize_response_is_the_native_payload(config):
    payload = {"summary": "terse summary", "input_chars": 400, "output_chars": 13, "input_tokens": 78}

    response = _transform_response(config, "scaledown/summarize", payload)

    assert json.loads(response.choices[0].message.content) == payload
    assert response.usage.prompt_tokens == 78


def test_compress_response_reports_top_level_input_tokens(config):
    payload = {
        "results": {"success": True, "compressed_prompt": "short", "original_prompt_tokens": 52},
        "input_tokens": 52,
    }

    response = _transform_response(config, "scaledown/compress", payload)

    assert json.loads(response.choices[0].message.content) == payload
    assert response.usage.prompt_tokens == 52


def test_compress_response_falls_back_to_original_prompt_tokens(config):
    payload = {"results": {"compressed_prompt": "short", "original_prompt_tokens": 900}}

    assert _transform_response(config, "scaledown/compress", payload).usage.prompt_tokens == 900


def test_extract_request_maps_schema_to_entities_including_nesting(config):
    schema = {
        "type": "object",
        "properties": {
            "vendor": {"type": "string", "description": "company that issued the invoice"},
            "notes": {"type": "string"},
            "address": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}}},
            "line_items": {
                "type": "array",
                "items": {"type": "object", "properties": {"sku": {"type": "string", "description": "item code"}}},
            },
        },
    }

    body = config.transform_request(
        model="scaledown/extract",
        messages=[{"role": "user", "content": "Invoice from Northwind."}],
        optional_params={"response_format": {"type": "json_schema", "json_schema": {"name": "i", "schema": schema}}},
        litellm_params={},
        headers={},
    )

    assert body == {
        "text": "Invoice from Northwind.",
        "entities": {
            "vendor": "company that issued the invoice",
            "notes": "notes",
            "address": {"city": "city name"},
            "line_items": [{"sku": "item code"}],
        },
    }


def test_extract_without_a_schema_is_rejected(config):
    with pytest.raises(ScaleDownError, match="response_format"):
        config.transform_request(
            model="scaledown/extract",
            messages=[{"role": "user", "content": "text"}],
            optional_params={},
            litellm_params={},
            headers={},
        )


@pytest.mark.parametrize("role", ["system", "developer"])
def test_summarize_request_splits_instructions_text_and_max_tokens(
    config: ScaleDownChatConfig, role: Literal["system", "developer"]
) -> None:
    body: Final = config.transform_request(
        model="scaledown/summarize",
        messages=[
            {"role": role, "content": "Be terse."},
            {"role": "user", "content": "a long document"},
        ],
        optional_params={"max_tokens": 40},
        litellm_params={},
        headers={},
    )

    assert body == {"text": "a long document", "instructions": "Be terse.", "max_tokens": 40}


def test_compress_request_puts_earlier_messages_in_context_and_defaults_rate_to_auto(config):
    body = config.transform_request(
        model="scaledown/compress",
        messages=[
            {"role": "system", "content": "background"},
            {"role": "user", "content": "the question"},
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert body == {"context": "background", "prompt": "the question", "scaledown": {"rate": "auto"}}


def test_decisions_response_ignores_the_upstream_cost_field(config):
    payload = {
        "model": "classify-1",
        "answers": {
            "category": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.93, "technical": 0.07},
                "confidence": 0.93,
            }
        },
        "usage": {"input_tokens": 62, "output_tokens": 1, "cost": 0.00543},
    }

    response = _transform_response(config, "scaledown/decisions", payload)

    assert json.loads(response.choices[0].message.content) == payload["answers"]
    assert response.usage.prompt_tokens == 62
    assert response.usage.completion_tokens == 1
    assert "response_cost" not in response._hidden_params
    assert response._hidden_params["scaledown_response"] == payload


def test_score_answer_survives_the_envelope_intact(config):
    answer = {
        "type": "score",
        "score": 1.43,
        "confidence": 0.57,
        "legend": {"0": "Cosmetic", "1": "Workaround", "2": "Blocking"},
        "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
    }
    payload = {
        "model": "classify-1",
        "answers": {"bug_severity": answer},
        "usage": {"input_tokens": 68, "output_tokens": 1, "cost": 0.0000029},
    }

    response = _transform_response(config, "scaledown/decisions", payload)

    assert json.loads(response.choices[0].message.content)["bug_severity"] == answer


def test_decisions_response_without_answers_is_an_error(config):
    with pytest.raises(ScaleDownError, match="no answers"):
        _transform_response(config, "scaledown/decisions", {"model": "classify-1"})


def test_empty_native_response_is_an_error(config):
    with pytest.raises(ScaleDownError, match="empty or malformed"):
        _transform_response(config, "scaledown/extract", {})


@pytest.mark.parametrize("upstream_status", [200, 502])
def test_non_json_response_is_an_upstream_error(config: ScaleDownChatConfig, upstream_status: int) -> None:
    raw_response: Final = httpx.Response(
        status_code=upstream_status,
        text="<html>bad gateway</html>",
        request=httpx.Request("POST", f"{BASE}/v1/scaledown"),
    )

    with pytest.raises(ScaleDownError) as exc:
        config.transform_response(
            model="scaledown/decisions",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert exc.value.status_code == 502


def test_decisions_takes_no_openai_sampling_params(config):
    assert config.get_supported_openai_params("scaledown/decisions") == ["stream"]


def test_unsupported_param_is_rejected_unless_dropped(config):
    with pytest.raises(ScaleDownError, match="does not support"):
        config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="scaledown/decisions",
            drop_params=False,
        )

    assert (
        config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="scaledown/decisions",
            drop_params=True,
        )
        == {}
    )


@respx.mock
def test_completion_sends_x_api_key_to_the_native_summarize_endpoint():
    route = respx.post(f"{BASE}/summarization/abstractive").mock(
        return_value=httpx.Response(200, json={"summary": "s", "input_tokens": 120})
    )

    response = litellm.completion(
        model="scaledown/summarize",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "a long document"},
        ],
    )

    request = route.calls[0].request
    assert request.headers["x-api-key"] == "sk-scaledown-test"
    assert "authorization" not in request.headers
    assert json.loads(request.content) == {"text": "a long document", "instructions": "Be terse."}
    assert json.loads(response.choices[0].message.content)["summary"] == "s"


@respx.mock
def test_completion_routes_decisions_to_the_decisions_endpoint():
    route = respx.post(f"{BASE}/v1/scaledown").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "classify-1",
                "answers": {
                    "category": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.93, "technical": 0.07},
                        "confidence": 0.93,
                    }
                },
                "usage": {"input_tokens": 62, "output_tokens": 1, "cost": 0.00543},
            },
        )
    )

    response = litellm.completion(
        model="scaledown/decisions",
        messages=[{"role": "user", "content": "I was charged twice."}],
        questions={
            "category": {
                "type": "choice",
                "criteria": {"billing": "A charge or refund.", "technical": "A bug."},
            }
        },
    )

    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "classify-1"
    assert sent["state"] == {"text": "I was charged twice."}
    assert "messages" not in sent
    assert json.loads(response.choices[0].message.content)["category"]["choice"] == "billing"


@respx.mock
def test_completion_extract_converts_response_format_to_entities():
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "invoice",
            "schema": {
                "type": "object",
                "properties": {"vendor": {"type": "string", "description": "company name"}},
            },
        },
    }
    extract_result = {"entities": [], "structured_result": {"vendor": "Acme Corp"}, "input_tokens": 20}
    route = respx.post(f"{BASE}/extract").mock(return_value=httpx.Response(200, json=extract_result))

    response = litellm.completion(
        model="scaledown/extract",
        messages=[{"role": "user", "content": "Acme Corp invoiced $500."}],
        response_format=schema,
    )

    assert json.loads(route.calls[0].request.content) == {
        "text": "Acme Corp invoiced $500.",
        "entities": {"vendor": "company name"},
    }
    assert json.loads(response.choices[0].message.content) == {"vendor": "Acme Corp"}
    assert response._hidden_params["scaledown_response"] == extract_result


@respx.mock
def test_completion_cost_comes_from_input_tokens_not_upstream_usage_cost():
    respx.post(f"{BASE}/v1/scaledown").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "classify-1",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 1000, "output_tokens": 1, "cost": 99.0},
            },
        )
    )

    response = litellm.completion(
        model="scaledown/decisions",
        messages=[{"role": "user", "content": "text"}],
        questions={"q": {"type": "noul", "instructions": "Positive?"}},
    )

    cost = litellm.completion_cost(completion_response=response, model="scaledown/decisions")
    assert cost == pytest.approx(1000 * litellm.model_cost["scaledown/decisions"]["input_cost_per_token"])


@respx.mock
def test_completion_honors_scaledown_api_base(monkeypatch):
    monkeypatch.setenv("SCALEDOWN_API_BASE", "https://staging.scaledown.xyz/v1")
    route = respx.post("https://staging.scaledown.xyz/v1/scaledown").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "classify-1",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.0},
            },
        )
    )

    litellm.completion(
        model="scaledown/decisions",
        messages=[{"role": "user", "content": "text"}],
        questions={"q": {"type": "noul", "instructions": "Positive?"}},
    )

    assert route.called


@respx.mock
def test_upstream_payment_error_surfaces_the_detail():
    respx.post(f"{BASE}/v1/scaledown").mock(return_value=httpx.Response(402, json={"detail": "Insufficient credits"}))

    with pytest.raises(litellm.exceptions.BadRequestError) as exc:
        litellm.completion(
            model="scaledown/decisions",
            messages=[{"role": "user", "content": "text"}],
            questions={"q": {"type": "noul", "instructions": "Positive?"}},
        )

    assert "Insufficient credits" in str(exc.value)


@respx.mock
def test_malformed_decisions_request_never_reaches_the_network():
    route = respx.post(f"{BASE}/v1/scaledown")

    with pytest.raises(Exception, match="questions"):
        litellm.completion(
            model="scaledown/decisions",
            messages=[{"role": "user", "content": "text"}],
        )

    assert not route.called


@respx.mock
def test_questions_passed_through_extra_body_reach_scaledown():
    route = respx.post(f"{BASE}/v1/scaledown").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "classify-1",
                "answers": {"q": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )
    )
    questions = {"q": {"type": "noul", "instructions": "Positive?"}}

    litellm.completion(
        model="scaledown/decisions",
        messages=[{"role": "user", "content": "text"}],
        extra_body={"questions": questions},
    )

    sent = json.loads(route.calls[0].request.content)
    assert sent == {"model": DECISIONS_UPSTREAM_MODEL, "state": {"text": "text"}, "questions": questions}


@respx.mock
def test_streaming_request_is_served_as_a_single_chunk():
    respx.post(f"{BASE}/compress/raw/").mock(
        return_value=httpx.Response(200, json={"compressed_prompt": "c", "original_prompt_tokens": 9})
    )

    chunks = list(
        litellm.completion(
            model="scaledown/compress",
            messages=[{"role": "user", "content": "the question"}],
            stream=True,
        )
    )

    assert "compressed_prompt" in "".join(chunk.choices[0].delta.content or "" for chunk in chunks)


@pytest.mark.parametrize(
    "structured,expected",
    [
        (None, {"vendor": "Northwind"}),
        (
            {"invoice": {"amount": 500, "amount_span_anchor": "$500"}},
            {"vendor": "Northwind", "invoice": {"amount": 500}},
        ),
    ],
)
def test_extract_merges_first_scalar_match_with_nested_fields(config, structured, expected):
    payload = {
        "entities": [
            {"type": "vendor", "text": "Northwind", "confidence": 1.0},
            {"type": "vendor", "text": "Contoso", "confidence": 0.8},
            {"type": "unrequested", "text": "ignore"},
        ],
        "structured_result": structured,
        "input_tokens": 161,
    }
    response = _transform_response(
        config,
        "scaledown/extract",
        payload,
        {"entities": {"vendor": "company name", "missing": "absent field", "invoice": {"amount": "total"}}},
    )

    assert json.loads(response.choices[0].message.content) == expected
    assert response._hidden_params["scaledown_response"] == payload
    assert response.usage.prompt_tokens == 161


@respx.mock
def test_strict_extraction_is_rejected_before_network():
    route = respx.post(f"{BASE}/extract")
    with pytest.raises(litellm.BadRequestError, match="strict JSON schemas"):
        litellm.completion(
            model="scaledown/extract",
            messages=[{"role": "user", "content": "Invoice from Northwind. Total: $500."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice",
                    "strict": True,
                    "schema": {"type": "object", "properties": {"amount": {"type": "string"}}, "required": ["amount"]},
                },
            },
        )
    assert not route.called


@pytest.mark.parametrize("model", ["classify", "decisions", "extract", "summarize", "compress"])
@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}],
        [
            {"type": "text", "text": "invoice"},
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,aGVsbG8="}},
        ],
    ],
)
def test_multimodal_inputs_are_rejected_instead_of_dropped(config, model, content):
    with pytest.raises(ScaleDownError, match="text only"):
        config.transform_request(f"scaledown/{model}", [{"role": "user", "content": content}], {}, {}, {})


@pytest.mark.parametrize("params", [{"document": "QQ=="}, {"state": {"document": "QQ=="}}, {"text": "override"}])
def test_native_input_overrides_are_rejected(config, params):
    with pytest.raises(ScaleDownError, match="text through messages only"):
        config.transform_request("scaledown/decisions", [{"role": "user", "content": "text"}], params, {}, {})


@pytest.mark.parametrize("model", ["classify", "decisions", "extract"])
def test_unsupported_system_instructions_are_rejected(config, model):
    with pytest.raises(ScaleDownError, match="does not support system messages"):
        config.transform_request(
            f"scaledown/{model}",
            [{"role": "system", "content": "instructions"}, {"role": "user", "content": "text"}],
            {},
            {},
            {},
        )


@respx.mock
@pytest.mark.parametrize("nested", [False, True])
def test_classify_uses_native_labels_and_preserves_scores(nested):
    payload = {"top_label": "urgent", "scores": {"urgent": 0.9, "routine": 0.1}, "input_tokens": 521}
    route = respx.post(f"{BASE}/classify").mock(return_value=httpx.Response(200, json=payload))
    labels = [
        {"name": "urgent", "rubric": "Service is unavailable."},
        {"name": "routine", "rubric": "A cosmetic issue."},
    ]
    options = {"extra_body": {"labels": labels}} if nested else {"labels": labels}

    response = litellm.completion(
        model="scaledown/classify",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Server is down."}]}],
        **options,
    )

    assert json.loads(route.calls[0].request.content) == {"text": "Server is down.", "labels": labels}
    assert json.loads(response.choices[0].message.content) == payload
    assert response.usage.prompt_tokens == 521
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        521 * litellm.model_cost["scaledown/classify"]["input_cost_per_token"]
    )


@respx.mock
@pytest.mark.parametrize("labels", [None, [], ["urgent"], [{"name": "urgent"}]])
def test_classify_requires_labels_before_network(labels):
    route = respx.post(f"{BASE}/classify")
    with pytest.raises(litellm.BadRequestError, match=r"labels|rubric"):
        litellm.completion(
            model="scaledown/classify", messages=[{"role": "user", "content": "text"}], extra_body={"labels": labels}
        )
    assert not route.called


@pytest.mark.asyncio
@respx.mock
async def test_async_extraction_keeps_flat_fields_and_usage():
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    route = respx.post(f"{BASE}/extract").mock(
        return_value=httpx.Response(
            200,
            json={
                "entities": [{"type": "vendor", "text": "Northwind"}],
                "structured_result": None,
                "input_tokens": 161,
            },
        )
    )
    client = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport())
    try:
        response = await litellm.acompletion(
            model="scaledown/extract",
            messages=[{"role": "user", "content": "Invoice from Northwind."}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "invoice",
                    "strict": False,
                    "schema": {"properties": {"vendor": {"type": "string"}}},
                },
            },
            client=client,
        )
    finally:
        await client.close()
    assert json.loads(response.choices[0].message.content) == {"vendor": "Northwind"}
    assert response.usage.prompt_tokens == 161
    assert json.loads(route.calls[0].request.content) == {
        "text": "Invoice from Northwind.",
        "entities": {"vendor": "vendor"},
    }


@respx.mock
def test_router_stream_includes_usage_without_sending_stream_options_upstream():
    route = respx.post(f"{BASE}/summarization/abstractive").mock(
        return_value=httpx.Response(200, json={"summary": "Fixed billing.", "input_tokens": 90})
    )
    router = litellm.Router(
        model_list=[
            {"model_name": "summary", "litellm_params": {"model": "scaledown/summarize", "api_key": "test-key"}}
        ],
        num_retries=0,
    )
    chunks = list(
        router.completion(
            model="summary",
            messages=[{"role": "user", "content": "Billing was fixed on Monday."}],
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert (
        json.loads("".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices))["summary"]
        == "Fixed billing."
    )
    assert chunks[-1].usage.prompt_tokens == 90
    assert json.loads(route.calls[0].request.content) == {"text": "Billing was fixed on Monday."}


@respx.mock
def test_pydantic_response_format_does_not_silently_claim_strict_support():
    from pydantic import BaseModel, ConfigDict

    class Invoice(BaseModel):
        model_config = ConfigDict(frozen=True)
        vendor: str

    route = respx.post(f"{BASE}/extract")
    with pytest.raises(litellm.BadRequestError, match="strict JSON schemas"):
        litellm.completion(
            model="scaledown/extract",
            messages=[{"role": "user", "content": "Invoice from Northwind."}],
            response_format=Invoice,
        )
    assert not route.called
