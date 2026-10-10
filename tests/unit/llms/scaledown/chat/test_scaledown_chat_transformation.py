import json

import httpx
import pytest
import respx

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
        model="scaledown/classify", messages=messages, optional_params=optional_params, litellm_params={}, headers={}
    )
    config.sign_request(
        headers={},
        optional_params=optional_params,
        request_data=body,
        api_base=f"{BASE}/v1/scaledown",
        model="scaledown/classify",
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
        ("scaledown/classify", "/v1/scaledown"),
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
        api_base=api_base, api_key="k", model="scaledown/classify", optional_params={}, litellm_params={}
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
        model="scaledown/classify",
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


def test_explicit_state_with_document_is_forwarded_verbatim(config):
    state = {"document": "BASE64", "document_mime_type": "image/jpeg"}

    body = config.transform_request(
        model="scaledown/classify",
        messages=[],
        optional_params={"state": state, "questions": {"q": {"type": "noul"}}},
        litellm_params={},
        headers={},
    )

    assert body["state"] == state


def test_document_fields_move_into_derived_state(config):
    body = config.transform_request(
        model="scaledown/classify",
        messages=[{"role": "user", "content": "invoice text"}],
        optional_params={
            "document": "BASE64",
            "document_mime_type": "application/pdf",
            "questions": {"q": {"type": "noul"}},
        },
        litellm_params={},
        headers={},
    )

    assert body["state"] == {
        "text": "invoice text",
        "document": "BASE64",
        "document_mime_type": "application/pdf",
    }


def test_decisions_without_questions_is_rejected(config):
    with pytest.raises(ScaleDownError, match="questions"):
        _build_decisions(config, [{"role": "user", "content": "text"}], {})


def test_decisions_without_text_or_document_is_rejected(config):
    with pytest.raises(ScaleDownError, match="text to decide on"):
        _build_decisions(
            config, [{"role": "system", "content": "no user message"}], {"questions": {"q": {"type": "noul"}}}
        )


def test_state_text_is_rejected_so_guardrails_always_see_the_text(config):
    with pytest.raises(ScaleDownError, match="may only carry"):
        _build_decisions(
            config,
            [{"role": "user", "content": "innocuous"}],
            {"state": {"text": "secret"}, "questions": {"q": {"type": "noul"}}},
        )


def test_image_message_becomes_a_document_in_state(config):
    body = _build_decisions(
        config,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this an invoice?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            }
        ],
        {"questions": {"q": {"type": "noul"}}},
    )

    assert body["state"] == {"text": "Is this an invoice?", "document": "aGVsbG8=", "document_mime_type": "image/png"}


def test_image_only_message_is_accepted(config):
    body = _build_decisions(
        config,
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}]}],
        {"questions": {"q": {"type": "noul"}}},
    )

    assert body["state"] == {"document": "aGVsbG8=", "document_mime_type": "image/png"}


def test_remote_image_url_is_rejected(config):
    with pytest.raises(ScaleDownError, match="base64 data URLs"):
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
        ("scaledown/classify", {"model": "other"}),
        ("scaledown/classify", {"text": "secret"}),
    ],
)
def test_extra_body_cannot_override_the_model_or_prompt_text(config, model, extra_body):
    with pytest.raises(ScaleDownError, match="may only set|is not accepted"):
        config.transform_extra_body(extra_body=extra_body, request={}, model=model, litellm_params={})


def test_compress_rate_via_extra_body_sets_the_scaledown_rate(config):
    merged = config.transform_extra_body(
        extra_body={"compression_rate": 0.5},
        request={"context": "", "prompt": "q", "scaledown": {"rate": "auto"}},
        model="scaledown/compress",
        litellm_params={},
    )

    assert merged == {"scaledown": {"rate": 0.5}}


def test_extra_body_decisions_cannot_override_the_upstream_model_or_state_text(config):
    request = {"model": DECISIONS_UPSTREAM_MODEL, "state": {"text": "from messages"}}

    merged = config.transform_extra_body(
        extra_body={"questions": {"q": {"type": "noul"}}, "state": {"document": "QQ=="}},
        request=request,
        model="scaledown/classify",
        litellm_params={},
    )

    assert "model" not in merged
    assert merged["state"] == {"text": "from messages", "document": "QQ=="}
    with pytest.raises(ScaleDownError, match="may only carry"):
        config.transform_extra_body(
            extra_body={"state": {"text": "smuggled"}}, request=request, model="scaledown/classify", litellm_params={}
        )


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
    config.validate_environment(api_key="own", api_base="https://elsewhere.example", **kwargs)
    config.validate_environment(api_base=f"{BASE}/v1", **kwargs)
    monkeypatch.setenv("SCALEDOWN_API_BASE", "https://staging.scaledown.xyz")
    config.validate_environment(api_base="https://staging.scaledown.xyz/", **kwargs)


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
            non_default_params={"max_retries": 2}, optional_params={}, model="scaledown/classify", drop_params=False
        )
        == {}
    )


def test_every_model_fakes_a_stream(config):
    assert config.should_fake_stream(model="scaledown/compress", stream=True)
    assert not config.should_fake_stream(model="scaledown/compress", stream=False)


def test_unknown_question_type_is_rejected(config):
    with pytest.raises(ScaleDownError, match="expected one of"):
        config.transform_request(
            model="scaledown/classify",
            messages=[{"role": "user", "content": "t"}],
            optional_params={"questions": {"q": {"type": "ranking"}}},
            litellm_params={},
            headers={},
        )


def test_choice_question_without_criteria_is_rejected(config):
    with pytest.raises(ScaleDownError, match="non-empty 'criteria' map"):
        config.transform_request(
            model="scaledown/classify",
            messages=[{"role": "user", "content": "t"}],
            optional_params={"questions": {"q": {"type": "choice", "criteria": {}}}},
            litellm_params={},
            headers={},
        )


@pytest.mark.parametrize("levels", [1, 11])
def test_score_criteria_outside_two_to_ten_is_rejected(config, levels):
    with pytest.raises(ScaleDownError, match="ordered list"):
        config.transform_request(
            model="scaledown/classify",
            messages=[{"role": "user", "content": "t"}],
            optional_params={"questions": {"q": {"type": "score", "criteria": ["l"] * levels}}},
            litellm_params={},
            headers={},
        )


@pytest.mark.parametrize("levels", [2, 10])
def test_score_criteria_within_two_to_ten_is_accepted(config, levels):
    criteria = [f"level {index}" for index in range(levels)]

    body = config.transform_request(
        model="scaledown/classify",
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


def test_extract_and_summarize_send_an_image_as_a_native_document(config):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:application/pdf;base64,aGVsbG8="}}],
        }
    ]
    schema = {
        "response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": {"properties": {"a": {}}}}}
    }

    extract = config.transform_request("scaledown/extract", messages, schema, {}, {})
    summarize = config.transform_request("scaledown/summarize", messages, {}, {}, {})

    for body in (extract, summarize):
        assert body["document"] == "aGVsbG8="
        assert body["document_mime_type"] == "application/pdf"
        assert "text" not in body


def test_more_than_one_image_is_rejected(config):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}

    with pytest.raises(ScaleDownError, match="one image or document"):
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


def test_extract_schema_follows_local_refs(config):
    schema = {
        "type": "object",
        "$defs": {
            "Address": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}}}
        },
        "properties": {"address": {"$ref": "#/$defs/Address"}, "name": {"type": "string"}},
    }

    body = config.transform_request(
        model="scaledown/extract",
        messages=[{"role": "user", "content": "text"}],
        optional_params={"response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": schema}}},
        litellm_params={},
        headers={},
    )

    assert body["entities"] == {"address": {"city": "city name"}, "name": "name"}


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


def test_summarize_request_splits_instructions_text_and_max_tokens(config):
    body = config.transform_request(
        model="scaledown/summarize",
        messages=[
            {"role": "system", "content": "Be terse."},
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

    response = _transform_response(config, "scaledown/classify", payload)

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
        _transform_response(config, "scaledown/classify", {"model": "classify-1"})


def test_empty_native_response_is_an_error(config):
    with pytest.raises(ScaleDownError, match="empty or malformed"):
        _transform_response(config, "scaledown/extract", {})


def test_non_json_response_reports_the_upstream_status(config):
    raw_response = httpx.Response(
        status_code=502,
        text="<html>bad gateway</html>",
        request=httpx.Request("POST", f"{BASE}/v1/scaledown"),
    )

    with pytest.raises(ScaleDownError) as exc:
        config.transform_response(
            model="scaledown/classify",
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
    assert config.get_supported_openai_params("scaledown/classify") == ["stream"]


def test_unsupported_param_is_rejected_unless_dropped(config):
    with pytest.raises(ScaleDownError, match="does not support"):
        config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="scaledown/classify",
            drop_params=False,
        )

    assert (
        config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="scaledown/classify",
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
def test_completion_routes_classify_to_the_decisions_endpoint():
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
        model="scaledown/classify",
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
        model="scaledown/classify",
        messages=[{"role": "user", "content": "text"}],
        questions={"q": {"type": "noul", "instructions": "Positive?"}},
    )

    cost = litellm.completion_cost(completion_response=response, model="scaledown/classify")
    assert cost == pytest.approx(1000 * litellm.model_cost["scaledown/classify"]["input_cost_per_token"])


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
        model="scaledown/classify",
        messages=[{"role": "user", "content": "text"}],
        questions={"q": {"type": "noul", "instructions": "Positive?"}},
    )

    assert route.called


@respx.mock
def test_upstream_payment_error_surfaces_the_detail():
    respx.post(f"{BASE}/v1/scaledown").mock(return_value=httpx.Response(402, json={"detail": "Insufficient credits"}))

    with pytest.raises(litellm.exceptions.BadRequestError) as exc:
        litellm.completion(
            model="scaledown/classify",
            messages=[{"role": "user", "content": "text"}],
            questions={"q": {"type": "noul", "instructions": "Positive?"}},
        )

    assert "Insufficient credits" in str(exc.value)


@respx.mock
def test_malformed_decisions_request_never_reaches_the_network():
    route = respx.post(f"{BASE}/v1/scaledown")

    with pytest.raises(Exception, match="questions"):
        litellm.completion(
            model="scaledown/classify",
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
        model="scaledown/classify",
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
