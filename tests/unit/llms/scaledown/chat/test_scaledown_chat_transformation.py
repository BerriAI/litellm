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

BASE = "https://api.scaledown.xyz/v1"


@pytest.fixture
def config() -> ScaleDownChatConfig:
    return ScaleDownChatConfig()


@pytest.fixture(autouse=True)
def scaledown_api_key(monkeypatch):
    monkeypatch.setenv("SCALEDOWN_API_KEY", "sk-scaledown-test")
    monkeypatch.delenv("SCALEDOWN_API_BASE", raising=False)


def _transform_response(config: ScaleDownChatConfig, model: str, payload: dict) -> ModelResponse:
    return config.transform_response(
        model=model,
        raw_response=httpx.Response(
            status_code=200,
            json=payload,
            request=httpx.Request("POST", f"{BASE}/chat/completions"),
        ),
        model_response=ModelResponse(),
        logging_obj=None,
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def _chat_envelope(model: str, content: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "created": 1704067200,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


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


@pytest.mark.parametrize("model", ["scaledown/extract", "scaledown/summarize", "scaledown/compress"])
def test_domain_models_target_chat_completions(config, model):
    url = config.get_complete_url(api_base=None, api_key="k", model=model, optional_params={}, litellm_params={})

    assert url == f"{BASE}/chat/completions"


@pytest.mark.parametrize("model", ["scaledown/classify", "scaledown/decisions"])
def test_decisions_models_target_the_decisions_endpoint(config, model):
    url = config.get_complete_url(api_base=None, api_key="k", model=model, optional_params={}, litellm_params={})

    assert url == f"{BASE}/scaledown"


@pytest.mark.parametrize(
    "api_base",
    ["https://staging.scaledown.xyz", "https://staging.scaledown.xyz/v1"],
)
def test_api_base_is_versioned_exactly_once(config, api_base):
    url = config.get_complete_url(
        api_base=api_base,
        api_key="k",
        model="scaledown/extract",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://staging.scaledown.xyz/v1/chat/completions"


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
        config.transform_request(
            model="scaledown/classify",
            messages=[{"role": "user", "content": "text"}],
            optional_params={},
            litellm_params={},
            headers={},
        )


def test_decisions_without_text_or_state_is_rejected(config):
    with pytest.raises(ScaleDownError, match="text to decide on"):
        config.transform_request(
            model="scaledown/classify",
            messages=[{"role": "system", "content": "no user message"}],
            optional_params={"questions": {"q": {"type": "noul"}}},
            litellm_params={},
            headers={},
        )


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


def test_domain_response_preserves_upstream_id_and_structured_content(config):
    domain_result = {"summary": "terse summary", "input_tokens": 123}
    payload = _chat_envelope("summarize", json.dumps(domain_result), 123, 0)

    response = _transform_response(config, "scaledown/summarize", payload)

    assert response.id == "chatcmpl-upstream"
    assert response.model == "scaledown/summarize"
    assert json.loads(response.choices[0].message.content) == domain_result
    assert response.usage.prompt_tokens == 123
    assert response.usage.total_tokens == 123


def test_decisions_response_exposes_answers_and_upstream_cost(config):
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
        "usage": {"input_tokens": 62, "output_tokens": 1, "cost": 0.0000026},
    }

    response = _transform_response(config, "scaledown/classify", payload)

    assert json.loads(response.choices[0].message.content) == payload["answers"]
    assert response.usage.prompt_tokens == 62
    assert response.usage.completion_tokens == 1
    assert response._hidden_params["response_cost"] == payload["usage"]["cost"]
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


def test_domain_response_without_choices_is_an_error(config):
    with pytest.raises(ScaleDownError, match="no choices"):
        _transform_response(config, "scaledown/extract", {"id": "x", "choices": []})


def test_non_json_response_reports_the_upstream_status(config):
    raw_response = httpx.Response(
        status_code=502,
        text="<html>bad gateway</html>",
        request=httpx.Request("POST", f"{BASE}/scaledown"),
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
    assert config.get_supported_openai_params("scaledown/classify") == []


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
def test_completion_sends_x_api_key_to_chat_completions():
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_envelope("summarize", json.dumps({"summary": "s"}), 120, 8))
    )

    litellm.completion(
        model="scaledown/summarize",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "a long document"},
        ],
    )

    request = route.calls[0].request
    assert request.headers["x-api-key"] == "sk-scaledown-test"
    assert "authorization" not in request.headers
    assert json.loads(request.content)["model"] == "summarize"


@respx.mock
def test_completion_routes_classify_to_the_decisions_endpoint():
    route = respx.post(f"{BASE}/scaledown").mock(
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
                "usage": {"input_tokens": 62, "output_tokens": 1, "cost": 0.0000026},
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
def test_completion_forwards_extract_response_format():
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
    extract_result = {"entities": [], "structured_result": {"vendor": "Acme Corp"}}
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_chat_envelope("extract", json.dumps(extract_result), 20, 0))
    )

    response = litellm.completion(
        model="scaledown/extract",
        messages=[{"role": "user", "content": "Acme Corp invoiced $500."}],
        response_format=schema,
    )

    assert json.loads(route.calls[0].request.content)["response_format"] == schema
    assert json.loads(response.choices[0].message.content) == extract_result


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
    respx.post(f"{BASE}/scaledown").mock(return_value=httpx.Response(402, json={"detail": "Insufficient credits"}))

    with pytest.raises(litellm.exceptions.BadRequestError) as exc:
        litellm.completion(
            model="scaledown/classify",
            messages=[{"role": "user", "content": "text"}],
            questions={"q": {"type": "noul", "instructions": "Positive?"}},
        )

    assert "Insufficient credits" in str(exc.value)


@respx.mock
def test_malformed_decisions_request_never_reaches_the_network():
    route = respx.post(f"{BASE}/scaledown")

    with pytest.raises(Exception, match="questions"):
        litellm.completion(
            model="scaledown/classify",
            messages=[{"role": "user", "content": "text"}],
        )

    assert not route.called
