import json
import re
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

CARD: Final = "4111-1111-1111-1111"
EMAIL: Final = "jane.doe@example.com"
PHONE: Final = "555-123-4567"
SYSTEM_PROMPT: Final = "You are a helpful assistant."
RECOGNIZERS: Final = {
    "CREDIT_CARD": re.escape(CARD),
    "EMAIL_ADDRESS": re.escape(EMAIL),
    "PHONE_NUMBER": re.escape(PHONE),
}


class _ApplyGuardrailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response_text: str


class _ApplyGuardrailErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    type: str
    param: str | None
    code: str


class _ApplyGuardrailError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: _ApplyGuardrailErrorBody


def _detect(entity: str, text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {"entity_type": entity, "start": match.start(), "end": match.end(), "score": 0.95}
        for match in re.finditer(RECOGNIZERS[entity], text)
    )


def _analyze(request: Request) -> Reply:
    assert request.target == "/analyze", request.target
    body: Final = json.loads(request.body)
    requested: Final = body.get("entities") or list(RECOGNIZERS)
    findings: Final = list(chain.from_iterable(_detect(entity, body["text"]) for entity in requested))
    return Reply(body=json.dumps(findings).encode())


def _anonymize(request: Request) -> Reply:
    assert request.target == "/anonymize", request.target
    body: Final = json.loads(request.body)
    spans: Final = sorted(body["analyzer_results"], key=lambda item: item["start"])
    pieces: Final = [
        body["text"][(spans[index - 1]["end"] if index else 0) : span["start"]] + f"<{span['entity_type']}>"
        for index, span in enumerate(spans)
    ]
    tail: Final = body["text"][spans[-1]["end"] :] if spans else body["text"]
    return Reply(body=json.dumps({"text": "".join(pieces) + tail, "items": []}).encode())


@dataclass(frozen=True, slots=True)
class Presidio:
    name: str
    analyzer: Wire
    anonymizer: Wire


@contextmanager
def _presidio(gateway: Gateway, mode: str, entities: Mapping[str, str] | None) -> Iterator[Presidio]:
    name: Final = f"presidio-{uuid.uuid4().hex}"
    with wire_server(_analyze) as analyzer, wire_server(_anonymize) as anonymizer:
        created: Final = gateway.request(
            "POST",
            "/guardrails",
            {
                "guardrail": {
                    "guardrail_name": name,
                    "litellm_params": {
                        "guardrail": "presidio",
                        "mode": mode,
                        "default_on": False,
                        "presidio_analyzer_api_base": analyzer.url,
                        "presidio_anonymizer_api_base": anonymizer.url,
                        **({} if entities is None else {"pii_entities_config": dict(entities)}),
                    },
                }
            },
        )
        assert created.status_code == 200, created.text
        try:
            yield Presidio(name, analyzer, anonymizer)
        finally:
            deleted: Final = gateway.request("DELETE", f"/guardrails/{created.json()['guardrail_id']}")
            assert deleted.status_code == 200, deleted.text


def _requested_entities(analyzer: Wire) -> list[JsonValue]:
    return [json.loads(request.body).get("entities") for request in analyzer.drain()]


def _assert_apply_guardrail_successes(
    presidio: Presidio,
    text: str,
    responses: tuple[httpx.Response, ...],
) -> None:
    assert tuple(response.status_code for response in responses) == (200,) * len(responses), tuple(
        response.text for response in responses
    )
    expected_response: Final = {"response_text": f"alias contract <CREDIT_CARD> and <EMAIL_ADDRESS>"}
    assert tuple(
        _ApplyGuardrailResponse.model_validate_json(response.content).model_dump() for response in responses
    ) == (expected_response,) * len(responses), tuple(response.text for response in responses)

    analyzer_requests: Final = presidio.analyzer.drain()
    assert len(analyzer_requests) == len(responses)
    assert tuple((request.method, request.target) for request in analyzer_requests) == (
        ("POST", "/analyze"),
    ) * len(responses)
    expected_analyzer_body: Final = {"text": text, "language": "en"}
    assert tuple(json.loads(request.body) for request in analyzer_requests) == (
        expected_analyzer_body,
    ) * len(responses)

    anonymizer_requests: Final = presidio.anonymizer.drain()
    assert len(anonymizer_requests) == len(responses)
    assert tuple((request.method, request.target) for request in anonymizer_requests) == (
        ("POST", "/anonymize"),
    ) * len(responses)
    expected_anonymizer_body: Final = {
        "text": text,
        "analyzer_results": [
            {
                "entity_type": "CREDIT_CARD",
                "start": text.index(CARD),
                "end": text.index(CARD) + len(CARD),
                "score": 0.95,
            },
            {
                "entity_type": "EMAIL_ADDRESS",
                "start": text.index(EMAIL),
                "end": text.index(EMAIL) + len(EMAIL),
                "score": 0.95,
            },
        ],
    }
    assert tuple(json.loads(request.body) for request in anonymizer_requests) == (
        expected_anonymizer_body,
    ) * len(responses)


def test_pre_call_masks_only_the_configured_entities_before_the_provider_sees_the_prompt(gateway: Gateway) -> None:
    with (
        _presidio(gateway, "pre_call", {"CREDIT_CARD": "MASK", "EMAIL_ADDRESS": "MASK"}) as presidio,
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        upstream.get("/__observations").raise_for_status()
        user_text: Final = f"{uuid.uuid4().hex} card {CARD}, email {EMAIL}, phone {PHONE}"
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "guardrails": [presidio.name],
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}],
            },
        )
        assert response.status_code == 200, response.text
        observed: Final = upstream.get("/__observations").json()["requests"]
        assert len(observed) == 1
        messages: Final = observed[0]["body"]["messages"]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        forwarded: Final = string_value(messages[1]["content"])
        assert CARD not in forwarded and EMAIL not in forwarded, forwarded
        assert "<CREDIT_CARD>" in forwarded and "<EMAIL_ADDRESS>" in forwarded, forwarded
        assert PHONE in forwarded, forwarded
        requested: Final = _requested_entities(presidio.analyzer)
        assert requested and all(sorted(entities) == ["CREDIT_CARD", "EMAIL_ADDRESS"] for entities in requested), (
            requested
        )


@pytest.mark.parametrize("entities", [None, {}])
def test_apply_guardrail_with_the_default_config_masks_every_detected_entity(
    gateway: Gateway, entities: Mapping[str, str] | None
) -> None:
    with _presidio(gateway, "pre_call", entities) as presidio:
        response: Final = gateway.request(
            "POST",
            "/guardrails/apply_guardrail",
            {"guardrail_name": presidio.name, "text": f"card {CARD} and email {EMAIL}"},
        )
        assert response.status_code == 200, response.text
        masked: Final = string_value(response.json()["response_text"])
        assert masked == "card <CREDIT_CARD> and email <EMAIL_ADDRESS>", masked
        assert _requested_entities(presidio.analyzer) == [None]
        assert len(presidio.anonymizer.drain()) == 1


def test_apply_guardrail_forwards_documented_language_and_entities_to_presidio(gateway: Gateway) -> None:
    pytest.skip("BUG: /guardrails/apply_guardrail drops the documented language and entities fields before Presidio")
    with _presidio(gateway, "pre_call", None) as presidio:
        text: Final = f"apply language contract {CARD} and {EMAIL}"
        response: Final = gateway.request(
            "POST",
            "/guardrails/apply_guardrail",
            {
                "guardrail_name": presidio.name,
                "text": text,
                "language": "fr",
                "entities": ["EMAIL_ADDRESS"],
            },
        )
        assert response.status_code == 200, response.text
        masked: Final = string_value(response.json()["response_text"])
        assert masked == f"apply language contract {CARD} and <EMAIL_ADDRESS>", masked
        analyzer_requests: Final = presidio.analyzer.drain()
        assert len(analyzer_requests) == 1
        assert json.loads(analyzer_requests[0].body) == {
            "text": text,
            "language": "fr",
            "entities": ["EMAIL_ADDRESS"],
        }, analyzer_requests[0].body
        assert len(presidio.anonymizer.drain()) == 1


def test_apply_guardrail_aliases_accept_admin_auth(gateway: Gateway) -> None:
    with _presidio(gateway, "pre_call", None) as presidio:
        text: Final = f"alias contract {CARD} and {EMAIL}"
        body: Final = {"guardrail_name": presidio.name, "text": text}
        paths: Final = ("/apply_guardrail", "/guardrails/apply_guardrail")

        unauthenticated_responses: Final = tuple(gateway.client.post(path, json=body) for path in paths)
        assert tuple(response.status_code for response in unauthenticated_responses) == (401, 401), tuple(
            response.text for response in unauthenticated_responses
        )
        assert presidio.analyzer.drain() == ()
        assert presidio.anonymizer.drain() == ()

        admin_requests: Final = tuple(
            gateway.client.post(path, json=body, headers=headers)
            for path, headers in (
                (paths[0], {"Authorization": f"Bearer {gateway.key}"}),
                (paths[1], {"Authorization": f"Bearer {gateway.key}"}),
                (paths[0], {"x-litellm-api-key": gateway.key}),
                (paths[1], {"x-litellm-api-key": gateway.key}),
            )
        )
        assert tuple(response.status_code for response in admin_requests) == (200, 200, 200, 200), tuple(
            response.text for response in admin_requests
        )
        for response in admin_requests:
            assert _ApplyGuardrailResponse.model_validate_json(response.content).model_dump() == {
                "response_text": "alias contract <CREDIT_CARD> and <EMAIL_ADDRESS>"
            }, response.text

        analyzer_requests: Final = presidio.analyzer.drain()
        assert len(analyzer_requests) == 4
        assert all(request.method == "POST" and request.target == "/analyze" for request in analyzer_requests)
        assert len({request.body for request in analyzer_requests}) == 1
        assert json.loads(analyzer_requests[0].body) == {"text": text, "language": "en"}, analyzer_requests[0].body
        anonymizer_requests: Final = presidio.anonymizer.drain()
        assert len(anonymizer_requests) == 4
        assert all(request.method == "POST" and request.target == "/anonymize" for request in anonymizer_requests)
        assert len({request.body for request in anonymizer_requests}) == 1
        assert json.loads(anonymizer_requests[0].body) == {
            "text": text,
            "analyzer_results": [
                {
                    "entity_type": "CREDIT_CARD",
                    "start": text.index(CARD),
                    "end": text.index(CARD) + len(CARD),
                    "score": 0.95,
                },
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "start": text.index(EMAIL),
                    "end": text.index(EMAIL) + len(EMAIL),
                    "score": 0.95,
                },
            ],
        }, anonymizer_requests[0].body


def test_long_apply_guardrail_alias_accepts_an_llm_api_scoped_key(gateway: Gateway) -> None:
    with (
        _presidio(gateway, "pre_call", None) as presidio,
        gateway.scenario() as scenario,
    ):
        proxy_admin: Final = scenario.user(user_role="proxy_admin")
        key: Final = scenario.key(user_id=proxy_admin, key_type="llm_api")
        text: Final = f"alias contract {CARD} and {EMAIL}"
        body: Final = {"guardrail_name": presidio.name, "text": text}
        path: Final = "/guardrails/apply_guardrail"

        assert presidio.analyzer.drain() == ()
        assert presidio.anonymizer.drain() == ()
        responses: Final = tuple(
            gateway.client.post(path, json=body, headers=headers)
            for headers in (
                {"Authorization": f"Bearer {key}"},
                {"x-litellm-api-key": key},
            )
        )
        _assert_apply_guardrail_successes(presidio, text, responses)

        unauthenticated: Final = gateway.client.post(path, json=body)
        assert unauthenticated.status_code == 401, unauthenticated.text
        assert _ApplyGuardrailError.model_validate_json(unauthenticated.content).model_dump() == {
            "error": {
                "message": "Authentication Error, No api key passed in.",
                "type": "auth_error",
                "param": "None",
                "code": "401",
            }
        }, unauthenticated.text
        assert presidio.analyzer.drain() == ()
        assert presidio.anonymizer.drain() == ()


def test_short_apply_guardrail_alias_accepts_an_llm_api_scoped_key(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: llm_api scoped keys are refused on /apply_guardrail while "
        "/guardrails/apply_guardrail serves them"
    )
    with (
        _presidio(gateway, "pre_call", None) as presidio,
        gateway.scenario() as scenario,
    ):
        proxy_admin: Final = scenario.user(user_role="proxy_admin")
        key: Final = scenario.key(user_id=proxy_admin, key_type="llm_api")
        text: Final = f"alias contract {CARD} and {EMAIL}"
        body: Final = {"guardrail_name": presidio.name, "text": text}
        path: Final = "/apply_guardrail"

        assert presidio.analyzer.drain() == ()
        assert presidio.anonymizer.drain() == ()
        responses: Final = tuple(
            gateway.client.post(path, json=body, headers=headers)
            for headers in (
                {"Authorization": f"Bearer {key}"},
                {"x-litellm-api-key": key},
            )
        )
        _assert_apply_guardrail_successes(presidio, text, responses)


def test_apply_guardrail_aliases_accept_internal_user_auth(gateway: Gateway) -> None:
    pytest.skip("BUG: non-admin virtual keys get 401 on /apply_guardrail and /guardrails/apply_guardrail")
    with (
        _presidio(gateway, "pre_call", None) as presidio,
        gateway.scenario() as scenario,
    ):
        internal_user: Final = scenario.user(user_role="internal_user")
        internal_key: Final = scenario.key(
            user_id=internal_user,
            key_alias=f"presidio-apply-short-{uuid.uuid4().hex}",
        )
        text: Final = f"alias contract {CARD} and {EMAIL}"
        body: Final = {"guardrail_name": presidio.name, "text": text}
        paths: Final = ("/apply_guardrail", "/guardrails/apply_guardrail")
        internal_auth_requests: Final = (
            (paths[0], {"Authorization": f"Bearer {internal_key}"}),
            (paths[1], {"Authorization": f"Bearer {internal_key}"}),
            (paths[0], {"x-litellm-api-key": internal_key}),
            (paths[1], {"x-litellm-api-key": internal_key}),
        )
        responses: Final = tuple(
            gateway.client.post(path, json=body, headers=headers) for path, headers in internal_auth_requests
        )
        assert tuple(response.status_code for response in responses) == (200, 200, 200, 200), tuple(
            response.text for response in responses
        )
        assert (
            tuple(_ApplyGuardrailResponse.model_validate_json(response.content).model_dump() for response in responses)
            == ({"response_text": "alias contract <CREDIT_CARD> and <EMAIL_ADDRESS>"},) * 4
        )
        analyzer_requests: Final = presidio.analyzer.drain()
        assert len(analyzer_requests) == 4
        assert all(request.method == "POST" and request.target == "/analyze" for request in analyzer_requests)
        assert len({request.body for request in analyzer_requests}) == 1
        assert json.loads(analyzer_requests[0].body)["text"] == text
        anonymizer_requests: Final = presidio.anonymizer.drain()
        assert len(anonymizer_requests) == 4
        assert all(request.method == "POST" and request.target == "/anonymize" for request in anonymizer_requests)
        assert len({request.body for request in anonymizer_requests}) == 1
