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
from pydantic import JsonValue

CARD: Final = "4111-1111-1111-1111"
EMAIL: Final = "jane.doe@example.com"
PHONE: Final = "555-123-4567"
SYSTEM_PROMPT: Final = "You are a helpful assistant."
RECOGNIZERS: Final = {
    "CREDIT_CARD": re.escape(CARD),
    "EMAIL_ADDRESS": re.escape(EMAIL),
    "PHONE_NUMBER": re.escape(PHONE),
}


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
