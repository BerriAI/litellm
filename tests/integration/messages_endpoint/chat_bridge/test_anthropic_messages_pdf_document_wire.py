import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import anthropic
import pytest
from integration._support.client import Gateway, Scenario, string_value
from integration._support.pdf_document import LETTER, Page, document, encoded, pdf_bytes
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-pdf-document-key"
_ANSWER: Final = "The codeword is PELICAN-42."
_TITLE: Final = "Quarterly report"
_URL: Final = "https://example.com/brief.pdf"
_SAP_TOKEN_TARGET: Final = "/oauth/token"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PARTS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _pdf(text: str) -> str:
    return encoded(pdf_bytes((Page(text=text), LETTER)))


def _chat_completion(identity: str, model: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": _ANSWER}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 914, "completion_tokens": 9, "total_tokens": 923},
            }
        ).encode()
    )


def _only_body(wire: Wire) -> dict[str, JsonValue]:
    received: Final = tuple(
        request
        for request in wire.drain()
        if request.method == "POST" and not request.target.startswith(_SAP_TOKEN_TARGET)
    )
    assert len(received) == 1, received
    return _JSON_OBJECT.validate_json(received[0].body)


def _user_parts(body: dict[str, JsonValue], key: str) -> list[dict[str, JsonValue]]:
    turns: Final = _PARTS.validate_python(body[key])
    assert len(turns) == 1, turns
    return _PARTS.validate_python(turns[0]["content" if key == "messages" else "parts"])


def test_anthropic_sdk_pdf_documents_reach_azure_as_file_parts_named_by_title(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    backend: Final = f"gpt-pdf-{marker[:8]}"
    identity: Final = f"chatcmpl-{marker}"
    titled: Final = _pdf(f"PELICAN-42 {marker}")
    untitled: Final = _pdf(f"ORCHID-77 {marker}")
    question: Final = f"What codeword does each document carry? {marker}"
    with wire_server(lambda _: _chat_completion(identity, backend)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"azure/{backend}", api_base=wire.url, api_key=_API_KEY, api_version="2024-10-21"
        )
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        message: Final = client.messages.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "title": _TITLE,
                            "source": {"type": "base64", "media_type": "application/pdf", "data": titled},
                        },
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": untitled},
                        },
                        {"type": "document", "source": {"type": "url", "url": _URL}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        assert message.id == identity
        assert [(block.type, getattr(block, "text", None)) for block in message.content] == [("text", _ANSWER)]
        assert _user_parts(_only_body(wire), "messages") == [
            {"type": "file", "file": {"filename": _TITLE, "file_data": f"data:application/pdf;base64,{titled}"}},
            {
                "type": "file",
                "file": {"filename": "document.pdf", "file_data": f"data:application/pdf;base64,{untitled}"},
            },
            {"type": "image_url", "image_url": {"url": _URL}},
            {"type": "text", "text": question},
        ]


@dataclass(frozen=True, slots=True)
class _Reader:
    deploy: Callable[[Scenario, Wire], str]
    reply: Callable[[Request], Reply]
    pdf_parts: Callable[[dict[str, JsonValue]], Sequence[JsonValue]]


def _gemini_reply(_: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": _ANSWER}], "role": "model"}, "finishReason": "STOP", "index": 0}
                ],
                "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 9, "totalTokenCount": 309},
                "modelVersion": "gemini-2.5-flash",
            }
        ).encode()
    )


def _bedrock_reply(_: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": _ANSWER}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 300, "outputTokens": 9, "totalTokens": 309},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _anthropic_reply(_: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "msg_pdf",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": _ANSWER}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 300, "output_tokens": 9},
            }
        ).encode()
    )


def _sap_reply(request: Request) -> Reply:
    if request.target.startswith(_SAP_TOKEN_TARGET):
        return Reply(body=json.dumps({"access_token": "scripted", "expires_in": 3600}).encode())
    match request.target:
        case "/v2/lm/deployments":
            deployment: Final = {
                "scenarioId": "orchestration",
                "configurationId": "orchestration-config",
                "deploymentUrl": f"http://{request.headers['host']}/sap-deployment",
                "createdAt": "2026-01-01T00:00:00Z",
            }
            return Reply(body=json.dumps({"resources": [deployment]}).encode())
        case "/v2/lm/configurations/orchestration-config":
            return Reply(body=json.dumps({"executableId": "orchestration"}).encode())
        case _:
            completion: Final = _JSON_OBJECT.validate_json(_chat_completion("chatcmpl-sap-pdf", "claude").body)
            return Reply(body=json.dumps({"final_result": completion}).encode())


def _snowflake(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model="snowflake/claude-sonnet-4-5", api_base=wire.url, api_key=_API_KEY)


def _sap(scenario: Scenario, wire: Wire) -> str:
    service_key: Final = {
        "clientid": "scripted",
        "clientsecret": _API_KEY,
        "url": wire.url,
        "serviceurls": {"AI_API_URL": wire.url},
    }
    return scenario.model(model="sap/anthropic--claude-4.5-sonnet", api_base=wire.url, api_key=json.dumps(service_key))


def _snowflake_pdf_parts(body: dict[str, JsonValue]) -> Sequence[JsonValue]:
    return tuple(part.get("source") for part in _user_parts(body, "messages") if part.get("type") == "document")


def _sap_pdf_parts(body: dict[str, JsonValue]) -> Sequence[JsonValue]:
    config: Final = _JSON_OBJECT.validate_python(body["config"])
    templating: Final = _JSON_OBJECT.validate_python(
        _JSON_OBJECT.validate_python(config["modules"])["prompt_templating"]
    )
    prompt: Final = _JSON_OBJECT.validate_python(templating["prompt"])
    turns: Final = _PARTS.validate_python(prompt["template"])
    assert len(turns) == 1, turns
    parts: Final = _PARTS.validate_python(turns[0]["content"])
    return tuple(
        _JSON_OBJECT.validate_python(part["image_url"]).get("url") for part in parts if part.get("type") == "image_url"
    )


def _gemini(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model="gemini/gemini-2.5-flash", api_base=wire.url, api_key=_API_KEY)


def _bedrock(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model="bedrock/converse/us.amazon.nova-2-lite-v1:0",
        api_key=None,
        aws_access_key_id="AKIDSYNTHETIC",
        aws_secret_access_key="synthetic-secret",
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=wire.url,
    )


def _gemini_pdf_parts(body: dict[str, JsonValue]) -> Sequence[JsonValue]:
    return tuple(part.get("inline_data") for part in _user_parts(body, "contents") if "inline_data" in part)


def _bedrock_pdf_parts(body: dict[str, JsonValue]) -> Sequence[JsonValue]:
    documents: Final = tuple(
        _JSON_OBJECT.validate_python(part["document"]) for part in _user_parts(body, "messages") if "document" in part
    )
    return tuple({"format": item.get("format"), "source": item.get("source")} for item in documents)


@pytest.mark.parametrize(
    ("reader", "expected"),
    [
        pytest.param(
            _Reader(_gemini, _gemini_reply, _gemini_pdf_parts),
            lambda data: [{"mime_type": "application/pdf", "data": data}],
            id="gemini",
        ),
        pytest.param(
            _Reader(_bedrock, _bedrock_reply, _bedrock_pdf_parts),
            lambda data: [{"format": "pdf", "source": {"bytes": data}}],
            id="bedrock-converse",
        ),
        pytest.param(
            _Reader(_snowflake, _anthropic_reply, _snowflake_pdf_parts),
            lambda data: [{"type": "base64", "media_type": "application/pdf", "data": data}],
            id="snowflake-claude",
        ),
        pytest.param(
            _Reader(_sap, _sap_reply, _sap_pdf_parts),
            lambda data: [f"data:application/pdf;base64,{data}"],
            id="sap-orchestration",
        ),
    ],
)
def test_messages_pdf_still_reaches_providers_that_read_pdfs_in_their_own_document_shape(
    gateway: Gateway, reader: _Reader, expected: Callable[[str], Sequence[JsonValue]]
) -> None:
    marker: Final = uuid.uuid4().hex
    data: Final = _pdf(f"PELICAN-42 {marker}")
    with wire_server(reader.reply) as wire, gateway.scenario() as scenario:
        model: Final = reader.deploy(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            document({"type": "base64", "media_type": "application/pdf", "data": data}, title=_TITLE),
                            {"type": "text", "text": f"What is the codeword? {marker}"},
                        ],
                    }
                ],
            },
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 200, response.text
        assert _JSON_OBJECT.validate_json(response.content)["content"] == [{"type": "text", "text": _ANSWER}]
        assert list(reader.pdf_parts(_only_body(wire))) == list(expected(data))


def _prompt_security(job: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        match (request.method, request.target.partition("?")[0]):
            case ("POST", "/api/sanitizeFile"):
                return Reply(body=json.dumps({"jobId": job}).encode())
            case ("GET", "/api/sanitizeFile"):
                return Reply(
                    body=json.dumps(
                        {
                            "status": "done",
                            "content": "",
                            "metadata": {"action": "block", "violations": ["Malicious File"]},
                        }
                    ).encode()
                )
            case _:
                return Reply(body=json.dumps({"result": {"prompt": {"action": "allow"}}}).encode())

    return respond


def _delete_guardrail(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", f"/guardrails/{identity}")
    assert response.status_code == 200, response.text


def test_prompt_security_still_sanitizes_a_messages_pdf_and_blocks_it_before_the_provider(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    pdf: Final = pdf_bytes((Page(text=f"PELICAN-42 {marker}"), LETTER))
    with (
        wire_server(lambda _: _chat_completion(f"chatcmpl-{marker}", "gpt-pdf")) as provider,
        wire_server(_prompt_security(f"job-{marker}")) as sink,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"azure/gpt-pdf-{marker[:8]}", api_base=provider.url, api_key=_API_KEY, api_version="2024-10-21"
        )
        name: Final = f"pdf-prompt-security-{marker}"
        created: Final = gateway.post(
            "/guardrails",
            {
                "guardrail": {
                    "guardrail_name": name,
                    "litellm_params": {
                        "guardrail": "prompt_security",
                        "mode": "pre_call",
                        "api_key": "synthetic-prompt-security-key",
                        "api_base": sink.url,
                    },
                }
            },
        )
        scenario.cleanups.callback(_delete_guardrail, gateway, string_value(created["guardrail_id"]))
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "guardrails": [name],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            document({"type": "base64", "media_type": "application/pdf", "data": encoded(pdf)}),
                            {"type": "text", "text": f"Summarize this PDF {marker}"},
                        ],
                    }
                ],
            },
            headers={"anthropic-version": "2023-06-01"},
        )
        assert response.status_code == 400, response.text
        assert "Malicious File" in response.text
        assert provider.drain() == ()
        uploads: Final = tuple(
            request for request in sink.drain() if (request.method, request.target) == ("POST", "/api/sanitizeFile")
        )
        assert len(uploads) == 1, uploads
        assert pdf in uploads[0].body
