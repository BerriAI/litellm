import base64
import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_GUARDRAIL_ID: Final = "synthetic-attachment-guard"
_APPLY_PATH: Final = f"/guardrail/{_GUARDRAIL_ID}/version/DRAFT/apply"
_CHECKS_PATH: Final = "/guardrail-checks/invoke"
_BEDROCK_MODEL: Final = "anthropic.claude-3-haiku-20240307-v1:0"
_SSN: Final = "123-45-6789"
_ENDPOINTS: Final = ("chat", "messages", "responses", "converse")


def _png(marker: bytes) -> bytes:
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + (64).to_bytes(4, "big") * 2 + b"\x08\x02" + marker


_THREAT_PNG: Final = _png(b"synthetic-threat")
_PLAIN_PNG: Final = _png(b"synthetic-plain")
_PDF: Final = b"%PDF-1.4 synthetic employee record"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


@dataclass(frozen=True, slots=True)
class _Rig:
    strict: Gateway
    lenient: Gateway
    checks: Gateway
    policy: Wire
    upstream: Wire
    outage: threading.Event


def _blocked(assessment: Mapping[str, JsonValue]) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": "blocked by synthetic policy"}],
                "assessments": [assessment],
                "usage": {"contentPolicyUnits": 1, "contentPolicyImageUnits": 1},
            }
        ).encode()
    )


def _guardrail(outage: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == _CHECKS_PATH:
            return Reply(body=json.dumps({"results": {}}).encode())
        assert request.target == _APPLY_PATH, request.target
        if outage.is_set():
            return Reply(status=500, body=b'{"message":"synthetic guardrail outage"}')
        content: Final = json.loads(request.body)["content"]
        images: Final = [base64.b64decode(item["image"]["source"]["bytes"]) for item in content if "image" in item]
        texts: Final = [item["text"]["text"] for item in content if "text" in item]
        if _THREAT_PNG in images:
            return _blocked(
                {"contentPolicy": {"filters": [{"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}]}}
            )
        if any(_SSN in text for text in texts):
            return _blocked(
                {
                    "sensitiveInformationPolicy": {
                        "piiEntities": [{"type": "US_SOCIAL_SECURITY_NUMBER", "match": _SSN, "action": "BLOCKED"}]
                    }
                }
            )
        return Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())

    return respond


def _upstream(request: Request) -> Reply:
    if request.target == "/v1/chat/completions":
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "served"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ).encode()
        )
    if request.target == "/v1/messages":
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "served"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ).encode()
        )
    if request.target == "/v1/responses":
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{uuid.uuid4().hex}",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_synthetic",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "served", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
                }
            ).encode()
        )
    assert request.target == f"/model/{_BEDROCK_MODEL}/converse", request.target
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "served"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _config(
    directory: Path, name: str, policy: Wire, upstream: Wire, skip_unscannable: bool, checks: bool = False
) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {}
    configuration["model_list"] = [
        {
            "model_name": "attach-chat",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": f"{upstream.url}/v1", "api_key": "sk-chat"},
        },
        {
            "model_name": "attach-messages",
            "litellm_params": {
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "api_base": upstream.url,
                "api_key": "sk-ant-synthetic",
            },
        },
        {
            "model_name": "attach-responses",
            "litellm_params": {"model": "openai/gpt-4.1-mini", "api_base": f"{upstream.url}/v1", "api_key": "sk-resp"},
        },
        {
            "model_name": "attach-converse",
            "litellm_params": {
                "model": f"bedrock/{_BEDROCK_MODEL}",
                "api_base": upstream.url,
                "aws_access_key_id": "AKIASYNTHETICRUNTIME",
                "aws_secret_access_key": "synthetic-runtime-secret",
                "aws_region_name": "us-east-1",
            },
        },
    ]
    target: Final[dict[str, JsonValue]] = (
        {"checks": {"contentFilter": {"categories": [{"category": "VIOLENCE"}]}}}
        if checks
        else {"guardrailIdentifier": _GUARDRAIL_ID, "guardrailVersion": "DRAFT"}
    )
    configuration["guardrails"] = [
        {
            "guardrail_name": f"bedrock-attachments-{name}",
            "litellm_params": {
                "guardrail": "bedrock",
                "mode": "pre_call",
                "default_on": True,
                "skip_unscannable_attachments": skip_unscannable,
                **target,
                "aws_region_name": "us-east-1",
                "aws_access_key_id": "AKIASYNTHETICGUARDRAIL",
                "aws_secret_access_key": "synthetic-guardrail-secret",
                "aws_bedrock_runtime_endpoint": policy.url,
            },
        }
    ]
    path: Final = directory / f"bedrock-attachments-{name}.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("bedrock-guardrail-attachments")
    outage: Final = threading.Event()
    with (
        wire_server(_guardrail(outage)) as policy,
        wire_server(_upstream) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared, directory, {}, config=_config(directory, "strict", policy, upstream, False), workers=2
        ) as strict,
        owned_proxy(
            shared, directory, {}, config=_config(directory, "lenient", policy, upstream, True), workers=2
        ) as lenient,
        owned_proxy(
            shared, directory, {}, config=_config(directory, "checks", policy, upstream, False, checks=True), workers=2
        ) as checks,
    ):
        yield _Rig(strict, lenient, checks, policy, upstream, outage)


@pytest.fixture(autouse=True)
def _drained(rig: _Rig) -> Iterator[None]:
    rig.policy.drain()
    rig.upstream.drain()
    rig.outage.clear()
    yield
    rig.outage.clear()


def _image_block(endpoint: str, encoded: str, mime: str = "image/png") -> dict[str, JsonValue]:
    if endpoint == "chat":
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    if endpoint == "messages":
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}}
    if endpoint == "responses":
        return {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}
    return {"image": {"format": mime.removeprefix("image/"), "source": {"bytes": encoded}}}


def _pdf_block(endpoint: str) -> dict[str, JsonValue]:
    if endpoint == "chat":
        return {"type": "file", "file": {"file_data": f"data:application/pdf;base64,{_b64(_PDF)}", "filename": "r.pdf"}}
    if endpoint == "messages":
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": _b64(_PDF)}}
    if endpoint == "responses":
        return {"type": "input_file", "file_data": f"data:application/pdf;base64,{_b64(_PDF)}", "filename": "r.pdf"}
    return {"document": {"format": "pdf", "name": "record", "source": {"bytes": _b64(_PDF)}}}


def _text_block(endpoint: str, text: str) -> dict[str, JsonValue]:
    if endpoint == "converse":
        return {"text": text}
    return {"type": "input_text" if endpoint == "responses" else "text", "text": text}


def _send(gateway: Gateway, endpoint: str, *blocks: JsonValue, stream: bool = False) -> httpx.Response:
    content: Final = list(blocks)
    if endpoint == "chat":
        body: dict[str, JsonValue] = {"model": "attach-chat", "messages": [{"role": "user", "content": content}]}
        return gateway.request("POST", "/v1/chat/completions", {**body, "stream": True} if stream else body)
    if endpoint == "messages":
        return gateway.request(
            "POST",
            "/v1/messages",
            {"model": "attach-messages", "max_tokens": 32, "messages": [{"role": "user", "content": content}]},
        )
    if endpoint == "responses":
        return gateway.request(
            "POST", "/v1/responses", {"model": "attach-responses", "input": [{"role": "user", "content": content}]}
        )
    return gateway.request(
        "POST", "/bedrock/model/attach-converse/converse", {"messages": [{"role": "user", "content": content}]}
    )


def _scans(policy: Wire) -> list[list[dict[str, JsonValue]]]:
    return [json.loads(request.body)["content"] for request in policy.drain()]


def _image_scans(scans: list[list[dict[str, JsonValue]]]) -> list[list[JsonValue]]:
    return [[item["image"] for item in content if "image" in item] for content in scans if content[0].get("image")]


def _text_scans(scans: list[list[dict[str, JsonValue]]]) -> list[str]:
    return [item["text"]["text"] for content in scans for item in content if "text" in item]


def _forwarded(upstream: Wire) -> list[dict[str, JsonValue]]:
    return [json.loads(request.body) for request in upstream.drain() if request.method == "POST"]


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_an_image_only_turn_is_scanned_and_a_threat_image_is_blocked(rig: _Rig, endpoint: str) -> None:
    response: Final = _send(rig.strict, endpoint, _image_block(endpoint, _b64(_THREAT_PNG)))

    assert response.status_code == 400, response.text
    assert "VIOLENCE" in response.text, response.text
    assert _image_scans(_scans(rig.policy)) == [[{"format": "png", "source": {"bytes": _b64(_THREAT_PNG)}}]]
    assert _forwarded(rig.upstream) == []


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_a_harmless_image_is_scanned_and_reaches_the_model(rig: _Rig, endpoint: str) -> None:
    response: Final = _send(rig.strict, endpoint, _image_block(endpoint, _b64(_PLAIN_PNG)))

    assert response.status_code == 200, response.text
    assert _image_scans(_scans(rig.policy)) == [[{"format": "png", "source": {"bytes": _b64(_PLAIN_PNG)}}]]
    assert len(_forwarded(rig.upstream)) == 1


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_a_pdf_is_refused_without_calling_bedrock_or_the_model(rig: _Rig, endpoint: str) -> None:
    question: Final = f"What is in the attachment {uuid.uuid4().hex}?"
    response: Final = _send(rig.strict, endpoint, _text_block(endpoint, question), _pdf_block(endpoint))

    assert response.status_code == 400, response.text
    assert "Bedrock guardrail cannot scan 1 attachment(s)" in response.text, response.text
    assert _scans(rig.policy) == []
    assert _forwarded(rig.upstream) == []


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_skip_unscannable_attachments_lets_a_pdf_through_and_still_scans_the_text(rig: _Rig, endpoint: str) -> None:
    question: Final = f"What is in the attachment {uuid.uuid4().hex}?"
    response: Final = _send(rig.lenient, endpoint, _text_block(endpoint, question), _pdf_block(endpoint))

    assert response.status_code == 200, response.text
    assert question in _text_scans(_scans(rig.policy))
    (forwarded,) = _forwarded(rig.upstream)
    assert _b64(_PDF) in json.dumps(forwarded)


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_skip_unscannable_attachments_still_blocks_a_threat_image(rig: _Rig, endpoint: str) -> None:
    response: Final = _send(rig.lenient, endpoint, _image_block(endpoint, _b64(_THREAT_PNG)))

    assert response.status_code == 400, response.text
    assert "VIOLENCE" in response.text, response.text
    assert _forwarded(rig.upstream) == []


def test_a_streaming_chat_request_with_a_pdf_is_refused_before_streaming(rig: _Rig) -> None:
    response: Final = _send(rig.strict, "chat", _text_block("chat", "summarize"), _pdf_block("chat"), stream=True)

    assert response.status_code == 400, response.text
    assert "Bedrock guardrail cannot scan 1 attachment(s)" in response.text, response.text
    assert _forwarded(rig.upstream) == []


def _anthropic_text_document(text: str) -> dict[str, JsonValue]:
    return {"type": "document", "title": "record", "source": {"type": "text", "media_type": "text/plain", "data": text}}


def _converse_text_document(source: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"document": {"format": "txt", "name": "record", "source": dict(source)}}


_TEXT_DOCUMENTS: Final = (
    pytest.param("messages", _anthropic_text_document, id="messages-text-source"),
    pytest.param("converse", lambda text: _converse_text_document({"text": text}), id="converse-text-source"),
    pytest.param(
        "converse",
        lambda text: _converse_text_document({"content": [{"text": "cover page"}, {"text": text}]}),
        id="converse-content-source",
    ),
)


@pytest.mark.parametrize("endpoint, document", _TEXT_DOCUMENTS)
def test_a_text_document_with_an_ssn_is_scanned_as_text_and_blocked(
    rig: _Rig, endpoint: str, document: Callable[[str], dict[str, JsonValue]]
) -> None:
    secret: Final = f"Employee SSN {_SSN} {uuid.uuid4().hex}"
    response: Final = _send(rig.strict, endpoint, _text_block(endpoint, "summarize"), document(secret))

    assert response.status_code == 400, response.text
    assert any(secret in text for text in _text_scans(_scans(rig.policy)))
    assert _forwarded(rig.upstream) == []


@pytest.mark.parametrize("endpoint, document", _TEXT_DOCUMENTS)
def test_a_harmless_text_document_is_scanned_as_text_and_reaches_the_model(
    rig: _Rig, endpoint: str, document: Callable[[str], dict[str, JsonValue]]
) -> None:
    note: Final = f"The meeting is at noon {uuid.uuid4().hex}"
    response: Final = _send(rig.strict, endpoint, _text_block(endpoint, "summarize"), document(note))

    assert response.status_code == 200, response.text
    assert any(note in text for text in _text_scans(_scans(rig.policy)))
    (forwarded,) = _forwarded(rig.upstream)
    assert note in json.dumps(forwarded)


def test_a_converse_txt_document_sent_as_bytes_stays_unscannable(rig: _Rig) -> None:
    document: Final = {"document": {"format": "txt", "name": "record", "source": {"bytes": _b64(b"notes")}}}
    response: Final = _send(rig.strict, "converse", _text_block("converse", "summarize"), document)

    assert response.status_code == 400, response.text
    assert "cannot scan 1 attachment(s) (document)" in response.text, response.text
    assert _forwarded(rig.upstream) == []


def test_a_converse_guard_content_threat_image_is_scanned_and_blocked(rig: _Rig) -> None:
    guarded: Final = {"guardContent": {"image": {"format": "png", "source": {"bytes": _b64(_THREAT_PNG)}}}}
    response: Final = _send(rig.strict, "converse", _text_block("converse", "describe"), guarded)

    assert response.status_code == 400, response.text
    assert "VIOLENCE" in response.text, response.text
    assert _forwarded(rig.upstream) == []


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_checks_mode_refuses_an_image_without_calling_bedrock(rig: _Rig, endpoint: str) -> None:
    response: Final = _send(rig.checks, endpoint, _image_block(endpoint, _b64(_PLAIN_PNG)))

    assert response.status_code == 400, response.text
    assert "cannot scan 1 attachment(s) (image)" in response.text, response.text
    assert list(rig.policy.drain()) == []
    assert _forwarded(rig.upstream) == []


def test_checks_mode_sends_a_converse_text_document_to_invoke_guardrail_checks(rig: _Rig) -> None:
    note: Final = f"The meeting is at noon {uuid.uuid4().hex}"
    response: Final = _send(rig.checks, "converse", _converse_text_document({"text": note}))

    assert response.status_code == 200, response.text
    calls: Final = list(rig.policy.drain())
    assert [call.target for call in calls] == [_CHECKS_PATH]
    assert note in calls[0].body.decode()
    (forwarded,) = _forwarded(rig.upstream)
    assert note in json.dumps(forwarded)


@pytest.mark.parametrize(
    "encoded",
    [
        pytest.param(_b64(_THREAT_PNG).rstrip("="), id="unpadded"),
        pytest.param(base64.urlsafe_b64encode(_THREAT_PNG).decode(), id="url-safe"),
    ],
)
def test_non_standard_base64_is_normalized_before_the_scan(rig: _Rig, encoded: str) -> None:
    response: Final = _send(rig.strict, "chat", _image_block("chat", encoded))

    assert response.status_code == 400, response.text
    assert _image_scans(_scans(rig.policy)) == [[{"format": "png", "source": {"bytes": _b64(_THREAT_PNG)}}]]


def test_the_image_format_sent_to_bedrock_comes_from_the_bytes(rig: _Rig) -> None:
    response: Final = _send(rig.strict, "chat", _image_block("chat", _b64(_PLAIN_PNG), mime="image/jpeg"))

    assert response.status_code == 200, response.text
    assert _image_scans(_scans(rig.policy)) == [[{"format": "png", "source": {"bytes": _b64(_PLAIN_PNG)}}]]


@pytest.mark.parametrize("endpoint", ("chat", "converse"))
def test_a_guardrail_outage_fails_closed_on_an_image_turn(rig: _Rig, endpoint: str) -> None:
    rig.outage.set()
    response: Final = _send(rig.strict, endpoint, _image_block(endpoint, _b64(_PLAIN_PNG)))

    assert response.status_code == 500, response.text
    assert "synthetic guardrail outage" in response.text, response.text
    assert len(_image_scans(_scans(rig.policy))) == 1
    assert _forwarded(rig.upstream) == []


def test_a_concurrent_mixed_burst_across_two_workers_keeps_each_verdict(rig: _Rig) -> None:
    cases: Final = tuple(
        (kind, uuid.uuid4().hex)
        for kind in ("threat", "plain", "pdf", "converse-note", "converse-ssn")
        for _ in range(5)
    )

    def send(case: tuple[str, str]) -> tuple[str, str, int]:
        kind, tag = case
        if kind == "threat":
            response = _send(rig.strict, "chat", _text_block("chat", tag), _image_block("chat", _b64(_THREAT_PNG)))
        elif kind == "plain":
            response = _send(
                rig.strict, "messages", _text_block("messages", tag), _image_block("messages", _b64(_PLAIN_PNG))
            )
        elif kind == "pdf":
            response = _send(rig.strict, "responses", _text_block("responses", tag), _pdf_block("responses"))
        elif kind == "converse-note":
            response = _send(rig.strict, "converse", _converse_text_document({"text": f"note {tag}"}))
        else:
            response = _send(rig.strict, "converse", _converse_text_document({"text": f"SSN {_SSN} {tag}"}))
        return kind, tag, response.status_code

    with ThreadPoolExecutor(max_workers=12) as pool:
        results: Final = tuple(pool.map(send, cases))

    expected: Final = {"threat": 400, "plain": 200, "pdf": 400, "converse-note": 200, "converse-ssn": 400}
    assert [(kind, status) for kind, _, status in results] == [(kind, expected[kind]) for kind, _, _ in results]
    served: Final = json.dumps(_forwarded(rig.upstream))
    assert sorted(tag for kind, tag, _ in results if kind in ("plain", "converse-note") and tag in served) == sorted(
        tag for kind, tag, _ in results if kind in ("plain", "converse-note")
    )
    assert not any(tag in served for kind, tag, _ in results if expected[kind] == 400)
