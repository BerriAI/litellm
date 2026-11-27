"""Run from the Grant repository root:

    python -m pytest integrations/litellm/test_grant_guardrail.py -q

No LiteLLM install is required: the module falls back to a tiny local shim while
still inheriting the real CustomGuardrail when installed upstream.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

try:  # package import when collected inside LiteLLM
    from .grant_guardrail import (
        CHECKPOINT,
        DEFAULT_GRANT_BASE_URL,
        STREAMING_REJECTION_MESSAGE,
        GrantApprovalRequired,
        GrantDenied,
        GrantGuardrail,
        _redacted_scope,
        resolve_grant_base_url,
    )
except ImportError:  # standalone invocation from the Grant repository
    from grant_guardrail import (
        CHECKPOINT,
        DEFAULT_GRANT_BASE_URL,
        STREAMING_REJECTION_MESSAGE,
        GrantApprovalRequired,
        GrantDenied,
        GrantGuardrail,
        _redacted_scope,
        resolve_grant_base_url,
    )

APPROVED = {
    "id": "req_approve1",
    "request_id": "req_approve1",
    "status": "approved",
    "decision_type": "continue_once",
    "signed_receipt": "eyJyZXF1ZXN0X2lkIjoicmVxX2FwcHJvdmUxIn0",
    "receipt_url": "/receipt/req_approve1",
}
PENDING = {
    "id": "req_pending1",
    "request_id": "req_pending1",
    "status": "pending",
    "approval_url": "https://grant-approve.vercel.app/a/req_pending1",
    "receipt_url": "/receipt/req_pending1",
    "expires_at": "2026-11-27T10:21:00Z",
}
TOOL_CALL = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "send_email",
                            "arguments": json.dumps(
                                {"to": "private@example.test", "api_key": "sk-secret-123", "body": "hello"}
                            ),
                        },
                    }
                ],
            }
        }
    ]
}
LEGACY_FUNCTION_CALL = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "function_call": {
                    "name": "send_email",
                    "arguments": json.dumps({"to": "private@example.test"}),
                },
            }
        }
    ]
}
ANTHROPIC_TOOL_USE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "I will send that now."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "send_email",
            "input": {"to": "private@example.test"},
        },
    ],
}
NO_TOOL_CALL = {"choices": [{"message": {"role": "assistant", "content": "Just a note."}}]}
STREAM_CHUNKS = [
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_stream",
                            "function": {"name": "send_email", "arguments": ""},
                        }
                    ]
                }
            }
        ]
    },
    {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {
                                "arguments": json.dumps({"to": "private@example.test"})
                            },
                        }
                    ]
                }
            }
        ]
    },
]
SCOPE_KEYS = {"tool_call_id", "argument_key_count", "arguments_hmac_sha256"}


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _chunks(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def test_base_url_ignores_vercel_project_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vercel = tmp_path / ".vercel"
    vercel.mkdir()
    (vercel / "project.json").write_text(
        json.dumps({"projectId": "attacker-controlled", "aliases": ["https://attacker.example"]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRANT_BASE_URL", raising=False)
    monkeypatch.delenv("GRANT_PUBLIC_BASE_URL", raising=False)

    assert resolve_grant_base_url() == DEFAULT_GRANT_BASE_URL
    monkeypatch.setenv("GRANT_PUBLIC_BASE_URL", "https://public.example/")
    assert resolve_grant_base_url() == "https://public.example"
    monkeypatch.setenv("GRANT_BASE_URL", "https://env.example/")
    assert resolve_grant_base_url() == "https://env.example"
    assert resolve_grant_base_url("https://custom.example/") == "https://custom.example"


def test_no_tool_calls_make_no_grant_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client)
            assert await guard.async_post_call_success_hook(data={}, response=NO_TOOL_CALL) is None

    asyncio.run(run())
    assert seen == []


def test_unprotected_tool_is_ignored() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json=APPROVED)

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, protected_tools={"issue_refund"})
            response = {"choices": [{"message": {"tool_calls": [{"function": {"name": "get_weather", "arguments": {}}}]}}]}
            assert await guard.async_post_call_success_hook(data={}, response=response) is None

    asyncio.run(run())
    assert seen == []


def test_approved_call_passes_through_once_without_raw_arguments() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(201, json=APPROVED)

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, approval_mode=CHECKPOINT)
            start = time.perf_counter()
            result = await guard.async_post_call_success_hook(data={}, response=TOOL_CALL)
            assert result is None
            assert guard.last_latency_ms is not None and guard.last_latency_ms < 150

    asyncio.run(run())
    assert [path for path, _ in seen] == ["/v1/approvals"]
    body = seen[0][1]
    scope = body["scope"]
    assert set(scope) == SCOPE_KEYS
    assert "argument_keys" not in scope
    assert scope["tool_call_id"] == "call_1"
    assert scope["argument_key_count"] == 3
    assert len(scope["arguments_hmac_sha256"]) == 64
    request_json = json.dumps(seen)
    assert "send_email" in request_json
    for leaked in ("sk-secret-123", "private@example.test", '"api_key"', '"body"', '"to"'):
        assert leaked not in request_json


def test_checkpoint_mode_raises_structured_approval_required() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(202, json=PENDING)

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, approval_mode=CHECKPOINT)
            with pytest.raises(GrantApprovalRequired) as exc:
                await guard.async_post_call_success_hook(data={}, response=TOOL_CALL)
            payload = exc.value.as_dict()
            assert payload["grant_approval_required"] is True
            assert payload["request_id"] == "req_pending1"
            assert payload["approval_url"] == PENDING["approval_url"]
            assert "Open the Grant checkpoint" in payload["message"]

    asyncio.run(run())
    assert seen == ["/v1/approvals"]


def test_blocking_mode_polls_until_approved() -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(202, json=PENDING)
        return httpx.Response(
            200,
            json={
                "request_id": "req_pending1",
                "status": "decided",
                "decision": "approve",
                "signed_receipt": "eyJyZXF1ZXN0X2lkIjoicmVxX3BlbmRpbmcxIn0",
            },
        )

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, approval_mode="blocking", poll_interval=0.01)
            assert await guard.async_post_call_success_hook(data={}, response=TOOL_CALL) is None

    asyncio.run(run())
    assert seen == [("POST", "/v1/approvals"), ("GET", "/v1/approvals/req_pending1")]


def test_blocking_mode_denies_on_reject() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json=PENDING)
        return httpx.Response(200, json={"request_id": "req_pending1", "status": "decided", "decision": "reject"})

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, approval_mode="blocking", poll_interval=0.01)
            with pytest.raises(GrantDenied, match="did not approve"):
                await guard.async_post_call_success_hook(data={}, response=TOOL_CALL)

    asyncio.run(run())


def test_wildcard_protects_every_tool_and_fail_closed_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "unavailable"})

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, protected_tools={"*"}, fail_open=False)
            with pytest.raises(GrantDenied, match="http_503"):
                await guard.async_post_call_success_hook(data={}, response=TOOL_CALL)
            permissive = GrantGuardrail(client=client, protected_tools={"*"}, fail_open=True)
            no_calls = {"choices": [{"message": {"tool_calls": [{"function": {"name": "anything", "arguments": {}}}]}}]}
            assert await permissive.async_post_call_success_hook(data={}, response=no_calls) is None

    asyncio.run(run())


def test_invalid_approval_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="approval_mode"):
        GrantGuardrail(approval_mode="maybe")


def test_stream_true_is_rejected_unless_explicitly_allowed() -> None:
    async def run() -> None:
        guard = GrantGuardrail(event_hook="post_call")
        assert guard.event_hook == ["pre_call", "post_call"]
        with pytest.raises(ValueError) as exc:
            await guard.async_pre_call_hook(data={"stream": True})
        assert str(exc.value) == STREAMING_REJECTION_MESSAGE
        assert await guard.async_pre_call_hook(data={"stream": False}) == {"stream": False}

        allowed = GrantGuardrail(allow_streaming=True)
        assert await allowed.async_pre_call_hook(data={"stream": True}) == {"stream": True}

    asyncio.run(run())


def test_streaming_iterator_also_rejects_unbuffered_stream() -> None:
    async def run() -> None:
        guard = GrantGuardrail()
        iterator = guard.async_post_call_streaming_iterator_hook(
            response=_chunks([{"choices": [{"delta": {"content": "partial"}}]}]),
            request_data={"stream": True},
        )
        with pytest.raises(ValueError) as exc:
            await iterator.__anext__()
        assert str(exc.value) == STREAMING_REJECTION_MESSAGE

    asyncio.run(run())


def test_allow_streaming_buffers_and_gates_before_yielding() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(202, json=PENDING)

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, allow_streaming=True)
            iterator = guard.async_post_call_streaming_iterator_hook(
                response=_chunks(STREAM_CHUNKS),
                request_data={"stream": True},
            )
            with pytest.raises(GrantApprovalRequired):
                await iterator.__anext__()

    asyncio.run(run())
    assert len(seen) == 1
    assert seen[0]["tool_name"] == "send_email"
    assert seen[0]["scope"]["tool_call_id"] == "call_stream"
    assert set(seen[0]["scope"]) == SCOPE_KEYS
    assert "private@example.test" not in json.dumps(seen)


def test_allow_streaming_yields_chunks_only_after_approval() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(201, json=APPROVED)

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client, allow_streaming=True)
            yielded = [
                chunk
                async for chunk in guard.async_post_call_streaming_iterator_hook(
                    response=_chunks(STREAM_CHUNKS),
                    request_data={"stream": True},
                )
            ]
            assert yielded == STREAM_CHUNKS

    asyncio.run(run())
    assert seen == ["/v1/approvals"]


@pytest.mark.parametrize(
    ("response", "call_id"),
    [
        (TOOL_CALL, "call_1"),
        (LEGACY_FUNCTION_CALL, ""),
        (ANTHROPIC_TOOL_USE, "toolu_1"),
    ],
)
def test_all_tool_call_formats_share_the_same_gate(response: dict[str, Any], call_id: str) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(201, json=APPROVED)

    async def run() -> None:
        async with _client(handler) as client:
            guard = GrantGuardrail(client=client)
            assert await guard.async_post_call_success_hook(data={}, response=response) is None

    asyncio.run(run())
    assert len(seen) == 1
    assert seen[0]["tool_name"] == "send_email"
    assert seen[0]["scope"]["tool_call_id"] == call_id
    assert "private@example.test" not in json.dumps(seen)


def test_hmac_scope_uses_local_secret_in_preference_to_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRANT_LOCAL_HMAC_SECRET", "local-secret")
    monkeypatch.setenv("GRANT_API_KEY", "api-secret")
    arguments = {"to": "private@example.test"}
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = hmac.new(b"local-secret", canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    scope = _redacted_scope(arguments, "call_1")
    assert set(scope) == SCOPE_KEYS
    assert scope["argument_key_count"] == 1
    assert scope["arguments_hmac_sha256"] == expected


def test_hmac_scope_falls_back_to_api_key_and_is_stable_across_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRANT_LOCAL_HMAC_SECRET", raising=False)
    monkeypatch.setenv("GRANT_API_KEY", "api-secret")
    arguments = {"b": 2, "a": 1}
    expected = hmac.new(b"api-secret", b'{"a":1,"b":2}', hashlib.sha256).hexdigest()

    first = _redacted_scope(arguments, "call_1")
    retry = _redacted_scope(json.dumps(arguments), "call_1")
    assert first == retry
    assert first["arguments_hmac_sha256"] == expected

    monkeypatch.setenv("GRANT_LOCAL_HMAC_SECRET", "rotated-secret")
    assert _redacted_scope(arguments, "call_1")["arguments_hmac_sha256"] != expected


def test_hmac_scope_uses_ephemeral_process_key_when_no_secret_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GRANT_LOCAL_HMAC_SECRET", raising=False)
    monkeypatch.delenv("GRANT_API_KEY", raising=False)
    arguments = {"to": "private@example.test"}
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    first = _redacted_scope(arguments, "call_1")
    assert first == _redacted_scope(arguments, "call_1")
    assert first["arguments_hmac_sha256"] != hashlib.sha256(canonical.encode("utf-8")).hexdigest()
