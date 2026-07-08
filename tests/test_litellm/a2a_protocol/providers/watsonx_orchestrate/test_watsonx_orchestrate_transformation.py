import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.a2a_protocol.providers.watsonx_orchestrate import handler as wxo_handler
from litellm.a2a_protocol.providers.watsonx_orchestrate.handler import (
    WatsonxOrchestrateHandler,
)
from litellm.a2a_protocol.providers.watsonx_orchestrate.transformation import (
    WatsonxOrchestrateTransformation,
)


class _JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _ShortTtlTokenClient:
    def __init__(self):
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        return _JsonResponse({"access_token": f"token-{self.calls}", "expires_in": 30})


class _SSELines:
    def __init__(self, lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _InvalidJsonStreamResponse:
    headers = {"content-type": "application/json"}

    def raise_for_status(self):
        pass

    async def aread(self):
        return b"not-json"


class _InvalidJsonStreamClient:
    def __init__(self):
        self.post_urls = []

    async def post(self, url, **kwargs):
        self.post_urls.append(url)
        if "identity/token" in url:
            return _JsonResponse({"access_token": "token", "expires_in": 3600})
        if url.endswith("/runs/stream"):
            return _InvalidJsonStreamResponse()
        if url.endswith("/runs"):
            return _JsonResponse({"status": "completed", "results": "fallback text"})
        raise AssertionError(url)


class _JsonStreamResponse:
    headers = {"content-type": "application/json"}

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    async def aread(self):
        return json.dumps(self.payload).encode()


class _JsonStreamClient:
    def __init__(self, stream_payload):
        self.stream_payload = stream_payload
        self.post_urls = []

    async def post(self, url, **kwargs):
        self.post_urls.append(url)
        if "identity/token" in url:
            return _JsonResponse({"access_token": "token", "expires_in": 3600})
        if url.endswith("/runs/stream"):
            return _JsonStreamResponse(self.stream_payload)
        raise AssertionError(url)


class TestWatsonxOrchestrateTransformation:
    def test_get_api_base_url(self):
        url = WatsonxOrchestrateTransformation.get_api_base_url(
            "https://cpd.example.com/",
            "1769134113217795",
        )
        assert (
            url == "https://cpd.example.com/orchestrate/cpd/instances/1769134113217795"
        )

    def test_extract_text_from_a2a_params(self):
        params = {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "text", "text": "Hello"},
                    {"kind": "text", "text": "world"},
                ],
            }
        }
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
            == "Hello world"
        )

    def test_extract_text_from_a2a_params_ignores_non_text_parts_with_text(self):
        params = {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "data", "text": "metadata label", "data": {}},
                    {"kind": "file", "text": "file label", "file": {}},
                    {"kind": "text", "text": "Hello"},
                    {"text": "legacy"},
                    {"kind": "", "text": "empty-kind"},
                ],
            }
        }
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_params(params)
            == "Hello legacy empty-kind"
        )

    def test_build_wxo_run_body_with_thread(self):
        body = WatsonxOrchestrateTransformation.build_wxo_run_body(
            wxo_agent_id="agent-uuid",
            text="Hi",
            thread_id="thread-1",
        )
        assert body["agent_id"] == "agent-uuid"
        assert body["thread_id"] == "thread-1"
        assert body["message"]["content"][0]["response_type"] == "text"
        assert body["message"]["content"][0]["text"] == "Hi"

    @pytest.mark.parametrize(
        "result,expected",
        [
            (
                {
                    "last_message": {
                        "content": [{"type": "text", "text": "from last_message"}]
                    }
                },
                "from last_message",
            ),
            (
                {
                    "result": {
                        "data": {
                            "message": {"content": [{"text": "from nested result"}]}
                        }
                    }
                },
                "from nested result",
            ),
            ({"results": "raw string"}, "raw string"),
        ],
    )
    def test_extract_text_from_wxo_result(self, result, expected):
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_wxo_result(result)
            == expected
        )

    def test_build_a2a_message_response(self):
        out = WatsonxOrchestrateTransformation.build_a2a_message_response(
            "req-1", "answer"
        )
        assert out["jsonrpc"] == "2.0"
        assert out["id"] == "req-1"
        assert out["result"]["kind"] == "message"
        assert out["result"]["parts"][0]["text"] == "answer"

    def test_extract_text_from_a2a_message_response(self):
        envelope = WatsonxOrchestrateTransformation.build_a2a_message_response(
            "req-1", "answer"
        )
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_message_response(
                envelope
            )
            == "answer"
        )
        assert (
            WatsonxOrchestrateTransformation.extract_text_from_a2a_message_response(
                {"result": {}}
            )
            == ""
        )


def test_cp4d_token_ttl_from_absolute_expiration():
    wall = 1_750_000_000.0
    assert (
        WatsonxOrchestrateHandler._cp4d_token_ttl_seconds(1_750_003_600, wall) == 3600
    )
    assert WatsonxOrchestrateHandler._cp4d_token_ttl_seconds(1_749_999_000, wall) == 0


@pytest.mark.asyncio
async def test_accumulate_wxo_sse_text_ignores_non_dict_json_events():
    response = _SSELines(
        [
            "data: null",
            "data: true",
            'data: {"results": "streamed text"}',
        ]
    )
    assert await WatsonxOrchestrateHandler._accumulate_wxo_sse_text(response) == (
        "streamed text"
    )


@pytest.mark.asyncio
async def test_short_lived_tokens_are_not_served_from_cache():
    client = _ShortTtlTokenClient()
    token_1 = await WatsonxOrchestrateHandler._get_bearer_token(
        cp4d_host="https://cpd.example.com",
        auth_mode="ibm_cloud",
        api_key="short-ttl-cache-key",
        client=client,
    )
    token_2 = await WatsonxOrchestrateHandler._get_bearer_token(
        cp4d_host="https://cpd.example.com",
        auth_mode="ibm_cloud",
        api_key="short-ttl-cache-key",
        client=client,
    )
    assert token_1 == "token-1"
    assert token_2 == "token-2"
    assert client.calls == 2


class _CP4DAuthClient:
    def __init__(self, expiration):
        self.expiration = expiration
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _JsonResponse({"token": "cp4d-token", "expiration": self.expiration})


@pytest.mark.asyncio
async def test_cp4d_auth_posts_to_authorize_and_caches_token():
    client = _CP4DAuthClient(int(time.time()) + 3600)
    token_1 = await WatsonxOrchestrateHandler._get_bearer_token(
        cp4d_host="https://cpd.example.com/",
        auth_mode="cp4d",
        api_key="cp4d-e2e-cache-key",
        username="cp4d-user",
        client=client,
    )
    token_2 = await WatsonxOrchestrateHandler._get_bearer_token(
        cp4d_host="https://cpd.example.com/",
        auth_mode="cp4d",
        api_key="cp4d-e2e-cache-key",
        username="cp4d-user",
        client=client,
    )

    assert token_1 == "cp4d-token"
    assert token_2 == "cp4d-token"
    assert len(client.calls) == 1
    url, kwargs = client.calls[0]
    assert url == "https://cpd.example.com/icp4d-api/v1/authorize"
    assert kwargs["json"] == {"username": "cp4d-user", "api_key": "cp4d-e2e-cache-key"}


@pytest.mark.asyncio
async def test_cp4d_auth_requires_username():
    client = _CP4DAuthClient(int(time.time()) + 3600)
    with pytest.raises(ValueError, match="username"):
        await WatsonxOrchestrateHandler._get_bearer_token(
            cp4d_host="https://cpd.example.com",
            auth_mode="cp4d",
            api_key="cp4d-missing-username-key",
            username=None,
            client=client,
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_expired_token_cache_entries_are_evicted():
    stale_key = "wxo-stale-cache-entry"
    wxo_handler._token_cache[stale_key] = ("stale-token", time.monotonic() - 1)

    class _FreshTokenClient:
        async def post(self, *args, **kwargs):
            return _JsonResponse({"access_token": "fresh", "expires_in": 3600})

    await WatsonxOrchestrateHandler._get_bearer_token(
        cp4d_host="https://cpd.example.com",
        auth_mode="ibm_cloud",
        api_key="wxo-eviction-trigger-key",
        client=_FreshTokenClient(),
    )

    assert stale_key not in wxo_handler._token_cache


@pytest.mark.asyncio
async def test_poll_run_raises_asyncio_timeout_when_never_terminal():
    class _NeverTerminalClient:
        def __init__(self):
            self.get_calls = 0

        async def get(self, url, headers=None):
            self.get_calls += 1
            return _JsonResponse({"status": "running"})

    client = _NeverTerminalClient()
    with pytest.raises(asyncio.TimeoutError):
        await WatsonxOrchestrateHandler._poll_run(
            base_url="https://cpd.example.com/orchestrate/cpd/instances/i",
            run_id="run-1",
            auth_headers={},
            client=client,
            max_attempts=2,
            interval_s=0,
        )
    assert client.get_calls == 2


@pytest.mark.asyncio
async def test_handle_streaming_polls_non_sse_json_until_complete(monkeypatch):
    client = _JsonStreamClient({"status": "running", "run_id": "run-1"})
    poll_calls = []

    async def poll_run(base_url, run_id, auth_headers, client, **kwargs):
        poll_calls.append((base_url, run_id, auth_headers, client))
        return {"status": "completed", "results": "polled text"}

    monkeypatch.setattr(
        WatsonxOrchestrateHandler,
        "_http_client",
        lambda timeout=90.0: client,
    )
    monkeypatch.setattr(WatsonxOrchestrateHandler, "_poll_run", poll_run)

    params = {
        "message": {
            "parts": [
                {"kind": "text", "text": "Hello"},
            ],
        }
    }
    litellm_params = {
        "cp4d_host": "https://cpd.example.com",
        "instance_id": "instance-id",
        "wxo_agent_id": "agent-id",
        "api_key": "pending-json-stream-cache-key",
        "auth_mode": "ibm_cloud",
    }

    events = [
        event
        async for event in WatsonxOrchestrateHandler.handle_streaming(
            request_id="req-1",
            params=params,
            litellm_params=litellm_params,
            delay_ms=0,
        )
    ]
    artifact_text = "".join(
        event["result"]["artifact"]["parts"][0]["text"]
        for event in events
        if event["result"].get("kind") == "artifact-update"
    )

    assert len(poll_calls) == 1
    assert poll_calls[0][1] == "run-1"
    assert artifact_text == "polled text"


@pytest.mark.asyncio
async def test_handle_streaming_raises_for_non_sse_json_failure(monkeypatch):
    client = _JsonStreamClient({"status": "failed", "run_id": "run-1"})
    monkeypatch.setattr(
        WatsonxOrchestrateHandler,
        "_http_client",
        lambda timeout=90.0: client,
    )

    params = {
        "message": {
            "parts": [
                {"kind": "text", "text": "Hello"},
            ],
        }
    }
    litellm_params = {
        "cp4d_host": "https://cpd.example.com",
        "instance_id": "instance-id",
        "wxo_agent_id": "agent-id",
        "api_key": "failed-json-stream-cache-key",
        "auth_mode": "ibm_cloud",
    }

    with pytest.raises(RuntimeError, match="non-success status 'failed'"):
        async for _ in WatsonxOrchestrateHandler.handle_streaming(
            request_id="req-1",
            params=params,
            litellm_params=litellm_params,
            delay_ms=0,
        ):
            pass


@pytest.mark.asyncio
async def test_handle_streaming_does_not_fallback_on_invalid_json(monkeypatch):
    client = _InvalidJsonStreamClient()
    monkeypatch.setattr(
        WatsonxOrchestrateHandler,
        "_http_client",
        lambda timeout=90.0: client,
    )

    params = {
        "message": {
            "parts": [
                {"kind": "text", "text": "Hello"},
            ],
        }
    }
    litellm_params = {
        "cp4d_host": "https://cpd.example.com",
        "instance_id": "instance-id",
        "wxo_agent_id": "agent-id",
        "api_key": "invalid-json-stream-cache-key",
        "auth_mode": "ibm_cloud",
    }

    with pytest.raises(json.JSONDecodeError):
        async for _ in WatsonxOrchestrateHandler.handle_streaming(
            request_id="req-1",
            params=params,
            litellm_params=litellm_params,
        ):
            pass

    assert not any(url.endswith("/runs") for url in client.post_urls)


@pytest.mark.asyncio
async def test_handle_streaming_does_not_resubmit_run_on_poll_transport_error(
    monkeypatch,
):
    class _RunSubmissionClient:
        def __init__(self):
            self.post_urls = []

        async def post(self, url, **kwargs):
            self.post_urls.append(url)
            if "identity/token" in url:
                return _JsonResponse({"access_token": "token", "expires_in": 3600})
            if url.endswith("/runs/stream"):
                return _JsonStreamResponse({"status": "running", "run_id": "run-1"})
            if url.endswith("/runs"):
                return _JsonResponse({"status": "completed", "results": "duplicate"})
            raise AssertionError(url)

    client = _RunSubmissionClient()

    async def poll_run(base_url, run_id, auth_headers, client, **kwargs):
        raise httpx.ConnectError("connection reset during poll")

    monkeypatch.setattr(
        WatsonxOrchestrateHandler,
        "_http_client",
        lambda timeout=90.0: client,
    )
    monkeypatch.setattr(WatsonxOrchestrateHandler, "_poll_run", poll_run)

    params = {"message": {"parts": [{"kind": "text", "text": "Hello"}]}}
    litellm_params = {
        "cp4d_host": "https://cpd.example.com",
        "instance_id": "instance-id",
        "wxo_agent_id": "agent-id",
        "api_key": "poll-transport-error-cache-key",
        "auth_mode": "ibm_cloud",
    }

    with pytest.raises(httpx.TransportError):
        async for _ in WatsonxOrchestrateHandler.handle_streaming(
            request_id="req-1",
            params=params,
            litellm_params=litellm_params,
            delay_ms=0,
        ):
            pass

    assert not any(url.endswith("/runs") for url in client.post_urls)


@pytest.mark.asyncio
async def test_handle_streaming_falls_back_when_initial_post_fails(monkeypatch):
    class _StreamPostFailsClient:
        def __init__(self):
            self.post_urls = []

        async def post(self, url, **kwargs):
            self.post_urls.append(url)
            if "identity/token" in url:
                return _JsonResponse({"access_token": "token", "expires_in": 3600})
            if url.endswith("/runs/stream"):
                raise httpx.ConnectError("cannot reach stream endpoint")
            if url.endswith("/runs"):
                return _JsonResponse({"status": "completed", "results": "fallback"})
            raise AssertionError(url)

    client = _StreamPostFailsClient()
    monkeypatch.setattr(
        WatsonxOrchestrateHandler,
        "_http_client",
        lambda timeout=90.0: client,
    )

    params = {"message": {"parts": [{"kind": "text", "text": "Hello"}]}}
    litellm_params = {
        "cp4d_host": "https://cpd.example.com",
        "instance_id": "instance-id",
        "wxo_agent_id": "agent-id",
        "api_key": "stream-post-fails-cache-key",
        "auth_mode": "ibm_cloud",
    }

    events = [
        event
        async for event in WatsonxOrchestrateHandler.handle_streaming(
            request_id="req-1",
            params=params,
            litellm_params=litellm_params,
            delay_ms=0,
        )
    ]
    artifact_text = "".join(
        event["result"]["artifact"]["parts"][0]["text"]
        for event in events
        if event["result"].get("kind") == "artifact-update"
    )

    assert artifact_text == "fallback"
    assert sum(url.endswith("/runs") for url in client.post_urls) == 1


def test_config_manager_returns_wxo_provider():
    config = A2AProviderConfigManager.get_provider_config(
        custom_llm_provider="watsonx_orchestrate"
    )
    assert config is not None
    assert config.__class__.__name__ == "WatsonxOrchestrateA2AConfig"


def test_wxo_dashboard_auth_fields():
    fields_path = (
        Path(__file__).resolve().parents[5]
        / "litellm/proxy/public_endpoints/agent_create_fields.json"
    )
    agent_fields = json.loads(fields_path.read_text())
    wxo_agent = next(
        agent for agent in agent_fields if agent["agent_type"] == "watsonx_orchestrate"
    )
    fields_by_key = {field["key"]: field for field in wxo_agent["credential_fields"]}

    assert fields_by_key["auth_mode"]["default_value"] == "cp4d"
    # Username is CP4D-only; UI does not require it so ibm_cloud users are not blocked.
    assert fields_by_key["username"]["required"] is False
    assert "cp4d" in fields_by_key["username"]["tooltip"].lower()
