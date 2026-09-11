import json
from pathlib import Path

import httpx
import pytest

from litellm.a2a_protocol.providers.base import A2A_PROVIDER_RESPONSE_COST_KEY
from litellm.a2a_protocol.providers.config_manager import A2AProviderConfigManager
from litellm.a2a_protocol.providers.tinyfish import handler as tinyfish_handler
from litellm.a2a_protocol.providers.tinyfish.handler import TinyfishAgentHandler
from litellm.a2a_protocol.providers.tinyfish.transformation import (
    TINYFISH_AGENT_DOCS_URL,
    TinyfishAgentTransformation,
)

_PARAMS = {
    "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Extract the page title"}],
        "metadata": {"url": "https://example.com"},
    }
}
_LITELLM_PARAMS = {"api_key": "sk-tf", "cost_per_step": 0.016}


class _JsonResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class _MaskedStatusError(httpx.HTTPStatusError):
    def __init__(self, body, status_code=402):
        request = httpx.Request("POST", "https://agent.tinyfish.ai/v1/automation/run-async")
        response = httpx.Response(status_code, request=request, text=body)
        super().__init__(body, request=request, response=response)
        self.message = body


class _SSEResponse:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    async def aiter_lines(self):
        for line in self.lines:
            yield line

    async def aclose(self):
        self.closed = True


class _FakeClient:
    """Serves /run-async, /run-sse, /v1/runs/{id}, and /cancel from canned payloads."""

    def __init__(self, run_payloads=(), sse_lines=None, submit_error=None, sse_error=None):
        self.run_payloads = list(run_payloads)
        self.sse_lines = sse_lines
        self.submit_error = submit_error
        self.sse_error = sse_error
        self.post_urls = []
        self.get_urls = []
        self.post_headers = []
        self.post_bodies = []

    async def post(self, url, json=None, headers=None, stream=False):
        self.post_urls.append(url)
        self.post_headers.append(headers or {})
        self.post_bodies.append(json)
        if url.endswith("/run-sse"):
            if self.sse_error is not None:
                raise self.sse_error
            return _SSEResponse(self.sse_lines or [])
        if url.endswith("/run-async"):
            if self.submit_error is not None:
                raise self.submit_error
            return _JsonResponse({"run_id": "run-123", "error": None})
        if url.endswith("/cancel"):
            return _JsonResponse({"run_id": "run-123", "status": "CANCELLED"})
        raise AssertionError(url)

    async def get(self, url, headers=None):
        self.get_urls.append(url)
        if not self.run_payloads:
            raise AssertionError(f"unexpected GET {url}")
        return _JsonResponse(self.run_payloads.pop(0))


def _install_client(monkeypatch, client):
    monkeypatch.setattr(TinyfishAgentHandler, "_http_client", staticmethod(lambda timeout: client))
    monkeypatch.setattr(tinyfish_handler, "_POLL_INTERVAL_S", 0.0)


class TestBuildRunBody:
    def test_goal_from_text_and_url_from_metadata(self):
        body = TinyfishAgentTransformation.build_run_body(_PARAMS, {}, allow_authenticated_runs=False)
        assert body == {"url": "https://example.com", "goal": "Extract the page title"}

    def test_metadata_fields_forwarded_and_override_defaults(self):
        params = {
            "message": {
                "parts": [{"kind": "text", "text": "go"}],
                "metadata": {"url": "https://a.dev", "browser_profile": "stealth", "output_schema": {"type": "object"}},
            }
        }
        body = TinyfishAgentTransformation.build_run_body(
            params, {"browser_profile": "lite", "proxy_config": {"enabled": True}}, allow_authenticated_runs=False
        )
        assert body["browser_profile"] == "stealth"
        assert body["proxy_config"] == {"enabled": True}
        assert body["output_schema"] == {"type": "object"}

    def test_single_json_text_part_is_the_body(self):
        text = json.dumps({"goal": "Buy the item", "url": "https://shop.dev", "agent_config": {"max_steps": 20}})
        params = {"message": {"parts": [{"kind": "text", "text": text}]}}
        body = TinyfishAgentTransformation.build_run_body(params, {}, allow_authenticated_runs=False)
        assert body == {"goal": "Buy the item", "url": "https://shop.dev", "agent_config": {"max_steps": 20}}

    def test_authenticated_fields_stripped_unless_allowed(self):
        params = {
            "message": {
                "parts": [{"kind": "text", "text": "go"}],
                "metadata": {"url": "https://a.dev", "use_vault": True, "profile_id": "prof_1", "use_profile": True},
            }
        }
        stripped = TinyfishAgentTransformation.build_run_body(params, {}, allow_authenticated_runs=False)
        assert "use_vault" not in stripped and "profile_id" not in stripped and "use_profile" not in stripped
        allowed = TinyfishAgentTransformation.build_run_body(params, {}, allow_authenticated_runs=True)
        assert allowed["use_vault"] is True and allowed["profile_id"] == "prof_1"

    def test_admin_default_authenticated_fields_survive_stripping(self):
        body = TinyfishAgentTransformation.build_run_body(
            _PARAMS, {"use_profile": True, "profile_id": "prof_admin"}, allow_authenticated_runs=False
        )
        assert body["use_profile"] is True and body["profile_id"] == "prof_admin"

    def test_missing_goal_raises(self):
        with pytest.raises(ValueError, match="no goal"):
            TinyfishAgentTransformation.build_run_body(
                {"message": {"parts": [], "metadata": {"url": "https://a.dev"}}}, {}, False
            )

    def test_missing_url_raises_with_docs_pointer(self):
        with pytest.raises(ValueError, match="metadata"):
            TinyfishAgentTransformation.build_run_body(
                {"message": {"parts": [{"kind": "text", "text": "go"}]}}, {}, False
            )


class TestResponseTransform:
    def test_dict_result_gets_data_and_text_parts(self):
        run = {"run_id": "r1", "status": "COMPLETED", "num_of_steps": 5, "result": {"title": "T"}}
        response = TinyfishAgentTransformation.build_a2a_message_response("req1", run)
        result = response["result"]
        assert result["kind"] == "message" and result["role"] == "agent"
        assert result["parts"][0] == {"kind": "data", "data": {"title": "T"}}
        assert json.loads(result["parts"][1]["text"]) == {"title": "T"}
        assert result["metadata"] == {
            "tinyfish_run_id": "r1",
            "tinyfish_status": "COMPLETED",
            "tinyfish_num_of_steps": 5,
        }

    def test_string_result_is_single_text_part(self):
        run = {"run_id": "r1", "status": "COMPLETED", "result": "plain answer"}
        response = TinyfishAgentTransformation.build_a2a_message_response("req1", run)
        assert response["result"]["parts"] == ({"kind": "text", "text": "plain answer"},)

    def test_run_failure_message_carries_error_details(self):
        run = {
            "run_id": "r1",
            "status": "FAILED",
            "error": {"message": "Site blocked", "category": "AGENT_FAILURE", "retry_after": 60},
        }
        message = TinyfishAgentTransformation.run_failure_message(run)
        assert "TinyFish Agent: Site blocked" in message
        assert "category=AGENT_FAILURE" in message and "retry_after=60" in message
        assert TINYFISH_AGENT_DOCS_URL in message

    def test_wrap_error_unwraps_tinyfish_envelope(self):
        wrapped = TinyfishAgentTransformation.wrap_error_message(
            json.dumps({"error": {"code": "INVALID_API_KEY", "message": "The provided API key is invalid"}})
        )
        assert wrapped == f"TinyFish Agent: The provided API key is invalid. See {TINYFISH_AGENT_DOCS_URL} for details."


@pytest.mark.asyncio
class TestHandleNonStreaming:
    async def test_submit_poll_complete_with_step_cost(self, monkeypatch):
        client = _FakeClient(
            run_payloads=[
                {"run_id": "run-123", "status": "RUNNING"},
                {"run_id": "run-123", "status": "COMPLETED", "num_of_steps": 7, "result": {"title": "T"}},
            ]
        )
        _install_client(monkeypatch, client)

        response = await TinyfishAgentHandler.handle_non_streaming("req1", _PARAMS, _LITELLM_PARAMS)

        assert client.post_urls == ["https://agent.tinyfish.ai/v1/automation/run-async"]
        assert client.post_headers[0]["X-API-Key"] == "sk-tf"
        assert client.post_bodies[0] == {"url": "https://example.com", "goal": "Extract the page title"}
        assert all(url.endswith("/v1/runs/run-123?screenshots=none") for url in client.get_urls)
        assert response[A2A_PROVIDER_RESPONSE_COST_KEY] == pytest.approx(7 * 0.016)
        assert response["result"]["metadata"]["tinyfish_num_of_steps"] == 7

    async def test_no_cost_key_without_cost_per_step(self, monkeypatch):
        client = _FakeClient(run_payloads=[{"run_id": "run-123", "status": "COMPLETED", "result": {}}])
        _install_client(monkeypatch, client)
        response = await TinyfishAgentHandler.handle_non_streaming("req1", _PARAMS, {"api_key": "sk-tf"})
        assert A2A_PROVIDER_RESPONSE_COST_KEY not in response

    async def test_failed_run_raises_attributed_error(self, monkeypatch):
        client = _FakeClient(
            run_payloads=[
                {"run_id": "run-123", "status": "FAILED", "error": {"message": "boom", "category": "SYSTEM_FAILURE"}}
            ]
        )
        _install_client(monkeypatch, client)
        with pytest.raises(RuntimeError, match="TinyFish Agent: boom"):
            await TinyfishAgentHandler.handle_non_streaming("req1", _PARAMS, _LITELLM_PARAMS)

    async def test_submit_http_error_is_attributed(self, monkeypatch):
        body = json.dumps({"error": {"code": "INSUFFICIENT_CREDITS", "message": "wallet balance is too low"}})
        client = _FakeClient(submit_error=_MaskedStatusError(body))
        _install_client(monkeypatch, client)
        with pytest.raises(RuntimeError, match="TinyFish Agent: wallet balance is too low"):
            await TinyfishAgentHandler.handle_non_streaming("req1", _PARAMS, _LITELLM_PARAMS)

    async def test_poll_timeout_cancels_run(self, monkeypatch):
        client = _FakeClient(run_payloads=[{"run_id": "run-123", "status": "RUNNING"}] * 50)
        _install_client(monkeypatch, client)
        params = {**_LITELLM_PARAMS, "polling_timeout_seconds": 0.0}
        with pytest.raises(RuntimeError, match="did not finish within"):
            await TinyfishAgentHandler.handle_non_streaming("req1", _PARAMS, params)
        assert client.post_urls[-1].endswith("/v1/runs/run-123/cancel")


@pytest.mark.asyncio
class TestHandleStreaming:
    async def test_sse_events_translate_to_a2a_stream(self, monkeypatch):
        lines = [
            'data: {"type":"STARTED","run_id":"run-123"}',
            "",
            'data: {"type":"HEARTBEAT"}',
            'data: {"type":"STREAMING_URL","run_id":"run-123","streaming_url":"https://live.example"}',
            'data: {"type":"PROGRESS","run_id":"run-123","purpose":"Clicking submit"}',
            'data: {"type":"COMPLETE","run_id":"run-123","status":"COMPLETED","result":{"title":"T"}}',
        ]
        client = _FakeClient(
            sse_lines=lines,
            run_payloads=[{"run_id": "run-123", "status": "COMPLETED", "num_of_steps": 4, "result": {"title": "T"}}],
        )
        _install_client(monkeypatch, client)

        events = [e async for e in TinyfishAgentHandler.handle_streaming("req1", _PARAMS, _LITELLM_PARAMS)]

        kinds = [e["result"]["kind"] for e in events]
        assert kinds == ["task", "status-update", "status-update", "artifact-update", "status-update"]
        assert events[0]["result"]["status"]["state"] == "submitted"
        assert "https://live.example" in events[1]["result"]["status"]["message"]["parts"][0]["text"]
        assert events[2]["result"]["status"]["message"]["parts"][0]["text"] == "Clicking submit"
        artifact = events[3]["result"]["artifact"]
        assert artifact["parts"][0] == {"kind": "data", "data": {"title": "T"}}
        assert events[4]["result"]["final"] is True
        assert events[4]["result"]["status"]["state"] == "completed"
        assert events[4][A2A_PROVIDER_RESPONSE_COST_KEY] == pytest.approx(4 * 0.016)

    async def test_failed_complete_yields_failed_final_event(self, monkeypatch):
        lines = [
            'data: {"type":"STARTED","run_id":"run-123"}',
            'data: {"type":"COMPLETE","run_id":"run-123","status":"FAILED","error":{"message":"blocked"}}',
        ]
        client = _FakeClient(sse_lines=lines, run_payloads=[{"run_id": "run-123", "status": "FAILED"}])
        _install_client(monkeypatch, client)
        events = [e async for e in TinyfishAgentHandler.handle_streaming("req1", _PARAMS, _LITELLM_PARAMS)]
        final = events[-1]["result"]
        assert final["final"] is True and final["status"]["state"] == "failed"
        assert "TinyFish Agent: blocked" in final["status"]["message"]["parts"][0]["text"]

    async def test_transport_error_falls_back_to_synthetic_stream(self, monkeypatch):
        client = _FakeClient(
            sse_error=httpx.ConnectError("boom"),
            run_payloads=[{"run_id": "run-123", "status": "COMPLETED", "num_of_steps": 2, "result": {"ok": 1}}],
        )
        _install_client(monkeypatch, client)
        events = [e async for e in TinyfishAgentHandler.handle_streaming("req1", _PARAMS, _LITELLM_PARAMS)]
        kinds = [e["result"]["kind"] for e in events]
        assert kinds == ["task", "artifact-update", "status-update"]
        assert events[-1][A2A_PROVIDER_RESPONSE_COST_KEY] == pytest.approx(2 * 0.016)
        assert any(url.endswith("/run-async") for url in client.post_urls)

    async def test_stream_ending_without_complete_yields_failed_advisory(self, monkeypatch):
        lines = ['data: {"type":"STARTED","run_id":"run-123"}']
        client = _FakeClient(sse_lines=lines)
        _install_client(monkeypatch, client)
        events = [e async for e in TinyfishAgentHandler.handle_streaming("req1", _PARAMS, {"api_key": "sk-tf"})]
        final = events[-1]["result"]
        assert final["status"]["state"] == "failed"
        assert "GET /v1/runs/run-123" in final["status"]["message"]["parts"][0]["text"]


def test_config_manager_returns_tinyfish_provider():
    config = A2AProviderConfigManager.get_provider_config(custom_llm_provider="tinyfish")
    assert config is not None
    assert config.__class__.__name__ == "TinyfishA2AConfig"


def test_tinyfish_dashboard_fields():
    fields_path = (
        Path(__file__).resolve().parents[5] / "litellm" / "proxy" / "public_endpoints" / "agent_create_fields.json"
    )
    entries = json.loads(fields_path.read_text())
    entry = next(e for e in entries if e["agent_type"] == "tinyfish")
    assert entry["litellm_params_template"] == {"custom_llm_provider": "tinyfish"}
    field_keys = {f["key"] for f in entry["credential_fields"]}
    assert {"api_key", "cost_per_step"} <= field_keys
    assert all(f["include_in_litellm_params"] for f in entry["credential_fields"])
