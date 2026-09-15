import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.tinyfish_passthrough_logging_handler import (
    TinyFishPassthroughLoggingHandler,
    is_tinyfish_agent_url,
    resolve_tinyfish_cost_per_step,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.tinyfish import is_allowed_tinyfish_endpoint

RUN_URL = "https://agent.tinyfish.ai/v1/automation/run"
RUN_ASYNC_URL = "https://agent.tinyfish.ai/v1/automation/run-async"


def _make_logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "test-call-id"
    logging_obj.model_call_details = {}
    return logging_obj


def _make_response(method: str, url: str, body: dict) -> httpx.Response:
    request = httpx.Request(method, url)
    return httpx.Response(200, request=request, text=json.dumps(body))


class _FakeClient:
    """Payload items are dicts served with status_code, or (status, dict) tuples for scripted failures."""

    def __init__(self, payloads: list, status_code: int = 200):
        self.payloads = payloads
        self.status_code = status_code
        self.requested_urls: list[str] = []

    async def get(self, url: str, headers: dict) -> httpx.Response:
        self.requested_urls.append(url)
        item = self.payloads[min(len(self.requested_urls) - 1, len(self.payloads) - 1)]
        status, payload = item if isinstance(item, tuple) else (self.status_code, item)
        return httpx.Response(status, text=json.dumps(payload), request=httpx.Request("GET", url))


@pytest.fixture
def tinyfish_env(monkeypatch):
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-tf-test")
    monkeypatch.delenv("TINYFISH_COST_PER_STEP", raising=False)
    monkeypatch.delenv("TINYFISH_AGENT_API_BASE", raising=False)


class TestCostResolution:
    def test_default_rate(self, tinyfish_env):
        assert resolve_tinyfish_cost_per_step() == pytest.approx(0.016)

    def test_env_override(self, tinyfish_env, monkeypatch):
        monkeypatch.setenv("TINYFISH_COST_PER_STEP", "0.02")
        assert resolve_tinyfish_cost_per_step() == pytest.approx(0.02)

    def test_invalid_env_falls_back_to_default(self, tinyfish_env, monkeypatch):
        monkeypatch.setenv("TINYFISH_COST_PER_STEP", "free")
        assert resolve_tinyfish_cost_per_step() == pytest.approx(0.016)


class TestBillingGate:
    @pytest.mark.parametrize(
        "method,url,expected",
        [
            ("POST", RUN_URL, True),
            ("POST", RUN_ASYNC_URL, True),
            ("POST", "https://agent.tinyfish.ai/v1/automation/run-sse", True),
            ("GET", "https://agent.tinyfish.ai/v1/runs", False),
            ("GET", "https://agent.tinyfish.ai/v1/runs/run-123?screenshots=none", False),
            ("POST", "https://agent.tinyfish.ai/v1/runs/run-123/cancel", False),
        ],
    )
    def test_only_run_submissions_are_billed(self, method, url, expected):
        assert TinyFishPassthroughLoggingHandler.should_log_request(method, url) is expected

    def test_polling_writes_no_spend_row(self, tinyfish_env):
        logging_obj = _make_logging_obj()
        logging_obj.dispatch_success_handlers = AsyncMock()
        poll_url = "https://agent.tinyfish.ai/v1/runs/run-123"

        asyncio.run(
            PassThroughEndpointLogging().pass_through_async_success_handler(
                httpx_response=_make_response("GET", poll_url, {"run_id": "run-123", "status": "RUNNING"}),
                response_body={"run_id": "run-123", "status": "RUNNING"},
                logging_obj=logging_obj,
                url_route=poll_url,
                result="",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
                request_body={},
                passthrough_logging_payload={"url": poll_url},
                custom_llm_provider="tinyfish",
            )
        )

        logging_obj.dispatch_success_handlers.assert_not_awaited()


class TestBlockingRunBilling:
    def _handle(self, response_body: dict, logging_obj: MagicMock):
        return TinyFishPassthroughLoggingHandler.tinyfish_passthrough_handler(
            httpx_response=_make_response("POST", RUN_URL, response_body),
            response_body=response_body,
            logging_obj=logging_obj,
            url_route=RUN_URL,
            result=json.dumps(response_body),
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"url": "https://scrapeme.live/shop", "goal": "extract products"},
        )

    def test_bills_steps_times_rate(self, tinyfish_env):
        logging_obj = _make_logging_obj()
        run = {"run_id": "run-1", "status": "COMPLETED", "num_of_steps": 3, "result": {"products": []}}

        handler_result = self._handle(run, logging_obj)

        assert handler_result["kwargs"]["model"] == "tinyfish/automation-run"
        assert handler_result["kwargs"]["custom_llm_provider"] == "tinyfish"
        assert handler_result["kwargs"]["response_cost"] == pytest.approx(0.048)
        assert "standard_logging_object" in handler_result["kwargs"]
        assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.048)

    def test_env_rate_override_applies(self, tinyfish_env, monkeypatch):
        monkeypatch.setenv("TINYFISH_COST_PER_STEP", "0.5")
        run = {"run_id": "run-1", "status": "COMPLETED", "num_of_steps": 2}

        handler_result = self._handle(run, _make_logging_obj())

        assert handler_result["kwargs"]["response_cost"] == pytest.approx(1.0)

    def test_failed_run_still_bills_steps_taken(self, tinyfish_env):
        run = {"run_id": "run-1", "status": "FAILED", "num_of_steps": 2, "error": {"code": "AGENT_FAILURE"}}

        handler_result = self._handle(run, _make_logging_obj())

        assert handler_result["kwargs"]["response_cost"] == pytest.approx(0.032)

    def test_null_steps_logs_without_cost(self, tinyfish_env):
        run = {"run_id": "run-1", "status": "RUNNING", "num_of_steps": None}

        handler_result = self._handle(run, _make_logging_obj())

        assert handler_result["kwargs"]["response_cost"] is None


class TestRunAsyncBilling:
    def test_poll_and_log_bills_once_terminal(self, tinyfish_env):
        logging_obj = _make_logging_obj()
        logging_obj.dispatch_success_handlers = AsyncMock()
        fake_client = _FakeClient(
            payloads=[{"run_id": "run-9", "status": "COMPLETED", "num_of_steps": 4, "result": "ok"}]
        )

        asyncio.run(
            TinyFishPassthroughLoggingHandler._poll_and_log(
                run_id="run-9",
                logging_obj=logging_obj,
                result="",
                start_time=datetime.now(),
                cache_hit=False,
                kwargs={},
                client=fake_client,
            )
        )

        logging_obj.dispatch_success_handlers.assert_awaited_once()
        awaited_kwargs = logging_obj.dispatch_success_handlers.await_args.kwargs
        assert awaited_kwargs["response_cost"] == pytest.approx(0.064)
        assert awaited_kwargs["model"] == "tinyfish/automation-run"
        assert fake_client.requested_urls == ["https://agent.tinyfish.ai/v1/runs/run-9?screenshots=none"]
        assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.064)

    def test_transient_poll_failure_keeps_polling(self, tinyfish_env):
        fake_client = _FakeClient(
            payloads=[(500, {}), {"run_id": "run-9", "status": "COMPLETED", "num_of_steps": 3}]
        )

        run = asyncio.run(
            TinyFishPassthroughLoggingHandler._poll_until_terminal("run-9", fake_client, poll_interval_seconds=0.0)
        )

        assert run is not None
        assert run["num_of_steps"] == 3
        assert len(fake_client.requested_urls) == 2

    def test_gives_up_after_consecutive_poll_failures(self, tinyfish_env):
        fake_client = _FakeClient(payloads=[(500, {})])

        run = asyncio.run(
            TinyFishPassthroughLoggingHandler._poll_until_terminal("run-9", fake_client, poll_interval_seconds=0.0)
        )

        assert run is None
        assert len(fake_client.requested_urls) == 3

    def test_traversal_run_id_is_rejected(self, tinyfish_env):
        fake_client = _FakeClient(payloads=[{}])

        run = asyncio.run(TinyFishPassthroughLoggingHandler._fetch_run("../vault/items", fake_client))

        assert run is None
        assert fake_client.requested_urls == []

    def test_upstream_error_status_returns_none(self, tinyfish_env):
        fake_client = _FakeClient(payloads=[{"error": {"code": "NOT_FOUND"}}], status_code=404)

        run = asyncio.run(TinyFishPassthroughLoggingHandler._fetch_run("run-1", fake_client))

        assert run is None


class TestSseBilling:
    def test_collected_chunks_price_via_run_fetch(self, tinyfish_env):
        logging_obj = _make_logging_obj()
        chunks = [
            'data: {"type": "STARTED", "run_id": "run-7", "status": "RUNNING"}',
            'data: {"type": "PROGRESS", "run_id": "run-7"}',
            'data: {"type": "COMPLETE", "run_id": "run-7", "status": "COMPLETED", "result": "done"}',
        ]
        fake_client = _FakeClient(
            payloads=[{"run_id": "run-7", "status": "COMPLETED", "num_of_steps": 5, "result": "done"}]
        )

        payload = asyncio.run(
            TinyFishPassthroughLoggingHandler.handle_logging_tinyfish_collected_chunks(
                litellm_logging_obj=logging_obj,
                url_route="https://agent.tinyfish.ai/v1/automation/run-sse",
                start_time=datetime.now(),
                all_chunks=chunks,
                end_time=datetime.now(),
                client=fake_client,
            )
        )

        assert payload["kwargs"]["response_cost"] == pytest.approx(0.08)
        assert payload["kwargs"]["model"] == "tinyfish/automation-run"
        assert fake_client.requested_urls == ["https://agent.tinyfish.ai/v1/runs/run-7?screenshots=none"]

    def test_stream_without_run_id_logs_without_cost(self, tinyfish_env):
        fake_client = _FakeClient(payloads=[{}])

        payload = asyncio.run(
            TinyFishPassthroughLoggingHandler.handle_logging_tinyfish_collected_chunks(
                litellm_logging_obj=_make_logging_obj(),
                url_route="https://agent.tinyfish.ai/v1/automation/run-sse",
                start_time=datetime.now(),
                all_chunks=["data: not-json", ": keepalive"],
                end_time=datetime.now(),
                client=fake_client,
            )
        )

        assert payload["kwargs"]["response_cost"] is None
        assert fake_client.requested_urls == []


class TestRouteDetection:
    def test_provider_tag_claims_route(self):
        assert PassThroughEndpointLogging().is_tinyfish_route("https://example.com/x", "tinyfish")

    def test_agent_host_claims_route(self):
        assert PassThroughEndpointLogging().is_tinyfish_route("https://agent.tinyfish.ai/v1/runs", None)

    def test_other_providers_do_not_claim(self):
        assert not PassThroughEndpointLogging().is_tinyfish_route("https://api.openai.com/v1", "openai")

    def test_env_base_override_claims_route(self, monkeypatch):
        monkeypatch.setenv("TINYFISH_AGENT_API_BASE", "https://agent.staging.tinyfish.ai")
        assert is_tinyfish_agent_url("https://agent.staging.tinyfish.ai/v1/runs/x")
        assert not is_tinyfish_agent_url("https://agent.tinyfish.ai/v1/runs/x")


class TestEndpointAllowlist:
    @pytest.mark.parametrize(
        "method,path,expected",
        [
            ("POST", "/v1/automation/run", True),
            ("POST", "/v1/automation/run-async", True),
            ("POST", "/v1/automation/run-sse", True),
            ("GET", "/v1/runs", True),
            ("GET", "/v1/runs/run-abc-123", True),
            ("POST", "/v1/runs/run-abc-123/cancel", True),
            ("GET", "/v1/vault/items", False),
            ("GET", "/v1/wallet", False),
            ("POST", "/v1/browser-profiles", False),
            ("DELETE", "/v1/runs/run-abc-123", False),
            ("GET", "/v1/automation/run", False),
            ("POST", "/v1/runs", False),
            ("GET", "/v1/runs/..", False),
            ("POST", "/v1/runs/../automation/run/cancel", False),
        ],
    )
    def test_allowlist(self, method, path, expected):
        assert is_allowed_tinyfish_endpoint(method, path) is expected
