import asyncio
import datetime
from collections.abc import Iterator
from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.logging_only_input_scan import collect_input_scan
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.router_utils.embedding_batch_size_check import async_log_rejection_before_provider_call
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs
from litellm.utils import Rules, function_setup

PROMPT: Final = "hello from the user"
TENANT_HEADER: Final = {"x-tenant-id": "tenant-7"}


class _ScanRecorder(CustomGuardrail):
    def __init__(self, gate: asyncio.Event | None = None, block: bool = False) -> None:
        super().__init__(guardrail_name="audit-vendor", event_hook=GuardrailEventHooks.logging_only, default_on=True)
        self.gate = gate
        self.block = block
        self.scans: list[tuple[str, list[str], object]] = []
        self.cancelled = False

    @log_guardrail_information
    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        self.scans.append(
            (
                input_type,
                list(inputs.get("texts") or []),
                (request_data.get("proxy_server_request") or {}).get("headers"),
            )
        )
        if self.gate is not None:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.block:
            raise ValueError("vendor says no")
        return GenericGuardrailAPIInputs(texts=["[MASKED]" for _ in inputs.get("texts") or []])


class _OwnLoggingHookGuardrail(_ScanRecorder):
    async def async_logging_hook(self, kwargs, result, call_type):
        self.scans.append(("own-hook", [], None))
        return kwargs, result


class _PreCallObserver(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return data


class _LogRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged = asyncio.Event()
        self.events: list[tuple[str | None, list[dict]]] = []

    @property
    def status(self) -> str | None:
        return self.events[-1][0]

    @property
    def guardrail_information(self) -> list[dict]:
        return self.events[-1][1]

    def _record(self, kwargs: dict) -> None:
        payload: Final = kwargs.get("standard_logging_object") or {}
        self.events.append((payload.get("status"), list(payload.get("guardrail_information") or [])))
        self.logged.set()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs)


def _on_arrival(guardrail: _ScanRecorder, timeout: float = 5.0) -> _ScanRecorder:
    guardrail.logging_only_scope = "input"
    guardrail.logging_only_input_timing = "on_arrival"
    guardrail.logging_only_verdict_timeout = timeout
    return guardrail


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[object]]:
    callbacks: Final[list[object]] = []
    for name in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, name, [] if name != "callbacks" else callbacks)
    ProxyLogging._callback_capabilities_cache.clear()
    yield callbacks
    ProxyLogging._callback_capabilities_cache.clear()


def _proxy_request() -> dict:
    logging_obj, data = function_setup(
        original_function="acompletion",
        rules_obj=Rules(),
        start_time=datetime.datetime.now(),
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        litellm_call_id="call-on-arrival",
        proxy_server_request={"url": "http://proxy/v1/chat/completions", "headers": dict(TENANT_HEADER), "body": {}},
        metadata={"user_api_key_alias": "audited-key"},
    )
    return {**data, "litellm_logging_obj": logging_obj}


async def _pre_call(data: dict, skip_guardrails: bool = False) -> dict:
    processed: Final = await ProxyLogging(user_api_key_cache=DualCache()).pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(), data=data, call_type="acompletion", skip_guardrails=skip_guardrails
    )
    assert processed is not None
    for _ in range(5):
        await asyncio.sleep(0)
    return processed


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def _complete(
    data: dict, logs: _LogRecorder, mock_response: object = "general kenobi", stream: bool = False
) -> None:
    try:
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=data["messages"],
            mock_response=mock_response,
            stream=stream,
            litellm_logging_obj=data["litellm_logging_obj"],
            litellm_call_id=data["litellm_call_id"],
            metadata=data["metadata"],
            proxy_server_request=data["proxy_server_request"],
        )
    except litellm.InternalServerError:
        pass
    await asyncio.wait_for(logs.logged.wait(), timeout=10)


def _vendor_entries(logs: _LogRecorder, event: int = -1) -> list[tuple[object, object]]:
    return [
        (entry.get("guardrail_status"), entry.get("guardrail_response"))
        for entry in logs.events[event][1]
        if entry.get("guardrail_name") == "audit-vendor"
    ]


@pytest.mark.asyncio
async def test_on_arrival_sends_the_request_with_its_headers_before_the_model_is_called(registered):
    guardrail: Final = _on_arrival(_ScanRecorder(gate=asyncio.Event()))
    registered.append(guardrail)

    processed: Final = await _pre_call(_proxy_request())

    assert guardrail.scans == [("request", [PROMPT], TENANT_HEADER)]
    assert processed["messages"] == [{"role": "user", "content": PROMPT}], "the scan must never change the request"


@pytest.mark.asyncio
async def test_on_arrival_verdict_lands_in_the_success_log_and_the_response_is_not_scanned(registered):
    guardrail: Final = _on_arrival(_ScanRecorder())
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    data: Final = await _pre_call(_proxy_request())
    assert data["messages"] == [{"role": "user", "content": PROMPT}], "a finished scan must not mask the request"
    await _complete(data, logs)

    assert guardrail.scans == [("request", [PROMPT], TENANT_HEADER)]
    assert logs.status == "success"
    assert [status for status, _ in _vendor_entries(logs)] == ["success"]


@pytest.mark.asyncio
async def test_on_arrival_verdict_lands_in_the_failure_log(registered):
    guardrail: Final = _on_arrival(_ScanRecorder(block=True))
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    await _complete(
        await _pre_call(_proxy_request()),
        logs,
        mock_response=litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini"),
    )

    assert logs.status == "failure"
    assert [status for status, _ in _vendor_entries(logs)] == ["guardrail_failed_to_respond"]
    assert len(guardrail.scans) == 1


@pytest.mark.asyncio
async def test_a_scan_slower_than_the_verdict_timeout_is_cancelled_and_logged_as_not_run(registered):
    guardrail: Final = _on_arrival(_ScanRecorder(gate=asyncio.Event()), timeout=0.05)
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    await _complete(await _pre_call(_proxy_request()), logs)
    await asyncio.sleep(0)

    assert _vendor_entries(logs) == [("not_run", "no verdict within 0.05s")]
    assert guardrail.cancelled is True


@pytest.mark.asyncio
async def test_after_call_timing_sends_a_failed_calls_request_in_the_background_without_a_logged_verdict(registered):
    guardrail: Final = _ScanRecorder(gate=asyncio.Event())
    guardrail.logging_only_scope = "input"
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    await _complete(
        await _pre_call(_proxy_request()),
        logs,
        mock_response=litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini"),
    )
    await _settle()

    assert logs.status == "failure", "the error is logged while the vendor is still holding the scan"
    assert _vendor_entries(logs) == []
    assert guardrail.scans == [("request", [PROMPT], TENANT_HEADER)]


@pytest.mark.asyncio
async def test_an_on_arrival_guardrail_falls_back_to_scanning_after_the_call_when_pre_call_was_skipped(registered):
    guardrail: Final = _on_arrival(_ScanRecorder())
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs, _PreCallObserver()))

    data: Final = await _pre_call(_proxy_request(), skip_guardrails=True)
    assert guardrail.scans == [], "skip_guardrails must not start a scan"
    await _complete(data, logs)

    assert guardrail.scans == [("request", [PROMPT], TENANT_HEADER)], "the fallback sends the same headers"
    assert [status for status, _ in _vendor_entries(logs)] == ["success"]


@pytest.mark.asyncio
async def test_a_guardrail_with_its_own_logging_hook_is_not_run_on_failure(registered):
    guardrail: Final = _OwnLoggingHookGuardrail()
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    await _complete(
        await _pre_call(_proxy_request()),
        logs,
        mock_response=litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini"),
    )
    await _settle()

    assert guardrail.scans == []


@pytest.mark.asyncio
async def test_verdicts_recorded_before_the_scan_are_not_logged_twice(registered):
    guardrail: Final = _on_arrival(_ScanRecorder())
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))
    data: Final = _proxy_request()
    guardrail.add_standard_logging_guardrail_information_to_request_data(
        guardrail_json_response="earlier verdict", request_data=data, guardrail_status="success"
    )

    await _complete(await _pre_call(data), logs)

    assert _vendor_entries(logs).count(("success", "earlier verdict")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
async def test_a_key_opted_out_of_the_guardrail_is_never_scanned(registered, fails):
    guardrail: Final = _on_arrival(_ScanRecorder())
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))
    data: Final = _proxy_request()
    data["metadata"]["user_api_key_metadata"] = {"opted_out_global_guardrails": ["audit-vendor"]}

    await _complete(
        await _pre_call(data),
        logs,
        mock_response=litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini")
        if fails
        else "general kenobi",
    )
    await _settle()

    assert guardrail.scans == []
    assert _vendor_entries(logs) == []


@pytest.mark.asyncio
async def test_a_guardrail_without_logging_only_mode_is_not_run_on_failure(registered):
    guardrail: Final = _ScanRecorder()
    guardrail.event_hook = GuardrailEventHooks.post_call
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    await _complete(
        await _pre_call(_proxy_request()),
        logs,
        mock_response=litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini"),
    )

    assert guardrail.scans == []


@pytest.mark.asyncio
async def test_every_attempt_that_collects_a_timed_out_scan_logs_it_as_not_run():
    guardrail: Final = _on_arrival(_ScanRecorder(gate=asyncio.Event()), timeout=0.05)
    scan: Final = asyncio.get_running_loop().create_task(
        guardrail.apply_guardrail(inputs={"texts": [PROMPT]}, request_data={"metadata": {}}, input_type="request")
    )

    first: Final = await collect_input_scan(guardrail, scan)
    second: Final = await collect_input_scan(guardrail, scan)

    assert [(entry["guardrail_status"], entry["guardrail_response"]) for entry in (*first, *second)] == [
        ("not_run", "no verdict within 0.05s"),
        ("not_run", "no verdict within 0.05s"),
    ]


@pytest.mark.asyncio
async def test_a_request_the_proxy_rejects_before_the_model_is_not_sent_to_the_vendor(registered):
    guardrail: Final = _ScanRecorder()
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))
    logging_obj: Final = _proxy_request()["litellm_logging_obj"]
    logging_obj.model_call_details[LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL] = True

    await logging_obj.async_failure_handler(
        exception=HTTPException(status_code=401, detail="invalid key"), traceback_exception=""
    )
    await _settle()

    assert logs.status == "failure"
    assert guardrail.scans == []


@pytest.mark.asyncio
async def test_a_guardrail_scoped_to_non_streaming_calls_skips_a_failed_streaming_call(registered):
    guardrail: Final = _ScanRecorder()
    guardrail.apply_stream_scope("non_streaming")
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    await _complete(
        await _pre_call(_proxy_request()),
        logs,
        mock_response=litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini"),
        stream=True,
    )
    await _settle()

    assert guardrail.scans == []


@pytest.mark.asyncio
async def test_a_failed_attempt_leaves_the_scan_running_for_the_retry_that_succeeds(registered):
    gate: Final = asyncio.Event()
    guardrail: Final = _on_arrival(_ScanRecorder(gate=gate))
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))
    data: Final = await _pre_call(_proxy_request())

    await data["litellm_logging_obj"].async_failure_handler(
        exception=litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini"),
        traceback_exception="",
    )
    gate.set()
    logs.logged.clear()
    await _complete(data, logs)

    assert [status for status, _ in logs.events] == ["failure", "success"]
    assert _vendor_entries(logs, event=0) == [("not_run", "no verdict yet when the call failed")]
    assert [status for status, _ in _vendor_entries(logs, event=1)] == ["success"]
    assert guardrail.cancelled is False


@pytest.mark.asyncio
async def test_a_failed_attempts_verdict_reaches_the_payload_the_router_publishes(registered):
    guardrail: Final = _on_arrival(_ScanRecorder())
    registered.append(guardrail)
    data: Final = await _pre_call(_proxy_request())
    logging_obj: Final = data["litellm_logging_obj"]

    await logging_obj.dispatch_attempt_failure_handlers(
        litellm.InternalServerError(message="upstream down", llm_provider="openai", model="gpt-4o-mini"), ""
    )

    published: Final = logging_obj.model_call_details["standard_logging_object"]["guardrail_information"]
    assert [entry["guardrail_status"] for entry in published if entry["guardrail_name"] == "audit-vendor"] == [
        "success"
    ]


@pytest.mark.asyncio
async def test_a_log_masking_guardrail_still_runs_for_a_key_that_disabled_global_guardrails(registered):
    guardrail: Final = _OwnLoggingHookGuardrail()
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))
    data: Final = _proxy_request()
    data["metadata"]["user_api_key_metadata"] = {"disable_global_guardrails": True}

    await _complete(await _pre_call(data), logs)

    assert guardrail.scans == [("own-hook", [], None)], "an opted-out key must never turn log redaction off"


@pytest.mark.asyncio
async def test_a_call_the_router_rejects_before_routing_is_not_sent_to_the_vendor(registered):
    guardrail: Final = _ScanRecorder()
    logs: Final = _LogRecorder()
    registered.extend((guardrail, logs))

    await async_log_rejection_before_provider_call(
        _proxy_request()["litellm_logging_obj"],
        litellm.BadRequestError(message="batch too large", llm_provider="openai", model="gpt-4o-mini"),
    )
    await _settle()

    assert logs.status == "failure"
    assert guardrail.scans == []
