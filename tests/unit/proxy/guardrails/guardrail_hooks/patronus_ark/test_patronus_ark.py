import asyncio
import json
import threading
from typing import Final

import httpx
import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy.guardrails.guardrail_hooks.patronus_ark.patronus_ark import Control, PatronusArkGuardrail
from litellm.proxy.guardrails.guardrail_hooks.patronus_ark.wolf_api import WolfAPI
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.guardrails.guardrail_hooks.patronus_ark import ArkConfig, ArkWolfConfig

EMAIL: Final = "alice" + "@example.org"


class TestControl:
    __test__: Final = False

    def __init__(self, timeout_ms: int) -> None:
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()


class TestCore:
    __test__: Final = False

    def __init__(self, *, block: bool = False) -> None:
        self.calls: tuple[str, ...] = ()
        self.started = threading.Event()
        self.block = block

    def Scanner(self, config: dict[str, object], *, wolf_options: dict[str, int]) -> "TestCore":
        return self

    def ScanControl(self, timeout_ms: int) -> TestControl:
        return TestControl(timeout_ms)

    def scan(self, text: str, *, pii: bool, dlp: bool, injection: bool, control: Control) -> object:
        self.calls = (*self.calls, text)
        self.started.set()
        if self.block:
            assert isinstance(control, TestControl)
            control.cancelled.wait()
            raise RuntimeError("cancelled")
        start: Final = text.find(EMAIL)
        return {
            "input_bytes": len(text.encode()),
            "wolf": None,
            "native": [
                {
                    "category": category,
                    "class_name": "attack"
                    if category == "injection" and "INJECT" in text
                    else ("pii" if category == "pii" and start >= 0 else "safe"),
                    "evidence_spans": [{"start_char": start, "end_char": start + len(EMAIL)}]
                    if category == "pii" and start >= 0
                    else [],
                }
                for category in self._categories(pii, dlp, injection)
            ],
        }

    @staticmethod
    def _categories(pii: bool, dlp: bool, injection: bool) -> tuple[str, ...]:
        return tuple(name for name, enabled in (("pii", pii), ("dlp", dlp), ("injection", injection)) if enabled)


async def guard(rail: PatronusArkGuardrail, messages: list[object]) -> dict[str, object]:
    return await rail.async_pre_call_hook(
        UserAPIKeyAuth(), litellm.DualCache(), {"model": "test", "messages": messages}, "acompletion"
    )


def rail(config: ArkConfig | None = None, core: TestCore | None = None) -> PatronusArkGuardrail:
    return PatronusArkGuardrail(
        config, core_module=core or TestCore(), default_on=True, event_hook=GuardrailEventHooks.pre_call
    )


async def test_redacts_unicode_and_decoded_json_without_touching_reasoning() -> None:
    core: Final = TestCore()
    output: Final = await guard(
        rail(core=core),
        [
            {"role": "user", "content": [{"type": "text", "text": "😀 " + EMAIL}]},
            {
                "role": "assistant",
                "reasoning_content": "INJECT " + EMAIL,
                "content": None,
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"email":"alice\\u0040example.org","limit":2}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call", "content": EMAIL},
        ],
    )
    assert output["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "😀 [REDACTED]"}]},
        {
            "role": "assistant",
            "reasoning_content": "INJECT " + EMAIL,
            "content": None,
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"email":"[REDACTED]","limit":2}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": "[REDACTED]"},
    ]
    assert "INJECT " + EMAIL not in core.calls


@pytest.mark.parametrize("source", ["user", "tool"])
async def test_native_injection_blocks_selected_sources(source: str) -> None:
    with pytest.raises(HTTPException) as error:
        await guard(rail(), [{"role": source, "content": "INJECT"}])
    assert error.value.status_code == 400


async def test_pii_block_policy_never_returns_redacted_success() -> None:
    with pytest.raises(HTTPException) as error:
        await guard(rail(ArkConfig(pii="block")), [{"role": "user", "content": EMAIL}])
    assert error.value.status_code == 400
    assert EMAIL not in str(error.value.detail)


async def test_source_selection_leaves_unselected_content_unchanged() -> None:
    output: Final = await guard(
        rail(ArkConfig(pii_sources=("tool_result",))),
        [
            {"role": "user", "content": EMAIL},
            {"role": "tool", "content": EMAIL},
        ],
    )
    assert output["messages"] == [
        {"role": "user", "content": EMAIL},
        {"role": "tool", "content": "[REDACTED]"},
    ]


@pytest.mark.parametrize("arguments", ["{bad json", "null invalid", "[NaN]"])
async def test_malformed_tool_arguments_fail_closed(arguments: str) -> None:
    with pytest.raises(HTTPException) as error:
        await guard(
            rail(),
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "arguments": arguments},
                        }
                    ],
                }
            ],
        )
    assert error.value.status_code == 503


async def test_request_cancellation_reaches_core_and_releases_scanner() -> None:
    core: Final = TestCore(block=True)
    instance: Final = rail(core=core)
    task: Final = asyncio.create_task(guard(instance, [{"role": "user", "content": "hello"}]))
    await asyncio.to_thread(core.started.wait)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert instance._pending == 0
    assert not instance._scan_lock.locked()


async def test_full_queue_rejects_without_starting_another_scan() -> None:
    core: Final = TestCore(block=True)
    instance: Final = rail(ArkConfig(max_queued=0), core)
    task: Final = asyncio.create_task(guard(instance, [{"role": "user", "content": "first"}]))
    await asyncio.to_thread(core.started.wait)
    try:
        with pytest.raises(HTTPException) as error:
            await guard(instance, [{"role": "user", "content": "second"}])
        assert error.value.status_code == 429
        assert core.calls == ("first",)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_nested_configuration_validates_through_litellm_params() -> None:
    params: Final = LitellmParams(guardrail="patronus_ark", mode="pre_call", ark={"pii": "block"})
    assert params.ark == ArkConfig(pii="block")
    with pytest.raises(ValueError):  # noqa: PT011  # configuration promises ValueError without a stable Pydantic message
        LitellmParams(guardrail="patronus_ark", mode="pre_call", ark={"pii": "silently_allow"})


def job(label: str = "benign", state: str = "complete") -> dict[str, object]:
    return {
        "job_id": "job_" + "a" * 32,
        "status": "completed",
        "completion": {"state": state},
        "categories": {"injection": {"level": "L3", "final_result": {"class_name": label, "confidence": 0.99}}},
    }


@pytest.mark.parametrize("label, expected_status", [("benign", None), ("attack", 400)])
async def test_api_only_requests_wolf_and_applies_final_verdict(label: str, expected_status: int | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://control.patronus.studio/api/v1/scan"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "text": "hello",
            "config": {"categories": ["injection"], "max_level": "L3", "gates": {"l1": False, "l2": False, "l3": True}},
        }
        return httpx.Response(200, json={"jobs": [job(label)]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="api")),
            core_module=TestCore(),
            http_client=client,
            api_key="test-key",
            default_on=True,
            event_hook=GuardrailEventHooks.pre_call,
        )
        if expected_status is None:
            assert (await guard(instance, [{"role": "user", "content": "hello"}]))["messages"] == [
                {"role": "user", "content": "hello"},
            ]
        else:
            with pytest.raises(HTTPException) as error:
                await guard(instance, [{"role": "user", "content": "hello"}])
            assert error.value.status_code == expected_status


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": "invalid key"}),
        httpx.Response(200, json={"jobs": [job(state="degraded")]}),
        httpx.Response(200, json={"jobs": [{**job(), "categories": {}}]}),
        httpx.Response(200, json={"jobs": [job(label="unknown")]}),
    ],
)
async def test_api_errors_and_incomplete_results_never_allow(response: httpx.Response) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="api")),
            core_module=TestCore(),
            http_client=client,
            api_key="test-key",
            default_on=True,
            event_hook=GuardrailEventHooks.pre_call,
        )
        with pytest.raises(HTTPException) as error:
            await guard(instance, [{"role": "user", "content": "hello"}])
        assert error.value.status_code == 503


async def test_api_polls_only_the_submitted_job_on_the_configured_origin() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"jobs": [{"job_id": "job_" + "a" * 32, "status": "running"}]})
        assert request.url == "https://control.patronus.studio/api/v1/scan/job_" + "a" * 32
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json=job("attack"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        assert await WolfAPI(client, "https://control.patronus.studio", "test-key").scan("hello", 10)


async def test_api_transport_timeout_fails_closed() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="api")),
            core_module=TestCore(),
            http_client=client,
            api_key="test-key",
            default_on=True,
            event_hook=GuardrailEventHooks.pre_call,
        )
        with pytest.raises(HTTPException) as error:
            await guard(instance, [{"role": "user", "content": "hello"}])
        assert error.value.status_code == 503
