import asyncio
import json
import threading
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from litellm.exceptions import GuardrailRaisedException as HTTPException
from litellm.llms.bedrock.passthrough.guardrail_translation.handler import BedrockPassthroughGuardrailHandler

from litellm.llms.openai.chat.guardrail_translation.handler import OpenAIChatCompletionsHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy.guardrails.guardrail_hooks.patronus_ark.patronus_ark import PatronusArkGuardrail
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
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

    def scan(self, text: str, *, pii: bool, dlp: bool, injection: bool, control: TestControl) -> object:
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
    return await OpenAIChatCompletionsHandler().process_input_messages(
        data={"model": "test", "messages": messages}, guardrail_to_apply=rail, litellm_logging_obj=None
    )


def rail(config: ArkConfig | None = None, core: TestCore | None = None) -> PatronusArkGuardrail:
    return PatronusArkGuardrail(
        config,
        scanner=core or TestCore(),
        control_factory=TestControl,
        default_on=True,
        event_hook=GuardrailEventHooks.pre_call,
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
    assert EMAIL not in str(error.value.message)


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


@pytest.mark.parametrize("arguments", ["{bad json", "null invalid", "[NaN]", '{"value":1e999}'])
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

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with handler.client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="api")),
            scanner=TestCore(),
            control_factory=TestControl,
            async_handler=handler,
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
        httpx.Response(201, json={"jobs": [job()]}),
        httpx.Response(200, json={"jobs": [job(state="degraded")]}),
        httpx.Response(200, json={"jobs": [{**job(), "categories": {}}]}),
        httpx.Response(200, json={"jobs": [job(label="unknown")]}),
    ],
)
async def test_api_errors_and_incomplete_results_never_allow(response: httpx.Response) -> None:
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(lambda request: response))
    async with handler.client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="api")),
            scanner=TestCore(),
            control_factory=TestControl,
            async_handler=handler,
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

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with handler.client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="api")),
            scanner=TestCore(),
            control_factory=TestControl,
            async_handler=handler,
            api_key="test-key",
        )
        assert await instance._scan_wolf("hello", asyncio.get_running_loop().time() + 10)


async def test_api_transport_timeout_fails_closed() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with handler.client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="api")),
            scanner=TestCore(),
            control_factory=TestControl,
            async_handler=handler,
            api_key="test-key",
            default_on=True,
            event_hook=GuardrailEventHooks.pre_call,
        )
        with pytest.raises(HTTPException) as error:
            await guard(instance, [{"role": "user", "content": "hello"}])
        assert error.value.status_code == 503


async def test_standard_translation_records_one_masking_event() -> None:
    output: Final = await guard(rail(), [{"role": "user", "content": EMAIL}])
    entries: Final = output["metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    assert entries[0]["guardrail_status"] == "success"
    assert entries[0]["guardrail_response"] == "mask"
    assert entries[0]["guardrail_mode"] == GuardrailEventHooks.pre_call


async def test_queue_timeout_refuses_forwarding_and_keeps_capacity_available() -> None:
    instance: Final = rail()

    async def expired(awaitable, *, timeout):
        awaitable.close()
        raise TimeoutError("synthetic queue timeout")

    with patch("asyncio.wait_for", side_effect=expired):
        with pytest.raises(HTTPException) as error:
            await guard(instance, [{"role": "user", "content": "hello"}])
    assert error.value.status_code == 503
    assert (await guard(instance, [{"role": "user", "content": "hello"}]))["messages"] == [
        {"role": "user", "content": "hello"},
    ]


async def test_scan_timeout_cancels_active_core_before_releasing_capacity() -> None:
    core: Final = TestCore(block=True)
    instance: Final = rail(core=core)

    async def expired(awaitable, *, timeout):
        if isinstance(awaitable, asyncio.Future):
            await asyncio.to_thread(core.started.wait)
            raise TimeoutError("synthetic scan timeout")
        return await awaitable

    with patch("asyncio.wait_for", side_effect=expired):
        with pytest.raises(HTTPException) as error:
            await guard(instance, [{"role": "user", "content": "hello"}])
    assert error.value.status_code == 503
    assert instance._pending == 0
    assert not instance._scan_lock.locked()


async def test_expired_wolf_budget_never_submits_to_control() -> None:
    instance: Final = rail()
    with pytest.raises(TimeoutError, match="Wolf deadline exceeded"):
        await instance._scan_wolf("hello", float("-inf"))


async def test_configured_local_wolf_without_a_result_fails_closed() -> None:
    instance: Final = rail(ArkConfig(wolf=ArkWolfConfig(backend="local", bundle_dir="unused-by-test-core")))
    with pytest.raises(HTTPException) as error:
        await guard(instance, [{"role": "user", "content": "hello"}])
    assert error.value.status_code == 503


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_scan_deadlines_refuse_initialization(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        PatronusArkGuardrail(scanner=TestCore(), timeout=timeout)


def test_api_wolf_without_credentials_refuses_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PATRONUS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Wolf API requires"):
        PatronusArkGuardrail(ArkConfig(wolf=ArkWolfConfig(backend="api")), scanner=TestCore())


@pytest.mark.parametrize(
    "wolf, reason",
    [
        ({"backend": "local"}, "Local Wolf requires bundle_dir"),
        ({"window_tokens": 4, "overlap_tokens": 2}, "Wolf overlap must be smaller"),
    ],
)
def test_invalid_wolf_configuration_is_rejected_by_the_standard_config(
    wolf: Mapping[str, str | int], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        LitellmParams(guardrail="patronus_ark", mode="pre_call", ark={"wolf": wolf})


async def test_admin_configuration_exposes_ark_settings_and_display_name() -> None:
    from litellm.proxy.guardrails.guardrail_endpoints import get_provider_specific_params

    providers: Final = await get_provider_specific_params()
    assert providers["patronus_ark"]["ui_friendly_name"] == "Patronus Ark"
    assert "ark" in providers["patronus_ark"]


async def test_bedrock_converse_is_rejected_before_unscanned_input_can_forward() -> None:
    core: Final = TestCore()
    with pytest.raises(HTTPException) as error:
        await BedrockPassthroughGuardrailHandler().process_input_messages(
            data={
                "endpoint": "/model/test/converse",
                "data": {"messages": [{"role": "user", "content": [{"text": "hello"}]}]},
            },
            guardrail_to_apply=rail(core=core),
        )
    assert error.value.status_code == 400
    assert core.calls == ()


async def test_missing_structured_input_refuses_forwarding() -> None:
    with pytest.raises(HTTPException) as error:
        await rail().apply_guardrail(inputs={"texts": ["hello"]}, request_data={"messages": []}, input_type="request")
    assert error.value.status_code == 503


async def test_input_limit_counts_utf8_bytes_before_starting_core() -> None:
    core: Final = TestCore()
    with pytest.raises(HTTPException) as error:
        await guard(rail(ArkConfig(max_input_bytes=3), core), [{"role": "user", "content": "😀"}])
    assert error.value.status_code == 503
    assert core.calls == ()


async def test_expired_native_deadline_never_starts_core() -> None:
    core: Final = TestCore()
    with pytest.raises(TimeoutError, match="Ark deadline exceeded"):
        await rail(core=core)._scan("hello", "user_input", False, float("-inf"))
    assert core.calls == ()


async def test_core_detector_error_refuses_forwarding() -> None:
    core: Final = Mock(
        scan=Mock(
            return_value={"wolf": None, "native": [{"category": "pii", "class_name": "error", "evidence_spans": []}]}
        )
    )
    with pytest.raises(HTTPException) as error:
        await guard(rail(core=core), [{"role": "user", "content": "hello"}])
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    "category, label, config",
    [("injection", "attack", ArkConfig(native_pi=False)), ("pii", "pii", ArkConfig(pii="off"))],
)
async def test_disabled_detectors_do_not_change_input(category: str, label: str, config: ArkConfig) -> None:
    core: Final = Mock(
        scan=Mock(
            return_value={
                "wolf": None,
                "native": [
                    {"category": category, "class_name": label, "evidence_spans": [{"start_char": 0, "end_char": 5}]}
                ],
            }
        )
    )
    instance: Final = rail(config, core)
    assert (await guard(instance, [{"role": "user", "content": "hello"}]))["messages"] == [
        {"role": "user", "content": "hello"},
    ]


async def test_local_wolf_attack_is_blocked_without_model_assets_in_the_test() -> None:
    core: Final = Mock(scan=Mock(return_value={"wolf": {"matched": True}, "native": []}))
    config: Final = ArkConfig(native_pi=False, wolf=ArkWolfConfig(backend="local", bundle_dir="unused-by-test-core"))
    with pytest.raises(HTTPException) as error:
        await guard(rail(config, core), [{"role": "user", "content": "hello"}])
    assert error.value.status_code == 400


@pytest.mark.parametrize("changed_job", [False, True])
async def test_wolf_polling_refuses_invalid_or_changed_job_identifiers(changed_job: bool) -> None:
    submit: Final = {"job_id": "job_" + "a" * 32 if changed_job else "invalid", "status": "running"}
    respond: Final = Mock(
        side_effect=[
            httpx.Response(202, json={"jobs": [submit]}),
            httpx.Response(200, json={**job(), "job_id": "job_" + "b" * 32}),
        ]
    )
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with handler.client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(wolf=ArkWolfConfig(backend="api")),
            scanner=TestCore(),
            control_factory=TestControl,
            async_handler=handler,
            api_key="test-key",
        )
        with pytest.raises(HTTPException) as error:
            await guard(instance, [{"role": "user", "content": "hello"}])
        assert error.value.status_code == 503
    assert respond.call_count == (2 if changed_job else 1)


async def test_wolf_submission_exhausting_the_deadline_never_polls() -> None:
    respond: Final = Mock(
        return_value=httpx.Response(202, json={"jobs": [{"job_id": "job_" + "a" * 32, "status": "running"}]})
    )
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    clock: Final = Mock()
    clock.get_running_loop.return_value.time.side_effect = [0.0, 11.0]
    clock.sleep = AsyncMock()
    async with handler.client:
        instance: Final = PatronusArkGuardrail(
            ArkConfig(wolf=ArkWolfConfig(backend="api")),
            scanner=TestCore(),
            control_factory=TestControl,
            async_handler=handler,
            api_key="test-key",
        )
        with patch("litellm.proxy.guardrails.guardrail_hooks.patronus_ark.patronus_ark.asyncio", clock):
            with pytest.raises(TimeoutError, match="Wolf deadline exceeded"):
                await instance._scan_wolf("hello", 10.0)
    assert respond.call_count == 1


@pytest.mark.parametrize("local_wolf", [False, True])
async def test_optional_sdk_initializes_through_the_standard_registry_and_scans(local_wolf: bool) -> None:
    from litellm.proxy.guardrails.guardrail_hooks.patronus_ark import guardrail_initializer_registry

    core: Final = Mock(scan=Mock(return_value={"input_bytes": 5, "native": [], "wolf": {"matched": False}}))
    scanner_factory: Final = Mock(return_value=core)
    control_factory: Final = Mock(return_value=TestControl(1000))
    params: Final = LitellmParams(
        guardrail="patronus_ark",
        mode="pre_call",
        ark={"wolf": {"backend": "local", "bundle_dir": "local-model-bundle"}} if local_wolf else {},
    )
    with patch.dict("sys.modules", {"patronus_ark_core": Mock(Scanner=scanner_factory, ScanControl=control_factory)}):
        instance: Final = guardrail_initializer_registry[params.guardrail](
            params, {"guardrail_name": "optional-sdk-ark", "litellm_params": params}
        )
        messages: Final = [{"role": "user", "content": "hello"}]
        output: Final = await guard(instance, messages)
        assert output["messages"] == messages
    scanner_factory.assert_called_once()
    config: Final = scanner_factory.call_args.args[0]
    assert config["models"] == {"native:injection_l1": True}
    if local_wolf:
        assert config["wolf"]["bundle_dir"] == "local-model-bundle"
        assert config["wolf"]["backend"] == "cpu"
    else:
        assert "wolf" not in config
    control_factory.assert_called_once()
    core.scan.assert_called_once_with("hello", pii=True, dlp=True, injection=True, control=control_factory.return_value)
