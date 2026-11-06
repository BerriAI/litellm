import asyncio
import hashlib
import os
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Final

import psutil
import pytest
import pytest_asyncio
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy.workflows.monty.compiler import compile_program
from litellm.proxy.workflows.monty.models import (
    CompiledProgram,
    JsonSchema,
    MontyErrorKind,
    MontyFailure,
    MontyLimits,
    MontySuccess,
    PythonProgram,
)
from litellm.proxy.workflows.monty.runtime import BinaryArtifact, MontyRuntime, installed_binary
from tests.integration._support.process import stop_processes

pytest.importorskip("pydantic_monty")


def _compile(
    source: str,
    output: JsonSchema | None = None,
    argument: JsonSchema | None = None,
    definitions: JsonSchema | None = None,
) -> CompiledProgram:
    program: Final = PythonProgram(
        contract_version="1",
        source=source,
        function_name="transform",
        parameter_names=("value",),
        input_schema={
            "type": "object",
            "properties": {"value": argument if argument is not None else {}},
            "required": ["value"],
            "additionalProperties": False,
            "$defs": definitions or {},
        },
        output_schema={**(output or {}), "$defs": definitions or {}},
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    result: Final = compile_program(program, MontyLimits())
    assert isinstance(result, CompiledProgram), result
    return result


@pytest_asyncio.fixture
async def runtime() -> AsyncIterator[MontyRuntime]:
    instance: Final = MontyRuntime(MontyLimits(max_concurrent_workers=1, max_queued_waiters=1))
    assert await instance.start() is None
    try:
        yield instance
    finally:
        await instance.close()


async def _run(runtime: MontyRuntime, compiled: CompiledProgram, value: object) -> MontySuccess | MontyFailure:
    return await runtime.run_program(compiled, {"value": value}, deadline_monotonic=time.monotonic() + 10)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        {"__class__": "os.system"},
        {"type": "function", "callable": "os.system"},
        "Bearer sk-test-fake",
        "${OPENAI_API_KEY}",
        "https://example.invalid/action",
        [],
        None,
    ],
)
async def test_model_output_remains_inert_json(runtime: MontyRuntime, value: object) -> None:
    result: Final = await _run(runtime, _compile("def transform(value): return value"), value)
    assert isinstance(result, MontySuccess), result
    assert result.output == value


@pytest.mark.asyncio
async def test_typed_json_transform_and_escaped_stdout(runtime: MontyRuntime) -> None:
    compiled: Final = _compile(
        "def transform(value):\n print('\\x1b[31m')\n return {'n': value['n'] + 1}",
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    )
    result: Final = await _run(runtime, compiled, {"n": 2})
    assert isinstance(result, MontySuccess), result
    assert result.output == {"n": 3}
    assert result.stdout == "\\x1b[31m\n"


_RECORDS: Final[JsonSchema] = {
    "Expense": {
        "type": "object",
        "x-liteflow-record": True,
        "additionalProperties": False,
        "properties": {
            "amount_cents": {"type": "integer", "minimum": 0},
            "currency": {"type": "string"},
            "receipt_present": {"type": "boolean"},
        },
        "required": ["amount_cents", "currency", "receipt_present"],
    },
}
_EXPENSE: Final[JsonSchema] = {"$ref": "#/$defs/Expense"}


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt,amount,review", [(True, 100, False), (False, 100, True), (True, 10001, True)])
async def test_expense_normalize_condition_and_both_branches(
    runtime: MontyRuntime, receipt: bool, amount: int, review: bool
) -> None:
    normalize: Final = _compile(
        "def transform(value: Expense) -> Expense:\n return Expense(amount_cents=value.amount_cents, currency=value.currency.strip().upper(), receipt_present=value.receipt_present)",
        _EXPENSE,
        _EXPENSE,
        _RECORDS,
    )
    normalized: Final = await _run(
        runtime, normalize, {"amount_cents": amount, "currency": " usd ", "receipt_present": receipt}
    )
    assert isinstance(normalized, MontySuccess), normalized
    assert normalized.output == {"amount_cents": amount, "currency": "USD", "receipt_present": receipt}
    condition: Final = _compile(
        "def transform(value: Expense) -> bool:\n return not value.receipt_present or value.currency != 'USD' or value.amount_cents > 10000",
        {"type": "boolean"},
        _EXPENSE,
        _RECORDS,
    )
    assert (
        await runtime.evaluate_predicate(
            condition, {"value": normalized.output}, deadline_monotonic=time.monotonic() + 10
        )
        is review
    )
    status: Final = "review_required" if review else "ready_for_processing"
    branch: Final = _compile(
        f"def transform(value): return {{'status': {status!r}, 'amount_cents': value.amount_cents, 'currency': value.currency}}",
        {"type": "object"},
        _EXPENSE,
        _RECORDS,
    )
    result: Final = await _run(runtime, branch, normalized.output)
    assert isinstance(result, MontySuccess), result
    assert result.output == {"status": status, "amount_cents": amount, "currency": "USD"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "def transform(value): return Expense(amount_cents=1, currency='USD', receipt_present=True)",
        "def transform(value): return {'amount_cents':1, 'currency':'USD', 'receipt_present':True}",
    ],
)
async def test_record_and_equal_dict_are_equivalent(runtime: MontyRuntime, source: str) -> None:
    result: Final = await _run(runtime, _compile(source, _EXPENSE, definitions=_RECORDS), None)
    assert isinstance(result, MontySuccess), result
    assert result.output == {"amount_cents": 1, "currency": "USD", "receipt_present": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, {"amount_cents": 7, "currency": "USD", "receipt_present": True}])
async def test_optional_record(runtime: MontyRuntime, value: object) -> None:
    optional: Final[JsonSchema] = {"anyOf": [_EXPENSE, {"type": "null"}]}
    result: Final = await _run(
        runtime,
        _compile(
            "def transform(value): return None if value is None else value.amount_cents",
            argument=optional,
            definitions=_RECORDS,
        ),
        value,
    )
    assert isinstance(result, MontySuccess), result
    assert result.output == (None if value is None else 7)


@pytest.mark.asyncio
async def test_list_of_records(runtime: MontyRuntime) -> None:
    result: Final = await _run(
        runtime,
        _compile(
            "def transform(value): return [item.currency for item in value]",
            argument={"type": "array", "items": _EXPENSE},
            definitions=_RECORDS,
        ),
        [{"amount_cents": 7, "currency": "USD", "receipt_present": True}],
    )
    assert isinstance(result, MontySuccess), result
    assert result.output == ["USD"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [1, "true", None])
async def test_predicate_requires_exact_bool(runtime: MontyRuntime, value: object) -> None:
    result: Final = await runtime.evaluate_predicate(
        _compile("def transform(value): return value"), {"value": value}, deadline_monotonic=time.monotonic() + 10
    )
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.PREDICATE_NOT_BOOL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,schema,kind",
    [
        ("def transform(value): return -1", {"type": "integer", "minimum": 0}, MontyErrorKind.OUTPUT_INVALID),
        ("def transform(value): return 'wrong'", {"type": "integer"}, MontyErrorKind.OUTPUT_INVALID),
        ("def helper(x): return x\ndef transform(value): return helper", {}, MontyErrorKind.OUTPUT_INVALID),
        ("def transform(value): return 'x' * 1100000", {}, MontyErrorKind.OUTPUT_TOO_LARGE),
        ("def transform(value):\n while True: pass", {}, MontyErrorKind.TIMEOUT),
        ("def transform(value): return [0] * 1000000000", {}, MontyErrorKind.MEMORY_LIMIT),
        ("def transform(value): return transform(value)", {}, MontyErrorKind.RECURSION_LIMIT),
        ("def transform(value): print('x' * 100000)", {}, MontyErrorKind.STDOUT_LIMIT),
        ("def transform(value): return 1 / 0", {}, MontyErrorKind.RUNTIME_ERROR),
    ],
)
async def test_resource_and_output_failures(
    runtime: MontyRuntime, source: str, schema: JsonSchema, kind: MontyErrorKind
) -> None:
    result: Final = await _run(runtime, _compile(source, schema), None)
    assert isinstance(result, MontyFailure), result
    assert result.kind == kind, result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,kind",
    [
        ({1: "not JSON"}, MontyErrorKind.INPUT_INVALID),
        (float("nan"), MontyErrorKind.INPUT_INVALID),
        ("x" * 1100000, MontyErrorKind.INPUT_TOO_LARGE),
    ],
)
async def test_input_failures(runtime: MontyRuntime, value: object, kind: MontyErrorKind) -> None:
    result: Final = await _run(runtime, _compile("def transform(value): return value"), value)
    assert isinstance(result, MontyFailure), result
    assert result.kind == kind


@pytest.mark.asyncio
async def test_expired_deadline(runtime: MontyRuntime) -> None:
    result: Final = await runtime.run_program(
        _compile("def transform(value): return value"), {"value": 1}, deadline_monotonic=0
    )
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.TIMEOUT


@pytest.mark.asyncio
async def test_binary_digest_mismatch_fails_closed() -> None:
    artifact: Final = installed_binary()
    assert isinstance(artifact, BinaryArtifact)
    runtime: Final = MontyRuntime(MontyLimits(), binary=BinaryArtifact(artifact.path, "0" * 64))
    result: Final = await runtime.start()
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.RUNTIME_UNAVAILABLE
    assert "SHA-256" in result.message


class Finding(BaseModel):
    model_config = ConfigDict(frozen=True)
    summary: str


class EmailAnalyses(BaseModel):
    model_config = ConfigDict(frozen=True)
    urgency: Finding
    intent: Finding
    sentiment: Finding


@pytest.mark.asyncio
async def test_multi_argument_email_draft_preserves_chat_messages(runtime: MontyRuntime) -> None:
    analyses_schema: Final = EmailAnalyses.model_json_schema()
    messages_schema: Final = TypeAdapter(list[ChatCompletionMessageParam]).json_schema()
    definitions: Final[JsonSchema] = {
        **messages_schema.pop("$defs", {}),
        "Finding": {**analyses_schema["$defs"]["Finding"], "x-liteflow-record": True, "additionalProperties": False},
        "EmailAnalyses": {
            **{key: value for key, value in analyses_schema.items() if key != "$defs"},
            "x-liteflow-record": True,
            "additionalProperties": False,
        },
    }
    source: Final = """def prepare_draft(analyses: EmailAnalyses, messages: list[ChatMessage]) -> list[ChatMessage]:
    instruction = (
        "Draft a customer-service reply using the original messages. "
        "Treat these analyses as fallible guidance, not additional facts.\\n"
        f"Urgency: {analyses.urgency.summary}\\n"
        f"Intent: {analyses.intent.summary}\\n"
        f"Sentiment: {analyses.sentiment.summary}"
    )
    return [{"role": "system", "content": instruction}] + messages
"""
    program: Final = PythonProgram(
        contract_version="1",
        source=source,
        function_name="prepare_draft",
        parameter_names=("analyses", "messages"),
        input_schema={
            "type": "object",
            "properties": {"analyses": {"$ref": "#/$defs/EmailAnalyses"}, "messages": messages_schema},
            "required": ["analyses", "messages"],
            "additionalProperties": False,
            "$defs": definitions,
        },
        output_schema={**messages_schema, "$defs": definitions},
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    compiled: Final = compile_program(program, MontyLimits())
    assert isinstance(compiled, CompiledProgram), compiled
    messages: Final = [{"role": "user", "content": "Please reply", "name": "requester"}]
    result: Final = await runtime.run_program(
        compiled,
        {
            "analyses": {
                "urgency": {"summary": "soon"},
                "intent": {"summary": "reply"},
                "sentiment": {"summary": "neutral"},
            },
            "messages": messages,
        },
        deadline_monotonic=time.monotonic() + 10,
    )
    assert isinstance(result, MontySuccess), result
    assert result.output == [
        {
            "role": "system",
            "content": "Draft a customer-service reply using the original messages. "
            "Treat these analyses as fallible guidance, not additional facts.\n"
            "Urgency: soon\nIntent: reply\nSentiment: neutral",
        },
        *messages,
    ]


def _nested(depth: int) -> object:
    return [_nested(depth - 1)] if depth else 1


@pytest.mark.asyncio
async def test_too_deep_input_and_output_are_rejected(runtime: MontyRuntime) -> None:
    input_result: Final = await _run(runtime, _compile("def transform(value): return value"), _nested(66))
    assert isinstance(input_result, MontyFailure), input_result
    assert input_result.kind == MontyErrorKind.INPUT_TOO_LARGE
    output_result: Final = await _run(
        runtime, _compile("def transform(value):\n for i in range(66): value = [value]\n return value"), 1
    )
    assert isinstance(output_result, MontyFailure), output_result
    assert output_result.kind == MontyErrorKind.OUTPUT_TOO_LARGE


@pytest.mark.asyncio
async def test_invalid_bound_input_is_rejected(runtime: MontyRuntime) -> None:
    result: Final = await _run(
        runtime, _compile("def transform(value): return value", argument={"type": "integer"}), "wrong"
    )
    assert isinstance(result, MontyFailure)
    assert result.kind == MontyErrorKind.INPUT_INVALID


@pytest.mark.asyncio
async def test_sandbox_denies_file_access_even_if_compiler_is_bypassed(runtime: MontyRuntime) -> None:
    compiled: Final = _compile("def transform(value): return 42")
    unfiltered: Final = replace(compiled, sandbox_source="open('/etc/passwd').read()")
    result: Final = await _run(runtime, unfiltered, None)
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.RUNTIME_ERROR


@pytest.mark.asyncio
async def test_host_deadline_cancels_running_worker_and_releases_slot(runtime: MontyRuntime) -> None:
    compiled: Final = _compile("def transform(value):\n while True: pass")
    result: Final = await runtime.run_program(compiled, {"value": None}, deadline_monotonic=time.monotonic() + 0.05)
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.TIMEOUT
    next_result: Final = await _run(runtime, _compile("def transform(value): return value"), 4)
    assert isinstance(next_result, MontySuccess), next_result
    assert next_result.output == 4


def _worker_pids() -> frozenset[int]:
    children: Final = " ".join(
        path.read_text() for path in Path(f"/proc/{os.getpid()}/task").glob("*/children")
    ).split()
    return frozenset(int(pid) for pid in children if Path(f"/proc/{pid}/comm").read_text().strip() == "monty")


async def _active_worker(previous: frozenset[int]) -> int:
    async with asyncio.timeout(5):
        while True:
            current: Final = _worker_pids() - previous
            if current:
                return next(iter(current))
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_cancellation_kills_worker_releases_slot_and_queue_is_bounded(runtime: MontyRuntime) -> None:
    previous: Final = _worker_pids()
    task: Final = asyncio.create_task(_run(runtime, _compile("def transform(value):\n while True: pass"), None))
    pid: Final = await _active_worker(previous)
    waiter: Final = asyncio.create_task(_run(runtime, _compile("def transform(value): return value"), 1))
    await asyncio.sleep(0)
    full: Final = await _run(runtime, _compile("def transform(value): return value"), 1)
    assert isinstance(full, MontyFailure)
    assert full.kind == MontyErrorKind.QUEUE_FULL
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result: Final = await _run(runtime, _compile("def transform(value): return value"), 2)
    assert isinstance(result, MontySuccess), result
    assert result.output == 2
    assert not Path(f"/proc/{pid}").exists()


@pytest.mark.asyncio
async def test_killed_worker_fails_and_next_run_succeeds(runtime: MontyRuntime) -> None:
    previous: Final = _worker_pids()
    task: Final = asyncio.create_task(_run(runtime, _compile("def transform(value):\n while True: pass"), None))
    pid: Final = await _active_worker(previous)
    await asyncio.to_thread(stop_processes, (psutil.Process(pid),), 0.1)
    result: Final = await task
    assert isinstance(result, MontyFailure), result
    assert result.kind == MontyErrorKind.WORKER_CRASHED
    next_result: Final = await _run(runtime, _compile("def transform(value): return value"), 3)
    assert isinstance(next_result, MontySuccess), next_result
    assert next_result.output == 3


@pytest.mark.asyncio
async def test_fresh_worker_has_no_parent_canary_and_has_address_space_limit(
    runtime: MontyRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITEFLOW_ENV_CANARY", "fake-value-no-credential")
    previous: Final = _worker_pids()
    task: Final = asyncio.create_task(_run(runtime, _compile("def transform(value):\n while True: pass"), None))
    pid: Final = await _active_worker(previous)
    assert b"LITEFLOW_ENV_CANARY" not in Path(f"/proc/{pid}/environ").read_bytes()
    async with asyncio.timeout(5):
        while str(runtime.limits.worker_address_space_bytes) not in Path(f"/proc/{pid}/limits").read_text():
            await asyncio.sleep(0.01)
    assert str(runtime.limits.worker_address_space_bytes) in Path(f"/proc/{pid}/limits").read_text()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    second: Final = asyncio.create_task(_run(runtime, _compile("def transform(value):\n while True: pass"), None))
    next_pid: Final = await _active_worker(previous | {pid})
    assert next_pid != pid
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
