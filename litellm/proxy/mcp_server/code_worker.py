from __future__ import annotations

import importlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


@dataclass(frozen=True, slots=True)
class CodeExecutionLimits:
    timeout_seconds: float = 120
    memory_bytes: int = 64 * 1024 * 1024
    script_bytes: int = 128 * 1024
    message_bytes: int = 2 * 1024 * 1024
    output_bytes: int = 256 * 1024
    max_calls: int = 100
    max_concurrent_calls: int = 8
    max_workers: int = 4


@dataclass(frozen=True, slots=True)
class CodeExecutionResult:
    output: tuple[JsonValue, ...] = ()
    error: str | None = None
    calls_started: int = 0


class WireModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Start(WireModel):
    code: str
    limits: CodeExecutionLimits


class Call(WireModel):
    kind: Literal["call"] = "call"
    id: int = Field(ge=0)
    tool_id: str
    arguments: Mapping[str, JsonValue]


class Reply(WireModel):
    id: int = Field(ge=0)
    value: Mapping[str, JsonValue] | None = None
    error: str | None = None


class Done(WireModel):
    kind: Literal["done"] = "done"
    output: tuple[JsonValue, ...] = ()
    error: str | None = None


WORKER_MESSAGE: Final[TypeAdapter[Call | Done]] = TypeAdapter(Annotated[Call | Done, Field(discriminator="kind")])
JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
TEXT: Final[TypeAdapter[str]] = TypeAdapter(str)
MAX_SAFE_INTEGER: Final = 2**53 - 1


def _integer(value: str) -> int:
    result: Final = int(value)
    if abs(result) > MAX_SAFE_INTEGER:
        raise ValueError("JSON integers must be JavaScript-safe; represent larger integers as strings")
    return result


def _float(value: str) -> float:
    result: Final = float(value)
    if not math.isfinite(result) or (result.is_integer() and abs(result) > MAX_SAFE_INTEGER):
        raise ValueError("JSON numbers must be finite and integers JavaScript-safe")
    return result


def load_json(value: str) -> JsonValue:
    return JSON_VALUE.validate_python(json.loads(value, parse_int=_integer, parse_float=_float, parse_constant=_float))


def encode_message(value: WireModel, maximum: int) -> bytes:
    text: Final = json.dumps(value.model_dump(), allow_nan=False, ensure_ascii=False)
    load_json(text)
    encoded: Final = (text + "\n").encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError("Code execution message exceeds its byte limit")
    return encoded


BOOTSTRAP: Final = r"""
(rpc, code, limits) => {
    const stringify = JSON.stringify.bind(JSON), parse = JSON.parse.bind(JSON);
    const isSafe = Number.isSafeInteger.bind(Number), isFinite = Number.isFinite.bind(Number);
    const isInteger = Number.isInteger.bind(Number);
    const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
    const ErrorType = Error;
    const StringType = String, slice = Function.prototype.call.bind(String.prototype.slice);
    const output = [];
    let outputSize = 0, calls = 0, active = 0, fatal = null;
    const fail = message => { fatal = message; throw new ErrorType(message); };
    const encode = value => {
        const result = stringify(value, (key, item) => {
            if (typeof item === 'number' && (!isFinite(item) || (isInteger(item) && !isSafe(item))))
                fail('JSON numbers must be finite and integers JavaScript-safe; use strings for larger integers');
            if (['undefined', 'bigint', 'function', 'symbol'].includes(typeof item))
                fail('Output and tool arguments must be JSON values');
            return item;
        });
        if (typeof result !== 'string') fail('Output and tool arguments must be JSON values');
        return result;
    };
    const text = value => {
        const encoded = encode(value);
        outputSize += encoded.length;
        if (outputSize > limits.output_bytes) fail('Code output exceeds its byte limit');
        output.push(parse(encoded));
    };
    const tools = Object.freeze({call_tool: async request => {
        if (++calls > limits.max_calls) fail('Code execution tool-call limit exceeded');
        if (++active > limits.max_concurrent_calls) fail('Code execution concurrent-call limit exceeded');
        try {
            const encoded = encode(request);
            if (encoded.length > limits.message_bytes) fail('Tool arguments exceed their byte limit');
            const reply = parse(await rpc(encoded));
            if (reply.error) fail(reply.error);
            return reply.value;
        } finally { active--; }
    }});
    return (async () => {
        try {
            await new AsyncFunction('tools', 'text', code)(tools, text);
            if (active) fail('Script returned with unawaited tool calls');
            return stringify({kind:'done', output, error:fatal});
        } catch (error) {
            let detail = 'JavaScript execution failed';
            try { detail = slice(StringType(error), 0, 1024); } catch (_) {}
            return stringify({kind:'done', output, error:fatal || detail});
        }
    })();
}
"""


_BRIDGE: Final = (
    r"""
(runner => {
    const send = globalThis.__litellm_send;
    delete globalThis.__litellm_send;
    const parse = JSON.parse.bind(JSON), stringify = JSON.stringify.bind(JSON);
    const get = Function.prototype.call.bind(Map.prototype.get);
    const set = Function.prototype.call.bind(Map.prototype.set);
    const remove = Function.prototype.call.bind(Map.prototype.delete);
    const then = Function.prototype.call.bind(Promise.prototype.then);
    const PromiseType = Promise, ErrorType = Error, pending = new Map();
    let sequence = 0;
    const rpc = raw => new PromiseType(resolve => {
        const id = sequence++;
        set(pending, id, resolve);
        send(stringify({...parse(raw), kind:'call', id}));
    });
    return (kind, raw) => {
        const value = parse(raw);
        if (kind === 'start') {
            then(runner(rpc, value.code, value.limits), send);
        } else {
            const resolve = get(pending, value.id);
            if (!resolve) throw new ErrorType('Invalid parent reply');
            remove(pending, value.id);
            resolve(stringify(value.error ? {error:value.error} : {value:value.value}));
        }
    };
})("""
    + BOOTSTRAP
    + ")"
)


class _Runtime(Protocol):
    def set_memory_limit(self, limit: int) -> None: ...
    def set_max_stack_size(self, limit: int) -> None: ...
    def add_callable(self, name: str, callback: Callable[[str], None]) -> None: ...
    def eval(self, code: str) -> object: ...
    def execute_pending_job(self) -> bool: ...


@runtime_checkable
class _RuntimeModule(Protocol):
    Context: Callable[[], _Runtime]


@runtime_checkable
class _Receiver(Protocol):
    def __call__(self, kind: str, value: str) -> None: ...


class _Output:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.result: Done | None = None

    def send(self, raw: str) -> None:
        if len(raw.encode("utf-8")) > self.maximum:
            raise ValueError("Code execution message exceeds its byte limit")
        message: Final = WORKER_MESSAGE.validate_python(load_json(raw))
        if isinstance(message, Done):
            self.result = message
        else:
            sys.stdout.buffer.write(encode_message(message, self.maximum))
            sys.stdout.buffer.flush()


def _read_reply(maximum: int) -> Reply:
    line: Final = sys.stdin.buffer.readline(maximum + 1)
    if not line or len(line) > maximum:
        raise ValueError("Invalid parent message")
    return Reply.model_validate(load_json(line.decode("utf-8")))


def _run(start: Start) -> Done:
    module: Final = importlib.import_module("_quickjs")
    if not isinstance(module, _RuntimeModule):
        raise TypeError("Unsupported quickjs-ng runtime")
    runtime: Final = module.Context()
    runtime.set_memory_limit(start.limits.memory_bytes)
    runtime.set_max_stack_size(512 * 1024)
    output: Final = _Output(start.limits.message_bytes)
    runtime.add_callable("__litellm_send", output.send)
    receive: Final = runtime.eval(_BRIDGE)
    if not isinstance(receive, _Receiver):
        raise TypeError("Invalid JavaScript bridge")
    receive("start", start.model_dump_json())
    while output.result is None:
        while runtime.execute_pending_job():
            if output.result is not None:
                return output.result
        if output.result is None:
            receive("reply", _read_reply(start.limits.message_bytes).model_dump_json())
    return output.result


def main() -> None:
    try:
        maximum: Final = CodeExecutionLimits().message_bytes
        line: Final = sys.stdin.buffer.readline(maximum + 1)
        if len(line) > maximum:
            raise ValueError("Code execution start message exceeds its byte limit")
        start: Final = Start.model_validate(load_json(line.decode("utf-8")))
        result: Final = _run(start)
        sys.stdout.buffer.write(encode_message(result, start.limits.message_bytes))
        sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001 - Worker boundary emits only a fixed safe error before parent reaps it.
        sys.stdout.buffer.write(b'{"kind":"done","error":"JavaScript worker failed","output":[]}\n')
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
