from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from pydantic import JsonValue, TypeAdapter

from .compiler import escape_diagnostics
from .models import CompiledProgram, JsonSchema, MontyErrorKind, MontyFailure, MontyLimits, MontySuccess

if TYPE_CHECKING:
    from pydantic_monty import AsyncMonty, AsyncMontySession

_JSON: Final = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class BinaryArtifact:
    path: Path
    sha256: str


def installed_binary() -> BinaryArtifact | MontyFailure:
    try:
        distribution: Final = importlib.metadata.distribution("pydantic-monty-runtime")
        candidates: Final = tuple(file for file in distribution.files or () if file.name in {"monty", "monty.exe"})
        if len(candidates) != 1:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "runtime binary missing or ambiguous")
        file: Final = candidates[0]
        if file.hash is None or file.hash.mode != "sha256":
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "runtime binary has no SHA-256 RECORD hash")
        digest: Final = base64.urlsafe_b64decode(file.hash.value + "=" * (-len(file.hash.value) % 4)).hex()
        return BinaryArtifact(Path(distribution.locate_file(file)).resolve(), digest)
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as exc:
        return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, escape_diagnostics(str(exc)))


def _json_issue(value: object, depth: int, max_depth: int) -> MontyErrorKind | None:
    if depth > max_depth:
        return MontyErrorKind.INPUT_TOO_LARGE
    if value is None or type(value) in (str, int, bool):
        return None
    if type(value) is float:
        return None if math.isfinite(value) else MontyErrorKind.INPUT_INVALID
    if type(value) is list:
        return next((issue for item in value if (issue := _json_issue(item, depth + 1, max_depth)) is not None), None)
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            return MontyErrorKind.INPUT_INVALID
        return next(
            (issue for item in value.values() if (issue := _json_issue(item, depth + 1, max_depth)) is not None), None
        )
    return MontyErrorKind.INPUT_INVALID


def validate_json(
    value: object, schema: JsonSchema, *, max_bytes: int, max_depth: int, output: bool = False
) -> JsonValue | MontyFailure:
    invalid: Final = MontyErrorKind.OUTPUT_INVALID if output else MontyErrorKind.INPUT_INVALID
    too_large: Final = MontyErrorKind.OUTPUT_TOO_LARGE if output else MontyErrorKind.INPUT_TOO_LARGE
    issue: Final = _json_issue(value, 0, max_depth)
    if issue is not None:
        return MontyFailure(
            too_large if issue == MontyErrorKind.INPUT_TOO_LARGE else invalid, "JSON depth or type invalid"
        )
    try:
        encoded: Final = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > max_bytes:
            return MontyFailure(too_large, "JSON byte limit exceeded")
        parsed: Final = _JSON.validate_python(value, strict=True)
        Draft202012Validator(schema).validate(parsed)
        return parsed
    except (ValidationError, ValueError, OverflowError, RecursionError) as exc:
        return MontyFailure(invalid, escape_diagnostics(str(exc))[:4096])


def _kill(session: AsyncMontySession) -> None:
    pid: Final = session.worker_pid
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


class MontyRuntime:
    def __init__(self, limits: MontyLimits, *, binary: BinaryArtifact | None = None) -> None:
        self.limits = limits
        self._binary = binary
        self._pool: AsyncMonty | None = None
        self._slots = asyncio.Semaphore(limits.max_concurrent_workers)
        self._admitted = 0

    async def start(self) -> MontyFailure | None:
        if self._pool is not None:
            return None
        try:
            from pydantic_monty import AsyncMonty

            artifact: Final = self._binary if self._binary is not None else installed_binary()
            if isinstance(artifact, MontyFailure):
                return artifact
            if hashlib.sha256(artifact.path.read_bytes()).hexdigest() != artifact.sha256:
                return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "runtime binary SHA-256 mismatch")
            pool: Final = AsyncMonty(
                binary_path=artifact.path,
                min_processes=0,
                max_processes=self.limits.max_concurrent_workers,
                max_checkouts_per_worker=1,
                checkout_timeout=self.limits.feed_seconds,
                request_timeout=self.limits.feed_seconds + 1,
                feed_duration_limit_grace=0.1,
            )
            await pool.__aenter__()
            self._pool = pool
            return None
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, escape_diagnostics(str(exc)))

    async def close(self) -> None:
        pool: Final = self._pool
        self._pool = None
        if pool is not None:
            await pool.__aexit__(None, None, None)

    async def run_program(
        self, compiled: CompiledProgram, input: object, *, deadline_monotonic: float
    ) -> MontySuccess | MontyFailure:
        started: Final = time.monotonic()
        if started >= deadline_monotonic:
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired")
        if self._pool is None:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "Monty runtime is not started")
        if self._admitted >= self.limits.max_concurrent_workers + self.limits.max_queued_waiters:
            return MontyFailure(MontyErrorKind.QUEUE_FULL, "Monty execution queue is full")
        self._admitted += 1
        try:
            async with asyncio.timeout_at(asyncio.get_running_loop().time() + deadline_monotonic - time.monotonic()):
                async with self._slots:
                    validated: Final = validate_json(
                        input,
                        compiled.input_schema,
                        max_bytes=self.limits.max_input_bytes,
                        max_depth=self.limits.max_input_depth,
                    )
                    if isinstance(validated, MontyFailure):
                        return validated
                    return await self._execute(compiled, validated, deadline_monotonic, started)
        except TimeoutError:
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired")
        finally:
            self._admitted -= 1

    async def _execute(
        self, compiled: CompiledProgram, input: JsonValue, deadline: float, started: float
    ) -> MontySuccess | MontyFailure:
        from pydantic_monty import CollectString, MontyCrashedError, MontyRuntimeError, MontySyntaxError, ResourceLimits

        pool: Final = self._pool
        if pool is None:
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "Monty runtime is closed")
        remaining: Final = deadline - time.monotonic()
        if remaining <= 0:
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired")
        collector: Final = CollectString(max_bytes=self.limits.max_stdout_bytes)
        session: Final = pool.checkout(
            script_name="workflow.py",
            limits=ResourceLimits(
                max_feed_duration_secs=min(self.limits.feed_seconds, remaining),
                max_memory=self.limits.max_memory,
                max_recursion_depth=self.limits.max_recursion_depth,
            ),
            os_policy={"sleep": "zero"},
        )
        try:
            async with session:
                try:
                    if sys.platform == "linux":
                        import resource

                        pid: Final = session.worker_pid
                        if pid is None:
                            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "worker PID unavailable")
                        try:
                            resource.prlimit(pid, resource.RLIMIT_AS, (self.limits.worker_address_space_bytes,) * 2)
                        except (OSError, ValueError) as exc:
                            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, escape_diagnostics(str(exc)))
                    result: Final[object] = await session.feed_run(
                        compiled.sandbox_source,
                        inputs={"_liteflow_input": input},
                        print_callback=collector,
                    )
                    if not isinstance(result, list) or len(result) != 2 or result[0] is not True:
                        return MontyFailure(MontyErrorKind.OUTPUT_INVALID, "return value is not strict JSON")
                    output: Final = validate_json(
                        result[1],
                        compiled.output_schema,
                        max_bytes=self.limits.max_output_bytes,
                        max_depth=self.limits.max_output_depth,
                        output=True,
                    )
                    if isinstance(output, MontyFailure):
                        return output
                    if time.monotonic() >= deadline:
                        return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired during validation")
                    return MontySuccess(
                        output, escape_diagnostics(collector.output), int((time.monotonic() - started) * 1000)
                    )
                except asyncio.CancelledError:
                    _kill(session)
                    raise
        except MontyRuntimeError as exc:
            error: Final = exc.exception()
            message: Final = escape_diagnostics(exc.display())[:4096]
            if isinstance(error, TimeoutError):
                return MontyFailure(MontyErrorKind.TIMEOUT, message)
            if isinstance(error, MemoryError):
                kind: Final = MontyErrorKind.STDOUT_LIMIT if "bytes >" in str(error) else MontyErrorKind.MEMORY_LIMIT
                return MontyFailure(kind, message)
            if isinstance(error, RecursionError):
                return MontyFailure(MontyErrorKind.RECURSION_LIMIT, message)
            return MontyFailure(MontyErrorKind.RUNTIME_ERROR, message)
        except MontyCrashedError as exc:
            return MontyFailure(
                MontyErrorKind.TIMEOUT if exc.timed_out else MontyErrorKind.WORKER_CRASHED,
                escape_diagnostics(str(exc))[:4096],
            )
        except MontySyntaxError as exc:
            return MontyFailure(MontyErrorKind.INVALID_PROGRAM, escape_diagnostics(str(exc))[:4096])
        except (ValueError, OSError) as exc:
            return MontyFailure(MontyErrorKind.RUNTIME_ERROR, escape_diagnostics(str(exc))[:4096])

    async def evaluate_predicate(
        self, compiled: CompiledProgram, input: object, *, deadline_monotonic: float
    ) -> bool | MontyFailure:
        result: Final = await self.run_program(compiled, input, deadline_monotonic=deadline_monotonic)
        if isinstance(result, MontyFailure):
            return result
        if type(result.output) is not bool:
            return MontyFailure(MontyErrorKind.PREDICATE_NOT_BOOL, "predicate must return exactly bool")
        return result.output
