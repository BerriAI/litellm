from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue, StrictBool, StrictStr

from ._validator_worker import MAX_MESSAGE_BYTES
from .compiler import escape_diagnostics
from .models import JsonSchema, MontyErrorKind, MontyFailure


class _Response(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    valid: StrictBool
    message: StrictStr


class ValidatorProcess:
    def __init__(self) -> None:
        self.busy = False
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> MontyFailure | None:
        if self.process is not None and self.process.returncode is None:
            return None
        try:
            async with asyncio.timeout(5):
                self.process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("_validator_worker.py")),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env={},
                    limit=65536,
                )
                if self.process.stdout is None or await self.process.stdout.readline() != b"ready\n":
                    await self.close()
                    return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, "validator startup failed")
        except asyncio.CancelledError:
            await self.close()
            raise
        except (OSError, ValueError, TimeoutError) as exc:
            await self.close()
            return MontyFailure(MontyErrorKind.RUNTIME_UNAVAILABLE, escape_diagnostics(str(exc)))
        return None

    async def close(self) -> None:
        process: Final = self.process
        self.process = None
        if process is not None:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()

    async def validate(
        self, value: JsonValue, schema: JsonSchema, *, deadline: float, output: bool = False
    ) -> MontyFailure | None:
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                if (failure := await self.start()) is not None:
                    return failure
                process: Final = self.process
                if process is None or process.stdin is None or process.stdout is None:
                    return MontyFailure(MontyErrorKind.WORKER_CRASHED, "validator pipes unavailable")
                request: Final = json.dumps({"value": value, "schema": schema}, ensure_ascii=True).encode() + b"\n"
                if len(request) > MAX_MESSAGE_BYTES:
                    return MontyFailure(
                        MontyErrorKind.OUTPUT_TOO_LARGE if output else MontyErrorKind.INPUT_TOO_LARGE,
                        "validator request byte limit exceeded",
                    )
                process.stdin.write(request)
                await process.stdin.drain()
                response: Final = _Response.model_validate_json(await process.stdout.readline())
                if not response.valid:
                    return MontyFailure(
                        MontyErrorKind.OUTPUT_INVALID if output else MontyErrorKind.INPUT_INVALID,
                        escape_diagnostics(response.message)[:4096],
                    )
        except asyncio.CancelledError:
            await self.close()
            raise
        except TimeoutError:
            await self.close()
            return MontyFailure(MontyErrorKind.TIMEOUT, "node deadline expired during schema validation")
        except (OSError, ValueError, RecursionError) as exc:
            await self.close()
            return MontyFailure(MontyErrorKind.WORKER_CRASHED, escape_diagnostics(str(exc))[:4096])
        return None
