"""
End-to-end reproduction of the aiohttp connector-pool leak that PR #30271 fixed.

The bug: AiohttpResponseStream.__aiter__ had no `finally` block. Streams that
ended abnormally (read timeout, client disconnect, GeneratorExit, CancelledError)
never released their TCPConnector slot. Once the pool was exhausted, every new
request to the same host blocked waiting for a slot and timed out with 408 forever.

This harness:

1. Starts a local aiohttp server that returns a slow chunked stream.
2. Builds a LiteLLMAiohttpTransport backed by a TCPConnector(limit=POOL_LIMIT).
3. Fires POOL_LIMIT streaming requests and cancels each one mid-stream
   (simulates a traffic spike where clients disconnect / time out).
4. Samples len(connector._acquired) (live in-use slots) on a timeline.
5. After the cancellations, fires a normal request with a tight pool wait
   timeout. With the leak, no slot is ever released and this final request
   hangs / errors; with the fix it returns 200 immediately.

The same script is reused for both `--scenario before` (monkey-patches out the
finally block so the bug returns) and `--scenario after` (current source).

Output: a JSON file with the slot-occupancy timeline + recovery probe result.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from typing import Final

import aiohttp
import httpx
from aiohttp import web

REPO_ROOT: Final = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from litellm.llms.custom_httpx import aiohttp_transport
from litellm.llms.custom_httpx.aiohttp_transport import (
    AiohttpResponseStream,
    LiteLLMAiohttpTransport,
)

POOL_LIMIT: Final = 4
LEAKING_REQUESTS: Final = POOL_LIMIT
CHUNK_INTERVAL_S: Final = 0.05
RECOVERY_TIMEOUT_S: Final = 2.0
SAMPLE_INTERVAL_S: Final = 0.02


@dataclass(frozen=True)
class Sample:
    t: float
    acquired: int
    phase: str


@dataclass
class Report:
    scenario: str
    pool_limit: int
    leaking_requests: int
    samples: list[dict]
    recovery_ok: bool
    recovery_elapsed_s: float
    recovery_error: str | None


def install_leaking_aiter() -> None:
    """
    Restore the pre-#30271 behavior of AiohttpResponseStream.__aiter__:
    no finally block, so abnormal exits never call response.close().
    """

    async def leaking_aiter(self):  # type: ignore[no-untyped-def]
        try:
            async for chunk in self._aiohttp_response.content.iter_chunked(
                self.CHUNK_SIZE
            ):
                yield chunk
        except (
            aiohttp.ClientPayloadError,
            aiohttp.client_exceptions.ClientPayloadError,
        ):
            return
        except RuntimeError as e:
            if "Connection closed" in str(e):
                return
            raise
        except aiohttp.http_exceptions.TransferEncodingError:
            return
        except Exception:
            with aiohttp_transport.map_aiohttp_exceptions():
                raise

    AiohttpResponseStream.__aiter__ = leaking_aiter  # type: ignore[assignment]


async def slow_stream_handler(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200, headers={"Content-Type": "application/octet-stream"}
    )
    await response.prepare(request)
    try:
        for i in range(10_000):
            await response.write(f"chunk{i:05d}\n".encode())
            await asyncio.sleep(CHUNK_INTERVAL_S)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


async def run_scenario(scenario: str) -> Report:
    app = web.Application()
    app.router.add_get("/stream", slow_stream_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    url = f"http://127.0.0.1:{port}/stream"

    connector = aiohttp.TCPConnector(limit=POOL_LIMIT, force_close=False)
    session = aiohttp.ClientSession(connector=connector)
    transport = LiteLLMAiohttpTransport(client=session, owns_session=True)

    samples: list[Sample] = []
    sampling = True
    t0 = time.monotonic()
    current_phase = "idle"

    async def sampler() -> None:
        while sampling:
            samples.append(
                Sample(
                    t=time.monotonic() - t0,
                    acquired=len(connector._acquired),
                    phase=current_phase,
                )
            )
            await asyncio.sleep(SAMPLE_INTERVAL_S)

    sampler_task = asyncio.create_task(sampler())

    async def cancel_mid_stream() -> None:
        req = httpx.Request("GET", url)
        req.extensions["timeout"] = {"connect": 5.0, "read": 5.0, "pool": 5.0}
        resp = await transport.handle_async_request(req)
        async for _ in resp.aiter_bytes():
            raise asyncio.CancelledError()

    await asyncio.sleep(0.1)
    current_phase = "leaking"
    for i in range(LEAKING_REQUESTS):
        with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError, Exception):
            await cancel_mid_stream()
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.2)

    current_phase = "recovery_probe"
    probe_req = httpx.Request("GET", url)
    probe_req.extensions["timeout"] = {
        "connect": 1.0,
        "read": 1.0,
        "pool": RECOVERY_TIMEOUT_S,
    }
    recovery_ok = False
    recovery_error: str | None = None
    probe_started = time.monotonic()
    try:
        async with asyncio.timeout(RECOVERY_TIMEOUT_S + 0.5):
            resp = await transport.handle_async_request(probe_req)
            async for _ in resp.aiter_bytes():
                recovery_ok = True
                await resp.aclose()
                break
    except (asyncio.TimeoutError, httpx.HTTPError, Exception) as exc:
        recovery_error = f"{type(exc).__name__}: {exc}"
    recovery_elapsed = time.monotonic() - probe_started

    current_phase = "done"
    await asyncio.sleep(0.1)
    sampling = False
    await sampler_task

    with contextlib.suppress(Exception):
        await transport.aclose()
    with contextlib.suppress(Exception):
        await session.close()
    await runner.cleanup()

    return Report(
        scenario=scenario,
        pool_limit=POOL_LIMIT,
        leaking_requests=LEAKING_REQUESTS,
        samples=[asdict(s) for s in samples],
        recovery_ok=recovery_ok,
        recovery_elapsed_s=recovery_elapsed,
        recovery_error=recovery_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("before", "after"), required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    if args.scenario == "before":
        install_leaking_aiter()

    report = asyncio.run(run_scenario(args.scenario))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2))
    print(
        f"[{report.scenario}] recovery_ok={report.recovery_ok} "
        f"elapsed={report.recovery_elapsed_s:.3f}s "
        f"err={report.recovery_error} "
        f"max_acquired={max(s['acquired'] for s in report.samples)}"
    )


if __name__ == "__main__":
    main()
