import asyncio
import importlib.metadata
import importlib.util
from collections.abc import Awaitable, Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Barrier
from typing import Final, Protocol, cast


class Encodable(Protocol):
    def encode(self) -> object: ...


class NativeBench(Protocol):
    def Payload(self, value: object) -> Encodable: ...
    def TypedResponse(self, value: object) -> Encodable: ...
    def decode(self, value: object) -> bool: ...
    def roundtrip(self, value: object) -> object: ...
    def gil_roundtrip(self) -> bool: ...
    def echo(self, body: object) -> object: ...
    def aecho(self, body: object) -> Awaitable[object]: ...


class NativeModule(Protocol):
    _bench: NativeBench


def extension_path() -> Path:
    package: Final = importlib.metadata.distribution("litellm")
    matches: Final = tuple(
        package.locate_file(file)
        for file in package.files or ()
        if file.name.startswith("_native.") and file.suffix in (".so", ".pyd")
    )
    if len(matches) != 1:
        raise RuntimeError("Install exactly one release wheel containing the native extension")
    return Path(str(matches[0]))


def load_native() -> NativeBench:
    spec: Final = importlib.util.spec_from_file_location("_native", extension_path())
    if spec is None or spec.loader is None:
        raise RuntimeError("Native extension cannot be loaded")
    module: Final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(NativeModule, cast(object, module))._bench  # pyright: ignore[reportPrivateUsage]  # Private, benchmark-only extension API


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    call: Callable[[], object]
    expected: object


def payloads() -> tuple[tuple[str, object], ...]:
    return (
        ("empty-object", {}),
        ("stream-chunk", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}),
        (
            "messages-tools",
            {
                "messages": [{"role": "user", "content": "서울 café"}],
                "tools": [
                    {"name": "weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
                ],
                "max_tokens": 128,
            },
        ),
        *(
            (f"image-{label}", {"type": "image_url", "image_url": "data:image/png;base64," + "A" * size})
            for label, size in (
                ("1KiB", 1024),
                ("64KiB", 64 * 1024),
                ("1MiB", 1024 * 1024),
                ("4MiB", 4 * 1024 * 1024),
                ("16MiB", 16 * 1024 * 1024),
            )
        ),
    )


async def async_batch(native: NativeBench, payload: object, concurrency: int) -> tuple[object, ...]:
    return tuple(await asyncio.gather(*(native.aecho(payload) for _ in range(concurrency))))


def run_async(loop: asyncio.AbstractEventLoop, native: NativeBench, payload: object, concurrency: int) -> object:
    return loop.run_until_complete(async_batch(native, payload, concurrency))


def sync_batch(pool: ThreadPoolExecutor, native: NativeBench, payload: object) -> object:
    futures: Final = tuple(pool.submit(native.echo, payload) for _ in range(4))
    return tuple(future.result(timeout=30) for future in futures)


@contextmanager
def boundary_cases() -> Generator[tuple[Case, ...]]:
    native: Final = load_native()
    loop: Final = asyncio.new_event_loop()
    fixtures: Final = payloads()
    response: Final = {
        "id": "msg_bench",
        "type": "message",
        "role": "assistant",
        "model": "benchmark",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            barrier: Final = Barrier(4)
            workers: Final = tuple(pool.submit(barrier.wait, 30) for _ in range(4))
            for worker in workers:
                worker.result(timeout=30)
            cases: Final = (
                *(
                    case
                    for label, payload in fixtures
                    for case in (
                        Case(f"decode/{label}", partial(native.decode, payload), True),
                        Case(f"encode/{label}", native.Payload(payload).encode, payload),
                        Case(f"roundtrip/{label}", partial(native.roundtrip, payload), payload),
                        Case(f"sync/{label}", partial(native.echo, payload), payload),
                        Case(f"async/{label}", partial(run_async, loop, native, payload, 1), (payload,)),
                    )
                ),
                Case("typed_response", native.TypedResponse(response).encode, response),
                Case("gil_roundtrip", native.gil_roundtrip, True),
                Case("sync_threads/4", partial(sync_batch, pool, native, fixtures[2][1]), (fixtures[2][1],) * 4),
                *(
                    Case(
                        f"async_concurrent/{count}",
                        partial(run_async, loop, native, fixtures[2][1], count),
                        (fixtures[2][1],) * count,
                    )
                    for count in (8, 32)
                ),
            )
            for case in cases:
                assert case.call() == case.expected, case.name
            yield cases
    finally:
        loop.close()
