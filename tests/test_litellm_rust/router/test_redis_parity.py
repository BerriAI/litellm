"""Both router backends write the same Redis keys, values and TTLs for the same failures."""

from __future__ import annotations

import random
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Generator, Mapping, Sequence
from typing import Final

import pytest
import redis

import litellm
from litellm.caching.redis_cache import RedisCache
from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_router import NATIVE_ROUTER, RustRouter
from litellm.types.router import RouterRateLimitError

pytestmark = pytest.mark.requires_rust_extension

MESSAGES: Final = ({"role": "user", "content": "hi"},)
_MINUTE: Final = re.compile(r":\d{2}-\d{2}$")
_TIMESTAMP: Final = re.compile(r"'timestamp': [0-9.e+-]+")


@pytest.fixture(scope="module")
def redis_port() -> Generator[int]:
    server: Final = shutil.which("redis-server")
    if server is None:
        pytest.skip("redis-server is not installed")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: Final = int(probe.getsockname()[1])
    process: Final = subprocess.Popen(
        [server, "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client: Final = redis.Redis(port=port)
    deadline: Final = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            client.ping()
            break
        except redis.ConnectionError:
            time.sleep(0.05)
    yield port
    process.terminate()
    process.wait(timeout=10)


@pytest.fixture
def store(redis_port: int) -> redis.Redis:
    client: Final = redis.Redis(port=redis_port, decode_responses=True)
    client.flushdb()
    return client


def _deployment(name: str, response: str) -> dict[str, object]:  # mutable-ok: PythonRouter consumes model_list dicts
    return {
        "model_name": "g",
        "litellm_params": {"model": f"openai/{name}", "api_key": "k", "mock_response": response},
        "model_info": {"id": name},
    }


def _snapshot(client: redis.Redis) -> Mapping[str, tuple[str, int]]:
    """Every key with its decoded value and TTL; volatile parts (timestamps, the UTC minute) normalized."""
    keys: Final = sorted(str(key) for key in client.keys("*"))
    return {_MINUTE.sub(":<minute>", key): (_stable(str(client.get(key))), int(client.ttl(key))) for key in keys}


def _stable(raw: str) -> str:
    """The stored text as is, with only the write time masked, so encoding differences still show."""
    return _TIMESTAMP.sub("'timestamp': <time>", raw)


async def _run(backend: object, calls: int) -> Sequence[str]:
    outcomes: Final[list[str]] = []  # mutable-ok: one outcome per call
    for _ in range(calls):
        try:
            await backend.acompletion("g", MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # both backends serve acompletion
            outcomes.append("ok")
        except RouterRateLimitError:
            outcomes.append("rejected")
        except litellm.RateLimitError:
            outcomes.append("rate_limited")
        except litellm.InternalServerError:
            outcomes.append("server_error")
    return outcomes


def _arguments(
    port: int, responses: Sequence[str], extra: Mapping[str, object]
) -> dict[str, object]:  # mutable-ok: Router(...) keyword arguments
    return {
        "model_list": [_deployment(f"d{index}", response) for index, response in enumerate(responses)],
        "redis_host": "127.0.0.1",
        "redis_port": port,
        "num_retries": 0,
        **extra,
    }


@pytest.mark.parametrize(
    ("responses", "extra", "calls"),
    (
        pytest.param(("litellm.RateLimitError", "litellm.RateLimitError"), {}, 3, id="rate-limit-cooldowns"),
        pytest.param(
            ("litellm.InternalServerError",), {"allowed_fails": 1, "cooldown_time": 30}, 3, id="allowed-fails"
        ),
        pytest.param(("litellm.InternalServerError", "ok"), {}, 4, id="failure-usage-only"),
    ),
)
async def test_both_backends_leave_the_same_redis_state(
    store: redis.Redis, redis_port: int, responses: Sequence[str], extra: Mapping[str, object], calls: int
) -> None:
    arguments: Final = _arguments(redis_port, responses, extra)
    random.seed(0)
    python_router: Final = PythonRouter(**arguments)
    python_outcomes: Final = await _run(python_router, calls)
    python_router.discard()
    python_state: Final = _snapshot(store)
    store.flushdb()

    native: Final = NATIVE_ROUTER.load()
    assert native is not None
    rust_outcomes: Final = await _run(RustRouter(_arguments(redis_port, responses, extra), native, 0), calls)
    rust_state: Final = _snapshot(store)

    assert rust_outcomes == python_outcomes
    assert python_state
    assert rust_state == python_state


async def test_a_rust_router_honors_cooldowns_a_python_router_wrote(store: redis.Redis, redis_port: int) -> None:
    responses: Final = ("litellm.RateLimitError", "litellm.RateLimitError")
    python_router: Final = PythonRouter(**_arguments(redis_port, responses, {}))
    assert await _run(python_router, 2) == ["rate_limited", "rate_limited"]
    python_router.discard()
    native: Final = NATIVE_ROUTER.load()
    assert native is not None

    rust_outcomes: Final = await _run(RustRouter(_arguments(redis_port, responses, {}), native, 0), 1)

    assert rust_outcomes == ["rejected"]


async def test_a_redis_attached_after_construction_carries_the_rust_routers_cooldowns(
    store: redis.Redis, redis_port: int
) -> None:
    """The proxy builds its router without Redis and attaches its own (`_update_redis_cache`) at startup."""
    responses: Final = ("litellm.RateLimitError", "litellm.RateLimitError")
    native: Final = NATIVE_ROUTER.load()
    assert native is not None
    without_redis: Final = {
        key: value for key, value in _arguments(redis_port, responses, {}).items() if not key.startswith("redis_")
    }
    rust_router: Final = RustRouter(without_redis, native, 0)

    rust_router._update_redis_cache(RedisCache(host="127.0.0.1", port=redis_port))  # pyright: ignore[reportPrivateUsage]  # the proxy's own hand-off
    rust_outcomes: Final = await _run(rust_router, 2)
    python_router: Final = PythonRouter(**_arguments(redis_port, responses, {}))
    python_outcomes: Final = await _run(python_router, 1)
    python_router.discard()

    assert rust_outcomes == ["rate_limited", "rate_limited"]
    assert sorted(key for key in store.keys("deployment:*:cooldown")) == [
        "deployment:d0:cooldown",
        "deployment:d1:cooldown",
    ]
    assert python_outcomes == ["rejected"]
    assert rust_router.cache.redis_cache is not None  # pyright: ignore[reportAttributeAccessIssue]  # a Router view
