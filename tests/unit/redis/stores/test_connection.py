import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final


from litellm.rust_bridge._native import NativeRedisClient, native_redis_breaker_states


def test_concurrent_first_connects_share_one_client_without_deadlocking() -> None:
    before: Final = len(native_redis_breaker_states())
    namespace: Final = f"concurrent-connect-{uuid.uuid4()}"
    pool: Final = ThreadPoolExecutor(max_workers=8)
    futures: Final = tuple(
        pool.submit(
            NativeRedisClient.connect,
            {"host": "127.0.0.1", "port": 1},
            namespace=namespace,
            breaker=(5, 60.0, 5.0),
        )
        for _ in range(8)
    )
    try:
        clients: Final = tuple(future.result(timeout=30) for future in futures)
    finally:
        pool.shutdown(wait=False)

    assert all(client is not None for client in clients)
    assert len(native_redis_breaker_states()) == before + 1
