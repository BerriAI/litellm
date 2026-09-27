"""In-memory Redis doubles for RedisBatch/RedisRequestPlan tests, injected through a RedisCache subclass."""

import json
from typing import Final

from redis.asyncio import RedisCluster

from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker


class RecordingPipeline:
    """Queues commands like a redis-py pipeline and replays scripted results on execute()."""

    def __init__(self, results: list | None = None, error: BaseException | None = None) -> None:
        self.calls: list[tuple] = []
        self.scripts: set = set()
        self.results = list(results or [])
        self.error = error
        self.execute_count = 0

    def mget(self, keys: list) -> "RecordingPipeline":
        self.calls.append(("mget", tuple(keys)))
        return self

    def get(self, name: str) -> "RecordingPipeline":
        self.calls.append(("get", name))
        return self

    def set(self, name: str, value: object, ex: object = None, **kwargs: object) -> "RecordingPipeline":
        self.calls.append(("set", name, value, ex))
        return self

    def incrbyfloat(self, name: str, amount: float) -> "RecordingPipeline":
        self.calls.append(("incrbyfloat", name, amount))
        return self

    def expire(self, name: str, time: int) -> "RecordingPipeline":
        self.calls.append(("expire", name, time))
        return self

    def delete(self, name: str) -> "RecordingPipeline":
        self.calls.append(("delete", name))
        return self

    def evalsha(self, sha: str, numkeys: int, *keys_and_args: object) -> "RecordingPipeline":
        self.calls.append(("evalsha", sha, numkeys, tuple(keys_and_args)))
        return self

    async def execute(self, raise_on_error: bool = True) -> list:
        self.execute_count += 1
        if self.error is not None:
            raise self.error
        return self.results


class RecordingRedisClient:
    """init_async_client() stand-in: hands out a prebuilt RecordingPipeline per call."""

    def __init__(self, pipelines: list[RecordingPipeline]) -> None:
        self.pipelines = pipelines
        self.pipeline_calls = 0
        self.registered_scripts: list[str] = []

    def pipeline(self, transaction: bool = True) -> RecordingPipeline:
        self.pipeline_calls += 1
        return self.pipelines[min(self.pipeline_calls - 1, len(self.pipelines) - 1)]

    def register_script(self, script: str) -> "FakeAsyncScript":
        self.registered_scripts.append(script)
        return FakeAsyncScript(sha=f"sha-{len(self.registered_scripts)}")


class FakeClusterClient(RedisCluster):
    """Passes the isinstance(client, RedisCluster) check without ever connecting."""

    def __init__(self, pipelines: list[RecordingPipeline]) -> None:
        self._delegate: Final = RecordingRedisClient(pipelines)

    def pipeline(self, transaction: bool = True) -> RecordingPipeline:
        return self._delegate.pipeline(transaction=transaction)

    def register_script(self, script: str) -> "FakeAsyncScript":
        return self._delegate.register_script(script)


class FakeAsyncScript:
    def __init__(self, sha: str) -> None:
        self.sha = sha


class RecordingServiceLogger:
    """service_logger_obj double that records the hooks RedisBatch fires."""

    def __init__(self) -> None:
        self.successes: list[str] = []
        self.failures: list[str] = []

    async def async_service_success_hook(self, call_type: str, **kwargs: object) -> None:
        self.successes.append(call_type)

    async def async_service_failure_hook(self, call_type: str, **kwargs: object) -> None:
        self.failures.append(call_type)


class FakeRedisCache(RedisCache):
    """A RedisCache whose Redis client is an in-test recorder; no sockets, no monkeypatching."""

    def __init__(self, client: RecordingRedisClient, namespace: str | None = None) -> None:
        self._fake_client: Final = client
        self.namespace = namespace
        self._circuit_breaker: Final = RedisCircuitBreaker(
            failure_threshold=2, recovery_timeout=60, enabled=True
        )
        self.service_logger_obj: Final = RecordingServiceLogger()

    def init_async_client(self) -> RecordingRedisClient:
        return self._fake_client

    def get_ttl(self, **kwargs: object) -> int | None:
        return None


def json_value(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")
