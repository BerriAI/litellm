from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from litellm.redis.health import RedisCircuitBreaker
from litellm.redis.transport.batch import PipelineBatch
from litellm.redis.transport.client import AsyncPubSubClient, AsyncRedisCommands
from litellm.redis.transport.contract import Admit, RedisTransport

if TYPE_CHECKING:
    from litellm.redis.transport.events import RedisServiceEvents


@dataclass(frozen=True, slots=True)
class RedisConnection:
    client: Callable[[], AsyncRedisCommands]
    pubsub_client: Callable[[], AsyncPubSubClient]
    namespace: Callable[[], str | None]
    breaker: Callable[[], RedisCircuitBreaker]
    admit: Admit
    events: Callable[[], "RedisServiceEvents"]
    multi_get: Callable[[Sequence[str]], Awaitable[object]]
    server_version: Callable[[], int]
    request_engine: Callable[[], PipelineBatch | None]
    post_call_engine: Callable[[], PipelineBatch | None]

    def transport(self) -> RedisTransport:
        from litellm.redis.transport.redis_py import RedisPyTransport

        return RedisPyTransport(
            client=self.client,
            pubsub_client=self.pubsub_client,
            namespace=self.namespace(),
            breaker=self.breaker(),
            admit=self.admit,
            events=self.events(),
            multi_get=self.multi_get,
            server_version=self.server_version,
            request_engine=self.request_engine,
            post_call_engine=self.post_call_engine,
        )
