from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from redis import Redis

    from rq.worker import Worker


WORKERS_SUSPENDED = 'rq:suspended'


def is_suspended(connection: 'Redis', worker: Optional['Worker'] = None):
    """Checks whether a Worker is suspendeed on a given connection
    PS: pipeline returns a list of responses
    Ref: https://github.com/andymccurdy/redis-py#pipelines

    Args:
        connection (Redis): The Redis Connection
        worker (Optional[Worker], optional): The Worker. Defaults to None.
    """
    with connection.pipeline() as pipeline:
        if worker is not None:
            worker.heartbeat(pipeline=pipeline)
        pipeline.exists(WORKERS_SUSPENDED)
        return pipeline.execute()[-1]


def suspend(connection: 'Redis', ttl: Optional[int] = None):
    """
    Suspends.
    TTL of 0 will invalidate right away.

    Args:
        connection (Redis): The Redis connection to use..
        ttl (Optional[int], optional): time to live in seconds. Defaults to `None`
    """
    connection.set(WORKERS_SUSPENDED, 1)
    if ttl is not None:
        connection.expire(WORKERS_SUSPENDED, ttl)


def resume(connection: 'Redis'):
    """
    Resumes.

    Args:
        connection (Redis): The Redis connection to use..
    """
    return connection.delete(WORKERS_SUSPENDED)
