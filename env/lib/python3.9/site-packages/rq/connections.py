import warnings
from contextlib import contextmanager
from typing import Optional, Tuple, Type

from redis import Connection as RedisConnection
from redis import Redis

from .local import LocalStack


class NoRedisConnectionException(Exception):
    pass


@contextmanager
def Connection(connection: Optional['Redis'] = None):  # noqa
    """The context manager for handling connections in a clean way.
    It will push the connection to the LocalStack, and pop the connection
    when leaving the context

    Example:

    ..codeblock:python::

        with Connection():
            w = Worker()
            w.work()

    This method is deprecated on version 1.12.0 and will be removed in the future.
    Pass the connection to the worker explicitly to handle Redis Connections.

    Args:
        connection (Optional[Redis], optional): A Redis Connection instance. Defaults to None.
    """
    warnings.warn(
        "The Connection context manager is deprecated. Use the `connection` parameter instead.",
        DeprecationWarning,
    )
    if connection is None:
        connection = Redis()
    push_connection(connection)
    try:
        yield
    finally:
        popped = pop_connection()
        assert (
            popped == connection
        ), 'Unexpected Redis connection was popped off the stack. Check your Redis connection setup.'


def push_connection(redis: 'Redis'):
    """
    Pushes the given connection to the stack.

    Args:
        redis (Redis): A Redis connection
    """
    warnings.warn(
        "The `push_connection` function is deprecated. Pass the `connection` explicitly instead.",
        DeprecationWarning,
    )
    _connection_stack.push(redis)


def pop_connection() -> 'Redis':
    """
    Pops the topmost connection from the stack.

    Returns:
        redis (Redis): A Redis connection
    """
    warnings.warn(
        "The `pop_connection` function is deprecated. Pass the `connection` explicitly instead.",
        DeprecationWarning,
    )
    return _connection_stack.pop()


def get_current_connection() -> 'Redis':
    """
    Returns the current Redis connection (i.e. the topmost on the
    connection stack).

    Returns:
        Redis: A Redis Connection
    """
    warnings.warn(
        "The `get_current_connection` function is deprecated. Pass the `connection` explicitly instead.",
        DeprecationWarning,
    )
    return _connection_stack.top


def resolve_connection(connection: Optional['Redis'] = None) -> 'Redis':
    """
    Convenience function to resolve the given or the current connection.
    Raises an exception if it cannot resolve a connection now.

    Args:
        connection (Optional[Redis], optional): A Redis connection. Defaults to None.

    Raises:
        NoRedisConnectionException: If connection couldn't be resolved.

    Returns:
        Redis: A Redis Connection
    """
    warnings.warn(
        "The `resolve_connection` function is deprecated. Pass the `connection` explicitly instead.",
        DeprecationWarning,
    )
    if connection is not None:
        return connection

    connection = get_current_connection()
    if connection is None:
        raise NoRedisConnectionException('Could not resolve a Redis connection')
    return connection


def parse_connection(connection: Redis) -> Tuple[Type[Redis], Type[RedisConnection], dict]:
    connection_pool_kwargs = connection.connection_pool.connection_kwargs.copy()
    connection_pool_class = connection.connection_pool.connection_class

    return connection.__class__, connection_pool_class, connection_pool_kwargs


_connection_stack = LocalStack()


__all__ = ['Connection', 'get_current_connection', 'push_connection', 'pop_connection']
