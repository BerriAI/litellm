# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

# s/o [@Frank Colson](https://www.linkedin.com/in/frank-colson-422b9b183/) for this redis implementation
import os
import inspect
import redis, litellm
import redis.asyncio as async_redis
from typing import List, Optional


def _get_redis_kwargs():
    arg_spec = inspect.getfullargspec(redis.Redis)

    # Only allow primitive arguments
    exclude_args = {
        "self",
        "connection_pool",
        "retry",
    }

    include_args = ["url"]

    available_args = [x for x in arg_spec.args if x not in exclude_args] + include_args

    return available_args


def _get_redis_env_kwarg_mapping():
    PREFIX = "REDIS_"

    return {f"{PREFIX}{x.upper()}": x for x in _get_redis_kwargs()}


def _redis_kwargs_from_environment():
    mapping = _get_redis_env_kwarg_mapping()

    return_dict = {}
    for k, v in mapping.items():
        value = litellm.get_secret(k, default_value=None)  # check os.environ/key vault
        if value is not None:
            return_dict[v] = value
    return return_dict


def get_redis_url_from_environment():
    if "REDIS_URL" in os.environ:
        return os.environ["REDIS_URL"]

    if "REDIS_HOST" not in os.environ or "REDIS_PORT" not in os.environ:
        raise ValueError(
            "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified for Redis."
        )

    if "REDIS_PASSWORD" in os.environ:
        redis_password = f":{os.environ['REDIS_PASSWORD']}@"
    else:
        redis_password = ""

    return (
        f"redis://{redis_password}{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}"
    )


def _get_redis_client_logic(**env_overrides):
    """
    Common functionality across sync + async redis client implementations
    """
    ### check if "os.environ/<key-name>" passed in
    for k, v in env_overrides.items():
        if isinstance(v, str) and v.startswith("os.environ/"):
            v = v.replace("os.environ/", "")
            value = litellm.get_secret(v)
            env_overrides[k] = value

    redis_kwargs = {
        **_redis_kwargs_from_environment(),
        **env_overrides,
    }

    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        redis_kwargs.pop("host", None)
        redis_kwargs.pop("port", None)
        redis_kwargs.pop("db", None)
        redis_kwargs.pop("password", None)
    elif "host" not in redis_kwargs or redis_kwargs["host"] is None:
        raise ValueError("Either 'host' or 'url' must be specified for redis.")
    litellm.print_verbose(f"redis_kwargs: {redis_kwargs}")
    return redis_kwargs


def get_redis_client(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        redis_kwargs.pop(
            "connection_pool", None
        )  # redis.from_url doesn't support setting your own connection pool
        return redis.Redis.from_url(**redis_kwargs)
    return redis.Redis(**redis_kwargs)


def get_redis_async_client(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        redis_kwargs.pop(
            "connection_pool", None
        )  # redis.from_url doesn't support setting your own connection pool
        return async_redis.Redis.from_url(**redis_kwargs)
    return async_redis.Redis(
        socket_timeout=5,
        **redis_kwargs,
    )


def get_redis_connection_pool(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        return async_redis.BlockingConnectionPool.from_url(
            timeout=5, url=redis_kwargs["url"]
        )
    return async_redis.BlockingConnectionPool(timeout=5, **redis_kwargs)
