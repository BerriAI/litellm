# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import inspect
import json

# s/o [@Frank Colson](https://www.linkedin.com/in/frank-colson-422b9b183/) for this redis implementation
import os
from typing import List, Optional

import redis  # type: ignore
import redis.asyncio as async_redis  # type: ignore

import litellm

from ._logging import verbose_logger


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


def _get_redis_url_kwargs(client=None):
    if client is None:
        client = redis.Redis.from_url
    arg_spec = inspect.getfullargspec(redis.Redis.from_url)

    # Only allow primitive arguments
    exclude_args = {
        "self",
        "connection_pool",
        "retry",
    }

    include_args = ["url"]

    available_args = [x for x in arg_spec.args if x not in exclude_args] + include_args

    return available_args


def _get_redis_cluster_kwargs(client=None):
    if client is None:
        client = redis.Redis.from_url
    arg_spec = inspect.getfullargspec(redis.RedisCluster)

    # Only allow primitive arguments
    exclude_args = {"self", "connection_pool", "retry", "host", "port", "startup_nodes"}

    available_args = [x for x in arg_spec.args if x not in exclude_args]
    available_args.append("password")

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

    _startup_nodes = redis_kwargs.get("startup_nodes", None) or litellm.get_secret(
        "REDIS_CLUSTER_NODES"
    )

    if _startup_nodes is not None:
        redis_kwargs["startup_nodes"] = json.loads(_startup_nodes)

    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        redis_kwargs.pop("host", None)
        redis_kwargs.pop("port", None)
        redis_kwargs.pop("db", None)
        redis_kwargs.pop("password", None)
    elif "startup_nodes" in redis_kwargs and redis_kwargs["startup_nodes"] is not None:
        pass
    elif "host" not in redis_kwargs or redis_kwargs["host"] is None:
        raise ValueError("Either 'host' or 'url' must be specified for redis.")
    # litellm.print_verbose(f"redis_kwargs: {redis_kwargs}")
    return redis_kwargs


def init_redis_cluster(redis_kwargs) -> redis.RedisCluster:
    _redis_cluster_nodes_in_env = litellm.get_secret("REDIS_CLUSTER_NODES")
    if _redis_cluster_nodes_in_env is not None:
        try:
            redis_kwargs["startup_nodes"] = json.loads(_redis_cluster_nodes_in_env)
        except json.JSONDecodeError:
            raise ValueError(
                "REDIS_CLUSTER_NODES environment variable is not valid JSON. Please ensure it's properly formatted."
            )

    verbose_logger.debug(
        "init_redis_cluster: startup nodes: ", redis_kwargs["startup_nodes"]
    )
    from redis.cluster import ClusterNode

    args = _get_redis_cluster_kwargs()
    cluster_kwargs = {}
    for arg in redis_kwargs:
        if arg in args:
            cluster_kwargs[arg] = redis_kwargs[arg]

    new_startup_nodes: List[ClusterNode] = []

    for item in redis_kwargs["startup_nodes"]:
        new_startup_nodes.append(ClusterNode(**item))

    redis_kwargs.pop("startup_nodes")
    return redis.RedisCluster(startup_nodes=new_startup_nodes, **cluster_kwargs)


def get_redis_client(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        args = _get_redis_url_kwargs()
        url_kwargs = {}
        for arg in redis_kwargs:
            if arg in args:
                url_kwargs[arg] = redis_kwargs[arg]

        return redis.Redis.from_url(**url_kwargs)

    if (
        "startup_nodes" in redis_kwargs
        or litellm.get_secret("REDIS_CLUSTER_NODES") is not None
    ):
        return init_redis_cluster(redis_kwargs)

    return redis.Redis(**redis_kwargs)


def get_redis_async_client(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        args = _get_redis_url_kwargs(client=async_redis.Redis.from_url)
        url_kwargs = {}
        for arg in redis_kwargs:
            if arg in args:
                url_kwargs[arg] = redis_kwargs[arg]
            else:
                litellm.print_verbose(
                    "REDIS: ignoring argument: {}. Not an allowed async_redis.Redis.from_url arg.".format(
                        arg
                    )
                )
        return async_redis.Redis.from_url(**url_kwargs)

    if "startup_nodes" in redis_kwargs:
        from redis.cluster import ClusterNode

        args = _get_redis_cluster_kwargs()
        cluster_kwargs = {}
        for arg in redis_kwargs:
            if arg in args:
                cluster_kwargs[arg] = redis_kwargs[arg]

        new_startup_nodes: List[ClusterNode] = []

        for item in redis_kwargs["startup_nodes"]:
            new_startup_nodes.append(ClusterNode(**item))
        redis_kwargs.pop("startup_nodes")
        return async_redis.RedisCluster(
            startup_nodes=new_startup_nodes, **cluster_kwargs
        )

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
    connection_class = async_redis.Connection
    if "ssl" in redis_kwargs and redis_kwargs["ssl"] is not None:
        connection_class = async_redis.SSLConnection
        redis_kwargs.pop("ssl", None)
        redis_kwargs["connection_class"] = connection_class
    redis_kwargs.pop("startup_nodes", None)
    return async_redis.BlockingConnectionPool(timeout=5, **redis_kwargs)
