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
from typing import List, Optional, Union

import redis  # type: ignore
import redis.asyncio as async_redis  # type: ignore

from litellm import get_secret, get_secret_str
from litellm.constants import REDIS_CONNECTION_POOL_TIMEOUT, REDIS_SOCKET_TIMEOUT

from ._logging import verbose_logger

# SSL Parameters - centralized list
SSL_PARAMS = [
    "ssl", "ssl_cert_reqs", "ssl_ca_certs", "ssl_certfile", 
    "ssl_keyfile", "ssl_check_hostname", "ssl_ca_cert_dir", 
    "ssl_ciphers", "ssl_crlfile"
]

def _get_redis_kwargs():
    arg_spec = inspect.getfullargspec(redis.Redis)
    exclude_args = {"self", "connection_pool", "retry"}
    available_args = [x for x in arg_spec.args if x not in exclude_args]
    return available_args + ["url"] + SSL_PARAMS

def _get_redis_url_kwargs(client=None):
    if client is None:
        client = redis.Redis.from_url
    arg_spec = inspect.getfullargspec(client)
    exclude_args = {"self", "connection_pool", "retry"}
    available_args = [x for x in arg_spec.args if x not in exclude_args]
    return available_args + ["url"] + SSL_PARAMS

def _get_redis_cluster_kwargs(client=None):
    if client is None:
        client = redis.Redis.from_url
    arg_spec = inspect.getfullargspec(redis.RedisCluster)
    exclude_args = {"self", "connection_pool", "retry", "host", "port", "startup_nodes"}
    available_args = [x for x in arg_spec.args if x not in exclude_args]
    return available_args + ["password", "username"] + SSL_PARAMS

def _get_redis_env_kwarg_mapping():
    """Simple environment variable mapping"""
    redis_kwargs = _get_redis_kwargs()
    mapping = {f"REDIS_{x.upper()}": x for x in redis_kwargs}
    return mapping

def _redis_kwargs_from_environment():
    """Get Redis parameters from environment variables"""
    mapping = _get_redis_env_kwarg_mapping()
    return_dict = {}
    
    for env_var, param_name in mapping.items():
        value = get_secret(env_var, default_value=None)
        if value is not None:
            # Convert string booleans to actual booleans for SSL params
            if param_name in ["ssl", "ssl_check_hostname"] and isinstance(value, str):
                return_dict[param_name] = value.lower() in ["true", "1", "yes", "on"]
            else:
                return_dict[param_name] = value
    
    return return_dict

def _extract_ssl_params(redis_kwargs):
    """Extract SSL parameters from kwargs"""
    return {k: v for k, v in redis_kwargs.items() if k in SSL_PARAMS}

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
    """Common functionality across sync + async redis client implementations"""
    # Handle os.environ/ references
    for k, v in env_overrides.items():
        if isinstance(v, str) and v.startswith("os.environ/"):
            v = v.replace("os.environ/", "")
            value = get_secret(v)
            env_overrides[k] = value

    redis_kwargs = {**_redis_kwargs_from_environment(), **env_overrides}

    # Handle special parameters
    _startup_nodes = redis_kwargs.get("startup_nodes") or get_secret("REDIS_CLUSTER_NODES")
    if _startup_nodes and isinstance(_startup_nodes, str):
        redis_kwargs["startup_nodes"] = json.loads(_startup_nodes)

    _sentinel_nodes = redis_kwargs.get("sentinel_nodes") or get_secret("REDIS_SENTINEL_NODES")
    if _sentinel_nodes and isinstance(_sentinel_nodes, str):
        redis_kwargs["sentinel_nodes"] = json.loads(_sentinel_nodes)

    _sentinel_password = redis_kwargs.get("sentinel_password") or get_secret_str("REDIS_SENTINEL_PASSWORD")
    if _sentinel_password:
        redis_kwargs["sentinel_password"] = _sentinel_password

    _service_name = redis_kwargs.get("service_name") or get_secret("REDIS_SERVICE_NAME")
    if _service_name:
        redis_kwargs["service_name"] = _service_name

    # Clean up conflicting parameters
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        redis_kwargs.pop("host", None)
        redis_kwargs.pop("port", None)
        redis_kwargs.pop("db", None)
        redis_kwargs.pop("password", None)
    elif "startup_nodes" in redis_kwargs and redis_kwargs["startup_nodes"] is not None:
        pass
    elif "sentinel_nodes" in redis_kwargs and redis_kwargs["sentinel_nodes"] is not None:
        pass
    elif "host" not in redis_kwargs or redis_kwargs["host"] is None:
        raise ValueError("Either 'host' or 'url' must be specified for redis.")

    return redis_kwargs

def init_redis_cluster(redis_kwargs) -> redis.RedisCluster:
    _redis_cluster_nodes_in_env = get_secret("REDIS_CLUSTER_NODES")
    if _redis_cluster_nodes_in_env is not None and isinstance(_redis_cluster_nodes_in_env, str):
        try:
            redis_kwargs["startup_nodes"] = json.loads(_redis_cluster_nodes_in_env)
        except json.JSONDecodeError:
            raise ValueError(
                "REDIS_CLUSTER_NODES environment variable is not valid JSON. Please ensure it's properly formatted."
            )

    verbose_logger.debug("init_redis_cluster: startup nodes are being initialized.")
    from redis.cluster import ClusterNode

    args = _get_redis_cluster_kwargs()
    cluster_kwargs = {arg: redis_kwargs[arg] for arg in redis_kwargs if arg in args}

    new_startup_nodes = [ClusterNode(**item) for item in redis_kwargs["startup_nodes"]]
    redis_kwargs.pop("startup_nodes")
    
    return redis.RedisCluster(startup_nodes=new_startup_nodes, **cluster_kwargs)

def _init_redis_sentinel(redis_kwargs) -> redis.Redis:
    sentinel_nodes = redis_kwargs.get("sentinel_nodes")
    sentinel_password = redis_kwargs.get("sentinel_password")
    service_name = redis_kwargs.get("service_name")

    if not sentinel_nodes or not service_name:
        raise ValueError(
            "Both 'sentinel_nodes' and 'service_name' are required for Redis Sentinel."
        )

    verbose_logger.debug("init_redis_sentinel: sentinel nodes are being initialized.")

    # Pass SSL parameters to Sentinel
    ssl_kwargs = _extract_ssl_params(redis_kwargs)
    sentinel = redis.Sentinel(
        sentinel_nodes,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        password=sentinel_password,
        **ssl_kwargs
    )

    return sentinel.master_for(service_name)

def _init_async_redis_sentinel(redis_kwargs) -> async_redis.Redis:
    sentinel_nodes = redis_kwargs.get("sentinel_nodes")
    sentinel_password = redis_kwargs.get("sentinel_password")
    service_name = redis_kwargs.get("service_name")

    if not sentinel_nodes or not service_name:
        raise ValueError(
            "Both 'sentinel_nodes' and 'service_name' are required for Redis Sentinel."
        )

    verbose_logger.debug("init_redis_sentinel: sentinel nodes are being initialized.")

    # Pass SSL parameters to Sentinel
    ssl_kwargs = _extract_ssl_params(redis_kwargs)
    sentinel = async_redis.Sentinel(
        sentinel_nodes,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        password=sentinel_password,
        **ssl_kwargs
    )

    return sentinel.master_for(service_name)

def get_redis_client(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        args = _get_redis_url_kwargs()
        url_kwargs = {arg: redis_kwargs[arg] for arg in redis_kwargs if arg in args}
        return redis.Redis.from_url(**url_kwargs)

    if "startup_nodes" in redis_kwargs or get_secret("REDIS_CLUSTER_NODES") is not None:
        return init_redis_cluster(redis_kwargs)

    if "sentinel_nodes" in redis_kwargs and "service_name" in redis_kwargs:
        return _init_redis_sentinel(redis_kwargs)

    return redis.Redis(**redis_kwargs)

def get_redis_async_client(**env_overrides) -> async_redis.Redis:
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        args = _get_redis_url_kwargs(client=async_redis.Redis.from_url)
        url_kwargs = {arg: redis_kwargs[arg] for arg in redis_kwargs if arg in args}
        return async_redis.Redis.from_url(**url_kwargs)

    if "startup_nodes" in redis_kwargs:
        from redis.cluster import ClusterNode
        args = _get_redis_cluster_kwargs()
        cluster_kwargs = {arg: redis_kwargs[arg] for arg in redis_kwargs if arg in args}
        new_startup_nodes = [ClusterNode(**item) for item in redis_kwargs["startup_nodes"]]
        redis_kwargs.pop("startup_nodes")
        return async_redis.RedisCluster(startup_nodes=new_startup_nodes, **cluster_kwargs)

    if "sentinel_nodes" in redis_kwargs and "service_name" in redis_kwargs:
        return _init_async_redis_sentinel(redis_kwargs)

    return async_redis.Redis(**redis_kwargs)

def get_redis_connection_pool(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    verbose_logger.debug("get_redis_connection_pool: redis_kwargs", redis_kwargs)
    
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        return async_redis.BlockingConnectionPool.from_url(
            timeout=REDIS_CONNECTION_POOL_TIMEOUT, url=redis_kwargs["url"]
        )
    
    # Set SSL connection class if SSL is enabled
    if redis_kwargs.get("ssl", False):
        redis_kwargs["connection_class"] = async_redis.SSLConnection
        verbose_logger.debug("Using SSL connection for Redis")
    else:
        redis_kwargs["connection_class"] = async_redis.Connection
    
    redis_kwargs.pop("startup_nodes", None)  # Not supported in connection pools
    
    return async_redis.BlockingConnectionPool(
        timeout=REDIS_CONNECTION_POOL_TIMEOUT, **redis_kwargs
    )
