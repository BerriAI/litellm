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
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

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
    available_args.append("username")
    available_args.append("ssl")

    return available_args


def _get_redis_env_kwarg_mapping():
    PREFIX = "REDIS_"

    return {f"{PREFIX}{x.upper()}": x for x in _get_redis_kwargs()}


def _redis_kwargs_from_environment():
    mapping = _get_redis_env_kwarg_mapping()

    return_dict = {}
    for k, v in mapping.items():
        value = get_secret(k, default_value=None)  # type: ignore
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
            value = get_secret(v)  # type: ignore
            env_overrides[k] = value

    redis_kwargs = {
        **_redis_kwargs_from_environment(),
        **env_overrides,
    }

    _startup_nodes: Optional[Union[str, list]] = redis_kwargs.get("startup_nodes", None) or get_secret(  # type: ignore
        "REDIS_CLUSTER_NODES"
    )

    if _startup_nodes is not None and isinstance(_startup_nodes, str):
        redis_kwargs["startup_nodes"] = json.loads(_startup_nodes)

    _sentinel_nodes: Optional[Union[str, list]] = redis_kwargs.get("sentinel_nodes", None) or get_secret(  # type: ignore
        "REDIS_SENTINEL_NODES"
    )

    if _sentinel_nodes is not None and isinstance(_sentinel_nodes, str):
        redis_kwargs["sentinel_nodes"] = json.loads(_sentinel_nodes)

    _sentinel_password: Optional[str] = redis_kwargs.get(
        "sentinel_password", None
    ) or get_secret_str("REDIS_SENTINEL_PASSWORD")

    if _sentinel_password is not None:
        redis_kwargs["sentinel_password"] = _sentinel_password

    _service_name: Optional[str] = redis_kwargs.get("service_name", None) or get_secret(  # type: ignore
        "REDIS_SERVICE_NAME"
    )

    if _service_name is not None:
        redis_kwargs["service_name"] = _service_name

    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        redis_kwargs.pop("host", None)
        redis_kwargs.pop("port", None)
        redis_kwargs.pop("db", None)
        redis_kwargs.pop("password", None)
    elif "startup_nodes" in redis_kwargs and redis_kwargs["startup_nodes"] is not None:
        pass
    elif (
        "sentinel_nodes" in redis_kwargs and redis_kwargs["sentinel_nodes"] is not None
    ):
        pass
    elif "host" not in redis_kwargs or redis_kwargs["host"] is None:
        raise ValueError("Either 'host' or 'url' must be specified for redis.")

    # litellm.print_verbose(f"redis_kwargs: {redis_kwargs}")
    return redis_kwargs


def init_redis_cluster(redis_kwargs) -> redis.RedisCluster:
    _redis_cluster_nodes_in_env: Optional[str] = get_secret("REDIS_CLUSTER_NODES")  # type: ignore
    if _redis_cluster_nodes_in_env is not None:
        try:
            redis_kwargs["startup_nodes"] = json.loads(_redis_cluster_nodes_in_env)
        except json.JSONDecodeError:
            raise ValueError(
                "REDIS_CLUSTER_NODES environment variable is not valid JSON. Please ensure it's properly formatted."
            )

    verbose_logger.debug("init_redis_cluster: startup nodes are being initialized.")
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
    return redis.RedisCluster(startup_nodes=new_startup_nodes, **cluster_kwargs)  # type: ignore


def _init_redis_sentinel(redis_kwargs) -> redis.Redis:
    sentinel_nodes = redis_kwargs.get("sentinel_nodes")
    sentinel_password = redis_kwargs.get("sentinel_password")
    service_name = redis_kwargs.get("service_name")

    if not sentinel_nodes or not service_name:
        raise ValueError(
            "Both 'sentinel_nodes' and 'service_name' are required for Redis Sentinel."
        )

    verbose_logger.debug("init_redis_sentinel: sentinel nodes are being initialized.")

    # Set up the Sentinel client
    sentinel = redis.Sentinel(
        sentinel_nodes,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        password=sentinel_password,
    )

    # Return the master instance for the given service

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

    # Set up the Sentinel client
    sentinel = async_redis.Sentinel(
        sentinel_nodes,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        password=sentinel_password,
    )

    # Return the master instance for the given service

    return sentinel.master_for(service_name)


def get_redis_client(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        args = _get_redis_url_kwargs()
        url_kwargs = {}
        for arg in redis_kwargs:
            if arg in args:
                url_kwargs[arg] = redis_kwargs[arg]

        return redis.Redis.from_url(**url_kwargs)

    if "startup_nodes" in redis_kwargs or get_secret("REDIS_CLUSTER_NODES") is not None:  # type: ignore
        return init_redis_cluster(redis_kwargs)

    # Check for Redis Sentinel
    if "sentinel_nodes" in redis_kwargs and "service_name" in redis_kwargs:
        return _init_redis_sentinel(redis_kwargs)

    return redis.Redis(**redis_kwargs)


def get_redis_async_client(
    **env_overrides,
) -> async_redis.Redis:
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        args = _get_redis_url_kwargs(client=async_redis.Redis.from_url)
        url_kwargs = {}
        for arg in redis_kwargs:
            if arg in args:
                url_kwargs[arg] = redis_kwargs[arg]
            else:
                verbose_logger.debug(
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
            startup_nodes=new_startup_nodes, **cluster_kwargs  # type: ignore
        )

    # Check for Redis Sentinel
    if "sentinel_nodes" in redis_kwargs and "service_name" in redis_kwargs:
        return _init_async_redis_sentinel(redis_kwargs)
    _pretty_print_redis_config(redis_kwargs=redis_kwargs)
    return async_redis.Redis(
        **redis_kwargs,
    )


def get_redis_connection_pool(**env_overrides):
    redis_kwargs = _get_redis_client_logic(**env_overrides)
    verbose_logger.debug("get_redis_connection_pool: redis_kwargs", redis_kwargs)
    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        return async_redis.BlockingConnectionPool.from_url(
            timeout=REDIS_CONNECTION_POOL_TIMEOUT, url=redis_kwargs["url"]
        )
    connection_class = async_redis.Connection
    if "ssl" in redis_kwargs:
        connection_class = async_redis.SSLConnection
        redis_kwargs.pop("ssl", None)
        redis_kwargs["connection_class"] = connection_class
    redis_kwargs.pop("startup_nodes", None)
    return async_redis.BlockingConnectionPool(
        timeout=REDIS_CONNECTION_POOL_TIMEOUT, **redis_kwargs
    )

def _pretty_print_redis_config(redis_kwargs: dict) -> None:
    """Pretty print the Redis configuration using rich with sensitive data masking"""
    try:
        import logging

        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        if not verbose_logger.isEnabledFor(logging.DEBUG):
            return

        console = Console()

        # Initialize the sensitive data masker
        masker = SensitiveDataMasker()
        
        # Mask sensitive data in redis_kwargs
        masked_redis_kwargs = masker.mask_dict(redis_kwargs)

        # Create main panel title
        title = Text("Redis Configuration", style="bold blue")

        # Create configuration table
        config_table = Table(
            title="🔧 Redis Connection Parameters",
            show_header=True,
            header_style="bold magenta",
            title_justify="left",
        )
        config_table.add_column("Parameter", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="yellow")

        # Add rows for each configuration parameter
        for key, value in masked_redis_kwargs.items():
            if value is not None:
                # Special handling for complex objects
                if isinstance(value, list):
                    if key == "startup_nodes" and value:
                        # Special handling for cluster nodes
                        value_str = f"[{len(value)} cluster nodes]"
                    elif key == "sentinel_nodes" and value:
                        # Special handling for sentinel nodes
                        value_str = f"[{len(value)} sentinel nodes]"
                    else:
                        value_str = str(value)
                else:
                    value_str = str(value)
                
                config_table.add_row(key, value_str)

        # Determine connection type
        connection_type = "Standard Redis"
        if masked_redis_kwargs.get("startup_nodes"):
            connection_type = "Redis Cluster"
        elif masked_redis_kwargs.get("sentinel_nodes"):
            connection_type = "Redis Sentinel"
        elif masked_redis_kwargs.get("url"):
            connection_type = "Redis (URL-based)"

        # Create connection type info
        info_table = Table(
            title="📊 Connection Info",
            show_header=True,
            header_style="bold green",
            title_justify="left",
        )
        info_table.add_column("Property", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="yellow")
        info_table.add_row("Connection Type", connection_type)

        # Print everything in a nice panel
        console.print("\n")
        console.print(Panel(title, border_style="blue"))
        console.print(info_table)
        console.print(config_table)
        console.print("\n")

    except ImportError:
        # Fallback to simple logging if rich is not available
        masker = SensitiveDataMasker()
        masked_redis_kwargs = masker.mask_dict(redis_kwargs)
        verbose_logger.info(f"Redis configuration: {masked_redis_kwargs}")
    except Exception as e:
        verbose_logger.error(f"Error pretty printing Redis configuration: {e}")

