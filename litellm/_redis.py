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
import ssl as _ssl_module
import threading
from typing import Callable, List, Optional, Union

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

    include_args = ["url", "redis_connect_func", "gcp_service_account", "gcp_ssl_ca_certs"]

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
    available_args.append("ssl_cert_reqs")
    available_args.append("ssl_check_hostname")
    available_args.append("ssl_ca_certs")
    available_args.append("redis_connect_func")  # Needed for sync clusters and IAM detection
    available_args.append("gcp_service_account")
    available_args.append("gcp_ssl_ca_certs")
    available_args.append("max_connections")

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


def _generate_gcp_iam_access_token(service_account: str) -> str:
    """
    Generate GCP IAM access token for Redis authentication.
    
    Args:
        service_account: GCP service account in format 'projects/-/serviceAccounts/name@project.iam.gserviceaccount.com'
    
    Returns:
        Access token string for GCP IAM authentication
    """
    try:
        from google.cloud import iam_credentials_v1
    except ImportError:
        raise ImportError(
            "google-cloud-iam is required for GCP IAM Redis authentication. "
            "Install it with: pip install google-cloud-iam"
        )
    
    client = iam_credentials_v1.IAMCredentialsClient()
    request = iam_credentials_v1.GenerateAccessTokenRequest(
        name=service_account,
        scope=['https://www.googleapis.com/auth/cloud-platform'],
    )
    response = client.generate_access_token(request=request)
    return str(response.access_token)


def create_gcp_iam_redis_connect_func(
    service_account: str,
    ssl_ca_certs: Optional[str] = None,
) -> Callable:
    """
    Creates a custom Redis connection function for GCP IAM authentication.
    
    Args:
        service_account: GCP service account in format 'projects/-/serviceAccounts/name@project.iam.gserviceaccount.com'
        ssl_ca_certs: Path to SSL CA certificate file for secure connections
    
    Returns:
        A connection function that can be used with Redis clients
    """
    def iam_connect(self):
        """Initialize the connection and authenticate using GCP IAM"""
        from redis.exceptions import (
            AuthenticationError,
            AuthenticationWrongNumberOfArgsError,
        )
        from redis.utils import str_if_bytes
        
        self._parser.on_connect(self)
        
        auth_args = (_generate_gcp_iam_access_token(service_account),)
        self.send_command("AUTH", *auth_args, check_health=False)
        
        try:
            auth_response = self.read_response()
        except AuthenticationWrongNumberOfArgsError:
            # Fallback to password auth if IAM fails
            if hasattr(self, 'password') and self.password:
                self.send_command("AUTH", self.password, check_health=False)
                auth_response = self.read_response()
            else:
                raise
        
        if str_if_bytes(auth_response) != "OK":
            raise AuthenticationError("GCP IAM authentication failed")
    
    return iam_connect


def get_redis_url_from_environment():
    if "REDIS_URL" in os.environ:
        return os.environ["REDIS_URL"]

    if "REDIS_HOST" not in os.environ or "REDIS_PORT" not in os.environ:
        raise ValueError(
            "Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified for Redis."
        )
    
    if "REDIS_SSL" in os.environ and os.environ["REDIS_SSL"].lower() == "true":
        redis_protocol = "rediss"
    else:
        redis_protocol = "redis"
    
    # Build authentication part of URL
    auth_part = ""
    if "REDIS_USERNAME" in os.environ and "REDIS_PASSWORD" in os.environ:
        auth_part = f"{os.environ['REDIS_USERNAME']}:{os.environ['REDIS_PASSWORD']}@"
    elif "REDIS_PASSWORD" in os.environ:
        auth_part = f"{os.environ['REDIS_PASSWORD']}@"
    
    return (
        f"{redis_protocol}://{auth_part}{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}"
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

    # Handle GCP IAM authentication
    _gcp_service_account = redis_kwargs.get("gcp_service_account") or get_secret_str("REDIS_GCP_SERVICE_ACCOUNT")
    _gcp_ssl_ca_certs = redis_kwargs.get("gcp_ssl_ca_certs") or get_secret_str("REDIS_GCP_SSL_CA_CERTS")
    
    if _gcp_service_account is not None:
        verbose_logger.debug("Setting up GCP IAM authentication for Redis with service account.")
        redis_kwargs["redis_connect_func"] = create_gcp_iam_redis_connect_func(
            service_account=_gcp_service_account,
            ssl_ca_certs=_gcp_ssl_ca_certs
        )
        # Store GCP service account in redis_connect_func for async cluster access
        redis_kwargs["redis_connect_func"]._gcp_service_account = _gcp_service_account
        
        # Remove GCP-specific kwargs that shouldn't be passed to Redis client
        redis_kwargs.pop("gcp_service_account", None)
        redis_kwargs.pop("gcp_ssl_ca_certs", None)
        
        # Only enable SSL if explicitly requested AND SSL CA certs are provided
        if _gcp_ssl_ca_certs and redis_kwargs.get("ssl", False):
            redis_kwargs["ssl_ca_certs"] = _gcp_ssl_ca_certs

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

    cluster_kwargs.pop("startup_nodes", None)
    return redis.RedisCluster(startup_nodes=new_startup_nodes, **cluster_kwargs)  # type: ignore


def _remove_sentinel_kwargs(redis_kwargs: dict) -> dict:
    """
    Remove Sentinel-specific kwargs that are not supported by standard Redis client.
    """
    sentinel_specific_keys = ["sentinel_nodes", "sentinel_password", "service_name"]
    cleaned_kwargs = redis_kwargs.copy()
    for key in sentinel_specific_keys:
        cleaned_kwargs.pop(key, None)
    return cleaned_kwargs


def _get_redis_ssl_context(ssl_cert_reqs=None) -> _ssl_module.SSLContext:
    """
    Create a new SSLContext configured for Redis connections.

    redis-py 5.2 creates a NEW ssl.SSLContext per connection via
    ssl.create_default_context(), which calls set_default_verify_paths()
    and parses ALL system CA certs into C-level structures each time.
    With 100+ connections in a Sentinel pool this causes 700+ MB memory
    spikes.

    One shared context eliminates this overhead entirely.
    """
    ctx = _ssl_module.create_default_context()
    if ssl_cert_reqs is not None:
        req_str = str(ssl_cert_reqs).lower()
        if req_str == "none":
            ctx.check_hostname = False
            ctx.verify_mode = _ssl_module.CERT_NONE
        elif req_str == "optional":
            ctx.check_hostname = False
            ctx.verify_mode = _ssl_module.CERT_OPTIONAL
    return ctx


# Process-wide caches — at most ~6 entries (3 cert_reqs variants × sync/async).
# Tests must call .clear() on these in setup_method for isolation.
_redis_ssl_context_cache: dict = {}
_redis_ssl_class_cache: dict = {}
_redis_ssl_context_lock = threading.RLock()


def _get_cached_redis_ssl_context(ssl_cert_reqs=None) -> _ssl_module.SSLContext:
    key = str(ssl_cert_reqs).lower() if ssl_cert_reqs is not None else "required"
    with _redis_ssl_context_lock:
        if key not in _redis_ssl_context_cache:
            _redis_ssl_context_cache[key] = _get_redis_ssl_context(ssl_cert_reqs)
        return _redis_ssl_context_cache[key]


def _make_shared_ssl_connection_class(base_cls, ssl_cert_reqs=None):
    """
    Create (or return cached) a connection class that reuses a shared SSLContext
    instead of creating a new one per connection.

    Sync SSLConnection._wrap_socket_with_ssl() calls ssl.create_default_context()
    every time.  We override it to use a cached context.

    Async SSLConnection stores self.ssl_context = RedisSSLContext(...) which
    creates context lazily.  We replace it after __init__ with a pre-built one.
    """
    cert_key = str(ssl_cert_reqs).lower() if ssl_cert_reqs is not None else "required"
    cache_key = (base_cls, cert_key)
    with _redis_ssl_context_lock:
        if cache_key in _redis_ssl_class_cache:
            return _redis_ssl_class_cache[cache_key]

        shared_ctx = _get_cached_redis_ssl_context(ssl_cert_reqs)

        class SharedSSLConnection(base_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Async path: replace RedisSSLContext lazy context with shared one.
                # Inject shared SSLContext into RedisSSLContext.context so that
                # .get() returns it instead of calling create_default_context().
                # Tested against redis-py 5.2.1 and 7.1.1. If redis-py changes
                # RedisSSLContext internals, the guard below will skip
                # injection and fall back to per-connection creation.
                if hasattr(self, "ssl_context") and hasattr(self.ssl_context, "get"):
                    sc = self.ssl_context
                    has_custom = (
                        getattr(sc, "certfile", None)
                        or getattr(sc, "keyfile", None)
                        or getattr(sc, "ca_certs", None)
                        or getattr(sc, "ca_data", None)
                        or getattr(sc, "min_version", None)
                        or getattr(sc, "ciphers", None)
                        or getattr(sc, "include_verify_flags", None)
                        or getattr(sc, "exclude_verify_flags", None)
                    )
                    if not has_custom:
                        sc.context = shared_ctx

            # Sync path: use shared SSLContext for vanilla TLS (server-cert only).
            # For any per-connection customization (client certs, custom CA, OCSP,
            # min_version, ciphers) fall back to the original redis-py method.
            def _wrap_socket_with_ssl(self, sock):
                has_custom = (
                    getattr(self, "certfile", None)
                    or getattr(self, "keyfile", None)
                    or getattr(self, "ca_certs", None)
                    or getattr(self, "ca_path", None)
                    or getattr(self, "ca_data", None)
                    or getattr(self, "ssl_validate_ocsp", False)
                    or getattr(self, "ssl_validate_ocsp_stapled", False)
                    or getattr(self, "ssl_min_version", None)
                    or getattr(self, "ssl_ciphers", None)
                    or getattr(self, "ssl_include_verify_flags", None)
                    or getattr(self, "ssl_exclude_verify_flags", None)
                )
                if has_custom:
                    return super()._wrap_socket_with_ssl(sock)
                return shared_ctx.wrap_socket(sock, server_hostname=self.host)

        SharedSSLConnection.__name__ = f"Shared{base_cls.__name__}"
        SharedSSLConnection.__qualname__ = f"Shared{base_cls.__qualname__}"
        _redis_ssl_class_cache[cache_key] = SharedSSLConnection
        return SharedSSLConnection


def _build_sentinel_master_kwargs(redis_kwargs: dict) -> dict:
    """
    Build kwargs for sentinel.master_for() from redis_kwargs.
    Forwards connection-relevant params, skips sentinel-specific ones.
    """
    master_kwargs = {}
    # NOTE: max_connections intentionally excluded — SentinelConnectionPool
    # should use its default (unlimited).  The max_connections value from
    # Helm/redis_kwargs is meant for BlockingConnectionPool, not Sentinel.
    # Passing it here caused "Too many connections" errors under burst load.
    # NOTE: ssl_certfile, ssl_keyfile, ssl_ca_certs etc. are intentionally
    # excluded — they reach connection instances through the connection_class
    # kwargs set by sentinel.master_for(). Only top-level ssl and ssl_cert_reqs
    # are needed to select SSL vs plain connection class.
    forward_keys = [
        "username",
        "password",
        "db",
        "socket_timeout",
        "socket_connect_timeout",
        "socket_keepalive",
        "socket_keepalive_options",
        "retry_on_timeout",
        "decode_responses",
        "encoding",
        "encoding_errors",
        "health_check_interval",
        "ssl",
        "ssl_cert_reqs",
    ]
    for key in forward_keys:
        if key in redis_kwargs and redis_kwargs[key] is not None:
            master_kwargs[key] = redis_kwargs[key]
    master_kwargs.setdefault("socket_timeout", REDIS_SOCKET_TIMEOUT)
    # Backward compatibility: if no explicit master password is set but
    # sentinel_password is provided, use it for master auth too.  The old
    # code passed sentinel_password as the positional `password` arg to
    # redis.Sentinel(), which redis-py treated as the default master auth.
    if "password" not in master_kwargs and redis_kwargs.get("sentinel_password"):
        master_kwargs["password"] = redis_kwargs["sentinel_password"]
    return master_kwargs


def _init_redis_sentinel(redis_kwargs) -> redis.Redis:
    sentinel_nodes = redis_kwargs.get("sentinel_nodes")
    sentinel_password = redis_kwargs.get("sentinel_password")
    service_name = redis_kwargs.get("service_name")
    ssl_enabled = redis_kwargs.get("ssl", False)
    ssl_cert_reqs = redis_kwargs.get("ssl_cert_reqs")

    if not sentinel_nodes or not service_name:
        raise ValueError(
            "Both 'sentinel_nodes' and 'service_name' are required for Redis Sentinel."
        )

    verbose_logger.debug("init_redis_sentinel: sentinel nodes are being initialized.")

    sentinel_kwargs_dict = {}
    if sentinel_password:
        sentinel_kwargs_dict["password"] = sentinel_password
    if ssl_enabled:
        sentinel_kwargs_dict["ssl"] = True
        if ssl_cert_reqs is not None:
            sentinel_kwargs_dict["ssl_cert_reqs"] = ssl_cert_reqs

    sentinel = redis.Sentinel(
        sentinel_nodes,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        sentinel_kwargs=sentinel_kwargs_dict,
    )

    master_kwargs = _build_sentinel_master_kwargs(redis_kwargs)
    if ssl_enabled:
        master_kwargs["connection_class"] = _make_shared_ssl_connection_class(
            redis.sentinel.SentinelManagedSSLConnection, ssl_cert_reqs
        )
        master_kwargs.pop("ssl", None)
        # Patch sentinel connection pools to reuse shared SSLContext.
        # Redis.__init__ creates SSLConnection per connection; we replace
        # connection_class on already-created pools before any connections
        # are established (pools start empty).
        _shared_sentinel_cls = _make_shared_ssl_connection_class(
            redis.connection.SSLConnection, ssl_cert_reqs
        )
        for s in sentinel.sentinels:
            s.connection_pool.connection_class = _shared_sentinel_cls

    return sentinel.master_for(service_name, **master_kwargs)


def _init_async_redis_sentinel(redis_kwargs) -> async_redis.Redis:
    sentinel_nodes = redis_kwargs.get("sentinel_nodes")
    sentinel_password = redis_kwargs.get("sentinel_password")
    service_name = redis_kwargs.get("service_name")
    ssl_enabled = redis_kwargs.get("ssl", False)
    ssl_cert_reqs = redis_kwargs.get("ssl_cert_reqs")

    if not sentinel_nodes or not service_name:
        raise ValueError(
            "Both 'sentinel_nodes' and 'service_name' are required for Redis Sentinel."
        )

    verbose_logger.debug("init_redis_sentinel: sentinel nodes are being initialized.")

    sentinel_kwargs_dict = {}
    if sentinel_password:
        sentinel_kwargs_dict["password"] = sentinel_password
    if ssl_enabled:
        sentinel_kwargs_dict["ssl"] = True
        if ssl_cert_reqs is not None:
            sentinel_kwargs_dict["ssl_cert_reqs"] = ssl_cert_reqs

    sentinel = async_redis.Sentinel(
        sentinel_nodes,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        sentinel_kwargs=sentinel_kwargs_dict,
    )

    master_kwargs = _build_sentinel_master_kwargs(redis_kwargs)
    if ssl_enabled:
        master_kwargs["connection_class"] = _make_shared_ssl_connection_class(
            async_redis.sentinel.SentinelManagedSSLConnection, ssl_cert_reqs
        )
        master_kwargs.pop("ssl", None)
        # Patch sentinel connection pools to reuse shared SSLContext
        _shared_sentinel_cls = _make_shared_ssl_connection_class(
            async_redis.connection.SSLConnection, ssl_cert_reqs
        )
        for s in sentinel.sentinels:
            s.connection_pool.connection_class = _shared_sentinel_cls

    return sentinel.master_for(service_name, **master_kwargs)


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

    return redis.Redis(**_remove_sentinel_kwargs(redis_kwargs))


def get_redis_async_client(
    connection_pool: Optional[async_redis.BlockingConnectionPool] = None, **env_overrides,
) -> Union[async_redis.Redis, async_redis.RedisCluster]:
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

        # Handle GCP IAM authentication for async clusters
        redis_connect_func = cluster_kwargs.pop("redis_connect_func", None)
        from litellm import get_secret_str

        # Get GCP service account - first try from redis_connect_func, then from environment
        gcp_service_account = None
        if redis_connect_func and hasattr(redis_connect_func, '_gcp_service_account'):
            gcp_service_account = redis_connect_func._gcp_service_account
        else:
            gcp_service_account = redis_kwargs.get("gcp_service_account") or get_secret_str("REDIS_GCP_SERVICE_ACCOUNT")
        
        verbose_logger.debug(f"DEBUG: Redis cluster kwargs: redis_connect_func={redis_connect_func is not None}, gcp_service_account_provided={gcp_service_account is not None}")
        
        # If GCP IAM is configured (indicated by redis_connect_func), generate access token and use as password
        if redis_connect_func and gcp_service_account:
            verbose_logger.debug("DEBUG: Generating IAM token for service account (value not logged for security reasons)")
            try:
                # Generate IAM access token using the helper function
                access_token = _generate_gcp_iam_access_token(gcp_service_account)
                cluster_kwargs["password"] = access_token
                verbose_logger.debug("DEBUG: Successfully generated GCP IAM access token for async Redis cluster")
            except Exception as e:
                verbose_logger.error(f"Failed to generate GCP IAM access token: {e}")
                from redis.exceptions import AuthenticationError
                raise AuthenticationError("Failed to generate GCP IAM access token")
        else:
            verbose_logger.debug(f"DEBUG: Not using GCP IAM auth - redis_connect_func={redis_connect_func is not None}, gcp_service_account_provided={gcp_service_account is not None}")
        
        new_startup_nodes: List[ClusterNode] = []

        for item in redis_kwargs["startup_nodes"]:
            new_startup_nodes.append(ClusterNode(**item))
        cluster_kwargs.pop("startup_nodes", None)
        
        # Create async RedisCluster with IAM token as password if available
        cluster_client = async_redis.RedisCluster(
            startup_nodes=new_startup_nodes, **cluster_kwargs  # type: ignore
        )
            
        return cluster_client

    # Check for Redis Sentinel
    if "sentinel_nodes" in redis_kwargs and "service_name" in redis_kwargs:
        return _init_async_redis_sentinel(redis_kwargs)
    _pretty_print_redis_config(redis_kwargs=redis_kwargs)

    if connection_pool is not None:
        redis_kwargs["connection_pool"] = connection_pool

    redis_kwargs = _remove_sentinel_kwargs(redis_kwargs)
    return async_redis.Redis(
        **redis_kwargs,
    )


def get_redis_connection_pool(**env_overrides):
    # NOTE: This pool is unused when Sentinel is configured —
    # get_redis_async_client() detects sentinel_nodes and creates its own
    # client via _init_async_redis_sentinel(), ignoring connection_pool.
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
    redis_kwargs = _remove_sentinel_kwargs(redis_kwargs)
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

