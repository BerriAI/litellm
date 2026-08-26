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
from collections.abc import Callable, Mapping
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import redis
import redis.asyncio as async_redis
from redis.credentials import CredentialProvider

from litellm import get_secret, get_secret_str
from litellm._redis_credential_provider import (
    AzureADCredentialProvider,
    GCPIAMCredentialProvider,
    _generate_gcp_iam_access_token,
)
from litellm.constants import (
    REDIS_CLUSTER_HEALTH_CHECK_INTERVAL,
    REDIS_CONNECTION_POOL_TIMEOUT,
    REDIS_SOCKET_TIMEOUT,
)
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker

from ._logging import verbose_logger

AZURE_REDIS_SCOPE: Final = "https://redis.azure.com/.default"


def _get_redis_kwargs():
    arg_spec: Final = inspect.getfullargspec(redis.Redis)

    # Only allow primitive arguments
    exclude_args: Final = {
        "self",
        "connection_pool",
        "retry",
    }

    include_args: Final = {
        "url",
        "redis_connect_func",
        "credential_provider",
        "gcp_service_account",
        "gcp_ssl_ca_certs",
        "azure_redis_ad_token",
        "azure_client_id",
        "azure_tenant_id",
        "azure_client_secret",
    }

    available_args: Final = {x for x in arg_spec.args if x not in exclude_args} | include_args

    return available_args


def _init_arg_names(cls: type) -> frozenset[str]:
    """Every ``__init__`` parameter accepted anywhere in a class's MRO.

    Keyword-only parameters are included, and the MRO is walked because redis-py splits a
    connection's parameters between ``AbstractConnection`` and its concrete subclasses.

    Each ``__init__`` is unwrapped before introspection: redis-py >= 7.4 decorates
    ``AbstractConnection.__init__`` with ``@deprecated_args``, whose wrapper is declared
    ``(self, *args, **kwargs)`` — introspecting the wrapper directly loses every real
    parameter (``socket_timeout`` included), which silently emptied this allowlist and
    dropped the socket timeouts from url-configured connections. ``inspect.unwrap``
    follows the ``__wrapped__`` chain to the true signature and is a no-op on
    undecorated ``__init__``s.
    """
    return frozenset(
        name
        for klass in inspect.getmro(cls)
        if klass is not object
        for spec in (inspect.getfullargspec(inspect.unwrap(klass.__init__)),)
        for name in spec.args + spec.kwonlyargs
    )


def _get_redis_url_kwargs(client: type | None = None) -> tuple[str, ...]:
    """Connection kwargs that redis-py forwards from ``from_url`` down to the connection.

    ``from_url`` is declared as ``(cls, url, **kwargs)``, so introspecting it yields no
    connection kwargs at all. What it really does is hand its kwargs to the connection
    class, so that class's signature is the allowlist.

    Taking the client's signature instead would be wrong in both directions: it omits
    nothing useful, but it admits client-only parameters such as
    ``single_connection_client`` and ``auto_close_connection_pool``, plus the ``ssl_*``
    family that only ``SSLConnection`` accepts. Those reach ``AbstractConnection`` and
    raise ``TypeError`` the first time a connection is created. TLS on a url config is
    selected by the ``rediss://`` scheme, which picks ``SSLConnection`` on its own.
    """
    if client is None:
        client = redis.Redis
    connection_cls: Final = async_redis.Connection if client is async_redis.Redis else redis.Connection

    exclude_args: Final = frozenset(
        {
            "self",
            "connection_pool",
            "retry",
        }
    )

    # Only allow primitive arguments
    include_args: Final = ("url", "max_connections")

    return tuple(x for x in _init_arg_names(connection_cls) if x not in exclude_args) + include_args


def _get_redis_cluster_kwargs(client=None):
    if client is None:
        client = redis.Redis.from_url
    arg_spec: Final = inspect.getfullargspec(redis.RedisCluster)

    # Only allow primitive arguments
    exclude_args: Final = {"self", "connection_pool", "retry", "host", "port", "startup_nodes"}

    available_args = {x for x in arg_spec.args if x not in exclude_args}
    available_args |= {
        "password",
        "username",
        "ssl",
        "ssl_cert_reqs",
        "ssl_check_hostname",
        "ssl_ca_certs",
        "redis_connect_func",  # Needed for sync clusters and IAM detection
        "credential_provider",
        "gcp_service_account",
        "gcp_ssl_ca_certs",
        "azure_redis_ad_token",
        "azure_client_id",
        "azure_tenant_id",
        "azure_client_secret",
        "max_connections",
        "socket_timeout",
        "socket_connect_timeout",
        "health_check_interval",
        "socket_keepalive",
    }

    return available_args


def _get_redis_env_kwarg_mapping():
    PREFIX: Final = "REDIS_"

    exclude_from_environment: Final = frozenset({"credential_provider"})
    return {f"{PREFIX}{x.upper()}": x for x in _get_redis_kwargs() if x not in exclude_from_environment}


def _redis_kwargs_from_environment():
    mapping: Final = _get_redis_env_kwarg_mapping()

    return_dict: Final = {}
    for k, v in mapping.items():
        value = get_secret(k, default_value=None)
        if value is not None:
            return_dict[v] = value
    return return_dict


def create_gcp_iam_redis_connect_func(
    service_account: str,
    ssl_ca_certs: str | None = None,
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

        auth_args: Final = (_generate_gcp_iam_access_token(service_account),)
        self.send_command("AUTH", *auth_args, check_health=False)

        try:
            auth_response = self.read_response()
        except AuthenticationWrongNumberOfArgsError:
            # Fallback to password auth if IAM fails
            if hasattr(self, "password") and self.password:
                self.send_command("AUTH", self.password, check_health=False)
                auth_response = self.read_response()
            else:
                raise

        if str_if_bytes(auth_response) != "OK":
            raise AuthenticationError("GCP IAM authentication failed")

    return iam_connect


def _build_azure_credential(
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
):
    """
    Build a long-lived Azure credential object.

    Azure SDK credentials cache tokens internally and handle expiry/refresh
    transparently, so this should be called once and the result reused.
    """
    try:
        from azure.identity import (
            ClientSecretCredential,
            DefaultAzureCredential,
            ManagedIdentityCredential,
        )
    except ImportError:
        raise ImportError(
            "azure-identity is required for Azure AD Redis authentication. Install it with: pip install azure-identity"
        )

    _client_id: Final = azure_client_id or os.environ.get("AZURE_CLIENT_ID")
    _tenant_id: Final = azure_tenant_id or os.environ.get("AZURE_TENANT_ID")
    _client_secret: Final = azure_client_secret or os.environ.get("AZURE_CLIENT_SECRET")

    if _client_id and _tenant_id and _client_secret:
        return ClientSecretCredential(
            client_id=_client_id,
            tenant_id=_tenant_id,
            client_secret=_client_secret,
        )
    elif _client_id:
        return ManagedIdentityCredential(client_id=_client_id)
    else:
        return DefaultAzureCredential()


def _generate_azure_ad_redis_token(
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
) -> str:
    """
    One-shot helper that builds a credential and fetches a single Azure AD
    access token for Redis. Each call rebuilds the credential and performs a
    network round-trip, so it should not be used in steady-state Redis flows
    — the sync (``create_azure_ad_redis_connect_func``) and async paths
    (``AzureADCredentialProvider``) keep the credential alive across
    connections so the Azure SDK's internal cache + silent refresh apply.
    """
    credential: Final = _build_azure_credential(
        azure_client_id=azure_client_id,
        azure_tenant_id=azure_tenant_id,
        azure_client_secret=azure_client_secret,
    )
    token: Final = credential.get_token(AZURE_REDIS_SCOPE)
    return token.token


def create_azure_ad_redis_connect_func(
    azure_client_id: str | None = None,
    azure_tenant_id: str | None = None,
    azure_client_secret: str | None = None,
) -> Callable:
    """
    Creates a custom Redis connection function for Azure AD authentication.

    Used for sync Redis clients. The credential is created once (captured by the
    closure) and reused across connections — the Azure SDK handles token caching
    and silent renewal internally. Only ``get_token`` is called per connection.
    """
    credential: Final = _build_azure_credential(
        azure_client_id=azure_client_id,
        azure_tenant_id=azure_tenant_id,
        azure_client_secret=azure_client_secret,
    )

    def ad_connect(self):
        """Initialize the connection and authenticate using Azure AD"""
        from redis.exceptions import (
            AuthenticationError,
            AuthenticationWrongNumberOfArgsError,
        )
        from redis.utils import str_if_bytes

        self._parser.on_connect(self)

        access_token: Final = credential.get_token(AZURE_REDIS_SCOPE).token

        # Only include username when explicitly set — sending AUTH "" <token>
        # is invalid for most ACL-configured Azure Redis instances.
        username: Final = os.environ.get("REDIS_USERNAME", "")
        if username:
            auth_args = (username, access_token)
        else:
            auth_args = (access_token,)

        self.send_command("AUTH", *auth_args, check_health=False)

        try:
            auth_response = self.read_response()
        except AuthenticationWrongNumberOfArgsError:
            # Fallback: try with just the token (Redis < 6 / no ACL)
            self.send_command("AUTH", access_token, check_health=False)
            auth_response = self.read_response()

        if str_if_bytes(auth_response) != "OK":
            raise AuthenticationError("Azure AD authentication failed for Redis")

    # Attach the live credential object so async paths can wrap it in
    # AzureADCredentialProvider for refresh-aware token retrieval. The raw
    # client_id/tenant_id/secret are intentionally NOT exposed here — the
    # credential closure already holds them.
    ad_connect._azure_credential = credential
    return ad_connect


def get_redis_url_from_environment():
    if "REDIS_URL" in os.environ:
        return os.environ["REDIS_URL"]

    if "REDIS_HOST" not in os.environ or "REDIS_PORT" not in os.environ:
        raise ValueError("Either 'REDIS_URL' or both 'REDIS_HOST' and 'REDIS_PORT' must be specified for Redis.")

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

    return f"{redis_protocol}://{auth_part}{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}"


def _url_without_userinfo(url: str) -> str:
    parts: Final = urlsplit(url)
    netloc: Final = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _get_redis_client_logic(**env_overrides):
    """
    Common functionality across sync + async redis client implementations
    """
    ### check if "os.environ/<key-name>" passed in
    for k, v in env_overrides.items():
        if isinstance(v, str) and v.startswith("os.environ/"):
            v = v.replace("os.environ/", "")
            value = get_secret(v)
            env_overrides[k] = value

    environment_kwargs: Final = _redis_kwargs_from_environment()

    # An explicitly configured connection target outranks REDIS_URL from the
    # environment. Without this, the url branch below strips the caller's
    # host/port/password and silently connects to whatever REDIS_URL names.
    caller_named_a_target: Final = any(
        env_overrides.get(key) is not None for key in ("host", "startup_nodes", "sentinel_nodes")
    )
    if caller_named_a_target and env_overrides.get("url") is None:
        environment_kwargs.pop("url", None)

    redis_kwargs: Final = {
        **environment_kwargs,
        **env_overrides,
    }

    _startup_nodes: Final[str | list | None] = redis_kwargs.get("startup_nodes", None) or get_secret(
        "REDIS_CLUSTER_NODES"
    )

    # If startup_nodes resolved to None (not set by kwarg or env), remove the key
    # entirely so callers can rely on key presence as a reliable cluster-mode signal.
    if _startup_nodes is not None and isinstance(_startup_nodes, str):
        redis_kwargs["startup_nodes"] = json.loads(_startup_nodes)
    elif _startup_nodes is None:
        redis_kwargs.pop("startup_nodes", None)

    _sentinel_nodes: Final[str | list | None] = redis_kwargs.get("sentinel_nodes", None) or get_secret(
        "REDIS_SENTINEL_NODES"
    )

    if _sentinel_nodes is not None and isinstance(_sentinel_nodes, str):
        redis_kwargs["sentinel_nodes"] = json.loads(_sentinel_nodes)

    _sentinel_password: Final[str | None] = redis_kwargs.get("sentinel_password", None) or get_secret_str(
        "REDIS_SENTINEL_PASSWORD"
    )

    if _sentinel_password is not None:
        redis_kwargs["sentinel_password"] = _sentinel_password

    _service_name: Final[str | None] = redis_kwargs.get("service_name", None) or get_secret("REDIS_SERVICE_NAME")

    if _service_name is not None:
        redis_kwargs["service_name"] = _service_name

    if redis_kwargs.get("credential_provider") is None:
        # Handle GCP IAM authentication
        _gcp_service_account: Final = redis_kwargs.get("gcp_service_account") or get_secret_str(
            "REDIS_GCP_SERVICE_ACCOUNT"
        )
        _gcp_ssl_ca_certs: Final = redis_kwargs.get("gcp_ssl_ca_certs") or get_secret_str("REDIS_GCP_SSL_CA_CERTS")

        if _gcp_service_account is not None:
            verbose_logger.debug("Setting up GCP IAM authentication for Redis with service account.")
            redis_kwargs["redis_connect_func"] = create_gcp_iam_redis_connect_func(
                service_account=_gcp_service_account, ssl_ca_certs=_gcp_ssl_ca_certs
            )
            # Store GCP service account in redis_connect_func for async cluster access
            redis_kwargs["redis_connect_func"]._gcp_service_account = _gcp_service_account

            # Only enable SSL if explicitly requested AND SSL CA certs are provided
            if _gcp_ssl_ca_certs and redis_kwargs.get("ssl", False):
                redis_kwargs["ssl_ca_certs"] = _gcp_ssl_ca_certs

        # Handle Azure AD authentication (after GCP IAM block)
        _azure_redis_ad_token: Final = redis_kwargs.get("azure_redis_ad_token") or get_secret("REDIS_AZURE_AD_TOKEN")

        _azure_ad_enabled: Final = _azure_redis_ad_token is not None and str(_azure_redis_ad_token).lower() == "true"

        if _azure_ad_enabled and _gcp_service_account is not None:
            verbose_logger.warning(
                "Both GCP IAM (gcp_service_account) and Azure AD (azure_redis_ad_token) are configured for Redis. "
                "Using GCP IAM. Remove one to avoid misconfiguration."
            )

        if _azure_ad_enabled and _gcp_service_account is None:
            _azure_client_id: Final = redis_kwargs.get("azure_client_id") or get_secret_str("AZURE_CLIENT_ID")
            _azure_tenant_id: Final = redis_kwargs.get("azure_tenant_id") or get_secret_str("AZURE_TENANT_ID")
            _azure_client_secret: Final = redis_kwargs.get("azure_client_secret") or get_secret_str(
                "AZURE_CLIENT_SECRET"
            )

            verbose_logger.debug("Setting up Azure AD authentication for Redis.")
            redis_kwargs["redis_connect_func"] = create_azure_ad_redis_connect_func(
                azure_client_id=_azure_client_id,
                azure_tenant_id=_azure_tenant_id,
                azure_client_secret=_azure_client_secret,
            )
            # Marker for async paths to detect Azure AD auth. The live credential
            # object is attached separately as `_azure_credential` by
            # `create_azure_ad_redis_connect_func`; the raw client_id/tenant_id/secret
            # are intentionally NOT exposed on the function to avoid leaking
            # credentials via inspection or logging.
            redis_kwargs["redis_connect_func"]._azure_redis_ad_token = True

    redis_kwargs.pop("gcp_service_account", None)
    redis_kwargs.pop("gcp_ssl_ca_certs", None)

    # Always remove Azure-specific kwargs that shouldn't be passed to Redis client
    redis_kwargs.pop("azure_redis_ad_token", None)
    redis_kwargs.pop("azure_client_id", None)
    redis_kwargs.pop("azure_tenant_id", None)
    redis_kwargs.pop("azure_client_secret", None)

    if redis_kwargs.get("credential_provider") is not None:
        redis_kwargs.pop("redis_connect_func", None)
        redis_kwargs.pop("username", None)
        redis_kwargs.pop("password", None)
        if redis_kwargs.get("url") is not None:
            redis_kwargs["url"] = _url_without_userinfo(redis_kwargs["url"])

    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        # Only strip host/port/db/password when not routing to a cluster.
        # When startup_nodes is also present the cluster path takes priority and
        # needs the password for authentication.
        if not redis_kwargs.get("startup_nodes"):
            redis_kwargs.pop("host", None)
            redis_kwargs.pop("port", None)
            redis_kwargs.pop("db", None)
            redis_kwargs.pop("password", None)
    elif (
        "startup_nodes" in redis_kwargs
        and redis_kwargs["startup_nodes"] is not None
        or "sentinel_nodes" in redis_kwargs
        and redis_kwargs["sentinel_nodes"] is not None
    ):
        pass
    elif "host" not in redis_kwargs or redis_kwargs["host"] is None:
        raise ValueError("Either 'host' or 'url' must be specified for redis.")

    # litellm.print_verbose(f"redis_kwargs: {redis_kwargs}")
    return redis_kwargs


def init_redis_cluster(redis_kwargs) -> redis.RedisCluster:
    _redis_cluster_nodes_in_env: Final[str | None] = get_secret("REDIS_CLUSTER_NODES")
    if _redis_cluster_nodes_in_env is not None:
        try:
            redis_kwargs["startup_nodes"] = json.loads(_redis_cluster_nodes_in_env)
        except json.JSONDecodeError:
            raise ValueError(
                "REDIS_CLUSTER_NODES environment variable is not valid JSON. Please ensure it's properly formatted."
            )

    verbose_logger.debug("init_redis_cluster: startup nodes are being initialized.")
    from redis.cluster import ClusterNode

    args: Final = _get_redis_cluster_kwargs()
    cluster_kwargs: Final = {}
    for arg in redis_kwargs:
        if arg in args:
            cluster_kwargs[arg] = redis_kwargs[arg]

    new_startup_nodes: Final[list[ClusterNode]] = []

    for item in redis_kwargs["startup_nodes"]:
        new_startup_nodes.append(ClusterNode(**item))

    cluster_kwargs.pop("startup_nodes", None)
    return redis.RedisCluster(startup_nodes=new_startup_nodes, **cluster_kwargs)


def _get_redis_sentinel_connection_kwargs(redis_kwargs: dict) -> dict:
    connection_kwargs: Final = {}
    args: Final = _get_redis_kwargs()
    for arg in redis_kwargs:
        if arg in args:
            connection_kwargs[arg] = redis_kwargs[arg]

    return connection_kwargs


def _init_redis_sentinel(redis_kwargs) -> redis.Redis:
    sentinel_nodes: Final = redis_kwargs.get("sentinel_nodes")
    sentinel_password: Final = redis_kwargs.get("sentinel_password")
    service_name: Final = redis_kwargs.get("service_name")
    connection_kwargs: Final = _get_redis_sentinel_connection_kwargs(redis_kwargs)
    connection_kwargs.setdefault("socket_timeout", REDIS_SOCKET_TIMEOUT)
    sentinel_kwargs: Final = _sentinel_auth_kwargs(connection_kwargs, sentinel_password)

    if not sentinel_nodes or not service_name:
        raise ValueError("Both 'sentinel_nodes' and 'service_name' are required for Redis Sentinel.")

    verbose_logger.debug("init_redis_sentinel: sentinel nodes are being initialized.")

    # Set up the Sentinel client
    sentinel: Final = redis.Sentinel(
        sentinel_nodes,
        sentinel_kwargs=sentinel_kwargs,
    )

    # Return the master instance for the given service

    return sentinel.master_for(service_name, **connection_kwargs)


def _sentinel_auth_kwargs(connection_kwargs: dict, sentinel_password: str | None) -> dict:
    """The Sentinel monitors are separate servers that authenticate with their own password, so the
    data node's credential provider never belongs on them: leaving it there makes redis-py send the
    data node's token to a monitor, which fails whether the monitor is unauthenticated or has its
    own password."""
    kept: Final = ((k, v) for k, v in connection_kwargs.items() if k != "credential_provider")
    return dict(kept, password=sentinel_password)


def _init_async_redis_sentinel(redis_kwargs) -> async_redis.Redis:
    sentinel_nodes: Final = redis_kwargs.get("sentinel_nodes")
    sentinel_password: Final = redis_kwargs.get("sentinel_password")
    service_name: Final = redis_kwargs.get("service_name")
    connection_kwargs: Final = _get_redis_sentinel_connection_kwargs(redis_kwargs)
    connection_kwargs.setdefault("socket_timeout", REDIS_SOCKET_TIMEOUT)
    sentinel_kwargs: Final = _sentinel_auth_kwargs(connection_kwargs, sentinel_password)

    if not sentinel_nodes or not service_name:
        raise ValueError("Both 'sentinel_nodes' and 'service_name' are required for Redis Sentinel.")

    verbose_logger.debug("init_redis_sentinel: sentinel nodes are being initialized.")

    # Set up the Sentinel client
    sentinel: Final = async_redis.Sentinel(
        sentinel_nodes,
        sentinel_kwargs=sentinel_kwargs,
    )

    # Return the master instance for the given service

    return sentinel.master_for(service_name, **connection_kwargs)


def _async_credential_provider(redis_connect_func: object | None) -> CredentialProvider | None:
    """The Azure AD and GCP IAM connect funcs run their AUTH exchange with the blocking client
    API, so on an async connection their ``send_command``/``read_response`` calls return
    coroutines nobody awaits and every connect fails. Async paths authenticate through a
    ``CredentialProvider`` instead, which redis-py consults per connection so the token stays
    fresh. Any other ``redis_connect_func`` is left where it is, since redis-py awaits it
    itself when it is a coroutine function."""
    gcp_service_account: Final = getattr(redis_connect_func, "_gcp_service_account", None)
    if gcp_service_account is not None:
        return GCPIAMCredentialProvider(gcp_service_account)

    azure_credential: Final = getattr(redis_connect_func, "_azure_credential", None)
    if azure_credential is not None:
        return AzureADCredentialProvider(azure_credential, username=os.environ.get("REDIS_USERNAME") or None)

    return None


def _async_auth_kwargs(redis_kwargs: dict) -> dict:
    """Swaps a connect func an async path cannot run for the equivalent credential provider,
    which supersedes any static username or password redis-py would otherwise reject it with."""
    explicit_provider: Final = redis_kwargs.get("credential_provider")
    credential_provider: Final = (
        explicit_provider
        if explicit_provider is not None
        else _async_credential_provider(redis_kwargs.get("redis_connect_func"))
    )
    if credential_provider is None:
        return redis_kwargs

    superseded: Final = frozenset({"redis_connect_func", "username", "password"})
    kept: Final = ((k, v) for k, v in redis_kwargs.items() if k not in superseded)
    return dict(kept, credential_provider=credential_provider)  # mutable-ok: the branches below mutate these kwargs


def get_redis_client(**env_overrides):
    redis_kwargs: Final = _get_redis_client_logic(**env_overrides)

    if "startup_nodes" in redis_kwargs:
        return init_redis_cluster(redis_kwargs)

    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        args: Final = _get_redis_url_kwargs()
        url_kwargs: Final = {}
        for arg in redis_kwargs:
            if arg in args:
                url_kwargs[arg] = redis_kwargs[arg]

        return redis.Redis.from_url(**url_kwargs)

    # Check for Redis Sentinel
    if "sentinel_nodes" in redis_kwargs and "service_name" in redis_kwargs:
        return _init_redis_sentinel(redis_kwargs)

    return redis.Redis(**redis_kwargs)


def get_redis_async_client(
    connection_pool: async_redis.BlockingConnectionPool | None = None,
    **env_overrides,
) -> async_redis.Redis | async_redis.RedisCluster:
    redis_kwargs: Final = _async_auth_kwargs(_get_redis_client_logic(**env_overrides))

    if "startup_nodes" in redis_kwargs:
        from redis.cluster import ClusterNode

        args = _get_redis_cluster_kwargs()
        cluster_kwargs: Final = {}
        for arg in redis_kwargs:
            if arg in args:
                cluster_kwargs[arg] = redis_kwargs[arg]

        new_startup_nodes: Final[list[ClusterNode]] = []

        for item in redis_kwargs["startup_nodes"]:
            new_startup_nodes.append(ClusterNode(**item))
        cluster_kwargs.pop("startup_nodes", None)
        cluster_kwargs.pop("redis_connect_func", None)

        # Default to a periodic health check + TCP keepalive so a connection silently dropped
        # by a cluster restart (e.g. ElastiCache Serverless maintenance) is revalidated and
        # reconnected before reuse instead of stalling in re-initialization; an explicit value
        # from config still wins.
        cluster_kwargs.setdefault("health_check_interval", REDIS_CLUSTER_HEALTH_CHECK_INTERVAL)
        cluster_kwargs.setdefault("socket_keepalive", True)

        # A single node's client-side timeout must reset only that node's connections,
        # not tear down the whole cluster client for every concurrent caller.
        from litellm.caching.redis_cluster_node_isolation import (
            get_litellm_async_redis_cluster_class,
        )

        async_redis_cluster_class: Final = get_litellm_async_redis_cluster_class()

        # Create async RedisCluster with IAM token as password if available
        cluster_client: Final = async_redis_cluster_class(
            startup_nodes=new_startup_nodes,
            **cluster_kwargs,
        )

        return cluster_client

    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        if connection_pool is not None:
            return async_redis.Redis(connection_pool=connection_pool)
        args = _get_redis_url_kwargs(client=async_redis.Redis)
        url_kwargs: Final = {}
        for arg in redis_kwargs:
            if arg in args:
                url_kwargs[arg] = redis_kwargs[arg]
            else:
                verbose_logger.debug(
                    "REDIS: ignoring argument: %s. Not an allowed async_redis.Redis.from_url arg.", arg
                )
        return async_redis.Redis.from_url(**url_kwargs)

    # Check for Redis Sentinel
    if "sentinel_nodes" in redis_kwargs and "service_name" in redis_kwargs:
        return _init_async_redis_sentinel(redis_kwargs)

    _pretty_print_redis_config(redis_kwargs=redis_kwargs)

    if connection_pool is not None:
        redis_kwargs["connection_pool"] = connection_pool

    return async_redis.Redis(
        **redis_kwargs,
    )


def get_redis_connection_pool(
    **env_overrides,
) -> async_redis.BlockingConnectionPool | None:
    redis_kwargs: Final = _async_auth_kwargs(_get_redis_client_logic(**env_overrides))
    verbose_logger.debug("get_redis_connection_pool: redis_kwargs", redis_kwargs)

    if "startup_nodes" in redis_kwargs:
        return None

    if "url" in redis_kwargs and redis_kwargs["url"] is not None:
        allowed_args: Final = _get_redis_url_kwargs(client=async_redis.Redis)
        pool_kwargs: Final = {k: v for k, v in redis_kwargs.items() if k in allowed_args and k != "max_connections"}
        pool_kwargs["timeout"] = REDIS_CONNECTION_POOL_TIMEOUT
        pool_kwargs["url"] = redis_kwargs["url"]
        if "max_connections" in redis_kwargs:
            try:
                pool_kwargs["max_connections"] = int(redis_kwargs["max_connections"])
            except (TypeError, ValueError):
                verbose_logger.warning(
                    "REDIS: invalid max_connections value %r, ignoring",
                    redis_kwargs["max_connections"],
                )
        return async_redis.BlockingConnectionPool.from_url(**pool_kwargs)

    if redis_kwargs.pop("ssl", None):
        redis_kwargs["connection_class"] = async_redis.SSLConnection
    return async_redis.BlockingConnectionPool(timeout=REDIS_CONNECTION_POOL_TIMEOUT, **redis_kwargs)


def _redis_kwargs_for_logging(redis_kwargs: Mapping[str, object]) -> Mapping[str, object]:
    return {
        key: "<credential provider>"
        if key == "credential_provider" and value is not None
        else "<redis connect function>"
        if key == "redis_connect_func" and value is not None
        else value
        for key, value in redis_kwargs.items()
    }


def _pretty_print_redis_config(redis_kwargs: dict) -> None:
    """Pretty print the Redis configuration using rich with sensitive data masking"""
    redis_kwargs_for_logging: Final = _redis_kwargs_for_logging(redis_kwargs)
    try:
        import logging

        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        if not verbose_logger.isEnabledFor(logging.DEBUG):
            return

        console: Final = Console()

        # Initialize the sensitive data masker
        masker = SensitiveDataMasker()

        # Mask sensitive data in redis_kwargs
        masked_redis_kwargs = masker.mask_dict(redis_kwargs_for_logging)

        # Create main panel title
        title: Final = Text("Redis Configuration", style="bold blue")

        # Create configuration table
        config_table: Final = Table(
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
        info_table: Final = Table(
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
        masked_redis_kwargs = masker.mask_dict(redis_kwargs_for_logging)
        verbose_logger.info("Redis configuration: %s", masked_redis_kwargs)
    except Exception as e:
        verbose_logger.error("Error pretty printing Redis configuration: %s", e)
