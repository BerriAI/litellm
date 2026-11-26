# conftest.py

import _hashlib
import asyncio
import hashlib
import importlib
import json
import logging
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from types import ModuleType
from typing import Dict, Final, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import responses
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prisma.errors import ClientNotConnectedError

import litellm
import litellm.proxy.auth.user_api_key_auth
import litellm.proxy.proxy_server
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.fips import enforce_fips_mode
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    InitPassThroughEndpointHelpers,
    _registered_pass_through_routes,
)
from litellm._service_logger import ServiceTypes
from litellm.integrations.otel.model.payloads import ServiceSpanData
from litellm.integrations.otel.model.spans import service_span_name
from tests.unit.litellm_core_utils.fake_secret_vault import FakeSecretVault
from tests._master_key import MASTER_KEY
from tests.unit.proxy.spend_tracking.native_budgets_replica import native_budgets_enabled
from tests.unit.proxy.rust_auth_stores import audit_log_store, rust_auth_stores  # noqa: F401  # fixtures
from tests._module_state import collect_flushable_caches, flush_caches, restore_mutable_state, snapshot_mutable_state


class StubClientNotConnectedError(ClientNotConnectedError):
    pass


class DisconnectedPrisma:
    def is_connected(self) -> bool:
        return False

    @property
    def _engine(self) -> None:
        raise StubClientNotConnectedError()


@pytest.fixture
def disconnected_prisma() -> DisconnectedPrisma:
    return DisconnectedPrisma()


_LITELLM_STATE: Final = snapshot_mutable_state(litellm)
_PROXY_SERVER_STATE: Final = snapshot_mutable_state(litellm.proxy.proxy_server)
_FLUSHABLE_CACHES: Final = collect_flushable_caches((litellm, litellm.proxy.proxy_server))
_PROXY_APP_LIFESPAN: Final = litellm.proxy.proxy_server.app.router.lifespan_context


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """Reset mutable module state on litellm and proxy_server before each test.

    Replaces a previous importlib.reload(litellm) approach that cost ~17s
    per test (re-executing the full litellm __init__ import chain).

    What IS reset:
      - Top-level module attributes of type list / dict / set / tuple
        / str / int / float / bool / bytes, and None-valued attributes.
        These cover callback lists, general_settings, master_key,
        premium_user, prisma_client, etc. — anything the old reload() reset
        by re-executing the module body.
      - Any module-level object instance that exposes flush_cache() (the
        DualCache and LLMClientCache family), which handles cache state
        that can't round-trip through deepcopy because of internal locks.

    What is NOT reset:
      - Class instances without flush_cache() (e.g. ProxyLogging,
        LiteLLM_JWTAuth, FastAPI routers, loggers). If a test mutates such an
        instance in-place (setattr on the instance, appending to one of
        its internal lists, etc.), the mutation will leak into later tests.
        Use pytest's monkeypatch.setattr() or a local fixture for those
        cases — don't rely on this autouse fixture to undo them.
    """
    restore_mutable_state(litellm, _LITELLM_STATE)
    restore_mutable_state(litellm.proxy.proxy_server, _PROXY_SERVER_STATE)
    flush_caches(_FLUSHABLE_CACHES)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests


_PROXY_MODULE_GLOBALS_TO_ISOLATE = (
    "master_key",
    "prisma_client",
    "llm_router",
)

_proxy_module_globals_snapshot = pytest.StashKey[Dict[str, object]]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    from litellm.proxy import proxy_server

    item.stash[_proxy_module_globals_snapshot] = {
        name: vars(proxy_server)[name]
        for name in _PROXY_MODULE_GLOBALS_TO_ISOLATE
        if name in vars(proxy_server)
    }
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    snapshot = item.stash.get(_proxy_module_globals_snapshot, None)
    if snapshot is None:
        return
    from litellm.proxy import proxy_server

    for name in _PROXY_MODULE_GLOBALS_TO_ISOLATE:
        if name in snapshot:
            setattr(proxy_server, name, snapshot[name])
        elif name in vars(proxy_server):
            delattr(proxy_server, name)


@pytest.fixture
def native_budgets_on(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    yield from native_budgets_enabled(monkeypatch)


@pytest.fixture
def secret_vault_factory() -> type[FakeSecretVault]:
    return FakeSecretVault


@contextmanager
def _eu_jev_route_configured_at(path: str) -> Iterator[None]:
    saved_registry: Final = dict(_registered_pass_through_routes)
    InitPassThroughEndpointHelpers.add_exact_path_route(
        app=FastAPI(),
        path=path,
        target="https://operator.example/decide",
        custom_headers=None,
        forward_headers=None,
        merge_query_params=None,
        dependencies=None,
        cost_per_request=None,
        endpoint_id="operator-eu-jev",
        methods=["POST"],
    )
    try:
        yield
    finally:
        _registered_pass_through_routes.clear()
        _registered_pass_through_routes.update(saved_registry)


@pytest.fixture
def configured_eu_jev_route() -> Iterator[None]:
    with _eu_jev_route_configured_at("/eu_jev/v1/systemone"):
        yield


@pytest.fixture
def configured_eu_jev_route_with_trailing_slash() -> Iterator[None]:
    with _eu_jev_route_configured_at("/eu_jev/v1/systemone/"):
        yield


VERTEX_CLAUDE_REPLY: Final = {
    "id": "msg_vertex_region",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "Hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


@pytest.fixture
def vertex_credentials() -> Iterator[str]:
    token_uri: Final = "https://oauth2.googleapis.com/token"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as google_token_endpoint:
        google_token_endpoint.post(
            token_uri, json={"access_token": "region-access-token", "expires_in": 3600, "token_type": "Bearer"}
        )
        yield json.dumps(
            {
                "type": "authorized_user",
                "client_id": "region-client",
                "client_secret": "region-client-secret",
                "refresh_token": "region-refresh-token",
                "token_uri": token_uri,
            }
        )


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.fixture(autouse=True)
def _reset_graceful_shutdown_state():
    from litellm.proxy.shutdown.graceful_shutdown_manager import (
        GracefulShutdownManager,
    )

    GracefulShutdownManager.reset()
    yield
    GracefulShutdownManager.reset()


def build_cache_config(enable_cache: bool = True) -> Optional[Dict]:
    """
    Build Redis cache configuration from environment variables.

    Args:
        enable_cache: Whether to enable cache (default: True)

    Returns:
        dict: Cache configuration dict with 'cache' and 'cache_params' keys, or None
    """
    if not enable_cache:
        return None

    redis_host = os.getenv("REDIS_HOST")
    if not redis_host:
        return None

    redis_port = os.getenv("REDIS_PORT", "6379")
    cache_params = {
        "type": "redis",
        "host": redis_host,
        "port": int(redis_port) if redis_port.isdigit() else redis_port,
    }

    redis_password = os.getenv("REDIS_PASSWORD")
    if redis_password:
        cache_params["password"] = redis_password

    return {"cache": True, "cache_params": cache_params}


def build_minimal_proxy_config(
    database_url: Optional[str] = None, **init_options
) -> Dict:
    """
    Build a minimal proxy configuration YAML.

    Args:
        database_url: Optional database URL (falls back to DATABASE_URL env var)
        **init_options: Additional configuration options:
            - master_key: API key for authentication (default: MASTER_KEY)
            - enable_cache: Whether to enable Redis cache (default: True)
            - success_callback: Callback function for success events

    Returns:
        dict: Configuration dictionary ready to be written as YAML
    """
    config = {
        "general_settings": {"master_key": init_options.get("master_key", MASTER_KEY)},
        "litellm_settings": {},
    }

    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        config["general_settings"]["database_url"] = db_url

    enable_cache = init_options.get("enable_cache", True)
    cache_config = build_cache_config(enable_cache=enable_cache)
    if cache_config:
        config["litellm_settings"].update(cache_config)

    if init_options.get("success_callback") is not None:
        config["litellm_settings"]["success_callback"] = init_options[
            "success_callback"
        ]

    excluded_keys = {
        "master_key",
        "debug",
        "success_callback",
        "database_url",
        "enable_cache",
    }
    for key, value in init_options.items():
        if key not in excluded_keys and key not in config["litellm_settings"]:
            config["litellm_settings"][key] = value

    return config


def set_proxy_environment_variables(
    monkeypatch, database_url: Optional[str] = None
) -> None:
    """
    Set environment variables for database and Redis.

    Args:
        monkeypatch: pytest monkeypatch fixture
        database_url: Optional database URL (falls back to DATABASE_URL env var)
    """
    db_url = database_url or os.getenv("DATABASE_URL")
    if db_url:
        monkeypatch.setenv("DATABASE_URL", db_url)

    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        monkeypatch.setenv("REDIS_HOST", redis_host)
        monkeypatch.setenv("REDIS_PORT", os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")
        if redis_password:
            monkeypatch.setenv("REDIS_PASSWORD", redis_password)


def create_proxy_test_client(
    monkeypatch, database_url: Optional[str] = None, **init_options
) -> TestClient:
    """
    Create a proxy TestClient with optional database and Redis cache configuration.

    Args:
        monkeypatch: pytest monkeypatch fixture
        database_url: Optional database URL (falls back to DATABASE_URL env var)
        **init_options: Additional configuration options:
            - master_key: API key for authentication (default: MASTER_KEY)
            - enable_cache: Whether to enable Redis cache (default: True)
            - success_callback: Callback function for success events
            - debug: Enable debug mode

    Returns:
        TestClient: FastAPI test client for the proxy server
    """
    from litellm.proxy.proxy_server import (
        cleanup_router_config_variables,
        initialize,
        app,
    )

    cleanup_router_config_variables()
    monkeypatch.setattr(app.router, "lifespan_context", _PROXY_APP_LIFESPAN)

    filepath = os.path.dirname(os.path.abspath(__file__))
    default_config_fp = os.path.join(
        filepath, "test_configs", "test_config_hosted_vllm_embedding.yaml"
    )

    enable_cache = init_options.get("enable_cache", True)
    needs_redis = enable_cache and os.getenv("REDIS_HOST") is not None
    needs_db = (database_url or os.getenv("DATABASE_URL")) is not None

    if not os.path.exists(default_config_fp) or needs_redis or needs_db:
        minimal_config = build_minimal_proxy_config(
            database_url=database_url, **init_options
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_fp = f.name
    else:
        config_fp = default_config_fp

    set_proxy_environment_variables(monkeypatch, database_url=database_url)
    monkeypatch.setenv("LITELLM_DANGEROUSLY_PERMIT_WEAK_OR_UNSET_MASTER_KEY", "true")

    asyncio.run(initialize(config=config_fp, debug=init_options.get("debug", False)))
    return TestClient(app)


@pytest.fixture
def fresh_agent_read_through(monkeypatch):
    from litellm.proxy.common_utils import registry_read_through

    read_through = registry_read_through.RegistryReadThrough(
        resync=registry_read_through._resync_agents, is_loaded=registry_read_through._agent_is_loaded
    )
    monkeypatch.setattr(registry_read_through, "agent_registry_read_through", read_through)
    return read_through


@pytest.fixture
def fresh_vector_store_read_throughs(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.common_utils import registry_read_through

    index_table_reload: Final = registry_read_through.ThrottledTableReload(
        reload=registry_read_through._reload_vector_store_indexes
    )
    store_read_through: Final = registry_read_through.RegistryReadThrough(
        resync=registry_read_through._resync_vector_stores,
        is_loaded=registry_read_through._vector_store_is_loaded,
    )
    monkeypatch.setattr(registry_read_through, "vector_store_index_table_reload", index_table_reload)
    monkeypatch.setattr(registry_read_through, "vector_store_registry_read_through", store_read_through)
    monkeypatch.setattr(litellm, "vector_store_index_registry", None)
    monkeypatch.setattr(litellm, "vector_store_registry", None)


class FakeVectorStoreIndexRow:
    def __init__(self, index_name: str, vector_store_name: str = "config-vector-store") -> None:
        self.index_name: Final = index_name
        self.vector_store_name: Final = vector_store_name

    def model_dump(self) -> dict[str, object]:
        return dict(self)

    def __iter__(self) -> Iterator[tuple[str, object]]:
        return iter(
            {
                "id": f"{self.index_name}-id",
                "index_name": self.index_name,
                "litellm_params": {"vector_store_index": "physical-index", "vector_store_name": self.vector_store_name},
                "index_info": None,
                "created_at": None,
                "created_by": None,
                "updated_at": None,
                "updated_by": None,
            }.items()
        )


class FakeVectorStoreRow:
    def __init__(
        self,
        vector_store_name: str,
        custom_llm_provider: str,
        litellm_params: dict[str, str],
        vector_store_id: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.vector_store_name: Final = vector_store_name
        self.custom_llm_provider: Final = custom_llm_provider
        self.litellm_params: Final = litellm_params
        self.vector_store_id: Final = vector_store_id or f"{vector_store_name}-id"
        self.created_at: Final = created_at

    def __iter__(self) -> Iterator[tuple[str, object]]:
        return iter(
            {
                "vector_store_id": self.vector_store_id,
                "custom_llm_provider": self.custom_llm_provider,
                "vector_store_name": self.vector_store_name,
                "vector_store_description": None,
                "vector_store_metadata": None,
                "created_at": self.created_at,
                "updated_at": None,
                "litellm_credential_name": None,
                "litellm_params": dict(self.litellm_params),
                "team_id": None,
                "user_id": None,
            }.items()
        )


TOKEN_COUNT_LOG_FIELDS: Final = (
    "litellm_call_id",
    "request_model",
    "model_id",
    "model_used",
    "input_tokens",
    "count_source",
    "status_code",
)


@pytest.fixture
def token_count_log_lines(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[[], tuple[dict[str, object], ...]]]:
    """The structured fields of every INFO token count line the proxy logger writes while the test runs."""
    monkeypatch.setattr(verbose_proxy_logger, "propagate", True)
    with caplog.at_level(logging.INFO, logger=verbose_proxy_logger.name):
        yield lambda: tuple(
            {field: record.__dict__[field] for field in TOKEN_COUNT_LOG_FIELDS}
            for record in caplog.records
            if record.levelno == logging.INFO and "count_source" in record.__dict__
        )


@pytest.fixture
def postgres_span_names() -> Iterator[Callable[[], Awaitable[tuple[str, ...]]]]:
    """The ``postgres.{verb} {table}`` names OTel would render for every DB service event
    the code under test emits, in emission order, once the hook tasks have run."""
    success: Final = AsyncMock()
    service_logging: Final = MagicMock(async_service_success_hook=success, async_service_failure_hook=AsyncMock())

    async def rendered() -> tuple[str, ...]:
        await asyncio.sleep(0)
        return tuple(
            service_span_name(
                ServiceSpanData(
                    service_name="postgres",
                    call_type=call.kwargs["call_type"],
                    event_metadata=call.kwargs["event_metadata"] or {},
                )
            )
            for call in success.await_args_list
            if call.kwargs["service"] == ServiceTypes.DB
        )

    with patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock(service_logging_obj=service_logging)):
        yield rendered


@pytest.fixture
def fips_openssl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_hashlib, "get_fips_mode", lambda: 1)
    monkeypatch.setattr(_hashlib, "scrypt", MagicMock(side_effect=ValueError("[digital envelope routines] unsupported")))
    monkeypatch.setattr("litellm.proxy.common_utils.encrypt_decrypt_utils._FIPS_PROPERTIES", None)


@pytest.fixture
def fips_mode(fips_openssl: None, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr("litellm.proxy.common_utils.fips.cryptography_enforces_fips", lambda: True)
    with monkeypatch.context() as boot:
        boot.setenv("LITELLM_SALT_KEY", "sk-" + bytes(range(32)).hex())
        enforce_fips_mode({"fips_mode": True})
    yield
    enforce_fips_mode({})


@pytest.fixture
def pbkdf2_on_event_loop(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """One entry per PBKDF2 call: whether it ran on the event loop's thread."""
    real_pbkdf2: Final = hashlib.pbkdf2_hmac
    on_event_loop: Final[list[bool]] = []

    def recording_pbkdf2(hash_name: str, password: bytes, salt: bytes, iterations: int, dklen: int) -> bytes:
        on_event_loop.append(asyncio._get_running_loop() is not None)
        return real_pbkdf2(hash_name, password, salt, iterations, dklen)

    monkeypatch.setattr(hashlib, "pbkdf2_hmac", recording_pbkdf2)
    return on_event_loop


def _per_token_priced(model_name: str, backend: str, input_cost: float, output_cost: float) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": backend,
            "api_key": "sk-fixture",
            "input_cost_per_token": input_cost,
            "output_cost_per_token": output_cost,
        },
    }


@pytest.fixture
def free_and_paid_router() -> "litellm.Router":
    import litellm

    return litellm.Router(
        model_list=[
            _per_token_priced("free-model", "openai/free-backend", 0.0, 0.0),
            _per_token_priced("paid-model", "openai/paid-backend", 1e-6, 2e-6),
        ]
    )


@pytest.fixture
def import_without_redis_installed(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], ModuleType]:
    def _import(module_name: str) -> ModuleType:
        package_name, _, attribute = module_name.rpartition(".")
        monkeypatch.setattr(sys.modules[package_name], attribute, sys.modules[module_name])
        monkeypatch.setitem(sys.modules, "redis", None)
        for name in tuple(sys.modules):
            if name.startswith("redis."):
                monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, module_name)
        return importlib.import_module(module_name)

    return _import


@pytest.fixture(autouse=True)
def _forget_email_in_use_checks() -> Iterator[None]:
    from litellm.proxy.auth import auth_checks

    auth_checks.email_in_use_checked_at.clear()
    yield
    auth_checks.email_in_use_checked_at.clear()
