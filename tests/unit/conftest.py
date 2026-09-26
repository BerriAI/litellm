import asyncio
import base64
import importlib
import os
from collections.abc import Coroutine, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import boto3
import httpx
import pytest
from pytest_socket import enable_socket, socket_allow_hosts

HOST_ENVIRONMENT_ALLOWLIST: Final = frozenset(
    ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "VIRTUAL_ENV")
)
HOST_ENVIRONMENT_ALLOWED_PREFIXES: Final = ("PYTEST_", "PYTHON", "COV_CORE_", "COVERAGE_")

for _name in tuple(os.environ):
    if _name not in HOST_ENVIRONMENT_ALLOWLIST and not _name.startswith(HOST_ENVIRONMENT_ALLOWED_PREFIXES):
        del os.environ[_name]
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm  # noqa: E402  # litellm reads LITELLM_LOCAL_MODEL_COST_MAP at import
import litellm.router as litellm_router_module  # noqa: E402  # same import-time dependency
import litellm.utils as litellm_utils_module  # noqa: E402  # same import-time dependency
from litellm._logging import ALL_LOGGERS  # noqa: E402  # same import-time dependency
from litellm.anthropic_beta_headers_manager import reload_beta_headers_config  # noqa: E402  # same import-time dependency
from litellm.litellm_core_utils.prompt_templates import factory as prompt_factory_module  # noqa: E402  # same import-time dependency
from litellm.litellm_core_utils.prompt_templates import (  # noqa: E402  # same import-time dependency
    image_handling as image_handling_module,
)
from litellm.llms.gemini.chat import transformation as gemini_chat_transformation_module  # noqa: E402  # same import-time dependency
from litellm.llms.custom_httpx.async_client_cleanup import (  # noqa: E402  # same import-time dependency
    close_litellm_async_clients,
)
from litellm.proxy.db import tool_registry_writer as tool_registry_writer_module  # noqa: E402  # same import-time dependency

LOOPBACK_HOSTS: Final = ["127.0.0.1", "::1", "localhost"]
AMBIENT_AZURE_CREDENTIAL_ENV_VARS: Final = (
    "AZURE_AD_TOKEN",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_USERNAME",
    "AZURE_PASSWORD",
)
AMBIENT_AWS_ENV_VARS: Final = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_SESSION_TOKEN",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_REGION_NAME",
    "AWS_DEFAULT_REGION",
)
MODULES_WITH_AWS_AUTH_HANDLERS: Final = (
    "litellm.main",
    "litellm.files.main",
    "litellm.rerank_api.main",
    "litellm.realtime_api.main",
)
CALLBACK_LISTS: Final = (
    "callbacks",
    "success_callback",
    "failure_callback",
    "input_callback",
    "_async_success_callback",
    "_async_failure_callback",
    "_async_input_callback",
)
RESET_TO_NONE_GLOBALS: Final = ("model_fallbacks", "cache")
RESTORED_GLOBALS: Final = (
    "disable_aiohttp_transport",
    "force_ipv4",
    "drop_params",
    "secret_manager_client",
    "_key_management_system",
    "_key_management_settings",
    "api_base",
    "num_retries",
    "modify_params",
    "ssl_verify",
    "credential_list",
    "model_group_settings",
    "default_internal_user_params",
    "default_team_params",
    "prometheus_emit_stream_label",
    "vector_store_registry",
    "model_cost",
    "cost_margin_config",
    "cost_discount_config",
    "disable_hf_tokenizer_download",
    "disable_copilot_system_to_assistant",
    "cohere_models",
    "anthropic_models",
    "token_counter",
    "initialized_langfuse_clients",
)
MODULE_LEVEL_CLIENTS: Final = ("module_level_client", "module_level_aclient")
SESSION_CLIENTS: Final = ("base_llm_aiohttp_handler", "httpx_client", "aclient", "client")
ONE_PIXEL_PNG: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _allow_loopback_only() -> None:
    socket_allow_hosts(LOOPBACK_HOSTS, allow_unix_socket=True)


_allow_loopback_only()


def pytest_collectstart() -> None:
    _allow_loopback_only()


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup() -> None:
    _allow_loopback_only()


def _run_coroutine_if_needed(result: object) -> None:
    if not asyncio.iscoroutine(result):
        return
    coroutine: Final[Coroutine[object, object, object]] = result
    try:
        asyncio.run(coroutine)
    except RuntimeError:
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            coroutine.close()
            return
        loop.create_task(coroutine)


def _close_handler_if_needed(handler: object) -> None:
    close: Final = getattr(handler, "close", None)
    if not callable(close):
        return
    _run_coroutine_if_needed(close())


def _reset_aws_auth_caches() -> None:
    modules: Final = tuple(importlib.import_module(name) for name in MODULES_WITH_AWS_AUTH_HANDLERS)
    flushes: Final = (
        getattr(getattr(getattr(module, attr_name), "iam_cache", None), "flush_cache", None)
        for module in modules
        for attr_name in dir(module)
    )
    for flush in filter(callable, flushes):
        flush()
    boto3.DEFAULT_SESSION = None


def _flush_client_caches() -> None:
    litellm.in_memory_llm_clients_cache.flush_cache()
    image_handling_module.in_memory_cache.flush_cache()
    _reset_aws_auth_caches()


@pytest.fixture(scope="session")
def isolated_aws_config_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    aws_dir: Final = tmp_path_factory.mktemp("aws-config")
    credentials: Final = aws_dir / "credentials"
    config: Final = aws_dir / "config"
    credentials.write_text("", encoding="utf-8")
    config.write_text("", encoding="utf-8")
    return credentials, config


@pytest.fixture(autouse=True)
def isolate_host_environment(isolated_aws_config_files: tuple[Path, Path]) -> Iterator[None]:
    credentials, config = isolated_aws_config_files
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
        environment.setenv("AWS_CONFIG_FILE", str(config))
        environment.setenv("AWS_EC2_METADATA_DISABLED", "true")
        for name in AMBIENT_AWS_ENV_VARS:
            environment.delenv(name, raising=False)
        environment.delenv("PROXY_BASE_URL", raising=False)
        environment.setenv("LITELLM_CLI_DISABLE_KEYRING", "1")
        yield


@pytest.fixture(autouse=True)
def isolate_litellm_globals() -> Iterator[None]:
    original_callbacks: Final = {name: list(getattr(litellm, name) or []) for name in CALLBACK_LISTS}
    original_reset: Final = {name: getattr(litellm, name) for name in RESET_TO_NONE_GLOBALS}
    original_restored: Final = {name: getattr(litellm, name) for name in RESTORED_GLOBALS if hasattr(litellm, name)}
    original_clients: Final = {name: litellm.__dict__[name] for name in MODULE_LEVEL_CLIENTS if name in litellm.__dict__}
    original_loggers: Final = {
        logger: (logger.level, logger.disabled, logger.propagate, list(logger.handlers), list(logger.filters))
        for logger in ALL_LOGGERS
    }
    original_tool_policy_registry: Final = tool_registry_writer_module._tool_policy_registry
    _flush_client_caches()
    for name in CALLBACK_LISTS:
        setattr(litellm, name, [])
    for name in RESET_TO_NONE_GLOBALS:
        setattr(litellm, name, None)
    for name in MODULE_LEVEL_CLIENTS:
        litellm.__dict__.pop(name, None)
    tool_registry_writer_module._tool_policy_registry = None
    yield
    _flush_client_caches()
    leaked_clients: Final = tuple(litellm.__dict__.pop(name, None) for name in MODULE_LEVEL_CLIENTS)
    for name, client in zip(MODULE_LEVEL_CLIENTS, leaked_clients):
        if client is not original_clients.get(name):
            _close_handler_if_needed(client)
    litellm.__dict__.update(original_clients)
    for name, value in (original_callbacks | original_reset | original_restored).items():
        setattr(litellm, name, value)
    for logger, (level, disabled, propagate, handlers, filters) in original_loggers.items():
        logger.setLevel(level)
        logger.disabled = disabled
        logger.propagate = propagate
        logger.handlers = handlers
        logger.filters = filters
    tool_registry_writer_module._tool_policy_registry = original_tool_policy_registry


@pytest.fixture(autouse=True)
def isolate_router_model_cost_state() -> Iterator[None]:
    original_live_routers: Final = frozenset(litellm_router_module._live_routers)
    original_runtime_registered_model_cost: Final = {
        model_key: dict(model_value)
        for model_key, model_value in litellm_utils_module._runtime_registered_model_cost.items()
    }
    litellm_utils_module._invalidate_model_cost_lowercase_map()
    yield
    for router in tuple(litellm_router_module._live_routers):
        litellm_router_module._live_routers.discard(router)
    for router in original_live_routers:
        litellm_router_module._live_routers.add(router)
    litellm_utils_module._runtime_registered_model_cost.clear()
    litellm_utils_module._runtime_registered_model_cost.update(original_runtime_registered_model_cost)
    litellm_utils_module._invalidate_model_cost_lowercase_map()
    litellm.get_model_info.cache_clear()


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.fixture
def local_beta_headers_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    reload_beta_headers_config()
    yield
    monkeypatch.delenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", raising=False)
    reload_beta_headers_config()


@dataclass(slots=True)
class AsyncOnlyImageFetch:
    fetched: list[str] = field(default_factory=list)  # mutable-ok: tests assert on the URLs fetched, in order
    base64_png: str = base64.b64encode(ONE_PIXEL_PNG).decode()
    data_url: str = "data:image/png;base64," + base64.b64encode(ONE_PIXEL_PNG).decode()


@pytest.fixture
def async_only_image_fetch(monkeypatch: pytest.MonkeyPatch) -> AsyncOnlyImageFetch:
    fetch: Final = AsyncOnlyImageFetch()

    def forbid_sync_fetch(client: object, url: str, **kwargs: object) -> httpx.Response:
        raise litellm.ImageFetchError(f"sync image fetch ran on the event loop: {url}")

    async def serve_png(client: object, url: str, **kwargs: object) -> httpx.Response:
        fetch.fetched.append(url)
        return httpx.Response(
            200, content=ONE_PIXEL_PNG, headers={"content-type": "image/png"}, request=httpx.Request("GET", url)
        )

    def forbid_sync_convert(url: str, *args: object, **kwargs: object) -> str:
        if url.startswith(("http://", "https://")):
            raise litellm.ImageFetchError(f"sync convert_url_to_base64 ran on the request path: {url}")
        return url

    monkeypatch.setattr(image_handling_module, "safe_get", forbid_sync_fetch)
    monkeypatch.setattr(image_handling_module, "async_safe_get", serve_png)
    for module in (image_handling_module, prompt_factory_module, gemini_chat_transformation_module):
        monkeypatch.setattr(module, "convert_url_to_base64", forbid_sync_convert)
    return fetch


@pytest.fixture
def no_ambient_azure_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in AMBIENT_AZURE_CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def pytest_sessionfinish() -> None:
    for name in MODULE_LEVEL_CLIENTS:
        _close_handler_if_needed(litellm.__dict__.pop(name, None))
    for name in SESSION_CLIENTS:
        _close_handler_if_needed(getattr(litellm, name, None))
    _run_coroutine_if_needed(close_litellm_async_clients())
    enable_socket()
