import pytest

import litellm
from litellm.integrations.langfuse import langfuse_sdk
from litellm.integrations.langfuse.langfuse import LangFuseLogger, resolve_langfuse_credentials
from litellm.integrations.langfuse.langfuse_handler import LangFuseHandler
from litellm.litellm_core_utils.specialty_caches.dynamic_logging_cache import DynamicLoggingCache
from litellm.types.utils import StandardCallbackDynamicParams


def test_resolve_langfuse_credentials_does_not_use_env_for_dynamic_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "global-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    public_key, secret_key, host = resolve_langfuse_credentials(
        langfuse_host="https://attacker.example",
        allow_env_credentials=False,
    )

    assert public_key is None
    assert secret_key is None
    assert host == "https://attacker.example"


def test_resolve_langfuse_credentials_accepts_secret_key_alias_for_dynamic_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    public_key, secret_key, host = resolve_langfuse_credentials(
        langfuse_public_key="dynamic-public",
        langfuse_secret_key="dynamic-secret",
        langfuse_host="https://team-langfuse.example",
        allow_env_credentials=False,
    )

    assert public_key == "dynamic-public"
    assert secret_key == "dynamic-secret"
    assert host == "https://team-langfuse.example"


def test_resolve_langfuse_credentials_keeps_env_for_global_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "global-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")

    public_key, secret_key, host = resolve_langfuse_credentials(
        langfuse_host="https://admin-configured.example",
        allow_env_credentials=True,
    )

    assert public_key == "global-public"
    assert secret_key == "global-secret"
    assert host == "https://admin-configured.example"


def test_langfuse_handler_accepts_secret_key_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_MOCK", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "global-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "global-secret")
    monkeypatch.delenv("UPSTREAM_LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(litellm, "initialized_langfuse_clients", 0)
    monkeypatch.setattr(langfuse_sdk, "_TRACING", {})
    params: StandardCallbackDynamicParams = {
        "langfuse_secret_key": "dynamic-secret",
        "langfuse_host": "https://langfuse.example",
        "langfuse_environment": "dynamic-environment",
    }
    cache: DynamicLoggingCache = DynamicLoggingCache()

    logger: LangFuseLogger = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=params,
        in_memory_dynamic_logger_cache=cache,
    )
    cached_logger: LangFuseLogger = LangFuseHandler.get_langfuse_logger_for_request(
        standard_callback_dynamic_params=params,
        in_memory_dynamic_logger_cache=cache,
    )

    assert logger.public_key is None
    assert logger.secret_key == "dynamic-secret"
    assert logger.langfuse_host == "https://langfuse.example"
    assert logger.langfuse_environment == "dynamic-environment"
    assert cached_logger is logger
    logger.stop()
