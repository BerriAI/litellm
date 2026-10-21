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


def test_langfuse_handler_accepts_secret_key_alias(monkeypatch):
    captured = {}

    class FakeLangFuseLogger:
        def __init__(
            self,
            *,
            langfuse_public_key=None,
            langfuse_secret=None,
            langfuse_host=None,
            langfuse_environment=None,
            allow_env_credentials=True,
        ):
            captured["langfuse_public_key"] = langfuse_public_key
            captured["langfuse_secret"] = langfuse_secret
            captured["langfuse_host"] = langfuse_host
            captured["langfuse_environment"] = langfuse_environment
            captured["allow_env_credentials"] = allow_env_credentials

    class FakeDynamicLoggingCache:
        def set_cache(self, *, credentials, service_name, logging_obj):
            captured["cached_credentials"] = credentials
            captured["cached_service_name"] = service_name
            captured["cached_logging_obj"] = logging_obj

    monkeypatch.setattr(
        "litellm.integrations.langfuse.langfuse_handler.LangFuseLogger",
        FakeLangFuseLogger,
    )

    logger = LangFuseHandler._create_langfuse_logger_from_credentials(
        credentials={
            "langfuse_public_key": "dynamic-public",
            "langfuse_secret_key": "dynamic-secret",
            "langfuse_host": "https://langfuse.example",
            "langfuse_environment": "dynamic-environment",
        },
        in_memory_dynamic_logger_cache=FakeDynamicLoggingCache(),
    )

    assert captured["langfuse_public_key"] == "dynamic-public"
    assert captured["langfuse_secret"] == "dynamic-secret"
    assert captured["langfuse_host"] == "https://langfuse.example"
    assert captured["langfuse_environment"] == "dynamic-environment"
    assert captured["allow_env_credentials"] is False
    assert captured["cached_service_name"] == "langfuse"
    assert captured["cached_logging_obj"] is logger
