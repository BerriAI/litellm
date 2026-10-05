from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from litellm.proxy import proxy_server
from litellm.proxy.bug_report_config import build_proxy_bug_report, build_proxy_environment_report, safe_config_lines

CUSTOMER_STRINGS = (
    "acme",
    "sk-live-secret",
    "hunter2",
    "postgres://",
    "10.0.0.7",
)

CUSTOMER_CONFIG: Mapping[str, object] = {
    "model_list": [
        {
            "model_name": "acme-prod-gpt4",
            "litellm_params": {
                "model": "azure/acme-gpt4o-deployment",
                "api_base": "https://acme-eastus.openai.azure.com",
                "api_key": "sk-live-secret-1",
                "rpm": 600,
                "acme_extra_param": "acme",
            },
        },
        {
            "model_name": "acme-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-live-secret-2"},
        },
        {
            "model_name": "acme-backup",
            "litellm_params": {"model": "azure/acme-backup-deployment", "api_key": "sk-live-secret-3"},
        },
        {"model_name": "acme-bare", "litellm_params": {"model": "acme-custom-model"}},
    ],
    "litellm_settings": {
        "callbacks": ["langfuse", "acme_hooks.audit_logger"],
        "drop_params": True,
        "num_retries": 3,
        "acme_internal_flag": True,
        "cache": True,
        "cache_params": {
            "type": "redis",
            "host": "10.0.0.7",
            "port": 6379,
            "password": "hunter2",
            "acme_cache_option": "acme",
        },
    },
    "router_settings": {
        "routing_strategy": "latency-based-routing",
        "redis_host": "10.0.0.7",
        "acme_router_option": True,
    },
    "guardrails": [
        {"guardrail_name": "acme-pii-mask", "litellm_params": {"guardrail": "presidio", "mode": "pre_call"}},
        {
            "guardrail_name": "acme-policy",
            "litellm_params": {"guardrail": "acme_guardrails.PolicyCheck", "api_key": "sk-live-secret-4"},
        },
    ],
    "environment_variables": {"ACME_PROD_OPENAI_KEY": "sk-live-secret-5", "ACME_TENANT": "acme"},
}

CUSTOMER_GENERAL_SETTINGS: Mapping[str, object] = {
    "master_key": "sk-live-secret-master",
    "database_url": "postgres://user:hunter2@10.0.0.7/litellm",
    "key_management_system": "aws_secret_manager",
    "store_model_in_db": True,
    "health_check_interval": 300,
    "acme_sso_tenant": "acme-prod",
}


def test_safe_config_lines_keep_only_flags_and_litellm_defined_values():
    lines = safe_config_lines(CUSTOMER_CONFIG, CUSTOMER_GENERAL_SETTINGS)

    assert lines == (
        "general_settings.key_management_system = aws_secret_manager",
        "general_settings.store_model_in_db = true",
        "litellm_settings.callbacks = [langfuse]",
        "litellm_settings.drop_params = true",
        "litellm_settings.cache = true",
        "litellm_settings.cache_params.type = redis",
        "router_settings.routing_strategy = latency-based-routing",
        "guardrails[0].litellm_params.guardrail = presidio",
        "guardrails[0].litellm_params.mode = pre_call",
        "model_list[*].provider = [azure, openai]",
    )
    assert not any(customer_string in "\n".join(lines) for customer_string in CUSTOMER_STRINGS)


@pytest.mark.parametrize(
    ("config", "expected_lines"),
    [
        ({"router_settings": {"routing_strategy": "acme-strategy"}}, ()),
        ({"litellm_settings": {"cache_params": {"type": "acme-cache"}}}, ()),
        (
            {"litellm_settings": {"success_callback": ["acme_logger", "langsmith"]}},
            ("litellm_settings.success_callback = [langsmith]",),
        ),
        ({"litellm_settings": {"callbacks": ["acme_hooks.audit_logger"]}}, ()),
    ],
)
def test_string_values_show_only_when_litellm_defines_them(
    config: Mapping[str, object], expected_lines: tuple[str, ...]
):
    assert safe_config_lines(config, {}) == expected_lines


def test_secrets_numbers_and_unknown_values_leave_no_line():
    general_settings: Mapping[str, object] = {
        "master_key": "sk-live-secret-master",
        "health_check_interval": 300,
        "store_model_in_db": object(),
        "alerting": {"acme": "webhook"},
        "background_health_checks": False,
    }

    assert safe_config_lines({}, general_settings) == ("general_settings.background_health_checks = false",)


def test_credential_keys_never_render_even_when_the_secret_equals_a_litellm_token():
    config: Mapping[str, object] = {
        "litellm_settings": {
            "openai_key": "openai",
            "token": "langfuse",
            "api_base": "azure",
            "callbacks": ["langfuse"],
            "cache_params": {"type": "redis", "password": "redis", "host": "openai", "qdrant_api_key": "qdrant"},
        },
        "router_settings": {
            "routing_strategy": "simple-shuffle",
            "redis_password": "simple-shuffle",
            "redis_url": "redis",
        },
        "guardrails": [
            {
                "litellm_params": {
                    "guardrail": "presidio",
                    "api_key": "presidio",
                    "auth_token": "pre_call",
                    "client_secret": "openai",
                    "credentials": ["openai", "azure"],
                }
            }
        ],
    }
    general_settings: Mapping[str, object] = {
        "master_key": "redis",
        "database_url": "openai",
        "alert_to_webhook_url": "langfuse",
        "key_management_system": "aws_secret_manager",
        "use_azure_key_vault": True,
    }

    assert safe_config_lines(config, general_settings) == (
        "general_settings.key_management_system = aws_secret_manager",
        "general_settings.use_azure_key_vault = true",
        "litellm_settings.callbacks = [langfuse]",
        "litellm_settings.cache_params.type = redis",
        "router_settings.routing_strategy = simple-shuffle",
        "guardrails[0].litellm_params.guardrail = presidio",
    )


def test_malformed_sections_produce_no_lines():
    config: Mapping[str, object] = {
        "litellm_settings": "acme",
        "router_settings": ["acme"],
        "guardrails": {"acme": {"litellm_params": {"guardrail": "presidio"}}},
        "model_list": "acme",
        "environment_variables": None,
    }

    assert safe_config_lines(config, {}) == ()


@pytest.fixture
def loaded_proxy_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous_config = proxy_server.proxy_config.get_config_state()
    proxy_server.proxy_config.update_config_state(config=CUSTOMER_CONFIG)
    monkeypatch.setattr(proxy_server, "general_settings", dict(CUSTOMER_GENERAL_SETTINGS))
    yield
    proxy_server.proxy_config.update_config_state(config=previous_config)


@pytest.mark.usefixtures("loaded_proxy_config")
def test_build_proxy_bug_report_reads_the_loaded_proxy_config():
    report = build_proxy_bug_report(RuntimeError("boom"), stream=False)

    assert report.environment.surface == "proxy"
    assert report.stream is False
    assert report.environment.config_lines == safe_config_lines(CUSTOMER_CONFIG, CUSTOMER_GENERAL_SETTINGS)


@pytest.mark.usefixtures("loaded_proxy_config")
def test_proxy_environment_report_matches_the_bug_report_environment():
    environment = build_proxy_environment_report()

    assert environment == build_proxy_bug_report(RuntimeError("boom")).environment
    assert environment.config_lines == safe_config_lines(CUSTOMER_CONFIG, CUSTOMER_GENERAL_SETTINGS)
