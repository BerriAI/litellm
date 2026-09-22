from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from litellm.proxy import proxy_server
from litellm.proxy.bug_report_config import (
    MAX_CONFIG_DEPTH,
    build_proxy_bug_report,
    build_proxy_environment_report,
    safe_config_lines,
    verbose_config_lines,
)

CUSTOMER_STRINGS = (
    "acme",
    "sk-live-secret",
    "hunter2",
    "postgres://",
    "10.0.0.7",
)

CUSTOMER_VALUES = (
    "acme-prod",
    "acme-gpt4o",
    "acme-mini",
    "acme-backup",
    "acme-bare",
    "acme-custom",
    "acme-pii",
    "acme-policy",
    "acme_hooks",
    "acme_guardrails",
    "ACME_",
    "sk-live-secret",
    "hunter2",
    "postgres://",
    "10.0.0.7",
    "azure.com",
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


def test_verbose_lines_show_every_key_but_only_litellm_defined_or_numeric_values():
    lines = verbose_config_lines(CUSTOMER_CONFIG, CUSTOMER_GENERAL_SETTINGS)

    assert lines == (
        "model_list[0].model_name = <str>",
        "model_list[0].litellm_params.model = azure/<str>",
        "model_list[0].litellm_params.api_base = <str>",
        "model_list[0].litellm_params.api_key = <str>",
        "model_list[0].litellm_params.rpm = 600",
        "model_list[0].litellm_params.acme_extra_param = <str>",
        "model_list[1].model_name = <str>",
        "model_list[1].litellm_params.model = openai/<str>",
        "model_list[1].litellm_params.api_key = <str>",
        "model_list[2].model_name = <str>",
        "model_list[2].litellm_params.model = azure/<str>",
        "model_list[2].litellm_params.api_key = <str>",
        "model_list[3].model_name = <str>",
        "model_list[3].litellm_params.model = <str>",
        "litellm_settings.callbacks = [langfuse, <str>]",
        "litellm_settings.drop_params = true",
        "litellm_settings.num_retries = 3",
        "litellm_settings.acme_internal_flag = true",
        "litellm_settings.cache = true",
        "litellm_settings.cache_params.type = redis",
        "litellm_settings.cache_params.host = <str>",
        "litellm_settings.cache_params.port = 6379",
        "litellm_settings.cache_params.password = <str>",
        "litellm_settings.cache_params.acme_cache_option = <str>",
        "router_settings.routing_strategy = latency-based-routing",
        "router_settings.redis_host = <str>",
        "router_settings.acme_router_option = true",
        "guardrails[0].guardrail_name = <str>",
        "guardrails[0].litellm_params.guardrail = presidio",
        "guardrails[0].litellm_params.mode = pre_call",
        "guardrails[1].guardrail_name = <str>",
        "guardrails[1].litellm_params.guardrail = <str>",
        "guardrails[1].litellm_params.api_key = <str>",
        "environment_variables = <2 keys>",
        "general_settings.master_key = <str>",
        "general_settings.database_url = <str>",
        "general_settings.key_management_system = aws_secret_manager",
        "general_settings.store_model_in_db = true",
        "general_settings.health_check_interval = 300",
        "general_settings.acme_sso_tenant = <str>",
    )
    assert not any(customer_value in "\n".join(lines) for customer_value in CUSTOMER_VALUES)


def test_verbose_lines_count_operator_keyed_maps_and_type_unknown_leaves():
    config: Mapping[str, object] = {
        "router_settings": {
            "model_group_alias": {"acme-gpt4": "acme-prod-gpt4", "acme-fast": {"model": "acme-mini", "hidden": True}},
            "timeout": None,
            "fallbacks": [{"acme-prod-gpt4": ["acme-backup"]}],
        },
        "model_list": [
            {
                "model_name": "acme-prod-gpt4",
                "litellm_params": {"model": "acme/acme-custom", "extra_headers": {"x-acme-tenant": "acme"}},
                "model_info": {"metadata": {"owner": "acme"}, "supports_vision": True, "weight": 1.5},
            }
        ],
        "litellm_settings": {"callbacks": object(), "tags": ["acme", 7, False]},
    }

    assert verbose_config_lines(config, {}) == (
        "router_settings.model_group_alias = <2 keys>",
        "router_settings.timeout = null",
        "router_settings.fallbacks[0] = <1 keys>",
        "model_list[0].model_name = <str>",
        "model_list[0].litellm_params.model = <str>",
        "model_list[0].litellm_params.extra_headers = <1 keys>",
        "model_list[0].model_info.metadata = <1 keys>",
        "model_list[0].model_info.supports_vision = true",
        "model_list[0].model_info.weight = 1.5",
        "litellm_settings.callbacks = <object>",
        "litellm_settings.tags = [<str>, 7, false]",
    )


def test_verbose_lines_count_pass_through_headers_and_operator_named_budget_maps():
    config: Mapping[str, object] = {
        "litellm_settings": {
            "model_alias_map": {"acme-gpt4": "gpt-4o"},
            "tag_budget_config": {"acme-team": {"max_budget": 10}},
            "priority_reservation": {"acme-prod": 0.9},
        },
        "mcp_servers": {"acme_mcp": {"url": "https://mcp.acme.example", "static_headers": {"x-acme-token": "t"}}},
    }
    general_settings: Mapping[str, object] = {
        "pass_through_endpoints": [
            {
                "path": "/acme",
                "target": "https://acme.example",
                "headers": {"x-acme-tenant": "acme", "Authorization": "k"},
            }
        ]
    }

    lines = verbose_config_lines(config, general_settings)

    assert lines == (
        "litellm_settings.model_alias_map = <1 keys>",
        "litellm_settings.tag_budget_config = <1 keys>",
        "litellm_settings.priority_reservation = <1 keys>",
        "mcp_servers = <1 keys>",
        "general_settings.pass_through_endpoints[0].path = <str>",
        "general_settings.pass_through_endpoints[0].target = <str>",
        "general_settings.pass_through_endpoints[0].headers = <2 keys>",
    )
    assert "acme" not in "\n".join(lines)


def test_verbose_lines_stop_at_the_depth_cap_and_survive_self_referencing_config():
    deep: dict[str, object] = {"drop_params": True}
    for _ in range(MAX_CONFIG_DEPTH + 5):
        deep = {"nested": deep}
    loop: list[object] = []
    loop.append(loop)

    lines = verbose_config_lines({"litellm_settings": deep, "guardrails": loop}, {})

    assert lines == (
        "litellm_settings" + ".nested" * (MAX_CONFIG_DEPTH - 1) + " = <object>",
        "guardrails" + "[0]" * (MAX_CONFIG_DEPTH - 1) + " = <object>",
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
def test_proxy_environment_report_switches_between_safe_and_verbose_config_lines():
    safe = build_proxy_environment_report()
    verbose = build_proxy_environment_report(verbose=True)

    assert safe.config_lines == safe_config_lines(CUSTOMER_CONFIG, CUSTOMER_GENERAL_SETTINGS)
    assert verbose.config_lines == verbose_config_lines(CUSTOMER_CONFIG, CUSTOMER_GENERAL_SETTINGS)
    assert (safe.surface, safe.litellm_version, safe.python_version, safe.deployment) == (
        verbose.surface,
        verbose.litellm_version,
        verbose.python_version,
        verbose.deployment,
    )
