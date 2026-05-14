"""
Regression tests: ``s3://`` / ``gcs://`` values in DB-overlay config
must be stripped at the merge boundary so they never reach
``get_instance_fn`` with ``config_file_path`` set.

Without this scrub, a PROXY_ADMIN who persists e.g.
``litellm_settings.success_callback: ["s3://attacker/m.i"]`` via
``/config/update`` would have it merged into the in-memory config
during the next ``load_config`` cycle. The YAML-load chain is active
at that point, so the runtime gate in ``get_instance_fn`` (which
permits remote loads when ``config_file_path`` is non-None) would
pass and ``_load_instance_from_remote_storage`` would exec the
remote module.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.proxy.proxy_server import (  # noqa: E402
    _scrub_db_overlay_remote_module_loads,
)


@pytest.mark.parametrize(
    "field",
    ["callbacks", "success_callback", "failure_callback", "audit_log_callbacks"],
)
def test_litellm_settings_callback_list_strips_remote_urls(field):
    overlay = {field: ["langfuse", "s3://attacker/m.i", "gcs://attacker/m.i"]}
    cleaned = _scrub_db_overlay_remote_module_loads("litellm_settings", overlay)
    assert cleaned[field] == ["langfuse"]


@pytest.mark.parametrize(
    "field",
    [
        "custom_auth",
        "custom_key_generate",
        "custom_key_update",
        "custom_sso",
        "custom_ui_sso_sign_in_handler",
    ],
)
def test_general_settings_str_field_strips_remote_urls(field):
    overlay = {field: "s3://attacker/m.i"}
    cleaned = _scrub_db_overlay_remote_module_loads("general_settings", overlay)
    assert cleaned[field] is None


def test_litellm_settings_post_call_rules_str_stripped():
    overlay = {"post_call_rules": "gcs://attacker/m.i"}
    cleaned = _scrub_db_overlay_remote_module_loads("litellm_settings", overlay)
    assert cleaned["post_call_rules"] is None


def test_custom_provider_map_custom_handler_stripped():
    overlay = {
        "custom_provider_map": [
            {"provider": "ok", "custom_handler": "my_module.handler"},
            {"provider": "bad", "custom_handler": "s3://attacker/m.i"},
        ]
    }
    cleaned = _scrub_db_overlay_remote_module_loads("litellm_settings", overlay)
    assert cleaned["custom_provider_map"][0]["custom_handler"] == "my_module.handler"
    assert cleaned["custom_provider_map"][1]["custom_handler"] is None


def test_pass_through_endpoints_target_stripped():
    overlay = {
        "pass_through_endpoints": [
            {"path": "/ok", "target": "my_module.legit_handler"},
            {"path": "/bad-s3", "target": "s3://attacker/m.handler"},
            {"path": "/bad-gcs", "target": "gcs://attacker/m.handler"},
        ]
    }
    cleaned = _scrub_db_overlay_remote_module_loads("general_settings", overlay)
    # Legit dotted-name target preserved
    assert cleaned["pass_through_endpoints"][0]["target"] == "my_module.legit_handler"
    # Both remote URLs stripped to None — entry remains so the path
    # registration can still be skipped explicitly downstream
    assert cleaned["pass_through_endpoints"][1]["target"] is None
    assert cleaned["pass_through_endpoints"][2]["target"] is None
    # Sibling fields preserved
    assert cleaned["pass_through_endpoints"][0]["path"] == "/ok"
    assert cleaned["pass_through_endpoints"][1]["path"] == "/bad-s3"


def test_pass_through_endpoints_non_list_passthrough():
    # If pass_through_endpoints is mistyped (not a list), the scrub
    # must not raise.
    cleaned = _scrub_db_overlay_remote_module_loads(
        "general_settings", {"pass_through_endpoints": "not-a-list"}
    )
    assert cleaned["pass_through_endpoints"] == "not-a-list"


def test_litellm_jwtauth_custom_validate_stripped():
    overlay = {
        "litellm_jwtauth": {
            "user_id_jwt_field": "sub",
            "custom_validate": "s3://attacker/m.validator",
        }
    }
    cleaned = _scrub_db_overlay_remote_module_loads("general_settings", overlay)
    assert cleaned["litellm_jwtauth"]["custom_validate"] is None
    # Sibling fields preserved.
    assert cleaned["litellm_jwtauth"]["user_id_jwt_field"] == "sub"


def test_local_dotted_name_preserved():
    # The scrub only targets s3:// / gcs:// scheme prefixes — legitimate
    # dotted module names (the documented operator flow) must pass
    # through unchanged.
    overlay = {
        "success_callback": ["langfuse", "my_module.success_handler", "datadog"],
        "post_call_rules": "my_module.rule_fn",
    }
    cleaned = _scrub_db_overlay_remote_module_loads("litellm_settings", overlay)
    assert cleaned["success_callback"] == [
        "langfuse",
        "my_module.success_handler",
        "datadog",
    ]
    assert cleaned["post_call_rules"] == "my_module.rule_fn"


def test_non_dict_overlay_passthrough():
    # Some DB-overlay values are scalars (e.g. ``max_internal_user_budget:
    # 100.0``). The scrub must not break those.
    assert _scrub_db_overlay_remote_module_loads("litellm_settings", 100.0) == 100.0
    assert _scrub_db_overlay_remote_module_loads("litellm_settings", None) is None


def test_unknown_section_passthrough():
    overlay = {"success_callback": ["s3://anything"]}
    # ``router_settings`` isn't a section with module-loading fields —
    # the scrub leaves it alone.
    cleaned = _scrub_db_overlay_remote_module_loads("router_settings", overlay)
    assert cleaned == overlay


def test_scrub_does_not_mutate_input():
    original = {"success_callback": ["s3://attacker/m.i"]}
    _scrub_db_overlay_remote_module_loads("litellm_settings", original)
    assert original["success_callback"] == ["s3://attacker/m.i"]
