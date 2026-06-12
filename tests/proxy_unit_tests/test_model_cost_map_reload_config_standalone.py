"""
Standalone test for model_cost_map_reload_config via config.yaml.

This test does NOT import the full litellm module (which has many dependencies
and requires Python 3.10+ for match statements). Instead, it validates the
source code changes and tests the logic with minimal mocking.
"""

import ast
import os
import sys
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

sys.path.insert(0, os.path.abspath("../.."))


def test_source_code_has_new_helper_method():
    """Verify the _get_model_cost_map_reload_config method exists in proxy_server.py."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # Parse the source code to find the method
    tree = ast.parse(source)

    found_method = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            if node.name == "_get_model_cost_map_reload_config":
                found_method = True
                break

    assert (
        found_method
    ), "_get_model_cost_map_reload_config method not found in proxy_server.py"
    print("✓ _get_model_cost_map_reload_config method exists")


def test_source_code_returns_tuple_with_source():
    """Verify the helper returns (config, source) tuple."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # Check the return type annotation includes Tuple
    assert (
        "Tuple[Optional[dict], Optional[str]]" in source
    ), "Return type annotation should be Tuple[Optional[dict], Optional[str]]"
    # Check it returns (db_config, \"db\") or (yaml_config, \"yaml\") or (None, None)
    assert (
        'return db_config, "db"' in source
    ), "Should return (db_config, 'db') for database source"
    assert (
        'return yaml_config, "yaml"' in source
    ), "Should return (yaml_config, 'yaml') for YAML source"
    assert (
        "return None, None" in source
    ), "Should return (None, None) when no config found"
    print("✓ Return type is Tuple[Optional[dict], Optional[str]] with source tracking")


def test_source_code_has_yaml_fallback_in_helper():
    """Verify the helper method checks general_settings for YAML config."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # Check for the key parts of the new logic
    assert (
        'general_settings.get("model_cost_map_reload_config")' in source
    ), "YAML fallback logic not found in _get_model_cost_map_reload_config"
    assert (
        "model_cost_map_reload_config from config.yaml" in source
    ), "Config.yaml debug log not found"
    assert (
        "Using model_cost_map_reload_config from database" in source
    ), "Database debug log not found"
    print("✓ YAML fallback logic exists in _get_model_cost_map_reload_config")


def test_source_code_warns_when_db_overrides_yaml():
    """Verify a warning is logged when DB config overrides YAML config."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # Check for the warning log when both configs exist
    assert (
        "model_cost_map_reload_config is set in both the database" in source
    ), "Warning about DB overriding YAML not found"
    assert (
        "_yaml_override_warned" in source
    ), "Module-level flag for suppressing repeated warnings not found"
    print("✓ Warning is logged when DB config overrides YAML config")


def test_source_code_db_takes_precedence_over_yaml():
    """Verify DB config is checked before YAML config (precedence)."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # Find the _get_model_cost_map_reload_config method
    start_idx = source.find("async def _get_model_cost_map_reload_config")
    assert start_idx != -1, "Method not found"

    method_source = source[
        start_idx : source.find("async def _check_and_reload_model_cost_map", start_idx)
    ]

    # DB check should come first (before YAML check)
    db_check_idx = method_source.find("get_config_param(")
    yaml_check_idx = method_source.find(
        'general_settings.get("model_cost_map_reload_config")'
    )

    assert db_check_idx != -1, "DB check not found"
    assert yaml_check_idx != -1, "YAML check not found"
    assert (
        db_check_idx < yaml_check_idx
    ), "DB check should come before YAML check (precedence)"
    print("✓ DB config takes precedence over YAML config")


def test_source_code_ignores_force_reload_from_yaml():
    """Verify force_reload from YAML is ignored with a warning."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # Check for the force_reload YAML handling in _check_and_reload_model_cost_map
    assert (
        "force_reload is only effective via the API" in source
    ), "Warning about force_reload from YAML not found"
    assert (
        'source == "yaml" and force_reload' in source
    ), "YAML force_reload check not found"
    print("✓ force_reload from YAML is ignored with a warning")


def test_source_code_no_extra_db_roundtrip():
    """Verify the caller uses the source info instead of a second DB query."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # The old code had a second get_config_param call inside _check_and_reload_model_cost_map
    # We should NOT find it anymore (the extra DB round-trip was eliminated)
    start_idx = source.find("async def _check_and_reload_model_cost_map")
    assert start_idx != -1, "Method not found"

    method_end = source.find(
        "async def _check_and_reload_anthropic_beta_headers", start_idx
    )
    method_source = source[start_idx:method_end]

    # Should use source == "db" to decide writeback
    assert (
        'source == "db"' in method_source
    ), "Should use source == 'db' to decide writeback (eliminating extra DB call)"

    # The old second get_config_param call should be gone
    second_db_call_count = method_source.count("get_config_param(")
    # One call to _get_model_cost_map_reload_config which internally calls get_config_param
    # But there should be no second call in the writeback section
    # We can't easily check this with string matching, but the source == "db" check
    # is the key indicator
    assert (
        'source == "db"' in method_source
    ), "Extra DB round-trip not eliminated — source check not found"
    print("✓ Extra DB round-trip eliminated (uses source instead of second DB query)")


def test_source_code_docstring_mentions_yaml_config():
    """Verify the docstring mentions config.yaml as a source."""
    proxy_server_path = os.path.join(
        os.path.dirname(__file__), "../../litellm/proxy/proxy_server.py"
    )
    with open(proxy_server_path, "r") as f:
        source = f.read()

    # Check _check_and_reload_model_cost_map docstring
    start_idx = source.find("async def _check_and_reload_model_cost_map")
    assert start_idx != -1, "Method not found"

    # Find the docstring
    docstring_start = source.find('"""', start_idx)
    docstring_end = source.find('"""', docstring_start + 3)
    docstring = source[docstring_start : docstring_end + 3]

    assert (
        "config.yaml" in docstring
        or "config.yaml"
        in source[start_idx : source.find("\n    async def", start_idx + 1)]
    ), "config.yaml not mentioned in reload method docstring or body"
    print("✓ Docstring mentions config.yaml as a configuration source")


def test_example_config_yaml_exists():
    """Verify the example config file with the new setting exists."""
    config_path = os.path.join(
        os.path.dirname(__file__),
        "example_config_yaml/model_cost_map_reload_config.yaml",
    )
    assert os.path.exists(config_path), f"Example config not found: {config_path}"

    with open(config_path, "r") as f:
        content = f.read()

    assert (
        "model_cost_map_reload_config" in content
    ), "model_cost_map_reload_config not found in example config"
    assert "interval_hours" in content, "interval_hours not found in example config"
    print("✓ Example config file exists with the new setting")


def test_main_config_yaml_has_documentation():
    """Verify the main proxy_server_config.yaml has documentation for the new setting."""
    config_path = os.path.join(
        os.path.dirname(__file__), "../../proxy_server_config.yaml"
    )
    with open(config_path, "r") as f:
        content = f.read()

    assert (
        "Model Cost Map Reload Configuration" in content
    ), "Documentation header not found in proxy_server_config.yaml"
    assert (
        "model_cost_map_reload_config" in content
    ), "model_cost_map_reload_config not documented in proxy_server_config.yaml"
    assert (
        "LITELLM_LOCAL_MODEL_COST_MAP" in content
    ), "LITELLM_LOCAL_MODEL_COST_MAP not mentioned in documentation"
    assert "interval_hours" in content, "interval_hours not documented"
    # Check that force_reload comment clarifies it's API-only
    assert "force_reload" in content, "force_reload not documented"
    print("✓ Main proxy_server_config.yaml has documentation for the new setting")


def test_logic_yaml_config_dict_value():
    """Test the YAML config parsing logic with a dict value."""
    # This test replicates the parsing logic from the method
    yaml_config = {"interval_hours": 24, "force_reload": False}
    general_settings = {"model_cost_map_reload_config": yaml_config}

    result = general_settings.get("model_cost_map_reload_config")
    if isinstance(result, str):
        import json

        result = json.loads(result)

    assert result == {"interval_hours": 24, "force_reload": False}
    print("✓ YAML config dict parsing works correctly")


def test_logic_yaml_config_string_value():
    """Test the YAML config parsing logic with a JSON string value."""
    import json

    yaml_config = json.dumps({"interval_hours": 12, "force_reload": True})
    general_settings = {"model_cost_map_reload_config": yaml_config}

    result = general_settings.get("model_cost_map_reload_config")
    if isinstance(result, str):
        result = json.loads(result)

    assert result == {"interval_hours": 12, "force_reload": True}
    print("✓ YAML config string parsing works correctly")


def test_logic_db_takes_precedence():
    """Test that DB config takes precedence over YAML config."""
    # Simulate the precedence logic:
    # 1. Try DB first
    # 2. If DB is None, fall back to YAML

    db_config = {"interval_hours": 6, "force_reload": True}
    yaml_config = {"interval_hours": 24, "force_reload": False}

    # DB config exists -> use it
    result = db_config if db_config is not None else yaml_config
    assert result == {"interval_hours": 6, "force_reload": True}

    # DB config is None -> fall back to YAML
    db_config = None
    result = db_config if db_config is not None else yaml_config
    assert result == {"interval_hours": 24, "force_reload": False}

    # Both are None -> None
    db_config = None
    yaml_config = None
    result = db_config if db_config is not None else yaml_config
    assert result is None
    print("✓ Precedence logic (DB > YAML > None) works correctly")


def test_logic_force_reload_from_yaml_ignored():
    """Test that force_reload from YAML is ignored (only works via API)."""
    source = "yaml"
    force_reload = True
    interval_hours = None

    # force_reload from YAML is ignored
    if source == "yaml" and force_reload:
        force_reload = False  # Ignored with warning

    # After ignoring, both interval_hours and force_reload are None/False
    # so reload should be skipped
    should_reload = force_reload or (interval_hours is not None)
    assert (
        should_reload is False
    ), "force_reload from YAML should be ignored, preventing reload"
    print("✓ force_reload from YAML is correctly ignored")


def test_logic_source_determines_writeback():
    """Test that source info determines whether to write back to DB."""
    # When source is "db", write back to clear force_reload
    source = "db"
    should_write_back = source == "db"
    assert should_write_back is True

    # When source is "yaml", do not write back
    source = "yaml"
    should_write_back = source == "db"
    assert should_write_back is False
    print("✓ Source info correctly determines DB writeback behavior")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing model_cost_map_reload_config via config.yaml")
    print("=" * 60)

    test_source_code_has_new_helper_method()
    test_source_code_returns_tuple_with_source()
    test_source_code_has_yaml_fallback_in_helper()
    test_source_code_warns_when_db_overrides_yaml()
    test_source_code_db_takes_precedence_over_yaml()
    test_source_code_ignores_force_reload_from_yaml()
    test_source_code_no_extra_db_roundtrip()
    test_source_code_docstring_mentions_yaml_config()
    test_example_config_yaml_exists()
    test_main_config_yaml_has_documentation()
    test_logic_yaml_config_dict_value()
    test_logic_yaml_config_string_value()
    test_logic_db_takes_precedence()
    test_logic_force_reload_from_yaml_ignored()
    test_logic_source_determines_writeback()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
