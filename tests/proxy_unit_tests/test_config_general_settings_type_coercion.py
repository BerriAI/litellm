"""
Tests for ConfigGeneralSettings type coercion and dict compatibility.

This test ensures that the ConfigGeneralSettings Pydantic model properly
coerces types from YAML/dict input, providing consistent behavior with
environment variable type coercion.
"""

import pytest
from litellm.proxy._types import ConfigGeneralSettings


def test_config_general_settings_integer_type_coercion():
    """Test that integer fields are properly coerced from string values."""
    config_dict = {
        "proxy_batch_write_at": "10",  # String should be coerced to int
        "proxy_batch_polling_interval": "6000",
        "proxy_budget_rescheduler_min_time": "597",
        "proxy_budget_rescheduler_max_time": "605",
        "health_check_interval": "300",
        "alerting_threshold": "600",
        "max_parallel_requests": "100",
        "maximum_spend_logs_retention_period": "30",
    }

    settings = ConfigGeneralSettings(**config_dict)

    # Verify integers are properly coerced
    assert settings.proxy_batch_write_at == 10
    assert isinstance(settings.proxy_batch_write_at, int)
    assert settings.proxy_batch_polling_interval == 6000
    assert isinstance(settings.proxy_batch_polling_interval, int)
    assert settings.proxy_budget_rescheduler_min_time == 597
    assert isinstance(settings.proxy_budget_rescheduler_min_time, int)
    assert settings.health_check_interval == 300
    assert isinstance(settings.health_check_interval, int)
    assert settings.alerting_threshold == 600
    assert settings.max_parallel_requests == 100
    assert settings.maximum_spend_logs_retention_period == 30


def test_config_general_settings_boolean_type_coercion():
    """Test that boolean fields are properly coerced from string values."""
    config_dict = {
        "disable_spend_logs": "true",
        "store_model_in_db": "false",
        "use_shared_health_check": "True",
        "disable_reset_budget": "False",
        "health_check_details": "yes",  # Pydantic accepts yes/no
        "infer_model_from_keys": "1",  # Pydantic accepts 1/0
    }

    settings = ConfigGeneralSettings(**config_dict)

    # Verify booleans are properly coerced
    assert settings.disable_spend_logs is True
    assert isinstance(settings.disable_spend_logs, bool)
    assert settings.store_model_in_db is False
    assert isinstance(settings.store_model_in_db, bool)
    assert settings.use_shared_health_check is True
    assert settings.disable_reset_budget is False
    assert settings.health_check_details is True
    assert settings.infer_model_from_keys is True


def test_config_general_settings_float_type_coercion():
    """Test that float fields are properly coerced from string values."""
    config_dict = {
        "user_api_key_cache_ttl": "60.5",
        "database_connection_timeout": "120",
    }

    settings = ConfigGeneralSettings(**config_dict)

    assert settings.user_api_key_cache_ttl == 60.5
    assert isinstance(settings.user_api_key_cache_ttl, float)
    assert settings.database_connection_timeout == 120.0
    assert isinstance(settings.database_connection_timeout, float)


def test_config_general_settings_get_method_with_none():
    """Test that .get() method returns default when value is None."""
    settings = ConfigGeneralSettings()

    # When field is None, .get() should return the default
    assert settings.get("proxy_batch_write_at", 10) == 10
    assert settings.get("master_key", "default_key") == "default_key"
    assert settings.get("disable_spend_logs", True) is True
    assert settings.get("moderation_model", "gpt-4") == "gpt-4"


def test_config_general_settings_get_method_with_values():
    """Test that .get() method returns actual value when set."""
    settings = ConfigGeneralSettings(
        proxy_batch_write_at=15,
        master_key="test_key",
        disable_spend_logs=False,
    )

    assert settings.get("proxy_batch_write_at", 10) == 15
    assert settings.get("master_key", "default_key") == "test_key"
    assert settings.get("disable_spend_logs", True) is False


def test_config_general_settings_extra_fields_allowed():
    """Test that extra fields (not defined in model) are allowed and preserved."""
    config_dict = {
        "proxy_batch_write_at": 10,
        "custom_field_not_in_model": "some_value",
        "another_custom_field": 123,
    }

    settings = ConfigGeneralSettings(**config_dict)

    # Defined field should work
    assert settings.proxy_batch_write_at == 10

    # Extra fields should be accessible
    assert settings.get("custom_field_not_in_model") == "some_value"
    assert settings.get("another_custom_field") == 123


def test_config_general_settings_dict_style_access():
    """Test dict-style access methods work correctly."""
    settings = ConfigGeneralSettings(proxy_batch_write_at=20, master_key="test")

    # __getitem__
    assert settings["proxy_batch_write_at"] == 20
    assert settings["master_key"] == "test"

    # __contains__
    assert "proxy_batch_write_at" in settings
    assert "master_key" in settings
    assert "nonexistent_field" not in settings

    # __setitem__
    settings["new_field"] = "new_value"
    assert settings.get("new_field") == "new_value"

    # items()
    items = dict(settings.items())
    assert "proxy_batch_write_at" in items
    assert items["proxy_batch_write_at"] == 20


def test_config_general_settings_update_method():
    """Test that .update() method works like dict.update()."""
    settings = ConfigGeneralSettings(proxy_batch_write_at=10)

    # Update with dict
    settings.update({"proxy_batch_write_at": 20, "master_key": "new_key"})

    assert settings.proxy_batch_write_at == 20
    assert settings.master_key == "new_key"

    # Update should raise TypeError for non-dict
    with pytest.raises(TypeError):
        settings.update("not a dict")


def test_config_general_settings_end_to_end_yaml_simulation():
    """
    End-to-end test simulating loading from YAML config.
    This is the key test that validates the original issue is fixed.
    """
    # Simulate YAML config being loaded as dict (YAML parsers return strings)
    yaml_config = {
        "general_settings": {
            "proxy_batch_write_at": "10",  # String from YAML
            "master_key": "sk-1234",
            "disable_spend_logs": "false",  # String from YAML
            "user_api_key_cache_ttl": "60.5",  # String from YAML
        }
    }

    # Instantiate ConfigGeneralSettings from the dict
    general_settings = ConfigGeneralSettings(**yaml_config["general_settings"])

    # Verify type coercion happened correctly
    assert general_settings.proxy_batch_write_at == 10
    assert isinstance(general_settings.proxy_batch_write_at, int)

    # This is the critical part - can be used in arithmetic without errors
    # (Original issue was that strings from YAML weren't coerced)
    batch_interval = general_settings.proxy_batch_write_at + 5
    assert batch_interval == 15

    # Verify other coercions
    assert general_settings.disable_spend_logs is False
    assert isinstance(general_settings.disable_spend_logs, bool)
    assert general_settings.user_api_key_cache_ttl == 60.5
    assert isinstance(general_settings.user_api_key_cache_ttl, float)

    # Verify .get() works with fallback to defaults
    assert general_settings.get("proxy_batch_write_at", 999) == 10
    assert general_settings.get("nonexistent", "default") == "default"
