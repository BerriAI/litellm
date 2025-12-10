#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for LiteLLM Premium Features Management System

This script tests the premium features functionality without requiring pytest.
"""

import os
import sys
import tempfile
import shutil

# Add the parent directory to the path so we can import litellm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from litellm.proxy.auth.premium_features import (
    PremiumFeatures,
    is_premium_feature_enabled,
    require_premium_feature,
    has_premium_access
)


def clear_premium_env_vars():
    """Clear all premium features related environment variables."""
    env_vars_to_clear = [
        "LITELLM_LICENSE",
        "LITELLM_PREMIUM_FEATURES"
    ]

    # Clear individual feature flags
    for feature_name in PremiumFeatures.get_all_features():
        env_vars_to_clear.append("LITELLM_ENABLE_{}".format(feature_name.upper()))

    for env_var in env_vars_to_clear:
        if env_var in os.environ:
            del os.environ[env_var]

    # Clear the feature cache
    PremiumFeatures.clear_cache()
    PremiumFeatures._initialized = False


def test_individual_feature_flags():
    """Test enabling individual features via environment flags."""
    print("🧪 Testing Individual Feature Flags...")

    clear_premium_env_vars()

    # Test enabling a feature
    os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"

    result = is_premium_feature_enabled("advanced_rate_limiting")
    assert result is True, "Feature should be enabled when flag is set to 'true'"

    # Test disabling a feature
    os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "false"
    PremiumFeatures.clear_cache()

    result = is_premium_feature_enabled("advanced_rate_limiting")
    assert result is False, "Feature should be disabled when flag is set to 'false'"

    print("✅ Individual feature flags test passed!")


def test_batch_feature_activation():
    """Test enabling multiple features via LITELLM_PREMIUM_FEATURES."""
    print("🧪 Testing Batch Feature Activation...")

    clear_premium_env_vars()

    # Enable multiple features
    os.environ["LITELLM_PREMIUM_FEATURES"] = "advanced_rate_limiting,sso_auth,prometheus_metrics"

    assert is_premium_feature_enabled("advanced_rate_limiting") is True
    assert is_premium_feature_enabled("sso_auth") is True
    assert is_premium_feature_enabled("prometheus_metrics") is True
    assert is_premium_feature_enabled("custom_guardrails") is False  # Not in list

    print("✅ Batch feature activation test passed!")


def test_feature_flag_priority():
    """Test that individual flags have priority over batch list."""
    print("🧪 Testing Feature Flag Priority...")

    clear_premium_env_vars()

    # Set both batch list and individual flag (with different values)
    os.environ["LITELLM_PREMIUM_FEATURES"] = "advanced_rate_limiting"
    os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "false"

    result = is_premium_feature_enabled("advanced_rate_limiting")

    # Individual flag should take priority
    assert result is False, "Individual flag should override batch list"

    print("✅ Feature flag priority test passed!")


def test_unknown_feature():
    """Test handling of unknown features."""
    print("🧪 Testing Unknown Feature Handling...")

    clear_premium_env_vars()

    result = is_premium_feature_enabled("unknown_feature")
    assert result is False, "Unknown feature should return False"

    print("✅ Unknown feature handling test passed!")


def test_various_boolean_values():
    """Test various boolean values for feature flags."""
    print("🧪 Testing Various Boolean Values...")

    clear_premium_env_vars()

    test_values = {
        "true": True,
        "1": True,
        "yes": True,
        "on": True,
        "TRUE": True,
        "True": True,
        "false": False,
        "0": False,
        "no": False,
        "off": False,
        "": False,
        "invalid": False
    }

    for value, expected in test_values.items():
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = value
        PremiumFeatures.clear_cache()

        result = is_premium_feature_enabled("advanced_rate_limiting")
        assert result is expected, "Failed for value: {}".format(value)

    print("✅ Various boolean values test passed!")


def test_get_enabled_features():
    """Test getting list of enabled features."""
    print("🧪 Testing Get Enabled Features...")

    clear_premium_env_vars()

    # Enable multiple features
    os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"
    os.environ["LITELLM_ENABLE_PROMETHEUS_METRICS"] = "true"

    enabled_features = PremiumFeatures.get_enabled_features()

    assert "advanced_rate_limiting" in enabled_features
    assert "prometheus_metrics" in enabled_features
    assert "sso_auth" not in enabled_features  # Not enabled

    print("✅ Get enabled features test passed!")


def test_has_premium_access():
    """Test premium access detection."""
    print("🧪 Testing Premium Access Detection...")

    clear_premium_env_vars()

    # Test without any premium features
    result = has_premium_access()
    assert result is False, "Should not have premium access without license or features"

    # Test with feature enabled
    os.environ["LITELLM_ENABLE_ADVANCED_DASHBOARD"] = "true"
    PremiumFeatures.clear_cache()

    result = has_premium_access()
    assert result is True, "Should have premium access when a feature is enabled"

    print("✅ Premium access detection test passed!")


def test_require_premium_feature():
    """Test require_premium_feature functionality."""
    print("🧪 Testing Require Premium Feature...")

    clear_premium_env_vars()

    # Enable a feature
    os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"

    # Should not raise exception
    try:
        require_premium_feature("advanced_rate_limiting")
    except ValueError:
        assert False, "Should not raise exception when feature is enabled"

    # Try with disabled feature
    try:
        require_premium_feature("sso_auth")
        assert False, "Should raise exception when feature is not enabled"
    except ValueError as e:
        assert "PREMIUM FEATURE" in str(e)
        assert "sso_auth" in str(e)

    print("✅ Require premium feature test passed!")


def test_features_by_category():
    """Test getting features by category."""
    print("🧪 Testing Features by Category...")

    routing_features = PremiumFeatures.get_features_by_category("routing")
    security_features = PremiumFeatures.get_features_by_category("security")

    assert len(routing_features) > 0, "Should have routing features"
    assert len(security_features) > 0, "Should have security features"
    assert "advanced_rate_limiting" in routing_features, "Should have advanced_rate_limiting in routing"
    assert "sso_auth" in security_features, "Should have sso_auth in security"

    print("✅ Features by category test passed!")


def test_debug_status():
    """Test debug status output."""
    print("🧪 Testing Debug Status...")

    clear_premium_env_vars()

    # Enable some features
    os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"

    debug_info = PremiumFeatures.debug_status()

    assert isinstance(debug_info, dict), "Debug info should be a dictionary"
    assert "has_license" in debug_info, "Should have has_license field"
    assert "has_premium_access" in debug_info, "Should have has_premium_access field"
    assert "enabled_features" in debug_info, "Should have enabled_features field"
    assert "environment_flags" in debug_info, "Should have environment_flags field"
    assert debug_info["has_premium_access"] is True, "Should have premium access"
    assert "advanced_rate_limiting" in debug_info["enabled_features"], "Should list enabled feature"

    print("✅ Debug status test passed!")


def test_whitespace_handling():
    """Test whitespace handling in batch features."""
    print("🧪 Testing Whitespace Handling...")

    clear_premium_env_vars()

    os.environ["LITELLM_PREMIUM_FEATURES"] = "  advanced_rate_limiting , sso_auth  , prometheus_metrics  "

    assert is_premium_feature_enabled("advanced_rate_limiting") is True
    assert is_premium_feature_enabled("sso_auth") is True
    assert is_premium_feature_enabled("prometheus_metrics") is True

    print("✅ Whitespace handling test passed!")


def test_empty_batch_features():
    """Test handling of empty batch features list."""
    print("🧪 Testing Empty Batch Features...")

    clear_premium_env_vars()

    os.environ["LITELLM_PREMIUM_FEATURES"] = ""

    enabled_features = PremiumFeatures.get_enabled_features()

    # Should not have any features enabled from the empty list
    assert len(enabled_features) == 0, "Should not have any enabled features from empty list"

    print("✅ Empty batch features test passed!")


def test_case_sensitivity():
    """Test case sensitivity handling in batch features."""
    print("🧪 Testing Case Sensitivity...")

    clear_premium_env_vars()

    os.environ["LITELLM_PREMIUM_FEATURES"] = "ADVANCED_RATE_LIMITING,Sso_Auth"

    # Should work with lowercase conversion
    assert is_premium_feature_enabled("advanced_rate_limiting") is True
    assert is_premium_feature_enabled("sso_auth") is True

    print("✅ Case sensitivity test passed!")


def main():
    """Run all tests."""
    print("🚀 Starting LiteLLM Premium Features Tests")
    print("=" * 60)

    tests = [
        test_individual_feature_flags,
        test_batch_feature_activation,
        test_feature_flag_priority,
        test_unknown_feature,
        test_various_boolean_values,
        test_get_enabled_features,
        test_has_premium_access,
        test_require_premium_feature,
        test_features_by_category,
        test_debug_status,
        test_whitespace_handling,
        test_empty_batch_features,
        test_case_sensitivity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print("❌ {} failed: {}".format(test.__name__, e))
            failed += 1

    print("\n" + "=" * 60)
    print("📊 Test Results: {} passed, {} failed".format(passed, failed))

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())