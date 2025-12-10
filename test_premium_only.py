#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for Premium Features module only
"""

import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Test importing our module without importing the full litellm package
try:
    from litellm.proxy.auth.premium_features import PremiumFeatures
    print("✅ Successfully imported PremiumFeatures")
except Exception as e:
    print("❌ Failed to import PremiumFeatures:", e)
    sys.exit(1)

# Test basic functionality
def test_basic():
    """Test basic functionality without full litellm import."""

    # Test registry exists
    features = PremiumFeatures.get_all_features()
    assert len(features) > 0, "Features registry should not be empty"
    print("✅ Features registry loaded with {} features".format(len(features)))

    # Test category functionality
    routing_features = PremiumFeatures.get_features_by_category("routing")
    assert len(routing_features) > 0, "Should have routing features"
    print("✅ Found {} routing features".format(len(routing_features)))

    # Test feature enabling
    os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"
    PremiumFeatures.clear_cache()

    result = PremiumFeatures.is_feature_enabled("advanced_rate_limiting")
    assert result is True, "Feature should be enabled"
    print("✅ Feature enabling works")

    # Test get enabled features
    enabled = PremiumFeatures.get_enabled_features()
    assert "advanced_rate_limiting" in enabled, "Feature should be in enabled list"
    print("✅ Get enabled features works")

    # Test premium access
    access = PremiumFeatures.has_premium_access()
    assert access is True, "Should have premium access when feature is enabled"
    print("✅ Premium access detection works")

    # Clear environment
    del os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"]
    PremiumFeatures.clear_cache()

    print("🎉 All basic tests passed!")

if __name__ == "__main__":
    test_basic()