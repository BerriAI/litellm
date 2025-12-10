"""
Tests for LiteLLM Premium Features Management System

This test suite validates the premium features control system that allows
enabling/disabling premium features via environment variables.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from litellm.proxy.auth.premium_features import (
    PremiumFeatures,
    PremiumFeature,
    is_premium_feature_enabled,
    require_premium_feature,
    has_premium_access
)


class TestPremiumFeatures:
    """Test suite for the PremiumFeatures class."""

    def setup_method(self):
        """Setup method for each test."""
        # Clear any cached state
        PremiumFeatures._initialized = False
        PremiumFeatures._feature_status_cache.clear()
        PremiumFeatures._license_check = None

        # Clear relevant environment variables
        env_vars_to_clear = [
            "LITELLM_LICENSE",
            "LITELLM_PREMIUM_FEATURES"
        ]

        # Clear individual feature flags
        for feature_name in PremiumFeatures._FEATURES_REGISTRY:
            env_vars_to_clear.append("LITELLM_ENABLE_{}".format(feature_name.upper()))

        for env_var in env_vars_to_clear:
            if env_var in os.environ:
                del os.environ[env_var]

    def teardown_method(self):
        """Cleanup after each test."""
        PremiumFeatures.clear_cache()

    def test_features_registry_populated(self):
        """Test that the features registry is properly populated."""
        features = PremiumFeatures.get_all_features()

        assert len(features) > 0
        assert "advanced_rate_limiting" in features
        assert "sso_auth" in features
        assert "prometheus_metrics" in features

        # Check that all features are PremiumFeature objects
        for feature_name, feature in features.items():
            assert isinstance(feature, PremiumFeature)
            assert feature.name == feature_name
            assert feature.description is not None

    def test_get_features_by_category(self):
        """Test getting features by category."""
        routing_features = PremiumFeatures.get_features_by_category("routing")
        security_features = PremiumFeatures.get_features_by_category("security")

        assert len(routing_features) > 0
        assert len(security_features) > 0
        assert "advanced_rate_limiting" in routing_features
        assert "sso_auth" in security_features

    @patch('litellm.proxy.auth.premium_features.LicenseCheck')
    def test_has_premium_license_with_valid_license(self, mock_license_check_class):
        """Test premium license detection with valid license."""
        mock_license_check = MagicMock()
        mock_license_check.is_premium.return_value = True
        mock_license_check_class.return_value = mock_license_check

        # Set a license in environment
        os.environ["LITELLM_LICENSE"] = "test-license-key"

        result = PremiumFeatures.has_premium_license()

        assert result is True
        mock_license_check.is_premium.assert_called_once()

    @patch('litellm.proxy.auth.premium_features.LicenseCheck')
    def test_has_premium_license_without_license(self, mock_license_check_class):
        """Test premium license detection without license."""
        mock_license_check = MagicMock()
        mock_license_check.is_premium.return_value = False
        mock_license_check_class.return_value = mock_license_check

        # No license in environment
        result = PremiumFeatures.has_premium_license()

        assert result is False
        mock_license_check.is_premium.assert_called_once()

    def test_individual_feature_flag_enable(self):
        """Test enabling features with individual flags."""
        # Enable a specific feature
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"

        result = PremiumFeatures.is_feature_enabled("advanced_rate_limiting")

        assert result is True

    def test_individual_feature_flag_disable(self):
        """Test disabling features with individual flags."""
        # Set flag to false
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "false"

        result = PremiumFeatures.is_feature_enabled("advanced_rate_limiting")

        assert result is False

    def test_batch_feature_activation(self):
        """Test enabling multiple features via LITELLM_PREMIUM_FEATURES."""
        os.environ["LITELLM_PREMIUM_FEATURES"] = "advanced_rate_limiting,sso_auth,prometheus_metrics"

        assert PremiumFeatures.is_feature_enabled("advanced_rate_limiting") is True
        assert PremiumFeatures.is_feature_enabled("sso_auth") is True
        assert PremiumFeatures.is_feature_enabled("prometheus_metrics") is True
        assert PremiumFeatures.is_feature_enabled("custom_guardrails") is False  # Not in list

    def test_feature_flag_priority(self):
        """Test that individual flags have priority over batch list."""
        # Set both batch list and individual flag (with different values)
        os.environ["LITELLM_PREMIUM_FEATURES"] = "advanced_rate_limiting"
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "false"

        result = PremiumFeatures.is_feature_enabled("advanced_rate_limiting")

        # Individual flag should take priority
        assert result is False

    def test_unknown_feature_returns_false(self):
        """Test that unknown features return False."""
        result = PremiumFeatures.is_feature_enabled("unknown_feature")

        assert result is False

    def test_feature_flag_various_values(self):
        """Test various boolean values for feature flags."""
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
            result = PremiumFeatures.is_feature_enabled("advanced_rate_limiting")
            assert result is expected, f"Failed for value: {value}"

    @patch('litellm.proxy.auth.premium_features.LicenseCheck')
    def test_license_based_feature_activation(self, mock_license_check_class):
        """Test that features are enabled with valid license."""
        mock_license_check = MagicMock()
        mock_license_check.is_premium.return_value = True
        mock_license_check_class.return_value = mock_license_check

        # Set license but no individual flags
        os.environ["LITELLM_LICENSE"] = "test-license"

        # Features that require license should be enabled
        assert PremiumFeatures.is_feature_enabled("advanced_rate_limiting") is True
        assert PremiumFeatures.is_feature_enabled("sso_auth") is True

    @patch('litellm.proxy.auth.premium_features.LicenseCheck')
    def test_feature_without_license_requirement(self, mock_license_check_class):
        """Test that features not requiring license can be enabled without it."""
        mock_license_check = MagicMock()
        mock_license_check.is_premium.return_value = False
        mock_license_check_class.return_value = mock_license_check

        # Enable a feature that doesn't require license (e.g., advanced_dashboard)
        os.environ["LITELLM_ENABLE_ADVANCED_DASHBOARD"] = "true"

        result = PremiumFeatures.is_feature_enabled("advanced_dashboard")

        assert result is True

    def test_get_enabled_features(self):
        """Test getting list of enabled features."""
        # Enable multiple features
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"
        os.environ["LITELLM_ENABLE_PROMETHEUS_METRICS"] = "true"

        enabled_features = PremiumFeatures.get_enabled_features()

        assert "advanced_rate_limiting" in enabled_features
        assert "prometheus_metrics" in enabled_features
        assert "sso_auth" not in enabled_features  # Not enabled

    def test_has_premium_access_with_license(self):
        """Test premium access detection with license."""
        with patch('litellm.proxy.auth.premium_features.LicenseCheck') as mock_class:
            mock_instance = MagicMock()
            mock_instance.is_premium.return_value = True
            mock_class.return_value = mock_instance

            os.environ["LITELLM_LICENSE"] = "test-license"

            result = PremiumFeatures.has_premium_access()

            assert result is True

    def test_has_premium_access_with_feature_flags(self):
        """Test premium access detection with feature flags."""
        with patch('litellm.proxy.auth.premium_features.LicenseCheck') as mock_class:
            mock_instance = MagicMock()
            mock_instance.is_premium.return_value = False
            mock_class.return_value = mock_instance

            # Enable feature without license
            os.environ["LITELLM_ENABLE_ADVANCED_DASHBOARD"] = "true"

            result = PremiumFeatures.has_premium_access()

            assert result is True

    def test_has_premium_access_without_anything(self):
        """Test premium access detection without license or features."""
        with patch('litellm.proxy.auth.premium_features.LicenseCheck') as mock_class:
            mock_instance = MagicMock()
            mock_instance.is_premium.return_value = False
            mock_class.return_value = mock_instance

            result = PremiumFeatures.has_premium_access()

            assert result is False

    def test_require_premium_feature_success(self):
        """Test require_premium_feature when feature is enabled."""
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"

        # Should not raise exception
        PremiumFeatures.require_premium_feature("advanced_rate_limiting")

    def test_require_premium_feature_failure(self):
        """Test require_premium_feature when feature is not enabled."""
        with pytest.raises(ValueError) as exc_info:
            PremiumFeatures.require_premium_feature("advanced_rate_limiting")

        assert "PREMIUM FEATURE" in str(exc_info.value)
        assert "advanced_rate_limiting" in str(exc_info.value)
        assert "LITELLM_LICENSE" in str(exc_info.value)

    def test_debug_status(self):
        """Test debug status output."""
        # Enable some features
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"
        os.environ["LITELLM_LICENSE"] = "test-license"

        with patch('litellm.proxy.auth.premium_features.LicenseCheck') as mock_class:
            mock_instance = MagicMock()
            mock_instance.is_premium.return_value = True
            mock_class.return_value = mock_instance

            debug_info = PremiumFeatures.debug_status()

            assert isinstance(debug_info, dict)
            assert "has_license" in debug_info
            assert "has_premium_access" in debug_info
            assert "enabled_features" in debug_info
            assert "environment_flags" in debug_info
            assert "individual_flags" in debug_info
            assert debug_info["has_license"] is True
            assert "advanced_rate_limiting" in debug_info["enabled_features"]

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Enable a feature to cache its status
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"

        # First call should check and cache
        result1 = PremiumFeatures.is_feature_enabled("advanced_rate_limiting")
        assert result1 is True

        # Clear cache
        PremiumFeatures.clear_cache()

        # Cache should be empty now
        assert len(PremiumFeatures._feature_status_cache) == 0

    def test_convenience_functions(self):
        """Test the convenience functions."""
        os.environ["LITELLM_ENABLE_ADVANCED_RATE_LIMITING"] = "true"

        # Test is_premium_feature_enabled
        assert is_premium_feature_enabled("advanced_rate_limiting") is True
        assert is_premium_feature_enabled("sso_auth") is False

        # Test require_premium_feature (should not raise)
        require_premium_feature("advanced_rate_limiting")

        # Test has_premium_access
        assert has_premium_access() is True

    def test_whitespace_handling_in_batch_features(self):
        """Test whitespace handling in LITELLM_PREMIUM_FEATURES."""
        os.environ["LITELLM_PREMIUM_FEATURES"] = "  advanced_rate_limiting , sso_auth  , prometheus_metrics  "

        assert PremiumFeatures.is_feature_enabled("advanced_rate_limiting") is True
        assert PremiumFeatures.is_feature_enabled("sso_auth") is True
        assert PremiumFeatures.is_feature_enabled("prometheus_metrics") is True

    def test_empty_batch_features_list(self):
        """Test handling of empty batch features list."""
        os.environ["LITELLM_PREMIUM_FEATURES"] = ""

        enabled_features = PremiumFeatures.get_enabled_features()

        # Should not have any features enabled from the empty list
        assert len(enabled_features) == 0

    def test_case_sensitivity_in_batch_features(self):
        """Test case sensitivity handling in batch features."""
        os.environ["LITELLM_PREMIUM_FEATURES"] = "ADVANCED_RATE_LIMITING,Sso_Auth"

        # Should work with lowercase conversion
        assert PremiumFeatures.is_feature_enabled("advanced_rate_limiting") is True
        assert PremiumFeatures.is_feature_enabled("sso_auth") is True

    def test_initialization_caching(self):
        """Test that initialization is cached properly."""
        # First initialization
        PremiumFeatures._initialize()
        first_check = PremiumFeatures._initialized

        # Second initialization should not change anything
        PremiumFeatures._initialize()
        second_check = PremiumFeatures._initialized

        assert first_check is True
        assert second_check is True
        assert first_check == second_check


if __name__ == "__main__":
    pytest.main([__file__])