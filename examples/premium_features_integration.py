#!/usr/bin/env python3
"""
LiteLLM Premium Features Integration Example

This example demonstrates how to integrate the new premium features system
into existing LiteLLM proxy code and how to check for feature availability.
"""

import os
import sys
from typing import Optional

# Add the parent directory to the path so we can import litellm modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from litellm.proxy.auth.premium_features import PremiumFeatures, is_premium_feature_enabled, require_premium_feature


class PremiumFeatureManager:
    """
    Example class showing how to integrate premium features into your application.
    """

    def __init__(self):
        self._initialize_features()

    def _initialize_features(self):
        """Initialize features based on availability."""
        print("🚀 Initializing LiteLLM Premium Features...")

        # Check overall premium access
        if PremiumFeatures.has_premium_access():
            print("✅ Premium access detected")
        else:
            print("ℹ️  Using open-source features only")

        # Initialize individual features
        self._setup_routing_features()
        self._setup_security_features()
        self._setup_observability_features()
        self._setup_admin_features()

    def _setup_routing_features(self):
        """Setup routing-related premium features."""
        print("\n🛣️  Setting up Routing Features...")

        # Advanced Rate Limiting
        if is_premium_feature_enabled("advanced_rate_limiting"):
            print("  ✅ Advanced Rate Limiting v3 - ENABLED")
            self._enable_advanced_rate_limiting()
        else:
            print("  ❌ Advanced Rate Limiting - DISABLED (Premium Feature)")

        # Managed Files
        if is_premium_feature_enabled("managed_files"):
            print("  ✅ Managed Files - ENABLED")
            self._enable_managed_files()
        else:
            print("  ❌ Managed Files - DISABLED (Premium Feature)")

        # Tag Filtering
        if is_premium_feature_enabled("tag_filtering"):
            print("  ✅ Tag Filtering - ENABLED")
            self._enable_tag_filtering()
        else:
            print("  ❌ Tag Filtering - DISABLED (Premium Feature)")

    def _setup_security_features(self):
        """Setup security-related premium features."""
        print("\n🔒 Setting up Security Features...")

        # SSO Authentication
        if is_premium_feature_enabled("sso_auth"):
            print("  ✅ SSO Authentication - ENABLED")
            self._enable_sso_authentication()
        else:
            print("  ❌ SSO Authentication - DISABLED (Premium Feature)")

        # Custom Guardrails
        if is_premium_feature_enabled("custom_guardrails"):
            print("  ✅ Custom Guardrails - ENABLED")
            self._enable_custom_guardrails()
        else:
            print("  ❌ Custom Guardrails - DISABLED (Premium Feature)")

        # Secret Detection
        if is_premium_feature_enabled("secret_detection"):
            print("  ✅ Secret Detection - ENABLED")
            self._enable_secret_detection()
        else:
            print("  ❌ Secret Detection - DISABLED (Premium Feature)")

    def _setup_observability_features(self):
        """Setup observability-related premium features."""
        print("\n📊 Setting up Observability Features...")

        # Prometheus Metrics
        if is_premium_feature_enabled("prometheus_metrics"):
            print("  ✅ Prometheus Metrics - ENABLED")
            self._enable_prometheus_metrics()
        else:
            print("  ❌ Prometheus Metrics - DISABLED (Premium Feature)")

        # Custom Callbacks
        if is_premium_feature_enabled("custom_callbacks"):
            print("  ✅ Custom Callbacks - ENABLED")
            self._enable_custom_callbacks()
        else:
            print("  ❌ Custom Callbacks - DISABLED (Premium Feature)")

        # Email Notifications
        if is_premium_feature_enabled("email_notifications"):
            print("  ✅ Email Notifications - ENABLED")
            self._enable_email_notifications()
        else:
            print("  ❌ Email Notifications - DISABLED (Premium Feature)")

    def _setup_admin_features(self):
        """Setup administration-related premium features."""
        print("\n👥 Setting up Admin Features...")

        # Team Management
        if is_premium_feature_enabled("team_management"):
            print("  ✅ Team Management - ENABLED")
            self._enable_team_management()
        else:
            print("  ❌ Team Management - DISABLED (Premium Feature)")

        # Audit Logging
        if is_premium_feature_enabled("audit_logging"):
            print("  ✅ Audit Logging - ENABLED")
            self._enable_audit_logging()
        else:
            print("  ❌ Audit Logging - DISABLED (Premium Feature)")

        # Budget Tracking
        if is_premium_feature_enabled("budget_tracking"):
            print("  ✅ Budget Tracking - ENABLED")
            self._enable_budget_tracking()
        else:
            print("  ❌ Budget Tracking - DISABLED (Premium Feature)")

    # Feature implementation methods (simplified for example)
    def _enable_advanced_rate_limiting(self):
        """Enable advanced rate limiting functionality."""
        # In real implementation, this would setup the rate limiter
        print("    → Setting up dynamic rate limiter with priority reservations")

    def _enable_managed_files(self):
        """Enable managed files functionality."""
        # In real implementation, this would setup file management
        print("    → Setting up unified file ID system")

    def _enable_tag_filtering(self):
        """Enable tag-based filtering functionality."""
        # In real implementation, this would setup tag filtering
        print("    → Setting up tag-based model routing")

    def _enable_sso_authentication(self):
        """Enable SSO authentication functionality."""
        # In real implementation, this would setup SSO providers
        print("    → Setting up SSO authentication (Microsoft/Google/Generic)")

    def _enable_custom_guardrails(self):
        """Enable custom guardrails functionality."""
        # In real implementation, this would setup guardrail integrations
        print("    → Setting up enterprise guardrail integrations")

    def _enable_secret_detection(self):
        """Enable secret detection functionality."""
        # In real implementation, this would setup secret scanning
        print("    → Setting up advanced secret scanning")

    def _enable_prometheus_metrics(self):
        """Enable Prometheus metrics functionality."""
        # In real implementation, this would setup Prometheus integration
        print("    → Setting up Prometheus metrics collection")

    def _enable_custom_callbacks(self):
        """Enable custom callbacks functionality."""
        # In real implementation, this would setup callback controls
        print("    → Setting up dynamic callback controls")

    def _enable_email_notifications(self):
        """Enable email notifications functionality."""
        # In real implementation, this would setup email alerting
        print("    → Setting up enterprise email notifications")

    def _enable_team_management(self):
        """Enable team management functionality."""
        # In real implementation, this would setup team admin features
        print("    → Setting up advanced team management")

    def _enable_audit_logging(self):
        """Enable audit logging functionality."""
        # In real implementation, this would setup comprehensive logging
        print("    → Setting up comprehensive audit logging")

    def _enable_budget_tracking(self):
        """Enable budget tracking functionality."""
        # In real implementation, this would setup budget controls
        print("    → Setting up advanced budget tracking")

    def _enable_tag_filtering(self):
        """Enable tag-based filtering functionality."""
        # In real implementation, this would setup tag filtering
        print("    → Setting up tag-based model routing")


def example_usage_with_requirements():
    """
    Example showing how to use premium features with requirements checking.
    """
    print("\n🔧 Example: Feature Requirements Checking")
    print("=" * 50)

    try:
        # This will succeed if the feature is enabled, or raise an exception
        require_premium_feature("advanced_rate_limiting")
        print("✅ Advanced rate limiting is available - setting up priority queues...")

    except ValueError as e:
        print(f"❌ Cannot setup advanced rate limiting: {str(e)}")

    try:
        # This will succeed if the feature is enabled, or raise an exception
        require_premium_feature("sso_auth")
        print("✅ SSO authentication is available - setting up SSO providers...")

    except ValueError as e:
        print(f"❌ Cannot setup SSO authentication: {str(e)}")


def example_feature_discovery():
    """
    Example showing how to discover available and enabled features.
    """
    print("\n🔍 Example: Feature Discovery")
    print("=" * 30)

    # Get all available features
    all_features = PremiumFeatures.get_all_features()
    print(f"Total available features: {len(all_features)}")

    # Get features by category
    categories = {}
    for feature_name, feature in all_features.items():
        category = feature.category
        if category not in categories:
            categories[category] = []
        categories[category].append(feature_name)

    for category, features in categories.items():
        print(f"\n{category.title()} Features:")
        for feature in features:
            status = "✅ ENABLED" if is_premium_feature_enabled(feature) else "❌ DISABLED"
            feature_info = all_features[feature]
            print(f"  - {feature}: {status}")
            print(f"    {feature_info.description}")


def example_debug_status():
    """
    Example showing how to get debug information.
    """
    print("\n🐛 Example: Debug Status")
    print("=" * 25)

    debug_info = PremiumFeatures.debug_status()

    print(f"Has Premium License: {debug_info['has_license']}")
    print(f"Has Premium Access: {debug_info['has_premium_access']}")
    print(f"Enabled Features: {debug_info['enabled_features']}")
    print(f"Environment Flags: {debug_info['environment_flags']}")
    print(f"Individual Flags: {debug_info['individual_flags']}")


def main():
    """
    Main function demonstrating the premium features integration.
    """
    print("🎯 LiteLLM Premium Features Integration Example")
    print("=" * 60)

    # Show current environment configuration
    print("\n📋 Current Environment Configuration:")
    print("-" * 40)
    print(f"LITELLM_LICENSE: {'✅ Set' if os.getenv('LITELLM_LICENSE') else '❌ Not set'}")
    print(f"LITELLM_PREMIUM_FEATURES: {os.getenv('LITELLM_PREMIUM_FEATURES', 'Not set')}")

    # Show individual feature flags
    premium_features = PremiumFeatures.get_all_features()
    for feature_name in premium_features:
        flag_name = f"LITELLM_ENABLE_{feature_name.upper()}"
        flag_value = os.getenv(flag_name)
        if flag_value:
            print(f"{flag_name}: {flag_value}")

    # Initialize the feature manager
    feature_manager = PremiumFeatureManager()

    # Run examples
    example_usage_with_requirements()
    example_feature_discovery()
    example_debug_status()

    print("\n🎉 Premium Features Integration Example Complete!")
    print("\n💡 To enable premium features, set one or more of these environment variables:")
    print("   - LITELLM_LICENSE (your license key)")
    print("   - LITELLM_PREMIUM_FEATURES (comma-separated list)")
    print("   - LITELLM_ENABLE_<FEATURE_NAME>=true (individual features)")


if __name__ == "__main__":
    main()