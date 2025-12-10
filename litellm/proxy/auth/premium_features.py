"""
LiteLLM Premium Features Management System

This module provides a centralized way to manage premium features using environment variables.
It extends the existing license-based system to allow granular control over premium features.

Usage:
    from litellm.proxy.auth.premium_features import PremiumFeatures

    # Check if a specific feature is enabled
    if PremiumFeatures.is_feature_enabled("advanced_rate_limiting"):
        # Enable advanced rate limiting
        pass

    # Check if user has premium access (license or feature flags)
    if PremiumFeatures.has_premium_access():
        # Enable premium features
        pass
"""

import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.litellm_license import LicenseCheck


@dataclass
class PremiumFeature:
    """Represents a premium feature with its configuration."""
    name: str
    description: str
    requires_license: bool = True  # If True, requires LITELLM_LICENSE unless explicitly enabled
    default_enabled: bool = False
    category: str = "general"


class PremiumFeatures:
    """
    Centralized premium features management system.

    Features can be controlled via:
    1. LITELLM_LICENSE - Traditional license-based activation
    2. LITELLM_PREMIUM_FEATURES - Comma-separated list of enabled features
    3. LITELLM_ENABLE_<FEATURE_NAME> - Individual feature flags
    """

    # Registry of all available premium features
    _FEATURES_REGISTRY: Dict[str, PremiumFeature] = {
        # Advanced Routing & Performance
        "advanced_rate_limiting": PremiumFeature(
            name="advanced_rate_limiting",
            description="Dynamic rate limiter v3 with priority-based resource allocation",
            category="routing"
        ),
        "managed_files": PremiumFeature(
            name="managed_files",
            description="Unified file ID system across multiple models",
            category="routing"
        ),
        "tag_filtering": PremiumFeature(
            name="tag_filtering",
            description="Advanced tag-based model filtering and routing",
            category="routing"
        ),

        # Security & Guardrails
        "sso_auth": PremiumFeature(
            name="sso_auth",
            description="SSO authentication (Microsoft/Google/Generic) for unlimited users",
            category="security"
        ),
        "custom_guardrails": PremiumFeature(
            name="custom_guardrails",
            description="Enterprise-specific guardrail integrations",
            category="security"
        ),
        "blocked_users": PremiumFeature(
            name="blocked_users",
            description="Advanced user blocking and allowlist management",
            category="security"
        ),
        "secret_detection": PremiumFeature(
            name="secret_detection",
            description="Advanced secret scanning in requests",
            category="security"
        ),

        # Management & Administration
        "team_management": PremiumFeature(
            name="team_management",
            description="Advanced team admin assignments and role management",
            category="admin"
        ),
        "audit_logging": PremiumFeature(
            name="audit_logging",
            description="Comprehensive request/response logging with audit trails",
            category="admin"
        ),
        "budget_tracking": PremiumFeature(
            name="budget_tracking",
            description="Advanced spend tracking and budget limits",
            category="admin"
        ),

        # Observability & Integrations
        "prometheus_metrics": PremiumFeature(
            name="prometheus_metrics",
            description="Advanced Prometheus metrics and monitoring integration",
            category="observability"
        ),
        "custom_callbacks": PremiumFeature(
            name="custom_callbacks",
            description="Dynamic callback disabling via headers and advanced callback control",
            category="observability"
        ),
        "email_notifications": PremiumFeature(
            name="email_notifications",
            description="Enterprise alerting and email notification system",
            category="observability"
        ),

        # Database & Storage
        "vector_store_acl": PremiumFeature(
            name="vector_store_acl",
            description="Vector store access control and management",
            category="storage"
        ),

        # UI Features
        "advanced_dashboard": PremiumFeature(
            name="advanced_dashboard",
            description="Advanced UI dashboard features and analytics",
            category="ui",
            requires_license=False  # Can be enabled without license for testing
        )
    }

    # Cache for feature enablement status
    _feature_status_cache: Dict[str, bool] = {}
    _license_check: Optional[LicenseCheck] = None
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize the premium features system."""
        if cls._initialized:
            return

        cls._license_check = LicenseCheck()
        cls._initialized = True
        verbose_proxy_logger.info("Premium Features system initialized")

    @classmethod
    def has_premium_license(cls) -> bool:
        """
        Check if user has a valid premium license.

        Returns:
            bool: True if LITELLM_LICENSE is valid
        """
        cls._initialize()
        if cls._license_check is None:
            return False
        return cls._license_check.is_premium()

    @classmethod
    def has_premium_access(cls) -> bool:
        """
        Check if user has any premium access (license or enabled features).

        Returns:
            bool: True if user has license or any premium features enabled
        """
        # Check for valid license
        if cls.has_premium_license():
            return True

        # Check if any premium features are explicitly enabled
        enabled_features = cls.get_enabled_features()
        return len(enabled_features) > 0

    @classmethod
    def is_feature_enabled(cls, feature_name: str) -> bool:
        """
        Check if a specific premium feature is enabled.

        Priority order:
        1. Individual feature flag (LITELLM_ENABLE_<FEATURE_NAME>)
        2. Premium features list (LITELLM_PREMIUM_FEATURES)
        3. Valid license (if feature requires it)

        Args:
            feature_name: Name of the feature to check

        Returns:
            bool: True if feature is enabled
        """
        cls._initialize()

        # Check cache first
        if feature_name in cls._feature_status_cache:
            return cls._feature_status_cache[feature_name]

        feature = cls._FEATURES_REGISTRY.get(feature_name)
        if feature is None:
            verbose_proxy_logger.warning(f"Unknown premium feature: {feature_name}")
            return False

        enabled = False

        # Check individual feature flag first (highest priority)
        flag_name = f"LITELLM_ENABLE_{feature_name.upper()}"
        if os.getenv(flag_name, "").lower() in ("true", "1", "yes", "on"):
            enabled = True
            verbose_proxy_logger.info(f"Feature {feature_name} enabled via {flag_name}")

        # Check premium features list
        elif not enabled:
            premium_features_env = os.getenv("LITELLM_PREMIUM_FEATURES", "")
            enabled_features = [f.strip().lower() for f in premium_features_env.split(",") if f.strip()]
            if feature_name.lower() in enabled_features:
                enabled = True
                verbose_proxy_logger.info(f"Feature {feature_name} enabled via LITELLM_PREMIUM_FEATURES")

        # Check license (if feature requires it and not already enabled)
        elif not enabled and feature.requires_license:
            if cls.has_premium_license():
                enabled = True
                verbose_proxy_logger.info(f"Feature {feature_name} enabled via valid license")

        # Cache the result
        cls._feature_status_cache[feature_name] = enabled

        return enabled

    @classmethod
    def get_enabled_features(cls) -> List[str]:
        """
        Get list of all enabled premium features.

        Returns:
            List[str]: List of enabled feature names
        """
        cls._initialize()
        enabled_features = []

        for feature_name in cls._FEATURES_REGISTRY:
            if cls.is_feature_enabled(feature_name):
                enabled_features.append(feature_name)

        return enabled_features

    @classmethod
    def get_features_by_category(cls, category: str) -> List[str]:
        """
        Get all features in a specific category.

        Args:
            category: Category name (routing, security, admin, observability, etc.)

        Returns:
            List[str]: List of feature names in the category
        """
        return [
            feature_name for feature_name, feature in cls._FEATURES_REGISTRY.items()
            if feature.category == category
        ]

    @classmethod
    def require_premium_feature(cls, feature_name: str) -> None:
        """
        Raise an exception if a premium feature is not enabled.
        This is a convenience method for use in code that requires premium features.

        Args:
            feature_name: Name of the required feature

        Raises:
            ValueError: If feature is not enabled
        """
        if not cls.is_feature_enabled(feature_name):
            feature = cls._FEATURES_REGISTRY.get(feature_name)
            description = feature.description if feature else "Unknown feature"

            error_msg = (
                f"PREMIUM FEATURE: {feature_name} - {description}\n"
                f"This is a LiteLLM Enterprise feature. Please add a 'LITELLM_LICENSE' to your .env to enable this.\n"
                f"Alternatively, enable it with: LITELLM_ENABLE_{feature_name.upper()}=true\n"
                f"Or enable multiple features with: LITELLM_PREMIUM_FEATURES={feature_name},other_feature\n"
                f"Get a license: https://docs.litellm.ai/docs/proxy/enterprise"
            )
            raise ValueError(error_msg)

    @classmethod
    def get_all_features(cls) -> Dict[str, PremiumFeature]:
        """
        Get all available premium features.

        Returns:
            Dict[str, PremiumFeature]: Dictionary of all features
        """
        return cls._FEATURES_REGISTRY.copy()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the feature status cache."""
        cls._feature_status_cache.clear()
        verbose_proxy_logger.debug("Premium features cache cleared")

    @classmethod
    def debug_status(cls) -> Dict[str, any]:
        """
        Get debug information about premium features status.

        Returns:
            Dict[str, any]: Debug information
        """
        cls._initialize()

        return {
            "has_license": cls.has_premium_license(),
            "has_premium_access": cls.has_premium_access(),
            "enabled_features": cls.get_enabled_features(),
            "environment_flags": {
                "LITELLM_LICENSE": os.getenv("LITELLM_LICENSE") is not None,
                "LITELLM_PREMIUM_FEATURES": os.getenv("LITELLM_PREMIUM_FEATURES", ""),
            },
            "individual_flags": {
                feature_name: os.getenv(f"LITELLM_ENABLE_{feature_name.upper()}", "")
                for feature_name in cls._FEATURES_REGISTRY.keys()
                if os.getenv(f"LITELLM_ENABLE_{feature_name.upper()}")
            }
        }


# Convenience functions for common usage patterns
def is_premium_feature_enabled(feature_name: str) -> bool:
    """
    Convenience function to check if a premium feature is enabled.

    Args:
        feature_name: Name of the feature to check

    Returns:
        bool: True if feature is enabled
    """
    return PremiumFeatures.is_feature_enabled(feature_name)


def require_premium_feature(feature_name: str) -> None:
    """
    Convenience function that raises an exception if a premium feature is not enabled.

    Args:
        feature_name: Name of the required feature

    Raises:
        ValueError: If feature is not enabled
    """
    PremiumFeatures.require_premium_feature(feature_name)


def has_premium_access() -> bool:
    """
    Convenience function to check if user has any premium access.

    Returns:
        bool: True if user has license or any premium features enabled
    """
    return PremiumFeatures.has_premium_access()