#!/usr/bin/env python3
"""
Test script to verify that enterprise features are properly unlocked in the on-prem version.
"""

import sys
import importlib.util

def test_enterprise_imports():
    """Test that enterprise modules can be imported successfully."""
    try:
        # Try to import the enterprise module
        import litellm.enterprise
        print("✅ Successfully imported litellm.enterprise")

        # Try to import specific enterprise features
        from litellm.enterprise import *
        print("✅ Successfully imported all enterprise features")

        # Try to import enterprise callbacks
        import litellm.enterprise.enterprise_callbacks
        print("✅ Successfully imported enterprise callbacks")

        # Try to import enterprise integrations
        import litellm.enterprise.integrations
        print("✅ Successfully imported enterprise integrations")

        return True

    except ImportError as e:
        print(f"❌ Failed to import enterprise modules: {e}")
        return False

def test_proxy_features():
    """Test that proxy-specific enterprise features are available."""
    try:
        # Try to import proxy modules
        import litellm.enterprise.proxy
        print("✅ Successfully imported enterprise proxy features")

        return True

    except ImportError as e:
        print(f"❌ Failed to import enterprise proxy features: {e}")
        return False

def main():
    """Main test function."""
    print("Testing LiteLLM on-prem enterprise features...")
    print("=" * 50)

    success = True

    # Test enterprise imports
    success &= test_enterprise_imports()

    # Test proxy features
    success &= test_proxy_features()

    print("=" * 50)
    if success:
        print("🎉 All enterprise features are successfully unlocked!")
        print("✅ The on-prem version is ready for use.")
        return 0
    else:
        print("❌ Some enterprise features are not available.")
        print("⚠️  Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
