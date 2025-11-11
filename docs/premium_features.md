# LiteLLM Premium Features Configuration

This document explains how to configure and use LiteLLM premium features using environment variables.

## Overview

LiteLLM provides two ways to enable premium features:

1. **License-based**: Traditional activation using `LITELLM_LICENSE`
2. **Feature-based**: Granular control using environment variables

## Environment Variables

### Primary License Variable
- `LITELLM_LICENSE` - Your LiteLLM Enterprise license key

### Feature Control Variables

#### Individual Feature Flags
```bash
# Enable specific features
LITELLM_ENABLE_ADVANCED_RATE_LIMITING=true
LITELLM_ENABLE_MANAGED_FILES=true
LITELLM_ENABLE_SSO_AUTH=true
```

#### Batch Feature Activation
```bash
# Enable multiple features at once
LITELLM_PREMIUM_FEATURES=advanced_rate_limiting,managed_files,sso_auth,prometheus_metrics
```

## Available Premium Features

### 🚀 Advanced Routing & Performance

| Feature | Environment Flag | Description |
|---------|------------------|-------------|
| `advanced_rate_limiting` | `LITELLM_ENABLE_ADVANCED_RATE_LIMITING` | Dynamic rate limiter v3 with priority-based resource allocation |
| `managed_files` | `LITELLM_ENABLE_MANAGED_FILES` | Unified file ID system across multiple models |
| `tag_filtering` | `LITELLM_ENABLE_TAG_FILTERING` | Advanced tag-based model filtering and routing |

### 🔒 Security & Guardrails

| Feature | Environment Flag | Description |
|---------|------------------|-------------|
| `sso_auth` | `LITELLM_ENABLE_SSO_AUTH` | SSO authentication for unlimited users (Microsoft/Google/Generic) |
| `custom_guardrails` | `LITELLM_ENABLE_CUSTOM_GUARDRAILS` | Enterprise-specific guardrail integrations |
| `blocked_users` | `LITELLM_ENABLE_BLOCKED_USERS` | Advanced user blocking and allowlist management |
| `secret_detection` | `LITELLM_ENABLE_SECRET_DETECTION` | Advanced secret scanning in requests |

### 👥 Management & Administration

| Feature | Environment Flag | Description |
|---------|------------------|-------------|
| `team_management` | `LITELLM_ENABLE_TEAM_MANAGEMENT` | Advanced team admin assignments and role management |
| `audit_logging` | `LITELLM_ENABLE_AUDIT_LOGGING` | Comprehensive request/response logging with audit trails |
| `budget_tracking` | `LITELLM_ENABLE_BUDGET_TRACKING` | Advanced spend tracking and budget limits |

### 📊 Observability & Integrations

| Feature | Environment Flag | Description |
|---------|------------------|-------------|
| `prometheus_metrics` | `LITELLM_ENABLE_PROMETHEUS_METRICS` | Advanced Prometheus metrics and monitoring integration |
| `custom_callbacks` | `LITELLM_ENABLE_CUSTOM_CALLBACKS` | Dynamic callback disabling via headers and advanced callback control |
| `email_notifications` | `LITELLM_ENABLE_EMAIL_NOTIFICATIONS` | Enterprise alerting and email notification system |

### 💾 Database & Storage

| Feature | Environment Flag | Description |
|---------|------------------|-------------|
| `vector_store_acl` | `LITELLM_ENABLE_VECTOR_STORE_ACL` | Vector store access control and management |

### 🎨 UI Features

| Feature | Environment Flag | Description |
|---------|------------------|-------------|
| `advanced_dashboard` | `LITELLM_ENABLE_ADVANCED_DASHBOARD` | Advanced UI dashboard features and analytics |

## Usage Examples

### Method 1: License-based (Traditional)
```bash
# Set your license key - this enables ALL premium features
export LITELLM_LICENSE="your-license-key-here"
```

### Method 2: Individual Feature Flags
```bash
# Enable specific features
export LITELLM_ENABLE_ADVANCED_RATE_LIMITING=true
export LITELLM_ENABLE_PROMETHEUS_METRICS=true
export LITELLM_ENABLE_SSO_AUTH=true
```

### Method 3: Batch Feature Activation
```bash
# Enable multiple features
export LITELLM_PREMIUM_FEATURES="advanced_rate_limiting,prometheus_metrics,sso_auth,team_management"
```

### Method 4: Mixed Approach
```bash
# Combine license with additional feature flags
export LITELLM_LICENSE="your-license-key-here"
export LITELLM_ENABLE_ADVANCED_DASHBOARD=true  # For testing UI features
```

## Configuration Priority

The system checks for feature enablement in this order:

1. **Individual Feature Flags** (`LITELLM_ENABLE_<FEATURE_NAME>`) - Highest priority
2. **Batch Feature List** (`LITELLM_PREMIUM_FEATURES`)
3. **Valid License** (`LITELLM_LICENSE`) - If the feature requires a license

## Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  litellm:
    image: litellm/litellm:main
    environment:
      # License-based activation
      - LITELLM_LICENSE=${LITELLM_LICENSE}

      # Or feature-based activation
      - LITELLM_PREMIUM_FEATURES=advanced_rate_limiting,prometheus_metrics,sso_auth

      # Or individual flags
      - LITELLM_ENABLE_ADVANCED_RATE_LIMITING=true
      - LITELLM_ENABLE_PROMETHEUS_METRICS=true
```

## Environment File (.env)

```bash
# .env file

# Option 1: License-based (enables all premium features)
LITELLM_LICENSE=sk-your-license-key-here

# Option 2: Feature-based activation
LITELLM_PREMIUM_FEATURES=advanced_rate_limiting,prometheus_metrics,sso_auth,team_management

# Option 3: Individual feature flags
LITELLM_ENABLE_ADVANCED_RATE_LIMITING=true
LITELLM_ENABLE_PROMETHEUS_METRICS=true
LITELLM_ENABLE_SSO_AUTH=true
LITELLM_ENABLE_TEAM_MANAGEMENT=true

# Additional proxy settings
LITELLM_MASTER_KEY=sk-your-master-key
```

## Programming Interface

### Checking Feature Availability

```python
from litellm.proxy.auth.premium_features import PremiumFeatures, is_premium_feature_enabled, require_premium_feature

# Method 1: Direct check
if PremiumFeatures.is_feature_enabled("advanced_rate_limiting"):
    # Enable advanced rate limiting
    pass

# Method 2: Convenience function
if is_premium_feature_enabled("prometheus_metrics"):
    # Enable Prometheus metrics
    pass

# Method 3: Require feature (raises exception if not available)
require_premium_feature("sso_auth")  # Will raise ValueError if not enabled

# Check if user has any premium access
if PremiumFeatures.has_premium_access():
    # Show premium features in UI
    pass
```

### Feature Management

```python
from litellm.proxy.auth.premium_features import PremiumFeatures

# Get all enabled features
enabled_features = PremiumFeatures.get_enabled_features()
print(f"Enabled features: {enabled_features}")

# Get features by category
routing_features = PremiumFeatures.get_features_by_category("routing")
security_features = PremiumFeatures.get_features_by_category("security")

# Debug status
debug_info = PremiumFeatures.debug_status()
print(f"Premium features status: {debug_info}")
```

### Integration in Existing Code

```python
# In your proxy configuration
from litellm.proxy.auth.premium_features import require_premium_feature

def setup_dynamic_rate_limiter():
    """Setup advanced rate limiting if premium feature is available."""
    require_premium_feature("advanced_rate_limiting")
    # Your rate limiting setup code here
    pass

def setup_sso_authentication():
    """Setup SSO authentication if premium feature is available."""
    require_premium_feature("sso_auth")
    # Your SSO setup code here
    pass
```

## Testing Premium Features

For testing and development, you can enable features without a license:

```bash
# Enable specific features for testing
export LITELLM_ENABLE_ADVANCED_RATE_LIMITING=true
export LITELLM_ENABLE_PROMETHEUS_METRICS=true

# Or enable all features for testing (without license)
export LITELLM_PREMIUM_FEATURES=advanced_rate_limiting,managed_files,sso_auth,custom_guardrails,blocked_users,team_management,audit_logging,budget_tracking,prometheus_metrics,custom_callbacks,email_notifications,vector_store_acl,advanced_dashboard
```

## Troubleshooting

### Common Issues

1. **Feature not enabled**: Check the environment variable names and values
2. **License validation failing**: Verify your license key is valid
3. **Priority confusion**: Remember that individual flags override batch lists and license

### Debug Information

```python
from litellm.proxy.auth.premium_features import PremiumFeatures

# Print debug status
debug_info = PremiumFeatures.debug_status()
import json
print(json.dumps(debug_info, indent=2))
```

### Clear Cache

If you've changed environment variables and need to clear the feature cache:

```python
from litellm.proxy.auth.premium_features import PremiumFeatures
PremiumFeatures.clear_cache()
```

## Migration from License-Only

If you're currently using only `LITELLM_LICENSE`:

1. **No changes needed** - existing licenses continue to work
2. **Optional granular control** - add feature flags for more control
3. **Testing** - use feature flags to test specific features without full license

## Support

For questions about premium features:

- Documentation: https://docs.litellm.ai/docs/proxy/enterprise
- License requests: https://docs.litellm.ai/docs/proxy/enterprise
- Issues: https://github.com/BerriAI/litellm/issues