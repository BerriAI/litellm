# Scaling with Multiple Instances

Learn how to deploy LiteLLM across multiple instances while maintaining centralized administration and avoiding duplication of management overhead.

## Overview

When scaling LiteLLM for production use, you may want to deploy multiple instances across different regions or availability zones while maintaining a single point of administration. This guide covers how to set up a distributed LiteLLM deployment with:

- **Regional Worker Instances**: Handle LLM requests for users in specific regions
- **Centralized Admin Instance**: Manages configuration, users, keys, and monitoring

## Architecture Pattern: Regional + Admin Instances

### Typical Deployment Scenario

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Admin Instance    │    │   US Worker         │    │   EU Worker         │
│  admin.company.com  │    │  us.company.com     │    │  eu.company.com     │
│                     │    │                     │    │                     │
│  ✅ Admin UI        │    │  ❌ Admin UI        │    │  ❌ Admin UI        │
│  ✅ Management APIs │    │  ❌ Management APIs │    │  ❌ Management APIs │
│  ✅ User/Key Mgmt   │    │  ✅ LLM APIs        │    │  ✅ LLM APIs        │
│  ✅ Monitoring      │    │  ✅ Health Checks   │    │  ❌ Health Checks   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                              ┌───────────────┐
                              │  Shared DB    │
                              │   (Global)    │
                              └───────────────┘
```

### Benefits of This Architecture

1. **Reduced Management Overhead**: Only one instance needs admin capabilities
2. **Regional Performance**: Users get low-latency access from their region
3. **Centralized Control**: All administration happens from a single interface
4. **Security**: Limit admin access to designated instances only
5. **Cost Efficiency**: Avoid duplicating admin infrastructure

## Configuration

### Admin Instance Configuration

The admin instance handles all management operations and provides the UI.

**Environment Variables for Admin Instance:**
```bash
# Keep admin capabilities enabled (default behavior)
# DISABLE_ADMIN_UI=false          # Admin UI available
# DISABLE_ADMIN_ENDPOINTS=false   # Management APIs available

DATABASE_URL=postgresql://user:pass@global-db:5432/litellm
LITELLM_MASTER_KEY=your-master-key
```

### Worker Instance Configuration

Worker instances handle LLM requests but have admin capabilities disabled.

**Environment Variables for Worker Instances:**
```bash
# Disable admin capabilities
DISABLE_ADMIN_UI=true           # No admin UI
DISABLE_ADMIN_ENDPOINTS=true    # No management endpoints

DATABASE_URL=postgresql://user:pass@global-db:5432/litellm
LITELLM_MASTER_KEY=your-master-key
```

## Environment Variables Reference

### `DISABLE_ADMIN_UI`

Disables the LiteLLM Admin UI interface.

- **Default**: `false`
- **Worker Instances**: Set to `true`
- **Admin Instance**: Leave as `false` (or don't set)

```bash
# Worker instances
DISABLE_ADMIN_UI=true
```

**Effect**: When enabled, the web UI at `/ui` becomes unavailable.

### `DISABLE_ADMIN_ENDPOINTS`

Disables all management/admin API endpoints.

- **Default**: `false`  
- **Worker Instances**: Set to `true`
- **Admin Instance**: Leave as `false` (or don't set)

```bash
# Worker instances  
DISABLE_ADMIN_ENDPOINTS=true
```

**Disabled Endpoints Include**:
- `/key/*` - Key management
- `/user/*` - User management  
- `/team/*` - Team management
- `/config/*` - Configuration updates
- All other administrative endpoints

**Available Endpoints** (when disabled):
- `/chat/completions` - LLM requests
- `/v1/*` - OpenAI-compatible APIs
- `/vertex_ai/*` - Vertex AI pass-through APIs
- `/bedrock/*` - Bedrock pass-through APIs
- `/health` - Basic health check
- `/metrics` - Prometheus metrics
- All other LLM API endpoints


## Usage Patterns

### Client Usage

**For LLM Requests** (use regional endpoints):
```python
import openai

# US users
client_us = openai.OpenAI(
    base_url="https://us.company.com/v1",
    api_key="your-litellm-key"
)

# EU users  
client_eu = openai.OpenAI(
    base_url="https://eu.company.com/v1", 
    api_key="your-litellm-key"
)

response = client_us.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**For Administration** (use admin endpoint):
```python
import requests

# Create a new API key
response = requests.post(
    "https://admin.company.com/key/generate",
    headers={"Authorization": "Bearer sk-1234"},
    json={"duration": "30d"}
)
```

## Related Documentation

- [Virtual Keys](./virtual_keys.md) - Managing API keys and users
- [Health Checks](./health.md) - Monitoring instance health  
- [Prometheus Metrics](./logging.md#prometheus-metrics) - Collecting metrics
- [Production Deployment](./prod.md) - Production best practices 