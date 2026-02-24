---
slug: responses-api-encrypted-content-incident
title: "Incident Report: Encrypted Content Failures in Multi-Region Responses API Load Balancing"
date: 2026-02-24T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
tags: [incident-report, proxy, responses-api, load-balancing]
hide_table_of_contents: false
---

**Date:** Feb 24, 2026  
**Duration:** Ongoing (until fix deployed)  
**Severity:** High (for users load balancing Responses API across different API keys)  
**Status:** Resolved

## Summary

When load balancing OpenAI's Responses API across deployments with **different API keys** (e.g., different Azure regions or OpenAI organizations), follow-up requests containing encrypted content items (like `rs_...` reasoning items) would fail with:

```json
{
  "error": {
    "message": "The encrypted content for item rs_0d09d6e56879e76500699d6feee41c8197bd268aae76141f87 could not be verified. Reason: Encrypted content organization_id did not match the target organization.",
    "type": "invalid_request_error",
    "code": "invalid_encrypted_content"
  }
}
```

Encrypted content items are cryptographically tied to the API key's organization that created them. When the router load balanced a follow-up request to a deployment with a different API key, decryption failed.

- **Responses API calls with encrypted content:** Complete failure when routed to wrong deployment
- **Initial requests:** Unaffected — only follow-up requests containing encrypted items failed
- **Other API endpoints:** No impact — chat completions, embeddings, etc. functioned normally

{/* truncate */}

---

## Background

OpenAI's Responses API can return encrypted "reasoning items" (with IDs like `rs_...`) that contain intermediate reasoning steps. These items are encrypted with the organization's key and can only be decrypted by the same organization's API key.

When load balancing across deployments with different API keys, the existing affinity mechanisms were insufficient:

- **`responses_api_deployment_check`**: Requires `previous_response_id` which some clients (like Codex) don't provide
- **`deployment_affinity`**: Too broad — pins *all* requests from a user to one deployment, reducing effective quota by the number of users
- **`session_affinity`**: Requires explicit session IDs and still reduces quota

```mermaid
flowchart TD
    A["1. Initial request to Responses API
    router.aresponses()"] --> B["2. Router load balances to Deployment A
    (API Key 1, Azure East US)"]
    B --> C["3. Response contains encrypted item
    rs_abc123 (encrypted with Org 1 key)"]
    C --> D["4. Follow-up request includes rs_abc123 in input"]
    D --> E["5. Router load balances to Deployment B
    (API Key 2, Azure West Europe)"]
    E -->|"Different API key"| F["6. ❌ Deployment B cannot decrypt rs_abc123
    Error: invalid_encrypted_content"]
    
    D -.->|"With encrypted_content_affinity"| G["5b. Router detects rs_abc123 was created by Deployment A"]
    G --> H["6b. ✅ Routes to Deployment A (bypasses rate limits)
    Request succeeds"]

    style F fill:#f8d7da,stroke:#dc3545
    style H fill:#d4edda,stroke:#28a745
    style E fill:#fff3cd,stroke:#ffc107
    style G fill:#d4edda,stroke:#28a745
```

---

## Root Cause

LiteLLM's router had no mechanism to track which deployment created specific encrypted content items and route follow-up requests accordingly. The router treated all deployments as interchangeable, leading to decryption failures when encrypted content crossed organizational boundaries.

**The Problem Flow:**

1. User calls `router.aresponses()` with model `gpt-5.1-codex`
2. Router load balances to Deployment A (Azure East US, API Key 1)
3. Response contains encrypted reasoning item `rs_abc123` (encrypted with Org 1's key)
4. User makes follow-up request with `rs_abc123` in the input
5. Router load balances to Deployment B (Azure West Europe, API Key 2)
6. Deployment B tries to decrypt `rs_abc123` with Org 2's key → **fails**

**Why Existing Solutions Didn't Work:**

- **`previous_response_id`**: Not provided by all clients (e.g., Codex)
- **`deployment_affinity`**: Pins *all* user requests to one deployment → reduces quota to 1/N where N = number of deployments
- **`session_affinity`**: Requires explicit session management and still reduces quota

**Timeline:**

1. Users configured multi-region Responses API load balancing with different API keys
2. Initial requests succeeded, but follow-up requests with encrypted content failed intermittently
3. Error rate correlated with number of deployments (more deployments = higher chance of routing to wrong one)
4. Investigation revealed encrypted content was organization-bound
5. Existing affinity mechanisms deemed unsuitable (quota reduction, missing `previous_response_id`)
6. New solution designed and implemented: `encrypted_content_affinity`

---

## The Fix

Implemented a new `encrypted_content_affinity` pre-call check that intelligently tracks encrypted content and routes follow-up requests **only when necessary**.

### Implementation

**1. New `EncryptedContentAffinityCheck` Class** ([`encrypted_content_affinity_check.py`](https://github.com/BerriAI/litellm/blob/main/litellm/litellm/router_utils/pre_call_checks/encrypted_content_affinity_check.py))

```python
class EncryptedContentAffinityCheck(CustomLogger):
    """
    Routes follow-up Responses API requests to the deployment that produced
    the encrypted output items they reference.
    """
    
    async def async_log_success_event(self, kwargs, response_obj, ...):
        """Track: Extract item IDs from response output, cache item_id → deployment_id"""
        output = self._get_output_from_response(response_obj)
        item_ids = self._extract_item_ids_from_output(output)
        model_id = self._get_model_id_from_kwargs(kwargs)
        
        for item_id in item_ids:
            await self.cache.async_set_cache(
                f"encrypted_content_affinity:v1:{item_id}",
                model_id,
                ttl=86400,  # 24 hours
            )
    
    async def async_filter_deployments(self, model, healthy_deployments, ...):
        """Route: Check if input contains tracked items, pin to originating deployment"""
        input_item_ids = self._extract_item_ids_from_input(request_kwargs.get("input"))
        
        for item_id in input_item_ids:
            cached_model_id = await self.cache.async_get_cache(f"...:{item_id}")
            if cached_model_id:
                deployment = self._find_deployment_by_model_id(
                    healthy_deployments, cached_model_id
                )
                if deployment:
                    # Signal to bypass rate limits (encrypted content must go here)
                    request_kwargs["_encrypted_content_affinity_pinned"] = True
                    return [deployment]
        
        return healthy_deployments  # Normal load balancing
```

**2. Rate Limit Bypass** ([`router.py#L8656-L8660`](https://github.com/BerriAI/litellm/blob/main/litellm/litellm/router.py#L8656-L8660))

When encrypted content requires a specific deployment, RPM/TPM limits are bypassed (the request would fail on any other deployment anyway):

```python
# In async_get_available_deployment, after filtering healthy deployments:
if (
    request_kwargs.get("_encrypted_content_affinity_pinned")
    and len(healthy_deployments) == 1
):
    return healthy_deployments[0]  # Bypass routing strategy (RPM/TPM checks)
```

**3. Configuration**

```yaml
router_settings:
  routing_strategy: usage-based-routing-v2
  enable_pre_call_checks: true
  optional_pre_call_checks:
    - encrypted_content_affinity
  deployment_affinity_ttl_seconds: 86400  # 24 hours
```

### Key Benefits

✅ **No quota reduction**: Only pins requests containing tracked encrypted items  
✅ **Bypasses rate limits**: When encrypted content requires a specific deployment, RPM/TPM limits don't block it  
✅ **No `previous_response_id` required**: Works by tracking item IDs in response output  
✅ **Globally safe**: Can be enabled for all models; non-Responses-API calls are unaffected  
✅ **Surgical precision**: Normal requests continue to load balance freely

---

## Remediation

| # | Action | Status | Code |
|---|---|---|---|
| 1 | Create `EncryptedContentAffinityCheck` class with tracking and routing logic | ✅ Done | [`encrypted_content_affinity_check.py`](https://github.com/BerriAI/litellm/blob/main/litellm/litellm/router_utils/pre_call_checks/encrypted_content_affinity_check.py) |
| 2 | Add `encrypted_content_affinity` to `OptionalPreCallChecks` type | ✅ Done | [`router.py`](https://github.com/BerriAI/litellm/blob/main/litellm/litellm/types/router.py) |
| 3 | Wire up check in `Router.add_optional_pre_call_checks` | ✅ Done | [`router.py#L8656-L8660`](https://github.com/BerriAI/litellm/blob/main/litellm/litellm/router.py#L8656-L8660) |
| 4 | Implement rate limit bypass for affinity-pinned requests | ✅ Done | [`router.py#L8656-L8660`](https://github.com/BerriAI/litellm/blob/main/litellm/litellm/router.py#L8656-L8660) |
| 5 | Unit tests: tracking, routing, no-op for non-Responses-API, RPM bypass | ✅ Done | [`test_encrypted_content_affinity_check.py`](https://github.com/BerriAI/litellm/blob/main/litellm/tests/test_litellm/router_utils/pre_call_checks/test_encrypted_content_affinity_check.py) |
| 6 | Documentation: Responses API guide, load balancing guide, config reference | ✅ Done | [Docs](https://docs.litellm.ai/docs/response_api#encrypted-content-affinity-multi-region-load-balancing) |

---

## Migration Guide

### Before (Using `deployment_affinity`)

```yaml
router_settings:
  optional_pre_call_checks:
    - deployment_affinity  # ❌ Reduces quota by number of users
```

**Problem:** All requests from a user pin to one deployment, reducing effective quota to 1/N.

### After (Using `encrypted_content_affinity`)

```yaml
router_settings:
  optional_pre_call_checks:
    - encrypted_content_affinity  # ✅ Only pins requests with encrypted content
```

**Benefit:** Normal requests load balance freely, only encrypted content requests pin when necessary.

---
