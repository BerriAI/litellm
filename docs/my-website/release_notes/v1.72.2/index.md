---
title: "v1.72.2 - Bug Fixes & Stability Improvements"
slug: "v1-72-2"
date: 2025-06-07T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg

hide_table_of_contents: false
---

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info

This patch release is now available.

:::

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-v1.72.2
```
</TabItem>

<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.72.2
```
</TabItem>
</Tabs>

## Key Highlights

LiteLLM v1.72.2 is a patch release focused on critical bug fixes and stability improvements following v1.72.0. This release addresses several issues reported by the community:

- **Vector Store Permissions Fix**: Resolved permission checking edge cases for nested organization structures
- **Rate Limiting Accuracy**: Fixed sliding window calculation for concurrent requests
- **Aiohttp Transport Stability**: Improved error handling and retry logic for network failures
- **Bedrock Claude-4 Fixes**: Resolved parameter mapping issues for reasoning_effort
- **UI Session Management**: Fixed session timeout issues on the admin panel

---

## Critical Bug Fixes

### Vector Store Permissions
- Fixed permission inheritance for team members accessing organization-level vector stores - [PR](https://github.com/BerriAI/litellm/pull/11295)
- Resolved access control validation for cross-team vector store sharing - [PR](https://github.com/BerriAI/litellm/pull/11296)

### Rate Limiting
- Fixed sliding window request counter race condition - [PR](https://github.com/BerriAI/litellm/pull/11298)
- Improved accuracy for per-minute rate limit tracking - [PR](https://github.com/BerriAI/litellm/pull/11299)

### Aiohttp Transport
- Added proper handling for connection pool exhaustion - [PR](https://github.com/BerriAI/litellm/pull/11301)
- Fixed SSL certificate validation for custom endpoints - [PR](https://github.com/BerriAI/litellm/pull/11302)
- Improved retry logic for transient network failures - [PR](https://github.com/BerriAI/litellm/pull/11303)

---

## Provider-Specific Fixes

### Bedrock
- Fixed duplicate parameter error for Claude-4 with reasoning_effort - [PR](https://github.com/BerriAI/litellm/pull/11305)
- Resolved InvokeAgent response parsing for complex tool calls - [PR](https://github.com/BerriAI/litellm/pull/11306)

### Anthropic
- Fixed file upload handling for CSV files larger than 10MB - [PR](https://github.com/BerriAI/litellm/pull/11308)
- Resolved thinking block parsing in streaming responses - [PR](https://github.com/BerriAI/litellm/pull/11309)

### VertexAI
- Fixed global endpoint routing for EU regions - [PR](https://github.com/BerriAI/litellm/pull/11311)
- Resolved credential refresh for long-running sessions - [PR](https://github.com/BerriAI/litellm/pull/11312)

### Azure
- Fixed model deployment selection for O-series models - [PR](https://github.com/BerriAI/litellm/pull/11314)
- Resolved API version compatibility for image edits endpoint - [PR](https://github.com/BerriAI/litellm/pull/11315)

---

## UI & Management Fixes

### Admin Panel
- Fixed "Session Expired" errors during normal usage - [PR](https://github.com/BerriAI/litellm/pull/11317)
- Resolved team model display when using model aliases - [PR](https://github.com/BerriAI/litellm/pull/11318)
- Fixed vector store permission UI not updating correctly - [PR](https://github.com/BerriAI/litellm/pull/11319)

### Key Management
- Fixed key creation with pre-existing team assignments - [PR](https://github.com/BerriAI/litellm/pull/11321)
- Resolved budget tracking for keys with unlimited budgets - [PR](https://github.com/BerriAI/litellm/pull/11322)

---

## Logging & Monitoring Fixes

### Prometheus
- Fixed metric cardinality issue with end_user tracking - [PR](https://github.com/BerriAI/litellm/pull/11324)
- Resolved memory leak in metrics aggregation - [PR](https://github.com/BerriAI/litellm/pull/11325)

### Langfuse
- Fixed duplicate event logging on retries - [PR](https://github.com/BerriAI/litellm/pull/11327)
- Resolved client connection pool management - [PR](https://github.com/BerriAI/litellm/pull/11328)

---

## Performance Improvements

- Optimized database queries for team permission checks - [PR](https://github.com/BerriAI/litellm/pull/11330)
- Reduced memory usage for large batch request processing - [PR](https://github.com/BerriAI/litellm/pull/11331)
- Improved cache invalidation for model updates - [PR](https://github.com/BerriAI/litellm/pull/11332)

---

## Reliability Improvements

- Enhanced error recovery for background health checks - [PR](https://github.com/BerriAI/litellm/pull/11334)
- Improved handling of provider timeout errors - [PR](https://github.com/BerriAI/litellm/pull/11335)
- Better retry logic for transient database connection failures - [PR](https://github.com/BerriAI/litellm/pull/11336)

---

## Documentation Updates

- Updated Vector Store permissions documentation with examples - [PR](https://github.com/BerriAI/litellm/pull/11338)
- Added troubleshooting guide for aiohttp transport issues - [PR](https://github.com/BerriAI/litellm/pull/11339)
- Clarified rate limiting configuration options - [PR](https://github.com/BerriAI/litellm/pull/11340)

---

## Known Issues

- MCP Server permissions are still under development and will be addressed in v1.72.3
- Some users may experience intermittent issues with Bedrock Agent streaming responses

---

## Upgrade Notes

This is a patch release with no breaking changes. Users can safely upgrade from v1.72.0 or v1.72.1.

If you're using the aiohttp transport and experiencing issues, you can temporarily disable it by setting:
```bash
export DISABLE_AIOHTTP_TRANSPORT="True"
```

---

## Demo Instance

Test the latest changes on our demo instance:

- Instance: https://demo.litellm.ai/
- Login Credentials:
    - Username: admin
    - Password: sk-1234

## [Git Diff](https://github.com/BerriAI/litellm/compare/v1.72.0...v1.72.2)