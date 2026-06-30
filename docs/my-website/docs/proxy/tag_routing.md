import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Tag-Based Routing

Route requests to specific deployments based on tags sent in request metadata or the `x-litellm-tags` header. Use this to enforce cost tiers, provider preferences, or team-level isolation without maintaining separate proxy endpoints.

## Quick Start

### 1. Tag your deployments in config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      tags: ["free"]

  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      tags: ["paid"]

router_settings:
  enable_tag_filtering: true
```

### 2. Send a request with a tag

<Tabs>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="Tag via metadata"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"tags": ["paid"]}
  }'
```

</TabItem>
<TabItem value="header" label="x-litellm-tags header">

```bash showLineNumbers title="Tag via header"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -H "x-litellm-tags: paid" \
  -d '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}'
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python showLineNumbers title="Tag via Python SDK"
import litellm

response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"tags": ["paid"]},
)
```

</TabItem>
</Tabs>

The router picks only deployments whose `tags` list intersects the request tags. If no match is found and no default deployment is configured, the request fails with a `no_deployments_with_tag_routing` error.

## Default Deployments

Tag a deployment `default` to use it for requests that carry no tags. Requests with an explicit `default` tag also route here.

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      tags: ["default"]

  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      tags: ["paid"]
```

Untagged requests use the `default` deployment; requests tagged `paid` use the paid deployment.

## Negation Tags (Denylist)

Prefix any tag with `!` to exclude deployments that carry that exact tag. This is useful when you want to ban a provider or model family from a request without having to enumerate every allowed alternative.

```bash showLineNumbers title="Exclude a provider"
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"tags": ["!provider:anthropic"]}
  }'
```

Any deployment tagged `provider:anthropic` is removed from the candidate pool before routing begins. All remaining deployments are eligible.

### Combining positive and negation tags

```bash showLineNumbers title="Paid tier, not Anthropic"
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"tags": ["paid", "!provider:anthropic"]}
  }'
```

Positive tags (`paid`) still run the normal inclusion filter after the exclusion pass. Only paid-tier, non-Anthropic deployments are returned.

### Excluding multiple providers

Send multiple `!` tags to exclude more than one deployment group:

```bash showLineNumbers title="Exclude multiple providers"
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "metadata": {"tags": ["!provider:anthropic", "!provider:openai"]}
  }'
```

### Negation semantics

| Behavior | Detail |
|----------|--------|
| Matching | Exact tag string match. `!provider:anthropic` removes only deployments tagged exactly `provider:anthropic`, not `provider:anthropic-haiku` |
| Ban-only request | If the request carries only `!` tags and no positive tags, all non-banned deployments are eligible (no inclusion filter runs) |
| All excluded | If negation tags remove every deployment, the request fails with `no_deployments_with_tag_routing` |
| Untagged deployments | Deployments with no `tags` field are never excluded by negation tags |
| Bare `!` | A tag that is exactly `!` (nothing after it) is silently ignored |

## Fallback Chains with Negation

When the primary model group is banned by a negation tag, the router falls through to the configured fallback automatically — no extra configuration needed.

```yaml showLineNumbers title="config.yaml with fallbacks"
model_list:
  - model_name: primary
    litellm_params:
      model: anthropic/claude-haiku-4-5-20251001
      api_key: os.environ/ANTHROPIC_API_KEY
      tags: ["provider:anthropic"]

  - model_name: fallback
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      tags: ["provider:openai"]

router_settings:
  enable_tag_filtering: true
  fallbacks:
    - {"primary": ["fallback"]}
```

A request with `!provider:anthropic` on `primary` raises `no_deployments_with_tag_routing`, which triggers the fallback to `fallback` (tagged `provider:openai`).

## Reference

### Router settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_tag_filtering` | `bool` | `false` | Enable tag-based routing. Must be `true` for any tag filtering to apply |
| `tag_filtering_match_any` | `bool` | `true` | `true`: route if any request tag matches a deployment tag. `false`: all request tags must be present on the deployment |

### Deployment config

| Field | Type | Description |
|-------|------|-------------|
| `tags` | `List[str]` | Tags this deployment accepts. Positive request tags are matched against this list. Negation request tags check for exact membership in this list |
| `tag_regex` | `List[str]` | Operator-configured regex patterns matched against request headers (e.g. `User-Agent`). Independent of `tags` matching; runs after tag filtering |

### Request metadata

| Field | Description |
|-------|-------------|
| `metadata.tags` | List of tag strings. Prefix with `!` to exclude deployments carrying that exact tag |
| `x-litellm-tags` header | Comma-separated tags, equivalent to `metadata.tags` |
