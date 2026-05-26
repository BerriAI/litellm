# CONTEXT.md — Domain Glossary

This document defines the precise terminology used throughout the LiteLLM codebase. Use these terms consistently when discussing the system.

---

## Core Concepts

### Provider
An external LLM service that LiteLLM routes requests to. Examples: OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Google Vertex AI, Cohere.

Provider implementations live in `litellm/llms/<provider>/`.

### Deployment
A specific configuration of provider + model + credentials. The atomic unit of routing.

```python
# Example: Two deployments in the same model group
{
    "model_name": "gpt-4",                    # Model Group (public name)
    "litellm_params": {
        "model": "azure/gpt-4-turbo",         # Provider-specific model identifier
        "api_key": "...",
        "api_base": "https://my-resource.openai.azure.com/"
    },
    "model_info": {"id": "deployment-uuid-1"} # Deployment ID
}
```

Key fields:
- `model_name` — The **Model Group** this deployment belongs to
- `litellm_params.model` — The provider-specific model string (e.g., `azure/gpt-4`, `bedrock/anthropic.claude-v2`)
- `model_info.id` — Unique identifier for this deployment

Defined in `litellm/types/router.py:Deployment`.

### Model Group
The public-facing model name that clients use in API requests. A model group maps to one or more deployments.

When a client requests `model="gpt-4"`, the Router selects one deployment from all deployments where `model_name == "gpt-4"`.

Synonyms in code: `model_name` (on Deployment), `model_group` (in logs/metrics).

### Model Alias
A mapping that redirects one model name to another. Configured at the team level or globally via `model_group_alias`.

```yaml
model_group_alias:
  "gpt-4-alias": "gpt-4"  # Requests for gpt-4-alias route to gpt-4 deployments
```

### Router
The component that selects which deployment handles a request. Responsibilities:
- **Load balancing** — Distribute requests across deployments (strategies: simple-shuffle, least-busy, usage-based, latency-based, cost-based)
- **Fallbacks** — Try alternative model groups when the primary fails
- **Retries** — Retry failed requests with exponential backoff
- **Cooldowns** — Temporarily exclude failing deployments
- **Rate limiting** — Enforce TPM/RPM limits per deployment

Entry point: `litellm/router.py:Router`.

### Routing Strategy
The algorithm used to select a deployment from available options:
- `simple-shuffle` — Random selection (default)
- `least-busy` — Fewest in-flight requests
- `usage-based-routing` — Respect TPM/RPM capacity
- `latency-based-routing` — Lowest recent latency
- `cost-based-routing` — Lowest cost per token

---

## Access Control Hierarchy

```
Organization
    └── Team
            └── Project
                    └── Virtual Key
                            └── (End User)
```

### Organization
Top-level administrative boundary. Contains teams. Has its own budget and model access controls.

Table: `LiteLLM_OrganizationTable`.

### Team
A group of internal users who share budgets, model access, and rate limits. Teams belong to an organization (optional).

Table: `LiteLLM_TeamTable`.

### Project
An optional grouping between teams and virtual keys, used to organize keys by use case or application.

Table: `LiteLLM_ProjectTable`.

### Virtual Key
A proxy-issued API key that authenticates requests to the LiteLLM proxy. Virtual keys:
- Map to provider credentials internally
- Have their own budgets, rate limits, and model access
- Can belong to a team, project, or user
- Are stored as hashed values (never plaintext)

Virtual keys are **not** provider API keys. The proxy uses virtual keys to authorize access, then uses provider credentials to make the actual LLM call.

Table: `LiteLLM_VerificationToken` (the `token` column holds the hash).

### Internal User
A human or service account that accesses the proxy directly. Internal users:
- Can create and manage virtual keys
- Belong to teams/organizations
- Have roles (admin, team_admin, internal_user, internal_user_viewer)

Table: `LiteLLM_UserTable`.

### End User
The downstream customer whose requests flow through the proxy. Identified by the `user` parameter in OpenAI-compatible requests.

End users:
- Are **not** proxy users — they're your customers
- Can have per-customer budgets and model restrictions
- Enable usage tracking per customer

Table: `LiteLLM_EndUserTable`.

---

## Budgets and Spend

### Spend
Actual cost incurred, in dollars. Tracked at multiple levels:
- Per virtual key (`LiteLLM_VerificationToken.spend`)
- Per team (`LiteLLM_TeamTable.spend`)
- Per user (`LiteLLM_UserTable.spend`)
- Per end user (`LiteLLM_EndUserTable.spend`)
- Per organization (`LiteLLM_OrganizationTable.spend`)

### Budget
A spending limit. Can be:
- **Hard budget** (`max_budget`) — Requests are blocked when exceeded
- **Soft budget** (`soft_budget`) — Alerts are sent but requests continue

Budgets can reset periodically via `budget_duration` (e.g., `"30d"`, `"1mo"`).

Table: `LiteLLM_BudgetTable`.

### Rate Limit
Request throughput limits:
- **TPM** — Tokens per minute
- **RPM** — Requests per minute

Enforced at deployment, virtual key, team, and user levels.

---

## Request Lifecycle

### Callback
A hook that executes during the LLM request lifecycle. Callbacks receive events at specific points:
- `log_pre_api_call` — Before the provider API call
- `log_success_event` / `async_log_success_event` — After successful completion
- `log_failure_event` / `async_log_failure_event` — After failure
- `log_stream_event` — During streaming responses

Implement callbacks by subclassing `litellm.integrations.custom_logger.CustomLogger`.

### Integration
A third-party service connector for logging, observability, or analytics. Integrations are a category of callbacks.

Examples: Langfuse, Datadog, Prometheus, Helicone, Lunary.

Location: `litellm/integrations/`.

### Guardrail
Content filtering or safety checks that run during the request lifecycle:
- **Pre-call guardrails** — Validate input before the LLM call
- **Post-call guardrails** — Validate output after the LLM responds
- **During-call guardrails** — Validate streaming chunks

Guardrails can block, modify, or flag requests.

Examples: Lakera, Presidio, LLM Guard, custom validators.

Location: `litellm/proxy/guardrails/`, `litellm/integrations/custom_guardrail.py`.

### Policy
A named configuration that bundles guardrails with conditions for when they apply. Policies can:
- Inherit from parent policies
- Add or remove guardrails
- Apply based on model, team, or tags

Table: `LiteLLM_PolicyTable`, `LiteLLM_PolicyAttachmentTable`.

### Pass-through Endpoint
A proxy route that forwards requests to a provider API with minimal transformation:
1. **URL construction** — Build the provider-specific URL
2. **Auth replacement** — Swap virtual key for provider credentials

The request **body** passes through unchanged. Used for provider-specific APIs that don't need OpenAI-compatible translation.

Location: `litellm/proxy/pass_through_endpoints/`.

---

## Reliability

### Cooldown
A period during which a deployment is excluded from routing after failures. Configured via:
- `allowed_fails` — Failures before cooldown triggers
- `cooldown_time` — Duration in seconds

### Fallback
An alternative model group to try when the primary fails or is unavailable.

```yaml
fallbacks:
  - gpt-4: [gpt-3.5-turbo, claude-3-sonnet]  # Try these in order if gpt-4 fails
```

Types:
- `fallbacks` — General failure fallbacks
- `context_window_fallbacks` — When context length is exceeded
- `content_policy_fallbacks` — When content is blocked

### Retry
Automatic retry of failed requests with configurable behavior per exception type.

```python
retry_policy = RetryPolicy(
    RateLimitErrorRetries=3,
    TimeoutErrorRetries=2
)
```

---

## Additional Concepts

### Credential
Stored provider credentials that deployments reference by name instead of inline secrets.

```yaml
credential_name: "my-openai-creds"
# Referenced as: litellm_credential_name: "my-openai-creds"
```

Table: `LiteLLM_CredentialsTable`.

### Access Group
A named collection of resources (models, MCP servers, agents) that can be granted to teams or virtual keys.

Table: `LiteLLM_AccessGroupTable`.

### Tag
A label attached to requests for:
- **Routing** — Direct requests to specific deployments
- **Filtering** — Restrict access based on tags
- **Tracking** — Aggregate spend/metrics by tag

Tags are passed via `metadata.tags` or the `x-litellm-tags` header.

Table: `LiteLLM_TagTable`.

### MCP Server
A Model Context Protocol server that provides tools for LLM function calling. The proxy can route MCP tool calls and manage per-user credentials.

Table: `LiteLLM_MCPServerTable`.

---

## Code Locations

| Concept | Primary Location |
|---------|------------------|
| Router | `litellm/router.py` |
| Provider implementations | `litellm/llms/<provider>/` |
| Proxy server | `litellm/proxy/proxy_server.py` |
| Authentication | `litellm/proxy/auth/` |
| Database schema | `litellm/proxy/schema.prisma` |
| Type definitions | `litellm/types/` |
| Integrations/Callbacks | `litellm/integrations/` |
| Guardrails | `litellm/proxy/guardrails/` |
| Caching | `litellm/caching/` |
