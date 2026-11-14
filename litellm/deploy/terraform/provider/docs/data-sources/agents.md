# litellm_agents Data Source

Retrieves the list of A2A agents registered on the LiteLLM proxy.

## Example Usage

```hcl
data "litellm_agents" "all" {}

output "agent_ids" {
  value = data.litellm_agents.all.ids
}

# Only agents whose URL is currently reachable (or that have no URL)
data "litellm_agents" "healthy" {
  health_check = true
}
```

## Argument Reference

The following arguments are supported:

* `health_check` - (Optional, default `false`) When true, the proxy probes each agent's URL and only returns agents that are reachable or have no URL.

## Attribute Reference

The following attributes are exported:

* `ids` - List of agent IDs.
* `agents` - List of agents. Each entry exports:
  * `agent_id` - The unique agent ID.
  * `agent_name` - Name of the agent.
  * `tpm_limit` - Tokens per minute limit.
  * `rpm_limit` - Requests per minute limit.
  * `session_tpm_limit` - Per-session tokens per minute limit.
  * `session_rpm_limit` - Per-session requests per minute limit.
  * `spend` - Total spend recorded for the agent.
  * `created_at` - Timestamp when the agent was created.
  * `updated_at` - Timestamp when the agent was last updated.
  * `created_by` - User who created the agent.
  * `updated_by` - User who last updated the agent.
