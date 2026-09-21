# litellm_agent Data Source

Retrieves information about an existing A2A agent on the LiteLLM proxy.

## Example Usage

```hcl
data "litellm_agent" "existing" {
  agent_id = "123e4567-e89b-12d3-a456-426614174000"
}

output "agent_card" {
  value = jsondecode(data.litellm_agent.existing.agent_card_params)
}
```

## Argument Reference

The following arguments are supported:

* `agent_id` - (Required) Unique identifier of the agent to retrieve.

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `agent_name` - Name of the agent.
* `agent_card_params` - The A2A agent card as a JSON object string (decode with `jsondecode`).
* `object_permission` - Access control permissions as a JSON object string.
* `extra_headers` - List of incoming request header names forwarded to the agent.
* `tpm_limit` - Tokens per minute limit.
* `rpm_limit` - Requests per minute limit.
* `session_tpm_limit` - Per-session tokens per minute limit.
* `session_rpm_limit` - Per-session requests per minute limit.
* `spend` - Total spend recorded for this agent.
* `created_at` - Timestamp when the agent was created.
* `updated_at` - Timestamp when the agent was last updated.
* `created_by` - User who created the agent.
* `updated_by` - User who last updated the agent.

## Security Note

`litellm_params` and `static_headers` are not exposed through this data source because they may hold API keys or tokens.
