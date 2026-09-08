# litellm_agent Resource

Manages an A2A (Agent-to-Agent) agent on the LiteLLM proxy. Agents are AI-powered entities that can be discovered, invoked, and composed using the A2A protocol.

## Example Usage

```hcl
resource "litellm_agent" "hello_world" {
  agent_name = "hello-world-agent"

  agent_card_params = jsonencode({
    protocolVersion    = "1.0"
    name               = "Hello World Agent"
    description        = "Just a hello world agent"
    url                = "http://localhost:9999/"
    version            = "1.0.0"
    defaultInputModes  = ["text"]
    defaultOutputModes = ["text"]
    capabilities = {
      streaming = true
    }
    skills = [
      {
        id          = "hello_world"
        name        = "Returns hello world"
        description = "just returns hello world"
        tags        = ["hello world"]
        examples    = ["hi", "hello world"]
      }
    ]
  })

  litellm_params = jsonencode({
    make_public = false
  })

  object_permission = jsonencode({
    models      = ["gpt-4-proxy"]
    mcp_servers = ["my-mcp-server-id"]
  })

  static_headers = {
    "x-api-key" = var.agent_api_key
  }

  extra_headers = ["x-request-id"]

  tpm_limit         = 100000
  rpm_limit         = 1000
  session_tpm_limit = 10000
  session_rpm_limit = 100
}
```

## Argument Reference

The following arguments are supported:

* `agent_name` - (Required) Name of the agent. Must be unique on the proxy.
* `agent_card_params` - (Required) The A2A agent card as a JSON object string (use `jsonencode`). Supports the standard A2A card fields: `name`, `description`, `url`, `version`, `protocolVersion`, `capabilities`, `skills`, `defaultInputModes`, `defaultOutputModes`, `preferredTransport`, `iconUrl`, `provider`, `documentationUrl`, and more. The proxy merges LiteLLM-fronting fields (such as `supportedInterfaces`) into the stored card, so the value you configure stays authoritative in state.
* `litellm_params` - (Optional, Sensitive) LiteLLM-specific parameters as a JSON object string. May include secrets such as `api_key`, so the value is never read back from the API; the configured value is authoritative.
* `object_permission` - (Optional) Access control permissions as a JSON object string with keys `mcp_servers`, `mcp_access_groups`, `mcp_tool_permissions`, `models`, and `agents`.
* `static_headers` - (Optional, Sensitive) Map of static headers sent with agent requests. May hold tokens, so it is never read back from the API.
* `extra_headers` - (Optional) List of incoming request header names to forward to the agent.
* `tpm_limit` - (Optional) Tokens per minute limit for the agent.
* `rpm_limit` - (Optional) Requests per minute limit for the agent.
* `session_tpm_limit` - (Optional) Per-session tokens per minute limit.
* `session_rpm_limit` - (Optional) Per-session requests per minute limit.

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `id` - The agent ID assigned by LiteLLM.
* `created_at` - Timestamp when the agent was created.
* `updated_at` - Timestamp when the agent was last updated.
* `created_by` - User who created the agent.
* `updated_by` - User who last updated the agent.

## Import

Agents can be imported using the agent ID:

```shell
terraform import litellm_agent.example <agent_id>
```

Note: `litellm_params` and `static_headers` cannot be recovered on import because the API never returns their unmasked values; re-apply after import to set them.
