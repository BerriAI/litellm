# litellm_guardrail Data Source

Retrieves information about an existing LiteLLM guardrail by ID. Sensitive `litellm_params` are not exposed.

## Example Usage

```hcl
data "litellm_guardrail" "existing" {
  guardrail_id = "123e4567-e89b-12d3-a456-426614174000"
}

output "guardrail_name" {
  value = data.litellm_guardrail.existing.guardrail_name
}
```

## Argument Reference

* `guardrail_id` - (Required) Unique identifier of the guardrail to retrieve.

## Attribute Reference

* `guardrail_name` - Human-readable name of the guardrail.
* `guardrail_info` - Map of additional metadata for the guardrail.
* `guardrail_definition_location` - Where the guardrail is defined: `config` or `db`.
* `created_at` - Timestamp when the guardrail was created.
* `updated_at` - Timestamp when the guardrail was last updated.
