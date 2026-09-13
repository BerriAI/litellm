# litellm_guardrail Resource

Manages a guardrail in LiteLLM. Guardrails provide content filtering, PII detection, prompt injection protection, and more.

## Example Usage

```hcl
resource "litellm_guardrail" "bedrock_guard" {
  guardrail_name = "my-bedrock-guard"
  guardrail      = "bedrock"
  mode           = "pre_call"
  default_on     = true

  litellm_params = jsonencode({
    guardrailIdentifier = "ff6ujrregl1q"
    guardrailVersion    = "DRAFT"
  })

  guardrail_info = {
    description = "Bedrock content moderation guardrail"
  }
}
```

### Multiple Modes

```hcl
resource "litellm_guardrail" "pii_guard" {
  guardrail_name = "presidio-pii"
  guardrail      = "presidio"
  mode           = jsonencode(["pre_call", "post_call"])
}
```

## Argument Reference

* `guardrail_name` - (Required) Human-readable name for the guardrail.
* `guardrail` - (Required) The guardrail integration type (e.g. `bedrock`, `lakera`, `presidio`, `openai_moderation`, `hide_secrets`).
* `mode` - (Required) When to apply the guardrail. A single value (`pre_call`, `post_call`, `during_call`, `logging_only`) or a JSON array of values.
* `default_on` - (Optional) Whether the guardrail is enabled by default for all requests.
* `litellm_params` - (Optional, Sensitive) JSON string with additional provider-specific parameters merged into `litellm_params` (may contain API keys). The API masks these values, so the configured value stays authoritative in state.
* `guardrail_info` - (Optional) Map of additional metadata for the guardrail.

## Attribute Reference

* `id` - The guardrail ID assigned by LiteLLM.
* `created_at` - Timestamp when the guardrail was created.

## Import

Guardrails can be imported using the guardrail ID:

```shell
terraform import litellm_guardrail.example 123e4567-e89b-12d3-a456-426614174000
```

Note: `guardrail`, `mode`, `default_on` and `litellm_params` are not returned unmasked by the API, so after import you must set them in configuration to match the server.
