# litellm_jwt_key_mapping

Maps a JWT claim value to a LiteLLM virtual key. Every JWT client identified by a claim, typically `client_id`, `azp` or `sub`, then gets the model restrictions, budgets, rate limits, guardrails and spend tracking of the virtual key it maps to, without that key ever being handed to the client.

The mappings only take effect once JWT auth is enabled on the proxy, which is configuration rather than API state:

```yaml
general_settings:
  enable_jwt_auth: True
  litellm_jwtauth:
    virtual_key_claim_field: "client_id"
    unregistered_jwt_client_behavior: "fallback_team_mapping"
```

See [JWT to virtual key mapping](https://docs.litellm.ai/docs/proxy/jwt_key_mapping) for the proxy side of the feature

## Example Usage

The mapped virtual key has to exist already and its value has to be known to Terraform, so it comes from a variable or a secret manager rather than from a `litellm_key` resource. `litellm_key` deliberately made its generated `key` write-only, to avoid storing raw API keys in state, so referencing it here does not merely read back null: Terraform's write-only enforcement turns `key = litellm_key.foo.key` into a static `Missing required argument` error at `terraform plan`, before any API call, in every apply ordering, including a first apply where both resources are created together:

```hcl
variable "alice_key" {
  type      = string
  sensitive = true
}

resource "litellm_jwt_key_mapping" "alice" {
  jwt_claim_name  = "client_id"
  jwt_claim_value = "dev-alice"
  key             = var.alice_key
}
```

Per-client limits live on the virtual key, so one mapping per client is how each JWT client gets its own budget and quota:

```hcl
resource "litellm_jwt_key_mapping" "billing_service" {
  jwt_claim_name  = "client_id"
  jwt_claim_value = "billing-service"
  key             = var.billing_service_key
  description     = "Billing service JWT client"
  is_active       = true
}
```

Several clients at once, with the key values coming from a map of secrets:

```hcl
variable "jwt_client_keys" {
  type      = map(string)
  sensitive = true
}

resource "litellm_jwt_key_mapping" "developer" {
  for_each = var.jwt_client_keys

  jwt_claim_name  = "client_id"
  jwt_claim_value = each.key
  key             = each.value
  description     = "Developer JWT client ${each.key}"
}
```

## Argument Reference

- `jwt_claim_name` - (Required, ForceNew) Name of the JWT claim to match on, for example `client_id`, `azp` or `sub`. Must match `virtual_key_claim_field` in the proxy JWT config
- `jwt_claim_value` - (Required, ForceNew) Value of the claim identifying the JWT client. Unique together with `jwt_claim_name`, so a second mapping for the same pair fails with a 409
- `key` - (Required, Sensitive) The virtual key this claim value maps to. It has to exist already, otherwise the proxy rejects the mapping with `The provided key does not match an existing virtual key`
- `description` - (Optional) Description of the mapping
- `is_active` - (Optional) Whether the mapping is active. Inactive mappings are ignored during JWT auth. Defaults to `true`

## Attribute Reference

- `id` - The mapping ID assigned by LiteLLM
- `created_at` - Timestamp when the mapping was created
- `updated_at` - Timestamp when the mapping was last updated
- `created_by` - User who created the mapping
- `updated_by` - User who last updated the mapping

## Notes

The proxy stores only a hash of `key` and never returns it, so drift on that attribute cannot be detected and Terraform tracks the value from your configuration. Changing `key` rotates the mapping onto the new virtual key in place, with no replacement. Like the other secrets this provider accepts, such as `credential_values` and `model_api_key`, the configured value is kept in state, so treat the state as sensitive

Only proxy admins can create, update or delete mappings, so the provider `api_key` has to be a master key or an admin key

## Import

Mappings are imported by their mapping ID:

```shell
terraform import litellm_jwt_key_mapping.alice 297a5536-1aeb-4cf1-b666-b3809c2750a8
```

Because the API does not return the mapped key, `key` is empty in state right after an import, so the first plan shows an in-place update that pushes the configured key back to the proxy. That update is harmless, the proxy just rehashes the same value when the key has not actually changed
