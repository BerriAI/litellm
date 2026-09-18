# litellm_fallback (Data Source)

Retrieves the fallback configuration for a LiteLLM model. Use this to reference fallbacks that were configured outside of Terraform.

## Example Usage

```hcl
data "litellm_fallback" "gpt4" {
  model = "gpt-4"
}

output "gpt4_fallback_models" {
  value = data.litellm_fallback.gpt4.fallback_models
}
```

### Specific Fallback Type

```hcl
data "litellm_fallback" "gpt4_context_window" {
  model         = "gpt-4"
  fallback_type = "context_window"
}
```

## Argument Reference

The following arguments are supported:

* `model` - (Required) The model name to get fallbacks for.
* `fallback_type` - (Optional) Type of fallback to retrieve. One of `general` (default), `context_window`, or `content_policy`.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The primary model name.
* `fallback_models` - List of fallback model names in order of priority.
