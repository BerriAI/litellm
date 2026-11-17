# litellm_fallback Resource

Manages a fallback configuration for a model in LiteLLM. Fallbacks are triggered when a call to the primary model fails after retries.

## Example Usage

### Basic Fallback Configuration

```hcl
resource "litellm_fallback" "gpt4_fallbacks" {
  model           = "gpt-4"
  fallback_models = ["claude-3-sonnet", "gpt-3.5-turbo"]
}
```

### Context Window Fallback

```hcl
resource "litellm_fallback" "gpt4_context_window" {
  model           = "gpt-4"
  fallback_models = ["claude-3-sonnet"]
  fallback_type   = "context_window"
}
```

## Argument Reference

The following arguments are supported:

* `model` - (Required, Forces new resource) The model name to configure fallbacks for. The model must already exist on the proxy.
* `fallback_models` - (Required) List of fallback model names in order of priority. Each model must exist on the proxy, and the primary model cannot be its own fallback.
* `fallback_type` - (Optional, Forces new resource) Type of fallback. One of `general` (default), `context_window`, or `content_policy`.

## Attribute Reference

In addition to the arguments above, the following attribute is exported:

* `id` - The primary model name.

## Import

Fallback configurations can be imported using the primary model name:

```shell
terraform import litellm_fallback.example gpt-4
```

Note: import always reads the `general` fallback type. Fallbacks of type `context_window` or `content_policy` cannot be imported.
