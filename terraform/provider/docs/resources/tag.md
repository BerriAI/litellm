# litellm_tag Resource

Manages a tag in LiteLLM. Tags are used for spend tracking, budgets, and tag-based routing to specific model deployments

## Example Usage

```hcl
resource "litellm_tag" "production" {
  name        = "production"
  description = "Production traffic"
  models      = ["4a422a4c-e246-4d02-a1eb-13e835cd0725"]

  max_budget      = 500.0
  soft_budget     = 400.0
  budget_duration = "30d"
  tpm_limit       = 100000
  rpm_limit       = 1000
}
```

## Argument Reference

The following arguments are supported:

* `name` - (Required, Forces new resource) Unique name of the tag. Also used as the resource ID
* `description` - (Optional) Description of the tag
* `models` - (Optional) List of model IDs this tag applies to
* `budget_id` - (Optional) Existing budget ID to associate with this tag. If omitted and budget fields are set, the proxy creates a budget
* `max_budget` - (Optional) Max budget in USD for this tag
* `soft_budget` - (Optional) Soft budget in USD for this tag
* `max_parallel_requests` - (Optional) Max concurrent requests allowed for this tag
* `tpm_limit` - (Optional) Max tokens per minute for this tag
* `rpm_limit` - (Optional) Max requests per minute for this tag
* `budget_duration` - (Optional) Duration for budget reset, for example `1h`, `1d`, `30d`
* `model_max_budget` - (Optional) JSON object string with per-model budget configuration

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `id` - The tag name

## Import

Tags can be imported using the tag name:

```shell
terraform import litellm_tag.example production
```
