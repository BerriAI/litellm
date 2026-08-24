# litellm_key_block Resource

Manages the blocked state of an existing LiteLLM API key. Creating this resource blocks the key; destroying it unblocks the key.

If the key is unblocked outside of Terraform (or deleted), the resource is removed from state and Terraform plans to re-block it on the next apply.

## Example Usage

```hcl
resource "litellm_key" "example" {
  models = ["gpt-4"]
}

resource "litellm_key_block" "example" {
  key = litellm_key.example.key
}
```

## Argument Reference

The following arguments are supported:

* `key` - (Required, Forces new resource, Sensitive) The API key to block.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The key token.
* `blocked` - Whether the key is currently blocked. Always `true` while this resource exists.

## Import

Key blocks can be imported using the key token:

```shell
terraform import litellm_key_block.example sk-example-key
```
