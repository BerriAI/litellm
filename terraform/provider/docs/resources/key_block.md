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

* `key` - (Required, Forces new resource, Sensitive) The API key to block, as the raw `sk-` value or its SHA-256 token hash. The provider normalizes raw values to the hash before talking to the API, so the plaintext key never appears in request URLs, the resource ID, or plan output.

## Attribute Reference

In addition to the arguments above, the following attributes are exported:

* `id` - The SHA-256 token hash of the key.
* `blocked` - Whether the key is currently blocked. Always `true` while this resource exists.

If the same key is also managed by a `litellm_key` resource, that resource's `blocked` attribute will show drift while the block is active; either set `blocked` there instead of using this resource, or add `lifecycle { ignore_changes = [blocked] }` to the `litellm_key`.

## Import

Key blocks can be imported using the key's SHA-256 token hash (shown as the key's ID in `litellm_key` state and in `/key/info`):

```shell
terraform import litellm_key_block.example 88362cbb875f4b48b4b5b56b2ea45f66465e27d55a189816bd54e5643e5410eb
```
