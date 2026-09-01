# litellm_search_tool Data Source

Retrieves information about an existing search tool on the LiteLLM proxy.

## Example Usage

```hcl
data "litellm_search_tool" "existing" {
  search_tool_id = "123e4567-e89b-12d3-a456-426614174000"
}

output "search_tool_name" {
  value = data.litellm_search_tool.existing.search_tool_name
}
```

## Argument Reference

The following arguments are supported:

* `search_tool_id` - (Required) Unique identifier of the search tool to retrieve.

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `search_tool_name` - Name of the search tool.
* `search_tool_info` - Additional metadata as a JSON object string (decode with `jsondecode`).
* `created_at` - Timestamp when the search tool was created.
* `updated_at` - Timestamp when the search tool was last updated.

## Security Note

`litellm_params` is not exposed through this data source because it may hold provider API keys.
