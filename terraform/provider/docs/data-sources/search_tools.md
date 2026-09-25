# litellm_search_tools Data Source

Retrieves the list of search tools configured on the LiteLLM proxy, from both the database and the proxy config.

## Example Usage

```hcl
data "litellm_search_tools" "all" {}

output "search_tool_ids" {
  value = data.litellm_search_tools.all.ids
}
```

## Argument Reference

This data source takes no arguments.

## Attribute Reference

The following attributes are exported:

* `ids` - List of search tool IDs.
* `search_tools` - List of search tools. Each entry exports:
  * `search_tool_id` - The unique search tool ID.
  * `search_tool_name` - Name of the search tool.
  * `search_tool_info` - Additional metadata as a JSON object string.
  * `is_from_config` - Whether the search tool comes from the proxy config file rather than the database.
  * `created_at` - Timestamp when the search tool was created.
  * `updated_at` - Timestamp when the search tool was last updated.

## Security Note

`litellm_params` is not exposed through this data source because it may hold provider API keys.
