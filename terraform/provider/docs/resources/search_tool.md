# litellm_search_tool Resource

Manages a search tool configuration on the LiteLLM proxy. Search tools connect the proxy's `/search` endpoints to an external search provider such as Tavily, Perplexity, or Exa.

## Example Usage

```hcl
resource "litellm_search_tool" "tavily" {
  search_tool_name = "tavily-search"

  litellm_params = jsonencode({
    search_provider = "tavily"
    api_key         = var.tavily_api_key
  })

  search_tool_info = jsonencode({
    description = "Tavily web search"
  })
}
```

## Argument Reference

The following arguments are supported:

* `search_tool_name` - (Required) Name of the search tool.
* `litellm_params` - (Required, Sensitive) Search tool parameters as a JSON object string (use `jsonencode`). Must include `search_provider`, and typically an `api_key`; may also carry `api_base`, `timeout`, `max_retries`, and other provider options. The API only returns masked values, so this is never read back; the configured value is authoritative.
* `search_tool_info` - (Optional) Additional metadata as a JSON object string, e.g. a `description`.

## Attribute Reference

In addition to all arguments above, the following attributes are exported:

* `id` - The search tool ID assigned by LiteLLM.
* `created_at` - Timestamp when the search tool was created.
* `updated_at` - Timestamp when the search tool was last updated.

## Import

Search tools can be imported using the search tool ID:

```shell
terraform import litellm_search_tool.example <search_tool_id>
```

Note: `litellm_params` cannot be recovered on import because the API only returns masked values; re-apply after import to set it.
