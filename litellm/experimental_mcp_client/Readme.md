# LiteLLM MCP Client

LiteLLM MCP Client is a client that allows you to use MCP tools with LiteLLM.

## Gateway discovery caching

The MCP gateway caches each upstream server's prompt, resource, and resource-template lists for 60 seconds per worker. Set `LITELLM_MCP_DISCOVERY_CACHE_TTL` to a nonnegative number of seconds to change the lifetime, or `0` to disable caching. Invalid values use the 60-second default

Discovery results may remain unchanged until that lifetime expires. Server configuration updates invalidate the affected server's entries. Concurrent requests for the same list share one upstream fetch. Each list cache holds at most 1,024 entries per worker

User-dependent upstream authentication uses separate cache entries. Gateway access checks still run for every request. Successful empty lists and unsupported capabilities are cached; failed requests retain the existing empty-list response and are retried on the next request
