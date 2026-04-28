# Search API

LiteLLM supports team-aware search provider credentials for providers like Tavily, Perplexity, Brave, Exa, and Serper.

## Per-team search provider configuration

Set per-team credentials in team metadata:

```json
{
  "search_provider_config": {
    "tavily": {
      "api_key": "tvly-team-a-key",
      "api_base": "https://api.tavily.com"
    },
    "perplexity": {
      "api_key": "pplx-team-a-key"
    }
  }
}
```

Update via API:

```bash
curl -X POST "http://localhost:4000/team/search_provider_config/update" \
  -H "Authorization: Bearer sk-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-a",
    "provider": "tavily",
    "api_key": "tvly-team-a-key",
    "api_base": "https://api.tavily.com"
  }'
```

## Request flow and precedence

Search credentials resolve in this order:

1. Request metadata: `metadata.search_provider_config.<provider>`
2. Team DB metadata: `user_api_key_team_metadata.search_provider_config.<provider>`
3. YAML team settings: `default_team_settings[].search_provider_config.<provider>`
4. Search tool config: `search_tools[].litellm_params`
5. Provider env fallback (`TAVILY_API_KEY`, etc.)

## Calling search as an end-user

The caller only uses their team-bound virtual key.

```bash
curl -X POST "http://localhost:4000/v1/search" \
  -H "Authorization: Bearer sk-team-a-user-key" \
  -H "Content-Type: application/json" \
  -d '{
    "search_tool_name": "company-search",
    "query": "latest AI news",
    "max_results": 5
  }'
```

or with URL tool name:

```bash
curl -X POST "http://localhost:4000/v1/search/company-search" \
  -H "Authorization: Bearer sk-team-a-user-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI news",
    "max_results": 5
  }'
```

## YAML examples

```yaml
search_tools:
  - search_tool_name: company-search
    litellm_params:
      search_provider: tavily
      api_key: os.environ/TAVILY_DEFAULT_API_KEY

default_team_settings:
  - team_id: team-a
    search_provider_config:
      tavily:
        api_key: os.environ/TAVILY_TEAM_A_API_KEY
  - team_id: team-b
    search_provider_config:
      tavily:
        api_key: os.environ/TAVILY_TEAM_B_API_KEY
```
