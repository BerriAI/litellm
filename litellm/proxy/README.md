# litellm-proxy

A local, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

## usage 

```shell 
$ uv tool install litellm
```
```shell
$ litellm --model ollama/codellama 

#INFO: Ollama running on http://0.0.0.0:8000
```

## replace openai base
```python 
import openai # openai v1.0.0+
client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:8000") # set proxy to base_url
# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
``` 

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/simple_proxy)

## Publish MCP servers in the AI Hub

Pin each server's `server_id` and list that ID under `litellm_settings.public_mcp_servers` in your proxy config

```yaml
mcp_servers:
  example:
    server_id: example-hub-id
    url: https://example.invalid/mcp
    transport: http
    auth_type: none

litellm_settings:
  public_mcp_servers:
    - example-hub-id
```

Replace the example URL with your MCP endpoint, start or restart the proxy with `litellm --config config.yaml`, and check `GET /public/mcp_hub`. The publication list uses server IDs, not YAML map names. A pinned ID stays stable when the URL, transport, authentication or alias changes

The default strict hub mode lists only these IDs. An empty or absent list publishes no servers. The legacy `public_mcp_hub_strict_whitelist: false` setting also lists internet-accessible servers. `available_on_public_internet` controls network access policy separately from hub listing; publishing does not remove authentication or tool permissions

When `public_mcp_servers` is declared in YAML, edit that file and restart to change the list. Dashboard or `/v1/mcp/make_public` requests that change this config-owned value are rejected. To manage publication through the dashboard instead, remove the key from YAML and restart


---

### Folder Structure

**Routes**
- `proxy_server.py` - all openai-compatible routes - `/v1/chat/completion`, `/v1/embedding` + model info routes - `/v1/models`, `/v1/model/info`, `/v1/model_group_info` routes.
- `health_endpoints/` - `/health`, `/health/liveliness`, `/health/readiness`
- `management_endpoints/key_management_endpoints.py` - all `/key/*` routes
- `management_endpoints/team_endpoints.py` - all `/team/*` routes
- `management_endpoints/internal_user_endpoints.py` - all `/user/*` routes
- `management_endpoints/ui_sso.py` - all `/sso/*` routes
