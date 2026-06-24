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

## Balanced smart routing

Set `router_settings.routing_strategy: balanced-smart` in proxy config to route
by active requests, observed tokens per second, request latency, bounded queue
TTL, and recent failures.

```yaml
litellm_settings:
  cache: true
  cache_params:
    type: redis

router_settings:
  routing_strategy: balanced-smart
  routing_strategy_args:
    max_concurrent_requests: 10
    max_queue_ttl_s: 1
    queue_poll_s: 0.01
    default_tokens_per_second: 50
    active_request_weight: 10
    tokens_per_second_weight: 1
    ttft_weight: 1
    failure_penalty: 5
    failure_cooldown_s: 5
    ewma_alpha: 0.2
```

Use a shared Redis cache for multi-pod deployments so concurrency and metrics
are shared across LiteLLM pods. If multiple model entries target the same
physical backend, set the same `model_info.balanced_smart_backend_id` on those
entries so they share one capacity pool.


---

### Folder Structure

**Routes**
- `proxy_server.py` - all openai-compatible routes - `/v1/chat/completion`, `/v1/embedding` + model info routes - `/v1/models`, `/v1/model/info`, `/v1/model_group_info` routes.
- `health_endpoints/` - `/health`, `/health/liveliness`, `/health/readiness`
- `management_endpoints/key_management_endpoints.py` - all `/key/*` routes
- `management_endpoints/team_endpoints.py` - all `/team/*` routes
- `management_endpoints/internal_user_endpoints.py` - all `/user/*` routes
- `management_endpoints/ui_sso.py` - all `/sso/*` routes
