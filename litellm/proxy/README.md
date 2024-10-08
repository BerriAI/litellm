# litellm-proxy

A local, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

## usage 

```shell 
$ pip install litellm
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


---

### Folder Structure

**Routes**
- `proxy_server.py` - all openai-compatible routes - `/v1/chat/completion`, `/v1/embedding` + model info routes - `/v1/models`, `/v1/model/info`, `/v1/model_group_info` routes.
- `health_endpoints/` - `/health`, `/health/liveliness`, `/health/readiness`
- `management_endpoints/key_management_endpoints.py` - all `/key/*` routes
- `management_endpoints/team_endpoints.py` - all `/team/*` routes
- `management_endpoints/internal_user_endpoints.py` - all `/user/*` routes
- `management_endpoints/ui_sso.py` - all `/sso/*` routes