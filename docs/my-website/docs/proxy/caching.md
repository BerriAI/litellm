# Caching 
Cache LLM Responses

Caching can be enabled by adding the `cache` key in the `config.yaml`
#### Step 1: Add `cache` to the config.yaml
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo

litellm_settings:
  set_verbose: True
  cache:          # init cache
    type: redis   # tell litellm to use redis caching
```

#### Step 2: Add Redis Credentials to .env
LiteLLM requires the following REDIS credentials in your env to enable caching

  ```shell
  REDIS_HOST = ""       # REDIS_HOST='redis-18841.c274.us-east-1-3.ec2.cloud.redislabs.com'
  REDIS_PORT = ""       # REDIS_PORT='18841'
  REDIS_PASSWORD = ""   # REDIS_PASSWORD='liteLlmIsAmazing'
  ```
#### Step 3: Run proxy with config
```shell
$ litellm --config /path/to/config.yaml
```

#### Using Caching 
Send the same request twice:
```shell
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7
   }'

curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7
   }'
```

#### Control caching per completion request
Caching can be switched on/off per `/chat/completions` request
- Caching **on** for completion - pass `caching=True`:
  ```shell
  curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7,
     "caching": true
   }'
  ```
- Caching **off** for completion - pass `caching=False`:
  ```shell
  curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "write a poem about litellm!"}],
     "temperature": 0.7,
     "caching": false
   }'
  ```