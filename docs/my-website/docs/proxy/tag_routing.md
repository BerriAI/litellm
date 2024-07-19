# 💸 Tag Based Routing

Route requests based on tags. 
This is useful for implementing free / paid tiers for users

### 1. Define tags on config.yaml 

- A request with `tags=["free"]` will get routed to `openai/fake`
- A request with `tags=["paid"]`  will get routed to `openai/gpt-4o`

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
      tags: ["free"] # 👈 Key Change
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      tags: ["paid"] # 👈 Key Change

router_settings:
  enable_tag_filtering: True # 👈 Key Change
general_settings: 
  master_key: sk-1234 
```

### 2. Make Request with `tags=["free"]`

This request includes "tags": ["free"], which routes it to `openai/fake`

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ],
    "tags": ["free"]
  }'
```
**Expected Response**

Expect to see the following response header when this works
```shell
x-litellm-model-api-base: https://exampleopenaiendpoint-production.up.railway.app/
```

Response
```shell
{
 "id": "chatcmpl-33c534e3d70148218e2d62496b81270b",
 "choices": [
   {
     "finish_reason": "stop",
     "index": 0,
     "message": {
       "content": "\n\nHello there, how may I assist you today?",
       "role": "assistant",
       "tool_calls": null,
       "function_call": null
     }
   }
 ],
 "created": 1677652288,
 "model": "gpt-3.5-turbo-0125",
 "object": "chat.completion",
 "system_fingerprint": "fp_44709d6fcb",
 "usage": {
   "completion_tokens": 12,
   "prompt_tokens": 9,
   "total_tokens": 21
 }
}
```


### 3. Make Request with `tags=["paid"]`

This request includes "tags": ["paid"], which routes it to `openai/gpt-4`

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ],
    "tags": ["paid"]
  }'
```

**Expected Response**

Expect to see the following response header when this works
```shell
x-litellm-model-api-base: https://api.openai.com
```

Response
```shell
{
 "id": "chatcmpl-9maCcqQYTqdJrtvfakIawMOIUbEZx",
 "choices": [
   {
     "finish_reason": "stop",
     "index": 0,
     "message": {
       "content": "Good morning! How can I assist you today?",
       "role": "assistant",
       "tool_calls": null,
       "function_call": null
     }
   }
 ],
 "created": 1721365934,
 "model": "gpt-4o-2024-05-13",
 "object": "chat.completion",
 "system_fingerprint": "fp_c4e5b6fa31",
 "usage": {
   "completion_tokens": 10,
   "prompt_tokens": 12,
   "total_tokens": 22
 }
}
```