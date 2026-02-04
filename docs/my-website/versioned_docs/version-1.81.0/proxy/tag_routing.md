# Tag Based Routing

Route requests based on tags. 
This is useful for 
- Implementing free / paid tiers for users
- Controlling model access per team, example Team A can access gpt-4 deployment A, Team B can access gpt-4 deployment B (LLM Access Control For Teams )

:::info
## See here for spend tags
- [Track spend per tag](cost_tracking#-custom-tags)
- [Setup Budgets per Virtual Key, Team](users)
:::

## Quick Start

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
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
      tags: ["default"] # OPTIONAL - All untagged requests will get routed to this
  

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

## Calling via Request Header

You can also call via request header `x-litellm-tags`

```shell
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-H 'x-litellm-tags: free,my-custom-tag' \
-d '{
  "model": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": "Hey, how'\''s it going 123456?"
    }
  ]
}'
```

## Setting Default Tags 

Use this if you want all untagged requests to be routed to specific deployments

1. Set default tag on your yaml
```yaml
  model_list:
    - model_name: fake-openai-endpoint
      litellm_params:
        model: openai/fake
        api_key: fake-key
        api_base: https://exampleopenaiendpoint-production.up.railway.app/
        tags: ["default"] # 👈 Key Change - All untagged requests will get routed to this
      model_info:
        id: "default-model" # used for identifying model in response headers
```

2. Start proxy
```shell
$ litellm --config /path/to/config.yaml
```

3. Make request with no tags
```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "fake-openai-endpoint",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ]
  }'
```

Expect to see the following response header when this works
```shell
x-litellm-model-id: default-model
```

## ✨ Team based tag routing (Enterprise)

LiteLLM Proxy supports team-based tag routing, allowing you to associate specific tags with teams and route requests accordingly. Example **Team A can access gpt-4 deployment A, Team B can access gpt-4 deployment B** (LLM Access Control For Teams)

:::info

This is an enterprise feature, [Contact us here to get a free trial](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

:::

Here's how to set up and use team-based tag routing using curl commands:

1. **Enable tag filtering in your proxy configuration:**

   In your `proxy_config.yaml`, ensure you have the following setting:

   ```yaml
   model_list:
    - model_name: fake-openai-endpoint
      litellm_params:
        model: openai/fake
        api_key: fake-key
        api_base: https://exampleopenaiendpoint-production.up.railway.app/
        tags: ["teamA"] # 👈 Key Change
      model_info:
        id: "team-a-model" # used for identifying model in response headers
    - model_name: fake-openai-endpoint
      litellm_params:
        model: openai/fake
        api_key: fake-key
        api_base: https://exampleopenaiendpoint-production.up.railway.app/
        tags: ["teamB"] # 👈 Key Change
      model_info:
        id: "team-b-model" # used for identifying model in response headers
    - model_name: fake-openai-endpoint
      litellm_params:
        model: openai/fake
        api_key: fake-key
        api_base: https://exampleopenaiendpoint-production.up.railway.app/
        tags: ["default"] # OPTIONAL - All untagged requests will get routed to this

  router_settings:
    enable_tag_filtering: True # 👈 Key Change

  general_settings: 
    master_key: sk-1234 
    ```

2. **Create teams with tags:**

   Use the `/team/new` endpoint to create teams with specific tags:

   ```shell
   # Create Team A
   curl -X POST http://0.0.0.0:4000/team/new \
     -H "Authorization: Bearer sk-1234" \
     -H "Content-Type: application/json" \
     -d '{"tags": ["teamA"]}'
   ```

   ```shell
   # Create Team B
   curl -X POST http://0.0.0.0:4000/team/new \
     -H "Authorization: Bearer sk-1234" \
     -H "Content-Type: application/json" \
     -d '{"tags": ["teamB"]}'
   ```

   These commands will return JSON responses containing the `team_id` for each team.

3. **Generate keys for team members:**

   Use the `/key/generate` endpoint to create keys associated with specific teams:

   ```shell
   # Generate key for Team A
   curl -X POST http://0.0.0.0:4000/key/generate \
     -H "Authorization: Bearer sk-1234" \
     -H "Content-Type: application/json" \
     -d '{"team_id": "team_a_id_here"}'
   ```

   ```shell
   # Generate key for Team B
   curl -X POST http://0.0.0.0:4000/key/generate \
     -H "Authorization: Bearer sk-1234" \
     -H "Content-Type: application/json" \
     -d '{"team_id": "team_b_id_here"}'
   ```

   Replace `team_a_id_here` and `team_b_id_here` with the actual team IDs received from step 2.

4. **Verify routing:**

   Check the `x-litellm-model-id` header in the response to confirm that the request was routed to the correct model based on the team's tags. You can use the `-i` flag with curl to include the response headers:
  
   Request with Team A's key (including headers)
   ```shell
   curl -i -X POST http://0.0.0.0:4000/chat/completions \
     -H "Authorization: Bearer team_a_key_here" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "fake-openai-endpoint",
       "messages": [
         {"role": "user", "content": "Hello!"}
       ]
     }'
   ```

   In the response headers, you should see:
   ```
   x-litellm-model-id: team-a-model
   ```

   Similarly, when using Team B's key, you should see:
   ```
   x-litellm-model-id: team-b-model
   ```

By following these steps and using these curl commands, you can implement and test team-based tag routing in your LiteLLM Proxy setup, ensuring that different teams are routed to the appropriate models or deployments based on their assigned tags.


