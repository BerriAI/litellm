# Oauth 2.0 Authentication

Use this if you want to use an Oauth2.0 token to make `/chat`, `/embeddings` requests to the LiteLLM Proxy

:::info

This is an Enterprise Feature - [get in touch with us if you want a free trial to test if this feature meets your needs]((https://enterprise.litellm.ai/demo))

:::

## Usage 

1. Set env vars:

```bash
export OAUTH_TOKEN_INFO_ENDPOINT="https://your-provider.com/token/info"
export OAUTH_USER_ID_FIELD_NAME="sub"
export OAUTH_USER_ROLE_FIELD_NAME="role"
export OAUTH_USER_TEAM_ID_FIELD_NAME="team_id"
```

- `OAUTH_TOKEN_INFO_ENDPOINT`: URL to validate OAuth tokens
- `OAUTH_USER_ID_FIELD_NAME`: Field in token info response containing user ID
- `OAUTH_USER_ROLE_FIELD_NAME`: Field in token info for user's role
- `OAUTH_USER_TEAM_ID_FIELD_NAME`: Field in token info for user's team ID

2. Enable on litellm config.yaml

Set this on your config.yaml

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

general_settings: 
  master_key: sk-1234
  enable_oauth2_auth: true
```

3. Use token in requests to LiteLLM 

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "gpt-3.5-turbo",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```

## Debugging 

Start the LiteLLM Proxy with [`--detailed_debug` mode and you should see more verbose logs](cli.md#detailed_debug)

## Using OAuth2 + JWT Together

If both `enable_oauth2_auth` and `enable_jwt_auth` are enabled, LiteLLM can split auth paths:
- JWT validation for user tokens
- OAuth2 introspection for machine tokens

For JWT-shaped machine tokens, configure `litellm_jwtauth.routing_overrides`:

```yaml title="config.yaml"
general_settings:
  enable_jwt_auth: true
  enable_oauth2_auth: true
  litellm_jwtauth:
    routing_overrides:
      - iss: "machine-issuer.example.com"
        client_id: "MID_LITELLM"
        path: "oauth2"
```

For full `routing_overrides` behavior and list-based selectors, see [`/proxy/token_auth`](./token_auth.md#route-jwt-shaped-machine-tokens-to-oauth2).

