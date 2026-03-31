# [DEPRECATED] Team-based Routing

:::info

This is deprecated, please use [Tag Based Routing](./tag_routing.md) instead

:::


## Routing
Route calls to different model groups based on the team-id

### Config with model group 

Create a config.yaml with 2 model groups + connected postgres db

```yaml
model_list: 
  - model_name: gpt-4o-eu # 👈 Model Group 1
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE_EU
      api_key: os.environ/AZURE_API_KEY_EU
      api_version: "2023-07-01-preview"
  - model_name: gpt-4o-worldwide # 👈 Model Group 2
    litellm_params:
      model: azure/chatgpt-v-2
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
      api_version: "2023-07-01-preview"

general_settings: 
    master_key: sk-1234
    database_url: "postgresql://..." # 👈 Connect proxy to DB
```

Start proxy

```bash
litellm --config /path/to/config.yaml
```

### Create Team with Model Alias

```bash
curl --location 'http://0.0.0.0:4000/team/new' \
--header 'Authorization: Bearer sk-1234' \ # 👈 Master Key
--header 'Content-Type: application/json' \
--data '{
  "team_alias": "my-new-team_4",
  "model_aliases": {"gpt-4o": "gpt-4o-eu"}
}'

# Returns team_id: my-team-id
```

### Create Team Key 

```bash 
curl --location 'http://localhost:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "team_id": "my-team-id",  # 👈 YOUR TEAM ID
}'
```

### Call Model with alias 

```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-A1L0C3Px2LJl53sF_kTF9A' \
--data '{
  "model": "gpt-4o", # 👈 MODEL 
  "messages": [{"role": "system", "content": "You'\''re an expert at writing poems"}, {"role": "user", "content": "Write me a poem"}, {"role": "user", "content": "What'\''s your name?"}],
  "user": "usha"
}'
```

