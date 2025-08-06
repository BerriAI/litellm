# Using Gemini CLI as provider of LiteLLM

## Supported

- Only support Gemini CLI - Login with Google method
- Only support following models:
  - `gemini-2.5-pro`
  - `gemini-2.5-flash`

## Prerequisites

### Get credentials from Gemini CLI

1. **Install Gemini CLI**:

    <https://github.com/google-gemini/gemini-cli>

2. **Authenticate with Google**:

    Select `● Login with Google`.

    A browser window will now open prompting you to login with your Google account.

3. **Locate the credentials file**:

    **Windows:**

    ```bash
    C:\Users\USERNAME\.gemini\oauth_creds.json
    ```

    **macOS/Linux:**

    ```bash
    ~/.gemini/oauth_creds.json
    ```

4. **This credentials will be used as vertext_credentials on next step**:

    The file contains JSON in this format:

    ```json
    {
        "access_token": "ya29.a0AS3H6Nx...",
        "refresh_token": "1//09FtpJYpxOd...",
        "scope": "https://www.googleapis.com/auth/cloud-platform ...",
        "token_type": "Bearer",
        "id_token": "eyJhbGciOiJSUzI1NiIs...",
        "expiry_date": 1750927763467
    }
    ```

## Start LiteLLM Proxy

1. **Create a `config.yaml` file**:

    ```yaml
    model_list:
    - model_name: your-custom-model-name
      litellm_params:
        model: gemini-2.5-pro
        api_base: https://cloudcode-pa.googleapis.com/v1internal
        vertex_project: your-google-cloud-project-id # you can get project id from https://aistudio.google.com/apikey
        vertex_credentials: | # put credentials from above step here
        {
            "access_token": "ya29.a0AS3H6Nx...",
            "refresh_token": "1//09FtpJYpxOd...",
            "scope": "https://www.googleapis.com/auth/cloud-platform ...",
            "token_type": "Bearer",
            "id_token": "eyJhbGciOiJSUzI1NiIs...",
            "expiry_date": 1750927763467
        }

    litellm_settings:
      drop_params: true
    
    general_settings:
      master_key: sk-1234 # change to any key you want
    ```

2. **Start LiteLLM with config**

    ```bash
    docker run -d \
        --name litellm \
        --restart unless-stopped \
        -p 4000:4000 \
        -v ./config.yaml:/app/config.yaml \
        ghcr.io/josheplibra/litellm-database:main-stable \
        --config /app/config.yaml --detailed_debug
    ```

## Make Call

Note: Change `your-custom-model-name`, `sk-1234` to your value from `config.yaml`

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "your-custom-model-name",
    "messages": [
      {
        "role": "system",
        "content": "You are an LLM named gemini-2.5-pro"
      },
      {
        "role": "user",
        "content": "what is your name?"
      }
    ]
}'
```

### Expected Response

```text
data: {"id":"kM6SaLeqJM7lpfgPq4HBmQ0","created":1754451617,"model":"gemini-2.5-pro","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"My name is gemini-2.5-pro. I am a large language model.","role":"assistant"}}],"vertex_ai_grounding_metadata":[],"vertex_ai_url_context_metadata":[],"vertex_ai_safety_ratings":[],"vertex_ai_citation_metadata":[]}

data: {"id":"kM6SaLeqJM7lpfgPq4HBmQ0","created":1754451617,"model":"gemini-2.5-pro","object":"chat.completion.chunk","choices":[{"finish_reason":"stop","index":0,"delta":{}}]}

data: [DONE]
```

## Next Steps

### Integrate to [Roo Code](https://roocode.com/)/[Cline](https://cline.bot/)

1. Go to Roo Code Settings
2. Select or create new Profile
3. API Provider: OpenAI Compatible
4. Base URL: <http://0.0.0.0:4000> (This is URL of LiteLLM Proxy Server)
5. API Key: `sk-1234` (This is key you set on `config.yaml`)
6. Model: `your-custom-model-name` (This is model name you set on `config.yaml`)
7. Context Window Size: 1000000
8. Save, done.

### Add more models

You can add more models to `config.yaml`.

```yaml
model_list:
- model_name: your-custom-model-name
  litellm_params:
    model: gemini-2.5-pro
    api_base: https://cloudcode-pa.googleapis.com/v1internal
    vertex_project: your-project-id-1
    vertex_credentials: |
      your-credentials-1
- model_name: your-custom-model-name-2
  litellm_params:
    model: gemini-2.5-flash
    api_base: https://cloudcode-pa.googleapis.com/v1internal
    vertex_project: your-project-id-2
    vertex_credentials: |
      your-credentials-2
```

You can also load balance between multiple models by using same `model_name`.

```yaml
model_list:
- model_name: your-custom-model-name
  litellm_params:
    model: gemini-2.5-pro
    api_base: https://cloudcode-pa.googleapis.com/v1internal
    vertex_project: your-project-id-1
    vertex_credentials: |
      your-credentials-1
- model_name: your-custom-model-name
  litellm_params:
    model: gemini-2.5-pro
    api_base: https://cloudcode-pa.googleapis.com/v1internal
    vertex_project: your-project-id-2
    vertex_credentials: |
      your-credentials-2
```

For more information, please refer to [LiteLLM Routing, Loadbalancing & Fallbacks](https://docs.litellm.ai/docs/routing-load-balancing)
