import Image from '@theme/IdealImage';

# Call Responses API models on Claude Code

This tutorial shows how to call the Responses API models like `codex-mini` and `o3-pro` from the Claude Code endpoint on LiteLLM.


Pre-requisites:

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- LiteLLM v1.72.6-stable or higher


### 1. Setup config.yaml

```yaml
model_list:
    - model_name: codex-mini    
      litellm_params:
        model: openai/codex-mini
        api_key: sk-proj-1234567890
        api_base: https://api.openai.com/v1
```

### 2. Start proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Test it! (Curl)

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer sk-proj-1234567890" \
-H "Content-Type: application/json" \
-d '{
    "model": "codex-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

### 4. Test it! (Claude Code)

- Setup environment variables

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_API_KEY="sk-1234" # replace with your LiteLLM key
```

- Start a Claude Code session

```bash
claude --model codex-mini-latest
```

- Send a message

<Image img={require('../../img/release_notes/claude_code_demo.png')} style={{ width: '500px', height: 'auto' }} />