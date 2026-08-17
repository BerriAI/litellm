`/chat/completion` calls routed via `openai.py`.

## Azure AI Foundry Agents v2 (Responses API)

Foundry Agents v2 uses the Responses API with `agent_reference` instead of the Assistants thread/run flow.

**Model format:** `azure_ai/agents/<agent_name>:<version>` (e.g. `azure_ai/agents/my-agent:1`)

**API base:** project endpoint, e.g. `https://<resource>.services.ai.azure.com/api/projects/<project>`

**Auth:** Azure AD bearer token via `api_key` or `AZURE_AI_API_KEY` (Entra ID token from `az account get-access-token --resource 'https://ai.azure.com'`)

**Example:**

```python
import litellm

response = litellm.responses(
    model="azure_ai/agents/my-agent:1",
    input=[{"role": "user", "content": "Tell me what you can help with."}],
    api_base="https://<resource>.services.ai.azure.com/api/projects/<project>",
    api_key="<azure-ad-token>",
)
```

v1 Assistants agents (`azure_ai/agents/asst_*`, no `:` in the model) continue to use `litellm.completion()`.
