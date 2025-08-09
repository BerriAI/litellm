# LiteLLM Prompt Management (GitOps)

Store prompts as `.prompt` files in your repository and use them directly with LiteLLM. No external services required.

### Quick Start

<Tabs>

<TabItem value="sdk" label="SDK">

**1. Create a .prompt file**

Create `prompts/hello.prompt`:

```yaml
---
model: gpt-4
temperature: 0.7
---
System: You are a helpful assistant.

User: {{user_message}}
```

**2. Use with LiteLLM**

```python
import litellm

# Set the global prompt directory
litellm.global_prompt_directory = "prompts/"

response = litellm.completion(
    model="dotprompt/gpt-4",
    prompt_id="hello",
    prompt_variables={"user_message": "What is the capital of France?"}
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

**1. Create a .prompt file**

Create `prompts/hello.prompt`:

```yaml
---
model: gpt-4
temperature: 0.7
---
System: You are a helpful assistant.

User: {{user_message}}
```

**2. Setup config.yaml**

```yaml
model_list:
  - model_name: my-dotprompt-model
    litellm_params:
      model: dotprompt/gpt-4
      prompt_id: "hello"
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  global_prompt_directory: "./prompts"
```

**3. Start the proxy**

```bash
litellm --config config.yaml --detailed_debug
```

**4. Test it!**

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "my-dotprompt-model",
    "messages": [{"role": "user", "content": "IGNORED"}],
    "prompt_variables": {
        "user_message": "What is the capital of France?"
    }
}'
```

</TabItem>
</Tabs>

### .prompt File Format

`.prompt` files use YAML frontmatter for metadata and support Jinja2 templating:

```yaml
---
model: gpt-4                    # Model to use
temperature: 0.7                # Optional parameters
max_tokens: 1000
input:
  schema:
    user_message: string        # Input validation (optional)
---
System: You are a helpful {{role}} assistant.

User: {{user_message}}
```

### Advanced Features

**Multi-role conversations:**

```yaml
---
model: gpt-4
temperature: 0.3
---
System: You are a helpful coding assistant.

User: {{user_question}}
```

**Dynamic model selection:**

```yaml
---
model: "{{preferred_model}}"  # Model can be a variable
temperature: 0.7
---
System: You are a helpful assistant specialized in {{domain}}.

User: {{user_message}}
```

### API Reference

For dotprompt integration, use these parameters:

```
model: dotprompt/<base_model>     # required (e.g., dotprompt/gpt-4)
prompt_id: str                    # required - the .prompt filename without extension
prompt_variables: Optional[dict]  # optional - variables for template rendering
```

**Example API call:**

```python
response = litellm.completion(
    model="dotprompt/gpt-4",
    prompt_id="hello",
    prompt_variables={"user_message": "Hello world"},
    messages=[{"role": "user", "content": "This will be ignored"}]
)
```
