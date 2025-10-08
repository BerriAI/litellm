# LiteLLM gitlab Prompt Management

A powerful prompt management system for LiteLLM that fetches `.prompt` files from gitlab repositories. This enables team-based prompt management with gitlab's built-in access control and version control capabilities.

## Features

- **🏢 Team-based access control**: Leverage gitlab's workspace and repository permissions
- **📁 Repository-based prompt storage**: Store prompts in gitlab repositories
- **🔐 Multiple authentication methods**: Support for access tokens and basic auth
- **🎯 YAML frontmatter**: Define model, parameters, and schemas in file headers
- **🔧 Handlebars templating**: Use `{{variable}}` syntax with Jinja2 backend
- **✅ Input validation**: Automatic validation against defined schemas
- **🔗 LiteLLM integration**: Works seamlessly with `litellm.completion()`
- **💬 Smart message parsing**: Converts prompts to proper chat messages
- **⚙️ Parameter extraction**: Automatically applies model settings from prompts

## Quick Start

### 1. Set up gitlab Repository

Create a repository in your gitlab workspace and add `.prompt` files:

```
your-repo/
├── prompts/
│   ├── chat_assistant.prompt
│   ├── code_reviewer.prompt
│   └── data_analyst.prompt
```

### 2. Create a `.prompt` file

Create a file called `prompts/chat_assistant.prompt`:

```yaml
---
model: gpt-4
temperature: 0.7
max_tokens: 150
input:
  schema:
    user_message: string
    system_context?: string
---

{% if system_context %}System: {{system_context}}

{% endif %}User: {{user_message}}
```

### 3. Configure gitlab Access

#### Option A: Access Token (Recommended)

```python
import litellm

# Configure gitlab access
gitlab_config = {
    "project": "a/b/<repo_name>",
    "access_token": "your-access-token",
    "base_url": "gitlab url",
    "prompts_path": "src/prompts", # folder to point to, defaults to root
    "branch":"main"  # optional, defaults to main
}

# Set global gitlab configuration
litellm.set_global_gitlab_config(gitlab_config)
```

#### Option B: Basic Authentication

```python
import litellm

# Configure gitlab access with basic auth
gitlab_config = {
    "project": "a/b/<repo_name>",
    "base_url": "base url",
    "access_token": "your-app-password",  # Use app password for basic auth
    "branch": "main",
    "prompts_path": "src/prompts", # folder to point to, defaults to root
}

litellm.set_global_gitlab_config(gitlab_config)
```

### 4. Use with LiteLLM

```python
# Use with completion - the model prefix 'gitlab/' tells LiteLLM to use gitlab prompt management
response = litellm.completion(
    model="gitlab/gpt-4",  # The actual model comes from the .prompt file
    prompt_id="prompts/chat_assistant", # Location of the prompt file
    prompt_variables={
        "user_message": "What is machine learning?",
        "system_context": "You are a helpful AI tutor."
    },
    # Any additional messages will be appended after the prompt
    messages=[{"role": "user", "content": "Please explain it simply."}]
)

print(response.choices[0].message.content)
```

## Proxy Server Configuration

### 1. Create a `.prompt` file

Create `prompts/hello.prompt`:

```yaml
---
model: gpt-4
temperature: 0.7
---
System: You are a helpful assistant.

User: {{user_message}}
```

### 2. Setup config.yaml

```yaml
model_list:
  - model_name: my-gitlab-model
    litellm_params:
      model: gitlab/gpt-4
      prompt_id: "prompts/hello"
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  global_gitlab_config:
    workspace: "your-workspace"
    repository: "your-repo"
    access_token: "your-access-token"
    branch: "main"
```

### 3. Start the proxy

```bash
litellm --config config.yaml --detailed_debug
```

### 4. Test it!

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "my-gitlab-model",
    "messages": [{"role": "user", "content": "IGNORED"}],
    "prompt_variables": {
        "user_message": "What is the capital of France?"
    }
}'
```

## Prompt File Format

### Basic Structure

```yaml
---
# Model configuration
model: gpt-4
temperature: 0.7
max_tokens: 500

# Input schema (optional)
input:
  schema:
    user_message: string
    system_context?: string
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

## Team-Based Access Control

gitlab's built-in permission system provides team-based access control:

1. **Workspace-level permissions**: Control access to entire workspaces
2. **Repository-level permissions**: Control access to specific repositories
3. **Branch-level permissions**: Control access to specific branches
4. **User and group management**: Manage team members and their access levels

### Setting up Team Access

1. **Create workspaces for each team**:
   ```
   team-a-prompts/
   team-b-prompts/
   team-c-prompts/
   ```

2. **Configure repository permissions**:
   - Grant read access to team members
   - Grant write access to prompt maintainers
   - Use branch protection rules for production prompts

3. **Use different access tokens**:
   - Each team can have their own access token
   - Tokens can be scoped to specific repositories
   - Use app passwords for additional security

## API Reference

### gitlab Configuration

```python
gitlab_config = {
    "workspace": str,           # Required: gitlab workspace name
    "repository": str,          # Required: Repository name
    "access_token": str,        # Required: gitlab access token or app password
    "branch": str,              # Optional: Branch to fetch from (default: "main")
    "base_url": str,            # Optional: Custom gitlab API URL
    "auth_method": str,         # Optional: "token" or "basic" (default: "token")
    "username": str,            # Optional: Username for basic auth
    "base_url" : str            # Optional: Incase where the base url is not https://api.gitlab.org/2.0
}
```

### LiteLLM Integration

```python
response = litellm.completion(
    model="gitlab/<base_model>",  # required (e.g., gitlab/gpt-4)
    prompt_id=str,                   # required - the .prompt filename without extension
    prompt_variables=dict,           # optional - variables for template rendering
    gitlab_config=dict,           # optional - gitlab configuration (if not set globally)
    messages=list,                   # optional - additional messages
)
```

## Error Handling

The gitlab integration provides detailed error messages for common issues:

- **Authentication errors**: Invalid access tokens or credentials
- **Permission errors**: Insufficient access to workspace/repository
- **File not found**: Missing .prompt files
- **Network errors**: Connection issues with gitlab API

## Security Considerations

1. **Access Token Security**: Store access tokens securely using environment variables or secret management systems
2. **Repository Permissions**: Use gitlab's permission system to control access
3. **Branch Protection**: Protect main branches from unauthorized changes
4. **Audit Logging**: gitlab provides audit logs for all repository access

## Troubleshooting

### Common Issues

1. **"Access denied" errors**: Check your gitlab permissions for the workspace and repository
2. **"Authentication failed" errors**: Verify your access token or credentials
3. **"File not found" errors**: Ensure the .prompt file exists in the specified branch
4. **Template rendering errors**: Check your Handlebars syntax in the .prompt file

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import litellm
litellm.set_verbose = True

# Your gitlab prompt calls will now show detailed logs
response = litellm.completion(
    model="gitlab/gpt-4",
    prompt_id="your_prompt",
    prompt_variables={"key": "value"}
)
```

## Migration from File-Based Prompts

If you're currently using file-based prompts with the dotprompt integration, you can easily migrate to gitlab:

1. **Upload your .prompt files** to a gitlab repository
2. **Update your configuration** to use gitlab instead of local files
3. **Set up team access** using gitlab's permission system
4. **Update your code** to use `gitlab/` model prefix instead of `dotprompt/`

This provides better collaboration, version control, and team-based access control for your prompts.
