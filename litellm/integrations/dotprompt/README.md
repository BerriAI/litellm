# LiteLLM Dotprompt Manager

A powerful prompt management system for LiteLLM that supports [Google's Dotprompt specification](https://google.github.io/dotprompt/getting-started/). This allows you to manage your AI prompts in organized `.prompt` files with YAML frontmatter, Handlebars templating, and full integration with LiteLLM's completion API.

## Features

- **📁 File-based prompt management**: Organize prompts in `.prompt` files
- **🎯 YAML frontmatter**: Define model, parameters, and schemas in file headers
- **🔧 Handlebars templating**: Use `{{variable}}` syntax with Jinja2 backend
- **✅ Input validation**: Automatic validation against defined schemas
- **🔗 LiteLLM integration**: Works seamlessly with `litellm.completion()`
- **💬 Smart message parsing**: Converts prompts to proper chat messages
- **⚙️ Parameter extraction**: Automatically applies model settings from prompts

## Quick Start

### 1. Create a `.prompt` file

Create a file called `chat_assistant.prompt`:

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

### 2. Use with LiteLLM

```python
import litellm

litellm.set_global_prompt_directory("path/to/your/prompts")

# Use with completion - the model prefix 'dotprompt/' tells LiteLLM to use prompt management
response = litellm.completion(
    model="dotprompt/gpt-4",  # The actual model comes from the .prompt file
    prompt_id="chat_assistant",
    prompt_variables={
        "user_message": "What is machine learning?",
        "system_context": "You are a helpful AI tutor."
    },
    # Any additional messages will be appended after the prompt
    messages=[{"role": "user", "content": "Please explain it simply."}]
)

print(response.choices[0].message.content)
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
    name: string
    age: integer
    preferences?: array
---

# Template content using Handlebars syntax
Hello {{name}}! 

{% if age >= 18 %}
You're an adult, so here are some mature recommendations:
{% else %}
Here are some age-appropriate suggestions:
{% endif %}

{% for pref in preferences %}
- Based on your interest in {{pref}}, I recommend...
{% endfor %}
```

### Supported Frontmatter Fields

- **`model`**: The LLM model to use (e.g., `gpt-4`, `claude-3-sonnet`)
- **`input.schema`**: Define expected input variables and their types
- **`output.format`**: Expected output format (`json`, `text`, etc.)
- **`output.schema`**: Structure of expected output

### Additional Parameters

- **`temperature`**: Model temperature (0.0 to 1.0)
- **`max_tokens`**: Maximum tokens to generate
- **`top_p`**: Nucleus sampling parameter (0.0 to 1.0)
- **`frequency_penalty`**: Frequency penalty (0.0 to 1.0)
- **`presence_penalty`**: Presence penalty (0.0 to 1.0)
- any other parameters that are not model or schema-related will be treated as optional parameters to the model. 

### Input Schema Types

- `string` or `str`: Text values
- `integer` or `int`: Whole numbers
- `float`: Decimal numbers  
- `boolean` or `bool`: True/false values
- `array` or `list`: Lists of values
- `object` or `dict`: Key-value objects

Use `?` suffix for optional fields: `name?: string`

## Message Format Conversion

The dotprompt manager intelligently converts your rendered prompts into proper chat messages:

### Simple Text → User Message
```yaml
---
model: gpt-4
---
Tell me about {{topic}}.
```
Becomes: `[{"role": "user", "content": "Tell me about AI."}]`

### Role-Based Format → Multiple Messages
```yaml
---
model: gpt-4
---
System: You are a {{role}}.

User: {{question}}
```

Becomes:
```python
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"}
]
```


## Example Prompts

### Data Extraction
```yaml
# extract_info.prompt
---
model: gemini/gemini-1.5-pro
input:
  schema:
    text: string
output:
  format: json
  schema:
    title?: string
    summary: string
    tags: array
---

Extract the requested information from the given text. Return JSON format.

Text: {{text}}
```

### Code Assistant
```yaml
# code_helper.prompt
---
model: claude-3-5-sonnet-20241022
temperature: 0.2
max_tokens: 2000
input:
  schema:
    language: string
    task: string
    code?: string
---

You are an expert {{language}} programmer.

Task: {{task}}

{% if code %}
Current code:
```{{language}}
{{code}}
```
{% endif %}

Please provide a complete, well-documented solution.
```

### Multi-turn Conversation
```yaml
# conversation.prompt
---
model: gpt-4
temperature: 0.8
input:
  schema:
    personality: string
    context: string
---

System: You are a {{personality}}. {{context}}

User: Let's start our conversation.
```

## API Reference

### PromptManager

The core class for managing `.prompt` files.

#### Methods

- **`__init__(prompt_directory: str)`**: Initialize with directory path
- **`render(prompt_id: str, variables: dict) -> str`**: Render prompt with variables
- **`list_prompts() -> List[str]`**: Get all available prompt IDs
- **`get_prompt(prompt_id: str) -> PromptTemplate`**: Get prompt template object
- **`get_prompt_metadata(prompt_id: str) -> dict`**: Get prompt metadata
- **`reload_prompts() -> None`**: Reload all prompts from directory
- **`add_prompt(prompt_id: str, content: str, metadata: dict)`**: Add prompt programmatically

### DotpromptManager

LiteLLM integration class extending `PromptManagementBase`.

#### Methods

- **`__init__(prompt_directory: str)`**: Initialize with directory path
- **`should_run_prompt_management(prompt_id: str, params: dict) -> bool`**: Check if prompt exists
- **`set_prompt_directory(directory: str)`**: Change prompt directory
- **`reload_prompts()`**: Reload prompts from directory

### PromptTemplate

Represents a single prompt with metadata.

#### Properties

- **`content: str`**: The prompt template content
- **`metadata: dict`**: Full metadata from frontmatter
- **`model: str`**: Specified model name
- **`temperature: float`**: Model temperature
- **`max_tokens: int`**: Token limit
- **`input_schema: dict`**: Input validation schema
- **`output_format: str`**: Expected output format
- **`output_schema: dict`**: Output structure schema

## Best Practices

1. **Organize by purpose**: Group related prompts in subdirectories
2. **Use descriptive names**: `extract_user_info.prompt` vs `prompt1.prompt`
3. **Define schemas**: Always specify input schemas for validation
4. **Version control**: Store `.prompt` files in git for change tracking
5. **Test prompts**: Use the test framework to validate prompt behavior
6. **Keep templates focused**: One prompt should do one thing well
7. **Use includes**: Break complex prompts into reusable components

## Troubleshooting

### Common Issues

**Prompt not found**: Ensure the `.prompt` file exists and has correct extension
```python
# Check available prompts
from litellm.integrations.dotprompt import get_dotprompt_manager
manager = get_dotprompt_manager()
print(manager.prompt_manager.list_prompts())
```

**Template errors**: Verify Handlebars syntax and variable names
```python
# Test rendering directly
manager.prompt_manager.render("my_prompt", {"test": "value"})
```

**Model not working**: Check that model name in frontmatter is correct
```python
# Check prompt metadata
metadata = manager.prompt_manager.get_prompt_metadata("my_prompt")
print(metadata)
```

### Validation Errors

Input validation failures show helpful error messages:
```
ValueError: Invalid type for field 'age': expected int, got str
```

Make sure your variables match the defined schema types.

## Contributing

The LiteLLM Dotprompt manager follows the [Dotprompt specification](https://google.github.io/dotprompt/) for maximum compatibility. When contributing:

1. Ensure compatibility with existing `.prompt` files
2. Add tests for new features
3. Update documentation
4. Follow the existing code style

## License

This prompt management system is part of LiteLLM and follows the same license terms.