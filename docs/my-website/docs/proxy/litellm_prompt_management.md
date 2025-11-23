# LiteLLM AI Gateway Prompt Management

Use the LiteLLM AI Gateway to create, manage and version your prompts.

## Quick Start

### Accessing the Prompts Interface

1. Navigate to **Experimental > Prompts** in your LiteLLM dashboard
2. You'll see a table displaying all your existing prompts with the following columns:
   - **Prompt ID**: Unique identifier for each prompt
   - **Model**: The LLM model configured for the prompt
   - **Created At**: Timestamp when the prompt was created
   - **Updated At**: Timestamp of the last update
   - **Type**: Prompt type (e.g., db)
   - **Actions**: Delete and manage prompt options (admin only)

![Prompt Table](../../img/prompt_table.png)

## Create a Prompt

Click the **+ Add New Prompt** button to create a new prompt.

### Step 1: Select Your Model

Choose the LLM model you want to use from the dropdown menu at the top. You can select from any of your configured models (e.g., `aws/anthropic/bedrock-claude-3-5-sonnet`, `gpt-4o`, etc.).

### Step 2: Set the Developer Message 

The **Developer message** section allows you to set optional system instructions for the model. This acts as the system prompt that guides the model's behavior.

For example:

```
Respond as jack sparrow would
```

This will instruct the model to respond in the style of Captain Jack Sparrow from Pirates of the Caribbean.

![Add Prompt with Developer Message](../../img/add_prompt.png)

### Step 3: Add Prompt Messages

In the **Prompt messages** section, you can add the actual prompt content. Click **+ Add message** to add additional messages to your prompt template.

### Step 4: Use Variables in Your Prompts

Variables allow you to create dynamic prompts that can be customized at runtime. Use the `{{variable_name}}` syntax to insert variables into your prompts.

For example:

```
Give me a recipe for {{dish}}
```

The UI will automatically detect variables in your prompt and display them in the **Detected variables** section.

![Add Prompt with Variables](../../img/add_prompt_var.png)

### Step 5: Test Your Prompt

Before saving, you can test your prompt directly in the UI:

1. Fill in the template variables in the right panel (e.g., set `dish` to `cookies`)
2. Type a message in the chat interface to test the prompt
3. The assistant will respond using your configured model, developer message, and substituted variables

![Test Prompt with Variables](../../img/add_prompt_use_var1.png)

The result will show the model's response with your variables substituted:

![Prompt Test Results](../../img/add_prompt_use_var.png)

### Step 6: Save Your Prompt

Once you're satisfied with your prompt, click the **Save** button in the top right corner to save it to your prompt library.

## Using Your Prompts

Now that your prompt is published, you can use it in your application via the LiteLLM proxy API.

### Simple Usage

Call a prompt without any variables:

```python
import litellm

response = litellm.completion(
    model="prompt/your-prompt-id",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### With Custom Messages

You can override or add to the prompt's messages:

```python
import litellm

response = litellm.completion(
    model="prompt/your-prompt-id",
    messages=[
        {"role": "user", "content": "What's the weather like?"}
    ]
)
```

### With Prompt Variables

Pass variables to your prompt template using the `prompt_variables` parameter:

```python
import litellm

response = litellm.completion(
    model="prompt/your-prompt-id",
    messages=[{"role": "user", "content": "what is it ?"}],
    prompt_variables={"dish": "cookies"}
)
```

### Using the OpenAI SDK

You can also use the OpenAI SDK directly with the LiteLLM proxy:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-api-key",
    base_url="http://your-litellm-proxy:4000"
)

response = client.chat.completions.create(
    model="prompt/your-prompt-id",
    messages=[{"role": "user", "content": "what is it ?"}],
    extra_body={
        "prompt_variables": {"dish": "pasta"}
    }
)
```

### cURL Example

```bash
curl -X POST http://your-litellm-proxy:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-litellm-api-key" \
  -d '{
    "model": "prompt/your-prompt-id",
    "messages": [{"role": "user", "content": "what is it ?"}],
    "prompt_variables": {"dish": "cookies"}
  }'
```

## Prompt Versioning

LiteLLM automatically versions your prompts each time you update them. This allows you to maintain a complete history of changes and roll back to previous versions if needed.

### View Prompt Details

Click on any prompt ID in the prompts table to view its details page. This page shows:
- **Prompt ID**: The unique identifier for your prompt
- **Version**: The current version number (e.g., v4)
- **Prompt Type**: The storage type (e.g., db)
- **Created At**: When the prompt was first created
- **Last Updated**: Timestamp of the most recent update
- **LiteLLM Parameters**: The raw JSON configuration

![Prompt Details](../../img/edit_prompt.png)

### Update a Prompt

To update an existing prompt:

1. Click on the prompt you want to update from the prompts table
2. Click the **Prompt Studio** button in the top right
3. Make your changes to:
   - Model selection
   - Developer message (system instructions)
   - Prompt messages
   - Variables
4. Test your changes in the chat interface on the right
5. Click the **Update** button to save the new version

![Edit Prompt in Studio](../../img/edit_prompt2.png)

Each time you click **Update**, a new version is created (v1 → v2 → v3, etc.) while maintaining the same prompt ID.

### View Version History

To view all versions of a prompt:

1. Open the prompt in **Prompt Studio**
2. Click the **History** button in the top right
3. A **Version History** panel will open on the right side

![Version History Panel](../../img/edit_prompt3.png)

The version history panel displays:
- **Latest version** (marked with a "Latest" badge and "Active" status)
- All previous versions (v4, v3, v2, v1, etc.)
- Timestamps for each version
- Database save status ("Saved to Database")

### View and Restore Older Versions

To view or restore an older version:

1. In the **Version History** panel, click on any previous version (e.g., v2)
2. The prompt studio will load that version's configuration
3. You can see:
   - The developer message from that version
   - The prompt messages from that version
   - The model and parameters used
   - All variables defined at that time

![View Older Version](../../img/edit_prompt4.png)

The selected version will be highlighted with an "Active" badge in the version history panel.

To restore an older version:
1. View the older version you want to restore
2. Click the **Update** button
3. This will create a new version with the content from the older version

### Use Specific Versions in API Calls

By default, API calls use the latest version of a prompt. To use a specific version, append the version number to the prompt ID:

```python
import litellm

# Use latest version (default)
response = litellm.completion(
    model="prompt/jack-sparrow",
    messages=[{"role": "user", "content": "Hello"}]
)

# Use specific version v2
response = litellm.completion(
    model="prompt/jack-sparrow:v2",
    messages=[{"role": "user", "content": "Hello"}]
)
```

With the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-litellm-api-key",
    base_url="http://your-litellm-proxy:4000"
)

# Use version v3
response = client.chat.completions.create(
    model="prompt/jack-sparrow:v3",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Using cURL:

```bash
curl -X POST http://your-litellm-proxy:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-litellm-api-key" \
  -d '{
    "model": "prompt/jack-sparrow:v2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```






