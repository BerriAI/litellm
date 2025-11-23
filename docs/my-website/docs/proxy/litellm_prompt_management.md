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

