import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom Prompt Management

Follow this guide to implement custom hooks that allow connecting LiteLLM to your prompt management system.

## Quick Start

### 1. Implement a `CustomPromptManagement` Class

Create a class that inherits from `CustomPromptManagement` to manage prompts and their parameters. The key method to implement is `get_chat_completion_prompt`.

**Example Implementation**

Create a new file called `custom_prompt.py` and add this code:

```python
from typing import List, Tuple, Optional
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams

class MyCustomPromptManagement(CustomPromptManagement):
    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Args:
            model: The model name
            messages: List of message objects
            non_default_params: Optional parameters like temperature, max_tokens
            prompt_id: Identifier for the prompt to retrieve
            prompt_variables: Variables to format into the prompt
            dynamic_callback_params: Additional callback parameters

        Returns:
            - model: str - the model to use
            - messages: List[AllMessageValues] - the messages to use
            - non_default_params: dict - optional params (e.g. temperature)
        """
        # Example 1: Simple prompt retrieval
        if prompt_id == "welcome_prompt":
            messages = [
                {"role": "user", "content": "Welcome to our AI assistant! How can I help you today?"},
            ]
            return model, messages, non_default_params
        
        # Example 2: Prompt with variables
        elif prompt_id == "personalized_greeting":
            content = "Hello, {name}! You are {age} years old and live in {city}."
            content_with_variables = content.format(**(prompt_variables or {}))
            messages = [
                {"role": "user", "content": content_with_variables},
            ]
            return model, messages, non_default_params
        
        # Default: Return original messages if no prompt_id match
        return model, messages, non_default_params

prompt_management = MyCustomPromptManagement()
```

### 2. Configure Your Prompt Manager in LiteLLM `config.yaml`

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: custom_prompt.prompt_management  # sets litellm.callbacks = [prompt_management]
```

### 3. Start LiteLLM Gateway

<Tabs>
<TabItem value="docker" label="Docker Run">

Mount your `custom_logger.py` on the LiteLLM Docker container.

```shell
docker run -d \
  -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --name my-app \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  -v $(pwd)/custom_logger.py:/app/custom_logger.py \
  my-app:latest \
  --config /app/config.yaml \
  --port 4000 \
  --detailed_debug \
```

</TabItem>

<TabItem value="py" label="litellm pip">

```shell
litellm --config config.yaml --detailed_debug
```

</TabItem>
</Tabs>

### 4. Test Your Custom Prompt Manager

#### Example 1: Simple Prompt ID

When you pass `prompt_id="welcome_prompt"`, the custom prompt manager will replace your empty messages with a predefined welcome message: `"Welcome to our AI assistant! How can I help you today?"`.

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[],
    prompt_id="welcome_prompt"
)

print(response.choices[0].message.content)
```
</TabItem>

<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-4",
    openai_api_key="sk-1234",
    openai_api_base="http://0.0.0.0:4000",
    extra_body={
        "prompt_id": "welcome_prompt"
    }
)

messages = []
response = chat(messages)

print(response.content)
```
</TabItem>

<TabItem value="curl" label="Curl">

```shell
curl -X POST http://0.0.0.0:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
    "model": "gpt-4",
    "messages": [],
    "prompt_id": "welcome_prompt"
}'
```
</TabItem>
</Tabs>

Expected response:

```json
{
  "id": "chatcmpl-123",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "I'd be happy to assist you today! How can I help?",
        "role": "assistant"
      }
    }
  ]
}
```

#### Example 2: Prompt ID with Variables

When you pass `prompt_id="personalized_greeting"`, the custom prompt manager will:
1. Start with the template: `"Hello, {name}! You are {age} years old and live in {city}."`
2. Replace the variables with your provided values
3. Create a message with the formatted content: `"Hello, John! You are 30 years old and live in New York."`

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[],
    prompt_id="personalized_greeting",
    prompt_variables={
        "name": "John",
        "age": 30,
        "city": "New York"
    }
)

print(response.choices[0].message.content)
```
</TabItem>

<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

chat = ChatOpenAI(
    model="gpt-4",
    openai_api_key="sk-1234",
    openai_api_base="http://0.0.0.0:4000",
    extra_body={
        "prompt_id": "personalized_greeting",
        "prompt_variables": {
            "name": "John",
            "age": 30,
            "city": "New York"
        }
    }
)

messages = []
response = chat(messages)

print(response.content)
```
</TabItem>

<TabItem value="curl" label="Curl">

```shell
curl -X POST http://0.0.0.0:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
    "model": "gpt-4",
    "messages": [],
    "prompt_id": "personalized_greeting",
    "prompt_variables": {
        "name": "John",
        "age": 30,
        "city": "New York"
    }
}'
```
</TabItem>
</Tabs>

Expected response:

```json
{
  "id": "chatcmpl-123",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Hi John! I understand you're 30 years old and based in New York. How can I assist you today?",
        "role": "assistant"
      }
    }
  ]
}
```

The prompt manager will:
1. Intercept the request
2. Check the prompt_id to determine which template to use
3. If variables are provided, format them into the template
4. Send the final prompt to the LLM


