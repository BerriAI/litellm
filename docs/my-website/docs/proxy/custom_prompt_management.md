
import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom Prompt Management

Follow this guide to implement custom hooks that allow connecting LiteLLM to your prompt management system.

## Quick Start

### 1. Implement a `CustomLogger` Class

A `CustomLogger` class is used to manage prompts and their parameters. It has a key method to retrieve the chat completion prompt.

**Example `CustomLogger` Class**

Create a new file called `custom_logger.py` and add this code to it:

```python
from typing import List, Tuple, Optional
from litellm.integrations.prompt_management_base import PromptManagementBase
from litellm.integrations.custom_logger import CustomLogger
from litellm.types import AllMessageValues, StandardCallbackDynamicParams

class CustomPromptManagement(CustomLogger, PromptManagementBase):
    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Returns:
        - model: str - the model to use (can be pulled from prompt management tool)
        - messages: List[AllMessageValues] - the messages to use (can be pulled from prompt management tool)
        - non_default_params: dict - update with any optional params (e.g. temperature, max_tokens, etc.) to use (can be pulled from prompt management tool)
        """
        return model, messages, non_default_params
    
    @property
    def custom_logger_name(self) -> str:
        return "custom-prompt-management"

proxy_prompt_management_instance = CustomPromptManagement()
```

### 2. Configure Your Logger in LiteLLM `config.yaml`

In the configuration file, specify your custom logger class to manage prompts.

- Python Filename: `custom_logger.py`
- Logger class name: `CustomLogger`

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: custom_logger.proxy_prompt_management_instance # sets litellm.callbacks = [proxy_prompt_management_instance]

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

### 4. Test Your Custom Logger

#### Test `"custom-logger"`

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Retrieve Prompt" value="retrieve-prompt">

Use this to test the retrieval of prompts using your custom logger.

```shell
curl -i  -X POST http://localhost:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer sk-1234" \
-d '{
    "model": "custom-prompt-management/gpt-4",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how can I assist you today?"
        }
    ],
    "prompt_id": "1234",
    "prompt_variables": {
        "name": "John Doe"
    }
}'
```

```json
{
  "id": "chatcmpl-9zREDkBIG20RJB4pMlyutmi1hXQWc",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Hello! How can I help you today?",
        "role": "assistant"
      }
    }
  ],
  "created": 1724429701,
  "model": "gpt-4o-2024-05-13",
  "object": "chat.completion",
  "system_fingerprint": "fp_3aa7262c27",
  "usage": {
    "completion_tokens": 65,
    "prompt_tokens": 14,
    "total_tokens": 79
  },
  "service_tier": null
}
```

</TabItem>
</Tabs>


