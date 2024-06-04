import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Assistants API 

Covers Threads, Messages, Assistants. 

LiteLLM currently covers: 
- Get Assistants
- Create Thread
- Get Thread
- Add Messages
- Get Messages
- Run Thread

## Quick Start 

Call an existing Assistant. 

- Get the Assistant 

- Create a Thread when a user starts a conversation.

- Add Messages to the Thread as the user asks questions.

- Run the Assistant on the Thread to generate a response by calling the model and the tools.

<Tabs>
<TabItem value="sdk" label="SDK">

**Get the Assistant**

```python
from litellm import get_assistants, aget_assistants
import os 

# setup env
os.environ["OPENAI_API_KEY"] = "sk-.."
os.environ["AZURE_API_KEY"] = "sk-.."
os.environ["AZURE_API_BASE"] = "https://..."
os.environ["AZURE_API_VERSION"] = "2024-02-15-preview"

assistants = get_assistants(custom_llm_provider="openai")

assistants = get_assistants(custom_llm_provider="azure")

### ASYNC USAGE ### 
# assistants = await aget_assistants(custom_llm_provider="openai")
```

**Create a Thread**

```python
from litellm import create_thread, acreate_thread
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

new_thread = create_thread(
            custom_llm_provider="openai",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],  # type: ignore
        )

### ASYNC USAGE ### 
# new_thread = await acreate_thread(custom_llm_provider="openai",messages=[{"role": "user", "content": "Hey, how's it going?"}])
```

**Add Messages to the Thread**

```python
from litellm import create_thread, get_thread, aget_thread, add_message, a_add_message
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

## CREATE A THREAD
_new_thread = create_thread(
            custom_llm_provider="openai",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],  # type: ignore
        )

## OR retrieve existing thread
received_thread = get_thread(
            custom_llm_provider="openai",
            thread_id=_new_thread.id,
        )

### ASYNC USAGE ### 
# received_thread = await aget_thread(custom_llm_provider="openai", thread_id=_new_thread.id,)

## ADD MESSAGE TO THREAD
message = {"role": "user", "content": "Hey, how's it going?"}
added_message = add_message(
            thread_id=_new_thread.id, custom_llm_provider="openai", **message
        )

### ASYNC USAGE ### 
# added_message = await a_add_message(thread_id=_new_thread.id, custom_llm_provider="openai", **message)
```

**Run the Assistant on the Thread**

```python
from litellm import get_assistants, create_thread, add_message, run_thread, arun_thread
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."
assistants = get_assistants(custom_llm_provider="openai")

## get the first assistant ###
assistant_id = assistants.data[0].id

## GET A THREAD
_new_thread = create_thread(
            custom_llm_provider="openai",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],  # type: ignore
        )

## ADD MESSAGE
message = {"role": "user", "content": "Hey, how's it going?"}
added_message = add_message(
            thread_id=_new_thread.id, custom_llm_provider="openai", **message
        )

## 🚨 RUN THREAD
response = run_thread(
            custom_llm_provider="openai", thread_id=thread_id, assistant_id=assistant_id
        )

### ASYNC USAGE ### 
# response = await arun_thread(custom_llm_provider="openai", thread_id=thread_id, assistant_id=assistant_id)

print(f"run_thread: {run_thread}")
```
</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
$ export OPENAI_API_KEY="sk-..."

$ litellm

# RUNNING on http://0.0.0.0:4000
```

**Get the Assistant**

```bash
curl "http://0.0.0.0:4000/v1/assistants?order=desc&limit=20" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
```

**Create a Thread**

```bash
curl http://0.0.0.0:4000/v1/threads \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d ''
```

**Add Messages to the Thread**

```bash
curl http://0.0.0.0:4000/v1/threads/{thread_id}/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
      "role": "user",
      "content": "How does AI work? Explain it in simple terms."
    }'
```

**Run the Assistant on the Thread**

```bash
curl http://0.0.0.0:4000/v1/threads/thread_abc123/runs \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "asst_abc123"
  }'
```

</TabItem>
</Tabs>

## [👉 Proxy API Reference](https://litellm-api.up.railway.app/#/assistants)
