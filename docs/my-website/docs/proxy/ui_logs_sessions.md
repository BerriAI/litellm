import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# UI Logs Sessions Management

Group requests into sessions. This allows you to see all requests made by a user in a single session.


<Image img={require('../../img/ui_session_logs.png')}/>

## Usage 

### `/chat/completions`

To group multiple requests into a single session, pass the same `litellm_trace_id` in the metadata for each request. Here's how to do it:

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

**Request 1**
```python showLineNumbers
import openai
import uuid

# Create a session ID
session_id = str(uuid.uuid4())

client = openai.OpenAI(
    api_key="<your litellm api key>",
    base_url="http://0.0.0.0:4000"
)

# First request in session
response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Write a short story about a robot"
        }
    ],
    extra_body={
        "metadata": {
            "litellm_trace_id": session_id  # Pass the session ID
        }
    }
)
```

**Request 2**
```python showLineNumbers
# Second request using same session ID
response2 = client.chat.completions.create(
    model="gpt-4o", 
    messages=[
        {
            "role": "user",
            "content": "Now write a poem about that robot"
        }
    ],
    extra_body={
        "metadata": {
            "litellm_trace_id": session_id  # Reuse the same session ID
        }
    }
)
```

</TabItem>
<TabItem value="langchain" label="Langchain">

**Request 1**
```python showLineNumbers
from langchain.chat_models import ChatOpenAI
import uuid

# Create a session ID
session_id = str(uuid.uuid4())

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    api_key="<your litellm api key>",
    model="gpt-4o",
    extra_body={
        "metadata": {
            "litellm_trace_id": session_id  # Pass the session ID
        }
    }
)

# First request in session
response1 = chat.invoke("Write a short story about a robot")
```

**Request 2**
```python showLineNumbers
# Second request using same chat object and session ID
response2 = chat.invoke("Now write a poem about that robot")
```

</TabItem>
<TabItem value="curl" label="Curl">

**Request 1**
```bash showLineNumbers
# Create a session ID
SESSION_ID=$(uuidgen)

# Store your API key
API_KEY="<your litellm api key>"

# First request in session
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer $API_KEY" \
    --data '{
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": "Write a short story about a robot"
        }
    ],
    "metadata": {
        "litellm_trace_id": "'$SESSION_ID'"
    }
}'
```

**Request 2**
```bash showLineNumbers
# Second request using same session ID
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer $API_KEY" \
    --data '{
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": "Now write a poem about that robot"
        }
    ],
    "metadata": {
        "litellm_trace_id": "'$SESSION_ID'"
    }
}'
```

</TabItem>
<TabItem value="litellm" label="LiteLLM">

**Request 1**
```python showLineNumbers
import litellm
import uuid

# Create a session ID
session_id = str(uuid.uuid4())

# First request in session
response1 = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short story about a robot"}],
    api_base="http://0.0.0.0:4000",
    api_key="<your litellm api key>",
    metadata={
        "litellm_trace_id": session_id  # Pass the session ID
    }
)
```

**Request 2**
```python showLineNumbers
# Second request using same session ID
response2 = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Now write a poem about that robot"}],
    api_base="http://0.0.0.0:4000",
    api_key="<your litellm api key>",
    metadata={
        "litellm_trace_id": session_id  # Reuse the same session ID
    }
)
```

</TabItem>
</Tabs>

## `/responses`