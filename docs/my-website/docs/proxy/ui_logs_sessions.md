import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Session Logs

Group requests into sessions. This allows you to group related requests together.


<Image img={require('../../img/ui_session_logs.png')}/>

## Usage 

### `/chat/completions`

To group multiple requests into a single session, pass the same `litellm_session_id` in the metadata for each request. Here's how to do it:

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

**Request 1**
Create a new session with a unique ID and make the first request. The session ID will be used to track all related requests.

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
        "litellm_session_id": session_id  # Pass the session ID
    }
)
```

**Request 2**
Make another request using the same session ID to link it with the previous request. This allows tracking related requests together.

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
        "litellm_session_id": session_id  # Reuse the same session ID
    }
)
```

</TabItem>
<TabItem value="langchain" label="Langchain">

**Request 1**
Initialize a new session with a unique ID and create a chat model instance for making requests. The session ID is embedded in the model's configuration.

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
        "litellm_session_id": session_id  # Pass the session ID
    }
)

# First request in session
response1 = chat.invoke("Write a short story about a robot")
```

**Request 2**
Use the same chat model instance to make another request, automatically maintaining the session context through the previously configured session ID.

```python showLineNumbers
# Second request using same chat object and session ID
response2 = chat.invoke("Now write a poem about that robot")
```

</TabItem>
<TabItem value="curl" label="Curl">

**Request 1**
Generate a new session ID and make the initial API call. The session ID in the metadata will be used to track this conversation.

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
    "litellm_session_id": "'$SESSION_ID'"
}'
```

**Request 2**
Make a follow-up request using the same session ID to maintain conversation context and tracking.

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
    "litellm_session_id": "'$SESSION_ID'"
}'
```

</TabItem>
<TabItem value="litellm" label="LiteLLM Python SDK">

**Request 1**
Start a new session by creating a unique ID and making the initial request. This session ID will be used to group related requests together.

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
        "litellm_session_id": session_id  # Pass the session ID
    }
)
```

**Request 2**
Continue the conversation by making another request with the same session ID, linking it to the previous interaction.

```python showLineNumbers
# Second request using same session ID
response2 = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Now write a poem about that robot"}],
    api_base="http://0.0.0.0:4000",
    api_key="<your litellm api key>",
    metadata={
        "litellm_session_id": session_id  # Reuse the same session ID
    }
)
```

</TabItem>
</Tabs>

### `/responses`

For the `/responses` endpoint, use `previous_response_id` to group requests into a session. The `previous_response_id` is returned in the response of each request.

<Tabs>
<TabItem value="openai" label="OpenAI Python v1.0.0+">

**Request 1**
Make the initial request and store the response ID for linking follow-up requests.

```python showLineNumbers
from openai import OpenAI

client = OpenAI(
    api_key="<your litellm api key>",
    base_url="http://0.0.0.0:4000"
)

# First request in session
response1 = client.responses.create(
    model="anthropic/claude-3-sonnet-20240229-v1:0",
    input="Write a short story about a robot"
)

# Store the response ID for the next request
response_id = response1.id
```

**Request 2**
Make a follow-up request using the previous response ID to maintain the conversation context.

```python showLineNumbers
# Second request using previous response ID
response2 = client.responses.create(
    model="anthropic/claude-3-sonnet-20240229-v1:0",
    input="Now write a poem about that robot",
    previous_response_id=response_id  # Link to previous request
)
```

</TabItem>
<TabItem value="curl" label="Curl">

**Request 1**
Make the initial request. The response will include an ID that can be used to link follow-up requests.

```bash showLineNumbers
# Store your API key
API_KEY="<your litellm api key>"

# First request in session
curl http://localhost:4000/v1/responses \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer $API_KEY" \
    --data '{
        "model": "anthropic/claude-3-sonnet-20240229-v1:0",
        "input": "Write a short story about a robot"
    }'

# Response will include an 'id' field that you'll use in the next request
```

**Request 2**
Make a follow-up request using the previous response ID to maintain the conversation context.

```bash showLineNumbers
# Second request using previous response ID
curl http://localhost:4000/v1/responses \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer $API_KEY" \
    --data '{
        "model": "anthropic/claude-3-sonnet-20240229-v1:0",
        "input": "Now write a poem about that robot",
        "previous_response_id": "resp_abc123..."  # Replace with actual response ID from previous request
    }'
```

</TabItem>
<TabItem value="litellm" label="LiteLLM Python SDK">

**Request 1**
Make the initial request and store the response ID for linking follow-up requests.

```python showLineNumbers
import litellm

# First request in session
response1 = litellm.responses(
    model="anthropic/claude-3-sonnet-20240229-v1:0",
    input="Write a short story about a robot",
    api_base="http://0.0.0.0:4000",
    api_key="<your litellm api key>"
)

# Store the response ID for the next request
response_id = response1.id
```

**Request 2**
Make a follow-up request using the previous response ID to maintain the conversation context.

```python showLineNumbers
# Second request using previous response ID
response2 = litellm.responses(
    model="anthropic/claude-3-sonnet-20240229-v1:0",
    input="Now write a poem about that robot",
    api_base="http://0.0.0.0:4000",
    api_key="<your litellm api key>",
    previous_response_id=response_id  # Link to previous request
)
```

</TabItem>
</Tabs>