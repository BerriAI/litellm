import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# GigaChat
[https://developers.sber.ru/docs/en/gigachat/api/overview](https://developers.sber.ru/docs/en/gigachat/api/overview)

## API Key
```python
# env variable
import os
os.environ['GIGACHAT_SCOPE'] = "GIGACHAT_API_PERS" # or GIGACHAT_API_CORP or GIGACHAT_API_B2B
os.environ['GIGACHAT_CREDENTIALS'] = "<your_authorization_key>"
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"
```

```dotenv
export GIGACHAT_SCOPE=GIGACHAT_API_PERS
export GIGACHAT_CREDENTIALS="..."
export GIGACHAT_VERIFY_SSL_CERTS=False
```


## Sample Usage
```python
from litellm import completion
import os

os.environ['GIGACHAT_SCOPE'] = "GIGACHAT_API_PERS" # or GIGACHAT_API_CORP or GIGACHAT_API_B2B
os.environ['GIGACHAT_CREDENTIALS'] = "<your_authorization_key>"
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"

response = completion(
    model="gigachat/GigaChat-2",
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?",
        }
    ],
    max_tokens=100,
    temperature=0.2,
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['GIGACHAT_SCOPE'] = "GIGACHAT_API_PERS" # or GIGACHAT_API_CORP or GIGACHAT_API_B2B
os.environ['GIGACHAT_CREDENTIALS'] = "<your_authorization_key>"
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"

response = completion(
    model="gigachat/GigaChat-2",
    messages=[
        {
            "role": "user",
            "content": "How are you doing?",
        }
    ],
    max_tokens=100,
    temperature=0.2,
    stream=True
)
for chunk in response:
    print(chunk)
```

## Usage with LiteLLM Proxy Server

Here's how to call a GigaChat model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
    model_list:
      - model_name: GigaChat-2-Max
        litellm_params: 
          model: gigachat/GigaChat-2-Max 
          scope: os.environ/GIGACHAT_SCOPE
          credentials: os.environ/GIGACHAT_CREDENTIALS
          ssl_verify: os.environ/GIGACHAT_VERIFY_SSL_CERTS
  ```

2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4000/", api_key="sk-1234")

completion = client.chat.completions.create(
    model="GigaChat-2-Max",
    messages=[
        {"role": "user", "content": "How are you?"},
    ],
)
print(completion)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "GigaChat-2-Max",
      "messages": [
          {
          "role": "user",
          "content": "How are you?"
          }
      ]
  }'
  ```
  </TabItem>

  </Tabs>

## GigaChat - Tool Calling

```python
import json
import os

from litellm import completion

os.environ['GIGACHAT_SCOPE'] = "GIGACHAT_API_PERS" # or GIGACHAT_API_CORP or GIGACHAT_API_B2B
os.environ['GIGACHAT_CREDENTIALS'] = "<your_authorization_key>"
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_horoscope",
            "description": "Get today's horoscope for an astrological sign.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sign": {
                        "type": "string",
                        "description": "An astrological sign like Taurus or Aquarius",
                    },
                },
                "required": ["sign"],
            },
        },
    },
]


def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."

messages = [{"role": "user", "content": "What is my horoscope? I am an Aquarius."}]
first_response = completion(
    model="gigachat/GigaChat-2-Max",
    tools=tools,
    messages=messages
)
print("\nFirst LLM Response:\n", first_response)
response_message = first_response.choices[0].message
messages.append(response_message)
tool_calls = response_message.tool_calls
if tool_calls:
    for tool_call in tool_calls:
        if tool_call.function.name == "get_horoscope":
            args = json.loads(tool_call.function.arguments)
            horoscope = get_horoscope(args["sign"])

            # Add tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"get_horoscope": horoscope})
            })
    messages.append({"role": "user", "content": "Answer only with tool result"})
    print(messages)
    second_response = completion(
        model="gigachat/GigaChat-2-Max", messages=messages
    )  # get a new response from the model where it can see the function response
    print("second response\n", second_response)

```

## GigaChat - Vision Example

### Base64
```python
import base64
from litellm import completion
import os

os.environ['GIGACHAT_SCOPE'] = "GIGACHAT_API_PERS" # or GIGACHAT_API_CORP or GIGACHAT_API_B2B
os.environ['GIGACHAT_CREDENTIALS'] = "<your_authorization_key>"
os.environ["GIGACHAT_VERIFY_SSL_CERTS"] = "False"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "path_to_your_image.png"

# Getting the base64 string
base64_image = encode_image(image_path)
response = completion(
    model="gigachat/GigaChat-2-Max",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

### URL
```python
from litellm import completion
import os

url = "https://images.rawpixel.com/image_png_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTA4L3Jhd3BpeGVsX29mZmljZV8zMF9hX3N0dWRpb19zaG90X29mX2NhdF93YXZpbmdfaW1hZ2VzZnVsbF9ib2R5X182YzRmM2YyOC0wMGJjLTQzNTYtYjM3ZC05NDM0NTgwY2FmNDcucG5n.png"

response = completion(
    model="gigachat/GigaChat-2-Max",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"{url}"},
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```

## GigaChat - PDF File Parsing

```python
import base64

from litellm import completion

def encode_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


file_path = "path_to_your_file.pdf"

# Getting the base64 string
base64_pdf = encode_file(file_path)
response = completion(
    model="gigachat/GigaChat-2-Max",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Create a comprehensive summary of this pdf"},
                {
                    "type": "file",
                    "file": {
                        "filename": "Day_2_v6.pdf",
                        "file_data": f"data:application/pdf;base64,{base64_pdf}"
                    }
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)
```


## GigaChat - Structured Output
:::tip

**It's currently in beta testing, it might break unexpectedly.**

:::

### Pydantic BaseModel
  <Tabs>

  <TabItem value="openai" label="SDK">
    ```python
    from pydantic import BaseModel, Field
    from litellm import completion
    
    class ResponseFormat(BaseModel):
        """Model response format"""
    
        thinking: str = Field(description="Thoughts of model")
        output: str = Field(description="Final answer")
    
    
    response = completion(
        model="gigachat/GigaChat-2-Max",
        messages=[
            {"role": "system", "content": "You're math professor."},
            {
                "role": "user",
                "content": "Solve this equation 8x^2 - 20x + 6 = 0",
            },
        ],
        response_format=ResponseFormat,
    )
    
    message = response.choices[0].message
    print(message)
    print(message.parsed)
    ```
  </TabItem>

  <TabItem value="OpenAI" label="OpenAI client">

  ```python
from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI(base_url="http://localhost:4000/", api_key="sk-1234")


class ResponseFormat(BaseModel):
    """Model response format"""

    thinking: str = Field(description="Thoughts of model")
    output: str = Field(description="Final answer")


response = client.chat.completions.parse(
    model="GigaChat-2-Max",
    messages=[
        {"role": "system", "content": "You're math professor."},
        {
            "role": "user",
            "content": "Solve this equation 8x^2 - 20x + 6 = 0",
        },
    ],
    response_format=ResponseFormat,
)

message = response.choices[0].message
print(message)
print(message.parsed)
  ```
  </TabItem>

  </Tabs>

### Json object
```python
import litellm

response = litellm.completion(
    model="gigachat/GigaChat-2-Max",
    messages=[
        {
        "role": "system",
        "content": "You are an expert at structured data extraction. You will be given unstructured text should convert it into the given structure."
        },
        {
            "role": "user",
            "content": "the section 24 has appliances, and videogames"
        },
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "title": "data",
            "name": "data_extraction",
            "schema": {
            "type": "object",
            "properties": {
                "section": {
                        "type": "string" },
                "products": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["section", "products"],
            "additionalProperties": False
            },
            "strict": True
        }
        },
    stream=False
)

print(response.choices[0].message.content)
```

## GigaChat - Embeddings
  <Tabs>
  <TabItem value="litellm" label="SDK">
    ```python
    from litellm import embedding

    response = embedding(model="gigachat/EmbeddingsGigaR",
                     input=["Hello", "itsme"])
    print(response)
    ```
  </TabItem>
  
  <TabItem value="openai" label="OpenAI client">
    ```python
    from openai import OpenAI
 
    client = OpenAI(base_url="http://localhost:4000", api_key="sk-1234")
    
    response = client.embeddings.create(model="EmbeddingsGigaR", input=["Hello", "itsme"])
    
    print(response)
    ```
  </TabItem>

</Tabs>

### Usage with LiteLLM Proxy Server

Here's how to call a GigaChat embedding with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
    model_list:
      - model_name: EmbeddingsGigaR
        litellm_params: 
          model: gigachat/EmbeddingsGigaR
          scope: os.environ/GIGACHAT_SCOPE
          credentials: os.environ/GIGACHAT_CREDENTIALS
          ssl_verify: os.environ/GIGACHAT_VERIFY_SSL_CERTS
        model_info:
          mode: embeddings
  ```