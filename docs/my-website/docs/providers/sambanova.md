import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# SambaNova
[https://cloud.sambanova.ai/](http://cloud.sambanova.ai?utm_source=litellm&utm_medium=external&utm_campaign=cloud_signup)

:::tip

**We support ALL Sambanova models, just set `model=sambanova/<any-model-on-sambanova>` as a prefix when sending litellm requests. For the complete supported model list, visit https://docs.sambanova.ai/cloud/docs/get-started/supported-models **

:::

## API Key
```python
# env variable
os.environ['SAMBANOVA_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['SAMBANOVA_API_KEY'] = ""
response = completion(
    model="sambanova/Llama-4-Maverick-17B-128E-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What do you know about SambaNova Systems",
        }
    ],
    max_tokens=10,
    stop=[],
    temperature=0.2,
    top_p=0.9,
    user="user",
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['SAMBANOVA_API_KEY'] = ""
response = completion(
    model="sambanova/Llama-4-Maverick-17B-128E-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What do you know about SambaNova Systems",
        }
    ],
    stream=True,
    max_tokens=10,
    response_format={ "type": "json_object" },
    stop=[],
    temperature=0.2,
    top_p=0.9,
    tool_choice="auto",
    tools=[],
    user="user",
)

for chunk in response:
    print(chunk)
```


## Usage with LiteLLM Proxy Server

Here's how to call a Sambanova model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml
  model_list:
    - model_name: my-model
      litellm_params:
        model: sambanova/<your-model-name>  # add sambanova/ prefix to route as Sambanova provider
        api_key: api-key                 # api key to send your model
  ```


2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python
  import openai
  client = openai.OpenAI(
      api_key="sk-1234",             # pass litellm proxy key, if you're using virtual keys
      base_url="http://0.0.0.0:4000" # litellm-proxy-base url
  )

  response = client.chat.completions.create(
      model="my-model",
      messages = [
          {
              "role": "user",
              "content": "what llm are you"
          }
      ],
  )

  print(response)
  ```
  </TabItem>

  <TabItem value="curl" label="curl">

  ```shell
  curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "my-model",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ],
  }'
  ```
  </TabItem>

  </Tabs>

## SambaNova - Tool Calling

```python
import litellm

# Example dummy function

def get_current_weather(location, unit="fahrenheit"):
    if unit == "fahrenheit"
        return{"location": location, "temperature": "72", "unit": "fahrenheit"}
    else:
        return{"location": location, "temperature": "22", "unit": "celsius"}

messages = [{"role": "user", "content": "What's the weather like in San Francisco"}]

tools = [
    {
        "type": "function",
        "function": {
            "name": "import litellm",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = litellm.completion(
    model="sambanova/Meta-Llama-3.3-70B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto is default, but we'll be explicit
)

print("\nFirst LLM Response:\n", response)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    # Step 2: check if the model wanted to call a function
if tool_calls:
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    available_functions = {
        "get_current_weather": get_current_weather,
    }
    messages.append(
        response_message
    )  # extend conversation with assistant's reply
    print("Response message\n", response_message)
    # Step 4: send the info for each function call and function response to the model
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
    print(f"messages: {messages}")
    second_response = litellm.completion(
        model="sambanova/Meta-Llama-3.3-70B-Instruct", messages=messages
    )  # get a new response from the model where it can see the function response
    print("second response\n", second_response)
```

## SambaNova - Vision Example

```python
import litellm

# Auxiliary function to get b64 images
def data_url_from_image(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type of the file")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    data_url = f"data:{mime_type};base64,{encoded_string}"
    return data_url

response = litellm.completion(
    model = "sambanova/Llama-4-Maverick-17B-128E-Instruct", 
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url_from_image("your_image_path"),
                        "format": "image/jpeg"
                    }
                }
            ]
        }
    ],
    stream=False
)

print(response.choices[0].message.content)
```


## SambaNova - Structured Output

```python
import litellm

response = litellm.completion(
    model="sambanova/Meta-Llama-3.3-70B-Instruct",
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
            "strict": False
        }
        },
    stream=False
)

print(response.choices[0].message.content))
```
