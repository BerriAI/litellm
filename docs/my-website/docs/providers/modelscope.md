# ModelScope
LiteLLM supports running inference across multiple services for models hosted on the ModelScope Hub.

## Supported Models

### Serverless Inference Providers
You can check available models for an inference provider by going to [modelscope.cn/models](https://modelscope.cn/models), clicking the "API-Inference" and the "Other" filter tab, and selecting your desired provider.

For example, you can find all Qwen3 series models [here](https://modelscope.cn/models?filter=inference_type&model_type=qwen3&page=1&tabKey=other).


### Dedicated Inference Endpoints
Refer to the [Inference Endpoints catalog](https://modelscope.cn/models?filter=inference_type&page=1&tabKey=task) for a list of available models.

## Usage


### Authentication
With a single ModelScope token, you can access inference through multiple providers. Your calls are routed through ModelScope and the usage is free. 
However, please ensure you bind your Alibaba Cloud account before use. For details, refer to the following two links.
- [API-Infereference](https://modelscope.cn/docs/model-service/API-Inference/intro) or [API-Inference-EN](https://modelscope.cn/docs/model-service/API-Inference/intro)
- [Binding Alibaba Cloud Account](https://modelscope.cn/docs/accounts/aliyun-binding-and-authorization) or [Binding Alibaba Cloud Account-EN](https://modelscope.cn/docs/Organization-and-Personal/AccessToken)

Simply set the `MODELSCOPE_TOKEN` environment variable with your ModelScope token, you can create one here: https://modelscope.cn/my/myaccesstoken.

```bash
export MODELSCOPE_TOKEN="123xxxxxx"
```
or alternatively, you can pass your ModelScope token as a parameter:
```python
completion(..., api_key="123xxxxxx")
```

### Getting Started

To use a ModelScope model, specify the model you want to use in the following format:
```
<provider>/<ms_org_or_user>/<ms_model>
```
Where `<ms_org_or_user>/<ms_model>` is the ModelScope model ID.

Examples:

```python
# Run Llama-4-Scout-17B-16E-Instruct inference through LLM-Research
completion(model="modelscope/LLM-Research/Llama-4-Scout-17B-16E-Instruct",...)

# Run DeepSeek-R1 inference through DeepSeek
completion(model="modelscope/deepseek-ai/DeepSeek-R1-0528",...)

# Run Qwen3-8B inference through Qwen
completion(model="modelscope/Qwen/Qwen3-8B",...)
```


### Basic Completion
Here's an example of chat completion using the Qwen3-8B model through Qwen:

```python
import os
from litellm import completion

os.environ["MODELSCOPE_TOKEN"] = "123xxxxxx"

response = completion(
    model="modelscope/Qwen/Qwen3-Coder-480B-A35B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?",
        }
    ],
)
print(response)
```

### Streaming
Now, let's see what a streaming request looks like.

```python
import os
from litellm import completion

os.environ["MODELSCOPE_TOKEN"] = "123xxxxxx"

response = completion(
    model="modelscope/Qwen/Qwen3-Coder-480B-A35B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "How many r's are in the word `strawberry`?",
            
        }
    ],
    stream=True,
)

for chunk in response:
    print(chunk)
```

### Image Input
You can also pass images when the model supports it. Here is an example using [Qwen/Qwen2.5-VL-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct) model.

```python
from litellm import completion

# Set your ModelScope Token
os.environ["MODELSCOPE_TOKEN"] = "123xxxxxx"

messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg"
                    }
                }
            ]
        }
    ]

response = completion(
    model="modelscope/Qwen/Qwen2.5-VL-72B-Instruct", 
    messages=messages,
)
print(response.choices[0])
```

## Model Deployment
SwingDeploy deployment service is a one-stop model deployment solution launched by ModelScope, aiming to provide developers with end-to-end services from model selection to cloud deployment. Through standardized deployment processes and cloud resource adaptation capabilities, users can quickly deploy the rich models of the Moda community (in multiple fields such as voice, video, and NLP) to the target cloud environment, achieving efficient implementation of model inference services. You can refer to [SwingDeploy](https://modelscope.cn/docs/model-service/deployment/intro) for more details.

## LiteLLM Proxy Server with ModelScope models
You can set up a [LiteLLM Proxy Server](https://docs.litellm.ai/#litellm-proxy-server-llm-gateway) to serve ModelScope models through any of the supported Inference Providers. Here's how to do it:

### Step 1. Setup the config file

In this case, we are configuring a proxy to serve `Qwen3-Coder-480B-A35B-Instruct` from ModelScope.

```yaml
model_list:
  - model_name: my-model
    litellm_params:
      model: modelscope/Qwen/Qwen3-Coder-480B-A35B-Instruct
      api_key: os.environ/MODELSCOPE_TOKEN # ensure you have `MODELSCOPE_TOKEN` in your .env
```

### Step 2. Start the server
```bash
litellm --config /path/to/config.yaml
```

### Step 3. Make a request to the server
<Tabs>
<TabItem value="curl" label="curl">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "my-model",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
}'
```


## Usage with LiteLLM Proxy Server

Here's how to call a ModelScope model with the LiteLLM Proxy Server

1. Modify the config.yaml 

  ```yaml showLineNumbers
  model_list:
    - model_name: my-model
      litellm_params:
        model: modelscope/<ms_org_or_user>/<your-model-name> # add modelscope/ prefix to route as ModelScope provider
        api_key: api-key                 # api key to send your model
  ```


2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

  ```python showLineNumbers
  import openai
  client = openai.OpenAI(
      api_key="123xxxx",             # pass litellm proxy key, if you're using virtual keys
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
      --header 'Authorization: Bearer 1234xxxxx' \
      --header 'Content-Type: application/json' \
      --data '{
      "model": "my-model",
      "messages": [
          {
          "role": "user",
          "content": "what llm are you"
          }
      ]
  }'
  ```
  </TabItem>

  </Tabs>

