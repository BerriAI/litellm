import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure AI Studio

## Sample Usage

**Ensure the following:**
1. The API Base passed ends in the `/v1/` prefix
  example:
  ```python
  api_base = "https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com/v1/"
  ```

2. The `model` passed is listed in [supported models](#supported-models). You **DO NOT** Need to pass your deployment name to litellm. Example `model=azure/Mistral-large-nmefg`  


**Quick Start**
```python
import litellm
response = litellm.completion(
    model="azure/command-r-plus",
    api_base="<your-deployment-base>/v1/"
    api_key="eskk******"
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
)
```

## Sample Usage - LiteLLM Proxy

1. Add models to your config.yaml

  ```yaml
  model_list:
    - model_name: mistral
      litellm_params:
        model: azure/mistral-large-latest
        api_base: https://Mistral-large-dfgfj-serverless.eastus2.inference.ai.azure.com/v1/
        api_key: JGbKodRcTp****
    - model_name: command-r-plus
      litellm_params:
          model: azure/command-r-plus
          api_key: os.environ/AZURE_COHERE_API_KEY
          api_base: os.environ/AZURE_COHERE_API_BASE
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
      model="mistral",
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
      "model": "mistral",
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

## Supported Models

| Model Name               | Function Call                                                                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cohere command-r-plus | `completion(model="azure/command-r-plus", messages)` | 
| Cohere ommand-r | `completion(model="azure/command-r", messages)` | 
| mistral-large-latest | `completion(model="azure/mistral-large-latest", messages)` | 


