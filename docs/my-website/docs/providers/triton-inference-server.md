import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Triton Inference Server

LiteLLM supports Embedding Models on Triton Inference Servers


## Usage

<Tabs>
<TabItem value="sdk" label="SDK">


### Example Call

Use the `triton/` prefix to route to triton server
```python
from litellm import embedding
import os

response = await litellm.aembedding(
    model="triton/<your-triton-model>",                                                       
    api_base="https://your-triton-api-base/triton/embeddings", # /embeddings endpoint you want litellm to call on your server
    input=["good morning from litellm"],
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add models to your config.yaml

  ```yaml
  model_list:
    - model_name: my-triton-model
      litellm_params:
        model: triton/<your-triton-model>"
        api_base: https://your-triton-api-base/triton/embeddings
  ```


2. Start the proxy 

  ```bash
  $ litellm --config /path/to/config.yaml --detailed_debug
  ```

3. Send Request to LiteLLM Proxy Server

  <Tabs>

  <TabItem value="openai" label="OpenAI Python v1.0.0+">

    ```python
    import openai
    from openai import OpenAI

    # set base_url to your proxy server
    # set api_key to send to proxy server
    client = OpenAI(api_key="<proxy-api-key>", base_url="http://0.0.0.0:4000")

    response = client.embeddings.create(
        input=["hello from litellm"],
        model="my-triton-model"
    )

    print(response)

    ```

  </TabItem>

  <TabItem value="curl" label="curl">

  `--header` is optional, only required if you're using litellm proxy with Virtual Keys

    ```shell
    curl --location 'http://0.0.0.0:4000/embeddings' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer sk-1234' \
    --data ' {
    "model": "my-triton-model",
    "input": ["write a litellm poem"]
    }'

    ```
  </TabItem>

  </Tabs>


</TabItem>

</Tabs>
