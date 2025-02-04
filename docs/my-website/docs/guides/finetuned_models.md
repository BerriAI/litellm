import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Calling Finetuned Models

## OpenAI


| Model Name                | Function Call                                                          |
|---------------------------|-----------------------------------------------------------------|
| fine tuned `gpt-4-0613`    | `response = completion(model="ft:gpt-4-0613", messages=messages)`     |
| fine tuned `gpt-4o-2024-05-13` | `response = completion(model="ft:gpt-4o-2024-05-13", messages=messages)` |
| fine tuned `gpt-3.5-turbo-0125` | `response = completion(model="ft:gpt-3.5-turbo-0125", messages=messages)` |
| fine tuned `gpt-3.5-turbo-1106` | `response = completion(model="ft:gpt-3.5-turbo-1106", messages=messages)` |
| fine tuned `gpt-3.5-turbo-0613` | `response = completion(model="ft:gpt-3.5-turbo-0613", messages=messages)` |


## Vertex AI

Fine tuned models on vertex have a numerical model/endpoint id. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

## set ENV variables
os.environ["VERTEXAI_PROJECT"] = "hardy-device-38811"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

response = completion(
  model="vertex_ai/<your-finetuned-model>",  # e.g. vertex_ai/4965075652664360960
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  base_model="vertex_ai/gemini-1.5-pro" # the base model - used for routing
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add Vertex Credentials to your env 

```bash
!gcloud auth application-default login
```

2. Setup config.yaml 

```yaml
- model_name: finetuned-gemini
  litellm_params:
    model: vertex_ai/<ENDPOINT_ID>
    vertex_project: <PROJECT_ID>
    vertex_location: <LOCATION>
  model_info:
    base_model: vertex_ai/gemini-1.5-pro # IMPORTANT
```

3. Test it! 

```bash
curl --location 'https://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: <LITELLM_KEY>' \
--data '{"model": "finetuned-gemini" ,"messages":[{"role": "user", "content":[{"type": "text", "text": "hi"}]}]}'
```

</TabItem>
</Tabs>


