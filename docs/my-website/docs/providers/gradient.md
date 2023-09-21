# GRADIENT 

LiteLLM supports from [Gradient](https://www.gradient.ai).

### API KEYS
You can set the API keys in the following ways:

```python
import os 
os.environ["GRADIENT_ACCESS_TOKEN"] = ""
os.environ["GRADIENT_WORKSPACE_ID"] = ""
```

```python
import litellm 
litllm.gradient_key = ""
litllm.gradient_workspace_id = ""
```

### Usage

```python
import litellm 
import os

# set the secrets from https://auth.gradient.ai/select-workspace
os.environ["GRADIENT_ACCESS_TOKEN"] = "DUMMY"
os.environ["GRADIENT_WORKSPACE_ID"] = "821893_dummy_workspace"

response = litellm.completion(
    # set model to the ID (base or fine-tuned adapter) in gradient.ai
    model="cc2dafce-9e6e-4a23-a918-cad6ba89e42e_base_ml_model",
    custom_llm_provider='gradient',
    messages=[
        {"content": "Gradient is a API for fine tuning and inference of large language", "role": "user"}
    ],
    temperature=0.1,
    top_p=0.95,
    top_k=40,
)  
```