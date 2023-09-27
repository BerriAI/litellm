# Replicate

LiteLLM supports all models on Replicate

### API KEYS
```python
import os 
os.environ["REPLICATE_API_KEY"] = ""
```


### Example Call

```python
# !pip install litellm
from litellm import completion
import os
## set ENV variables
os.environ["REPLICATE_API_KEY"] = "replicate key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# replicate llama-2 call
response = completion(
        model="replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf", 
        messages=messages
    )
```

### Replicate Models
liteLLM supports all replicate LLMs

For replicate models ensure to add a `replicate/` prefix to the `model` arg. liteLLM detects it using this arg. 

Below are examples on how to call replicate LLMs using liteLLM 

Model Name                  | Function Call                                                  | Required OS Variables                |
-----------------------------|----------------------------------------------------------------|--------------------------------------|
 replicate/llama-2-70b-chat | `completion('replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf', messages)` | `os.environ['REPLICATE_API_KEY']`    |
 a16z-infra/llama-2-13b-chat| `completion('replicate/a16z-infra/llama-2-13b-chat:2a7f981751ec7fdf87b5b91ad4db53683a98082e9ff7bfd12c8cd5ea85980a52', messages)`| `os.environ['REPLICATE_API_KEY']`    |
 replicate/vicuna-13b  | `completion('replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b', messages)` | `os.environ['REPLICATE_API_KEY']` |
 daanelson/flan-t5-large    | `completion('replicate/daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f', messages)`    | `os.environ['REPLICATE_API_KEY']`    |
 custom-llm    | Ensure the `model` param has `replicate/` as a prefix <`completion('replicate/custom-llm-version-id', messages)`    | `os.environ['REPLICATE_API_KEY']`    |

