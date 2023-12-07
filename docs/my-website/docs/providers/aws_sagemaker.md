# AWS Sagemaker
LiteLLM supports All Sagemaker Huggingface Jumpstart Models

### API KEYS
```python
!pip install boto3 

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""
```

### Usage
```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b", 
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            temperature=0.2,
            max_tokens=80
        )
```

### Passing credentials as parameters - Completion()
Pass AWS credentials as parameters to litellm.completion
```python
import os 
from litellm import completion

response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            aws_access_key_id="",
            aws_secret_access_key="",
            aws_region_name="",
)
```

### Applying Prompt Templates
To apply the correct prompt template for your sagemaker deployment, pass in it's hf model name as well. 

```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b", 
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            hf_model_name="meta-llama/Llama-2-7b",
        )
```

You can also pass in your own [custom prompt template](../completion/prompt_formatting.md#format-prompt-yourself)

### Usage - Streaming
Sagemaker currently does not support streaming - LiteLLM fakes streaming by returning chunks of the response string

```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b", 
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            temperature=0.2,
            max_tokens=80,
            stream=True,
        )
for chunk in response:
    print(chunk)
```

### Completion Models 
Here's an example of using a sagemaker model with LiteLLM 

| Model Name                    | Function Call                                                                                       |
|-------------------------------|-------------------------------------------------------------------------------------------|
| Your Custom Huggingface Model               | `completion(model='sagemaker/<your-deployment-name>', messages=messages)`        | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`      
| Meta Llama 2 7B               | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b', messages=messages)`        | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 7B (Chat/Fine-tuned)  | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b-f', messages=messages)`      | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 13B              | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-13b', messages=messages)`       | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 13B (Chat/Fine-tuned) | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-13b-f', messages=messages)`     | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 70B              | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-70b', messages=messages)`       | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |
| Meta Llama 2 70B (Chat/Fine-tuned) | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-70b-b-f', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`              |

### Embedding Models

LiteLLM supports all Sagemaker Jumpstart Huggingface Embedding models. Here's how to call it: 

```python
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = litellm.embedding(model="sagemaker/<your-deployment-name>", input=["good morning from litellm", "this is another item"])
print(f"response: {response}")
```


