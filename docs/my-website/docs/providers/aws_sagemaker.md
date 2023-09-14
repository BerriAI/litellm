# AWS Sagemaker
LiteLLM supports Llama2 on Sagemaker

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

### AWS Sagemaker Models
Here's an example of using a sagemaker model with LiteLLM 

| Model Name       | Function Call                                  | Required OS Variables              |
|------------------|--------------------------------------------|------------------------------------|
| Llama2 7B        | `completion(model='sagemaker/jumpstart-dft-meta-textgeneration-llama-2-7b, messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |
| Custom LLM Endpoint        | `completion(model='sagemaker/your-endpoint, messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |
