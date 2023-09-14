# AWS Bedrock

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
            model="bedrock/amazon.titan-tg1-large", 
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            temperature=0.2,
            max_tokens=80,
)
```

### Supported AWS Bedrock Models
Here's an example of using a bedrock model with LiteLLM 

| Model Name       | Function Call                                  | Required OS Variables              |
|------------------|--------------------------------------------|------------------------------------|
| Titan Text Large       | `completion(model='bedrock/amazon.titan-tg1-large', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |
| AI21 J2-Ultra       | `completion(model='bedrock/ai21.j2-ultra', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |
| AI21 J2-Mid      | `completion(model='bedrock/ai21.j2-mid', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |

