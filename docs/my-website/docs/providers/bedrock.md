# AWS Bedrock

## API KEYS
```python
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""
```

## Usage
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

## Supported AWS Bedrock Models
Here's an example of using a bedrock model with LiteLLM 

| Model Name       | Function Call                                  | Required OS Variables              |
|------------------|--------------------------------------------|------------------------------------|
| Titan Text Large       | `completion(model='bedrock/amazon.titan-tg1-large', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |
| AI21 J2-Ultra       | `completion(model='bedrock/ai21.j2-ultra', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |
| AI21 J2-Mid      | `completion(model='bedrock/ai21.j2-mid', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']`     |

## Streaming
Bedrock currently supports streaming for the following llms:
* `bedrock/amazon.titan-tg1-large`

Example Usage
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
            stream=True
)

for chunk in response:
    print(chunk)
```

### Example Streaming Output Chunk
```json
{
  "choices": [
    {
      "finish_reason": null,
      "index": 0,
      "delta": {
        "content": "ase can appeal the case to a higher federal court. If a higher federal court rules in a way that conflicts with a ruling from a lower federal court or conflicts with a ruling from a higher state court, the parties involved in the case can appeal the case to the Supreme Court. In order to appeal a case to the Sup"
      }
    }
  ],
  "created": null,
  "model": "amazon.titan-tg1-large",
  "usage": {
    "prompt_tokens": null,
    "completion_tokens": null,
    "total_tokens": null
  }
}
```

## Troubleshooting
If creating a boto3 bedrock client fails with `Unknown service: 'bedrock'`
Try re installing boto3 using the following commands
```shell
pip install https://github.com/BerriAI/litellm/raw/main/cookbook/bedrock_resources/boto3-1.28.21-py3-none-any.whl
pip install https://github.com/BerriAI/litellm/raw/main/cookbook/bedrock_resources/botocore-1.31.21-py3-none-any.whl
```

See Page 26 on [Amazon Bedrock User Guide](https://d2eo22ngex1n9g.cloudfront.net/Documentation/BedrockUserGuide.pdf)