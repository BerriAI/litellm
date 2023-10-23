# openai-proxy

A simple, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

<p align="center" style="margin: 2%">
        <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank">
                <img src="https://render.com/images/deploy-to-render-button.svg" width="200"/>
        </a>
        <a href="https://deploy.cloud.run" target="_blank">
                <img src="https://deploy.cloud.run/button.svg" width="200"/>
        </a>
</p>

## usage 

```shell 
$ git clone https://github.com/BerriAI/litellm.git
```
```shell
$ cd ./litellm/openai-proxy
```

```shell
$ uvicorn main:app --host 0.0.0.0 --port 8000
```

## replace openai base
```python 
import openai 
openai.api_base = "http://0.0.0.0:8000"

# call cohere
openai.api_key = "my-cohere-key" # this gets passed as a header 

response = openai.ChatCompletion.create(model="command-nightly", messages=[{"role":"user", "content":"Hey!"}])

# call bedrock 
response = openai.ChatCompletion.create(
    model = "bedrock/anthropic.claude-instant-v1",
    messages = [
        {
            "role": "user",
            "content": "Hey!"
        }
    ],
    aws_access_key_id="",
    aws_secret_access_key="",
    aws_region_name="us-west-2",
)

print(response)
``` 

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/proxy_server)
