# litellm-proxy

A local, fast, and lightweight **OpenAI-compatible server** to call 100+ LLM APIs.

## usage 

```shell 
$ pip install litellm
```
```shell
$ litellm --model ollama/codellama 

#INFO: Ollama running on http://0.0.0.0:8000
```

## replace openai base
```python 
import openai 

openai.api_base = "http://0.0.0.0:8000"

print(openai.ChatCompletion.create(model="test", messages=[{"role":"user", "content":"Hey!"}]))
``` 

[**See how to call Huggingface,Bedrock,TogetherAI,Anthropic, etc.**](https://docs.litellm.ai/docs/proxy_server)

## configure proxy

To save API Keys, change model prompt, etc. you'll need to create a local instance of it:
```shell
$ litellm --create-proxy
```
This will create a local project called `litellm-proxy` in your current directory, that has: 
* **proxy_cli.py**: Runs the proxy
* **proxy_server.py**: Contains the API calling logic
    - `/chat/completions`: receives `openai.ChatCompletion.create` call.
    - `/completions`: receives `openai.Completion.create` call.
    - `/models`: receives `openai.Model.list()` call
* **secrets.toml**: Stores your api keys, model configs, etc.

Run it by doing:
```shell
$ cd litellm-proxy
```
```shell
$ python proxy_cli.py --model ollama/llama # replace with your model name
```