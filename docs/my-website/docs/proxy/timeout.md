import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Timeouts

The timeout set in router is for the entire length of the call, and is passed down to the completion() call level as well. 

### Global Timeouts

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 

model_list = [{...}]

router = Router(model_list=model_list, 
                timeout=30) # raise timeout error if call takes > 30s 

print(response)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
router_settings:
    timeout: 30 # sets a 30s timeout for the entire call
```

**Start Proxy** 

```shell
$ litellm --config /path/to/config.yaml
```

</TabItem>
</Tabs>

### Custom Timeouts & Stream Timeouts (Per Model)

For each model, you can set `timeout` and `stream_timeout` under `litellm_params`:

- **`timeout`** → maximum time for the *complete request*. Use this to cap the total end-to-end call duration.

- **`stream_timeout`** → timeout used for `stream=True` calls. With LiteLLM's default httpx client, this is how long LiteLLM waits for the next streamed read / chunk before timing out.

If `stream_timeout` is not set, LiteLLM falls back to the normal `timeout` value.

### Practical rule of thumb

- Use **`timeout`** to control how long the whole request is allowed to run.
- Use **`stream_timeout`** to control how quickly a stalled stream should fail.
- If you are debugging socket read timeouts or mid-stream disconnects in a remote setup, start by setting **both** values to the same number.
- Once the stream is stable, lower `stream_timeout` only if you want faster failover for stalled streams.
- For quick local smoke tests, sub-second values can still be useful.

### Recommended starting values

- **Remote / self-hosted streaming path** (`Open WebUI -> LiteLLM -> provider`, `OpenHands -> LiteLLM -> provider`, etc.): start with `timeout: 180` and `stream_timeout: 180`.
- **Normal interactive streaming**: start with `timeout: 180-300` and `stream_timeout: 30-60`.
- **Batch / non-streaming requests**: set `timeout`, and you can usually skip `stream_timeout`.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 
import asyncio

model_list = [{
    "model_name": "gpt-3.5-turbo",
    "litellm_params": {
        "model": "azure/chatgpt-v-2",
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_version": os.getenv("AZURE_API_VERSION"),
        "api_base": os.getenv("AZURE_API_BASE"),
        "timeout": 300, # total request timeout
        "stream_timeout": 60 # stream=True timeout while waiting for next chunk
    }
}]

# init router
router = Router(model_list=model_list, routing_strategy="least-busy")
async def router_acompletion():
    response = await router.acompletion(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": "Hey, how's it going?"}]
    )
    print(response)
    return response

asyncio.run(router_acompletion())
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-eu
      api_base: https://my-endpoint-europe-berri-992.openai.azure.com/
      api_key: <your-key>
      timeout: 180                      # total request timeout (seconds)
      stream_timeout: 60                # stream=True timeout while waiting for next chunk (seconds)
      max_retries: 5
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: azure/gpt-turbo-small-ca
      api_base: https://my-endpoint-canada-berri992.openai.azure.com/
      api_key: 
      timeout: 180                      # total request timeout (seconds)
      stream_timeout: 60                # stream=True timeout while waiting for next chunk (seconds)
      max_retries: 5

```


**Start Proxy**

```shell
$ litellm --config /path/to/config.yaml
```


</TabItem>
</Tabs>


### Setting Dynamic Timeouts - Per Request

LiteLLM supports setting a `timeout` per request 

**Example Usage**
<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import Router 

model_list = [{...}]
router = Router(model_list=model_list)

response = router.completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "what color is red"}],
    timeout=1
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
     --header 'Content-Type: application/json' \
     --data-raw '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "what color is red"}
        ],
        "logit_bias": {12481: 100},
        "timeout": 1
     }'
```
</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai


client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "what color is red"}
    ],
    logit_bias={12481: 100},
    extra_body={"timeout": 1} # 👈 KEY CHANGE
)

print(response)
```
</TabItem>
</Tabs>

</TabItem>
</Tabs>


## Testing timeout handling 

To test if your retry/fallback logic can handle timeouts, you can set `mock_timeout=True` for testing. 

This is currently only supported on `/chat/completions` and `/completions` endpoints. Please [let us know](https://github.com/BerriAI/litellm/issues) if you need this for other endpoints. 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer sk-1234' \
    --data-raw '{
        "model": "gemini/gemini-1.5-flash",
        "messages": [
        {"role": "user", "content": "hi my email is ishaan@berri.ai"}
        ],
        "mock_timeout": true # 👈 KEY CHANGE
    }'
```
