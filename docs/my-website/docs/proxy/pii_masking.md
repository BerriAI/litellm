import Image from '@theme/IdealImage';

# PII Masking

LiteLLM supports [Microsoft Presidio](https://github.com/microsoft/presidio/) for PII masking. 


## Quick Start
### Step 1. Add env

```bash
export PRESIDIO_ANALYZER_API_BASE="http://localhost:5002"
export PRESIDIO_ANONYMIZER_API_BASE="http://localhost:5001"
```

### Step 2. Set it as a callback in config.yaml

```yaml
litellm_settings: 
    callbacks = ["presidio", ...] # e.g. ["presidio", custom_callbacks.proxy_handler_instance]
```

### Step 3. Start proxy 


```
litellm --config /path/to/config.yaml
```


This will mask the input going to the llm provider

<Image img={require('../../img/presidio_screenshot.png')} />

## Output parsing 

LLM responses can sometimes contain the masked tokens. 

For presidio 'replace' operations, LiteLLM can check the LLM response and replace the masked token with the user-submitted values. 

Just set `litellm.output_parse_pii = True`, to enable this. 


```yaml
litellm_settings:
    output_parse_pii: true
```

**Expected Flow: **

1. User Input: "hello world, my name is Jane Doe. My number is: 034453334"

2. LLM Input: "hello world, my name is [PERSON]. My number is: [PHONE_NUMBER]"

3. LLM Response: "Hey [PERSON], nice to meet you!"

4. User Response: "Hey Jane Doe, nice to meet you!"

## Turn on/off per key 

Turn off PII masking for a given key. 

Do this by setting `permissions: {"pii": false}`, when generating a key. 

```shell 
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "permissions": {"pii": false}
}'
```


