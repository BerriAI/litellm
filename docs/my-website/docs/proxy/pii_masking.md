import Image from '@theme/IdealImage';

# PII Masking

LiteLLM supports [Microsoft Presidio](https://github.com/microsoft/presidio/) for PII masking. 

## Step 1. Add env

```bash
export PRESIDIO_ANALYZER_API_BASE="http://localhost:5002"
export PRESIDIO_ANONYMIZER_API_BASE="http://localhost:5001"
```

## Step 2. Set it as a callback in config.yaml

```yaml
litellm_settings: 
    litellm.callbacks = ["presidio"] 
```

## Start proxy 

```
litellm --config /path/to/config.yaml
```


This will mask the input going to the llm provider

<Image img={require('../../img/presidio_screenshot.png')} />