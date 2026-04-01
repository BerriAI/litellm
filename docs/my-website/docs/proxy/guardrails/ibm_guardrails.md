import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# IBM Guardrails

LiteLLM works with [IBM's FMS Guardrails](https://github.com/foundation-model-stack/fms-guardrails-orchestrator) for content safety. You can use it to detect jailbreaks, PII, hate speech, and more. 

## What it does

IBM's FMS Guardrails is a framework for invoking detectors on LLM inputs and outputs. To configure these detectors, you can use e.g. [TrustyAI detectors](https://github.com/trustyai-explainability/guardrails-detectors), an open-source project maintained by the Red Hat's [TrustyAI team](https://github.com/trustyai-explainability) that allows the user to configure detectors that are: 

- regex patterns
- file type validators
- custom Python functions
- Hugging Face [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForSequenceClassification), i.e. sequence classification models

Each detector outputs an API response based on the following [openapi schema](https://foundation-model-stack.github.io/fms-guardrails-orchestrator/docs/api/openapi_detector_api.yaml). 

You can run these checks:
- Before sending to the LLM (on user input)
- After getting LLM response (on output)  
- During the call (parallel to LLM)

## Quick Start

### 1. Add to your config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: ibm-jailbreak-detector
    litellm_params:
      guardrail: ibm_guardrails
      mode: pre_call
      auth_token: os.environ/IBM_GUARDRAILS_AUTH_TOKEN
      base_url: "https://your-detector-server.com"
      detector_id: "jailbreak-detector"
      is_detector_server: true
      default_on: true
      optional_params:
        score_threshold: 0.8
        block_on_detection: true
```

### 2. Set your auth token

```bash
export IBM_GUARDRAILS_AUTH_TOKEN="your-token"
```

### 3. Start the proxy

```shell
litellm --config config.yaml --detailed_debug
```

### 4. Make a request

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "guardrails": ["ibm-jailbreak-detector"]
  }'
```

## Configuration

### Required params

- `guardrail` - str - Set to `ibm_guardrails`
- `auth_token` - str - Your IBM Guardrails auth token. Can use `os.environ/IBM_GUARDRAILS_AUTH_TOKEN`
- `base_url` - str - URL of your IBM Detector or Guardrails server 
- `detector_id` - str - Which detector to use (e.g., "jailbreak-detector", "pii-detector")

### Optional params  

- `mode` - str or list[str] - When to run. Options: `pre_call`, `post_call`, `during_call`. Default: `pre_call`
- `default_on` - bool - Run automatically without specifying in request. Default: `false`
- `is_detector_server` - bool - `true` for detector server, `false` for orchestrator. Default: `true`
- `verify_ssl` - bool - Whether to verify SSL certificates. Default: `true`

### optional_params

These go under `optional_params`:

- `detector_params` - dict - Parameters to pass to your detector
- `extra_headers` - dict - Additional headers to inject into requests to IBM Guardrails, as a key-value dict.
- `score_threshold` - float - Only count detections above this score (0.0 to 1.0)
- `block_on_detection` - bool - Block the request when violations found. Default: `true`

## Server Types

IBM Guardrails has two APIs you can use:

### Detector Server (recommended)

[This Detectors API](https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Detector+API#/Text) uses `api/v1/text/contents` endpoint to run a single detector; it can accept multiple text inputs within a request. 

```yaml
guardrails:
  - guardrail_name: ibm-detector
    litellm_params:
      guardrail: ibm_guardrails
      mode: pre_call
      auth_token: os.environ/IBM_GUARDRAILS_AUTH_TOKEN
      base_url: "https://your-detector-server.com"
      detector_id: "jailbreak-detector"
      is_detector_server: true  # Use detector server
```

### Orchestrator

If you're using the IBM FMS Guardrails Orchestrator, you can use [FMS Orchestrator API](https://foundation-model-stack.github.io/fms-guardrails-orchestrator/?urls.primaryName=Orchestrator+API), specifically by leveraging the `api/v2/text/detection/content` to potentially run multiple detectors in a single request; however, this endpoint can only accept one text input per request.

```yaml
guardrails:
  - guardrail_name: ibm-orchestrator
    litellm_params:
      guardrail: ibm_guardrails
      mode: pre_call
      auth_token: os.environ/IBM_GUARDRAILS_AUTH_TOKEN
      base_url: "https://your-orchestrator-server.com"
      detector_id: "jailbreak-detector"
      is_detector_server: false  # Use orchestrator
```

## Examples

### Check for jailbreaks on input

```yaml
guardrails:
  - guardrail_name: jailbreak-check
    litellm_params:
      guardrail: ibm_guardrails
      mode: pre_call
      auth_token: os.environ/IBM_GUARDRAILS_AUTH_TOKEN
      base_url: "https://your-detector-server.com"
      detector_id: "jailbreak-detector"
      is_detector_server: true
      default_on: true
      optional_params:
        score_threshold: 0.8
```

### Check for PII in responses

```yaml
guardrails:
  - guardrail_name: pii-check
    litellm_params:
      guardrail: ibm_guardrails
      mode: post_call
      auth_token: os.environ/IBM_GUARDRAILS_AUTH_TOKEN
      base_url: "https://your-detector-server.com"
      detector_id: "pii-detector"
      is_detector_server: true
      optional_params:
        score_threshold: 0.5  # Lower threshold for PII
        block_on_detection: true
```

### Run multiple detectors

```yaml
guardrails:
  - guardrail_name: jailbreak-check
    litellm_params:
      guardrail: ibm_guardrails
      mode: pre_call
      auth_token: os.environ/IBM_GUARDRAILS_AUTH_TOKEN
      base_url: "https://your-detector-server.com"
      detector_id: "jailbreak-detector"
      is_detector_server: true
      
  - guardrail_name: pii-check
    litellm_params:
      guardrail: ibm_guardrails
      mode: post_call
      auth_token: os.environ/IBM_GUARDRAILS_AUTH_TOKEN
      base_url: "https://your-detector-server.com"
      detector_id: "pii-detector"
      is_detector_server: true
```

Then in your request:

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "guardrails": ["jailbreak-check", "pii-check"]
  }'
```

## How detection works

When IBM Guardrails finds something, it returns details about what it found:

```json
{
  "start": 0,
  "end": 31,
  "text": "You are now in Do Anything Mode",
  "detection_type": "jailbreak",
  "score": 0.858
}
```

- `score` - How confident it is (0.0 to 1.0)
- `text` - The specific text that triggered it
- `detection_type` - What kind of violation

If the score is above your `score_threshold`, the request gets blocked (if `block_on_detection` is true).

## Further Reading

- [Control Guardrails per API Key](./quick_start#-control-guardrails-per-api-key)
- [IBM FMS Guardrails on GitHub](https://github.com/foundation-model-stack/fms-guardrails-orchestr8)

