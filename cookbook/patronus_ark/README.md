# Patronus Ark

Ark Core scans Chat Completions before the provider call, including requests with streamed responses. PII, DLP and native prompt-injection detection run locally. Wolf Defender can run locally or through the Patronus Control API

Install the separately licensed Core package into the proxy's Python environment. Python 3.11 or later is required

```sh
pip install patronus-ark-core
```

Add this guardrail to an existing LiteLLM proxy configuration

```yaml
guardrails:
  - guardrail_name: ark
    litellm_params:
      guardrail: patronus_ark
      mode: pre_call
      default_on: true
      timeout: 10
      ark:
        pii: redact
        dlp: redact
        native_pi: true
        wolf:
          backend: local
          bundle_dir: /srv/models/wolf-defender
          precision: fp16
          threshold: 0.8
```

Provide existing Wolf assets before starting the proxy. The configured scanner warms at initialization and is reused. Request scans never download models. Calibrate the local Wolf threshold for your traffic. Omit `wolf` to use native detection alone

To run Wolf through the API, replace the Wolf section and supply your key through the environment

```yaml
      api_key: os.environ/PATRONUS_API_KEY
      api_base: https://control.patronus.studio
      ark:
        pii: redact
        dlp: redact
        native_pi: true
        wolf:
          backend: api
```

The adapter sends only selected PI text to `/api/v1/scan`, with category `injection`, L3 enabled and L1/L2 disabled. It handles completed synchronous responses and asynchronous jobs. Remote Wolf uses the service's final verdict. Local PII/DLP redaction is separate and is not submitted as an API category

PII and DLP accept `off`, `block` or `redact`. `pii_sources` and `dlp_sources` default to `[user_input, tool_call, tool_result]`. `pi_sources` defaults to `[user_input, tool_result]`. Set these lists under `ark` to change coverage. System/developer instructions, assistant prose and reasoning are outside this integration's selected sources

Tool arguments are decoded and redacted as JSON, including escaped string values. The provider receives valid JSON. All selected checks use the original text before redaction. A policy violation returns HTTP 400, a full queue returns 429, and failed or incomplete analysis returns 503 without forwarding the request

Responses API, Anthropic Messages and response-stream redaction are outside this version's scope. Calls on unsupported endpoints are refused when this guardrail applies. Limit the guardrail to your Chat Completions traffic if the proxy serves other endpoints
