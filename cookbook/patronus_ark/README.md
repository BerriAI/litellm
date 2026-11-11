![Patronus Ark — PII, DLP and prompt-injection guardrails for LiteLLM](assets/ark-litellm-banner.png)

# Patronus Ark for LiteLLM

Redact personal information and secrets before a provider call. Block prompt
injection with native Ark checks and **Wolf Defender**, running **locally** or
through the **Patronus Control API**.

PII, DLP and native PI checks always run locally through `patronus-ark-core`.
The full Ark runtime is not required.

[Choose a mode](#choose-a-mode) · [Install](#install) ·
[Quickstart](#quickstart) · [Coverage](#scan-coverage) ·
[Settings](#configuration-reference) · [Errors](#errors-and-runtime-behavior)

## Choose a mode

| Mode | PII, DLP and native PI | Wolf Defender | Sent to Patronus Control |
|---|---|---|---|
| **Native only**: `wolf.backend: off` | Local Core checks | Disabled | Nothing |
| **Local Wolf**: `wolf.backend: local` | Local Core checks | Your local model bundle, on CPU | Nothing |
| **API Wolf**: `wolf.backend: api` | Local Core checks | Control API, L3 only | Original text from fields selected for PI scanning |

All three modes support local PII/DLP redaction or blocking. Native-only is the
default. The Wolf API does **not** perform this adapter's PII or DLP detection.

**API data boundary:** API-mode Wolf checks the selected original text before
local redaction. That text may contain personal information or secrets. Choosing
API mode therefore sends those selected fields to Control, even when the
forwarded provider request is subsequently redacted.

## Install

Use Python **3.11 or later** and install the Core Python package into the
**same environment as the LiteLLM proxy**.

Install the published Core Python bindings:

```sh
python -m pip install 'patronus-ark-core>=0.1.14,<0.2'
python -c "from patronus_ark_core import Scanner"
```

The adapter uses the standalone Core package; it does not import the full
`patronus_ark` Python runtime. Local Wolf weights are separate model assets.

## Quickstart

Choose **one** of the following complete `guardrails` sections and add it to a
LiteLLM proxy configuration that already has your provider's `model_list`.
Each example enables the guardrail by default and runs it **before** forwarding.

### A. Native checks only — no Wolf model or API key

```yaml
guardrails:
  - guardrail_name: patronus-ark
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
          backend: "off"
```

For example, detected personal information in `Contact: alice@example.org` is
forwarded as `Contact: [REDACTED]`. Native prompt-injection findings block the
request. Set `pii: block` or `dlp: block` to refuse sensitive content instead.

### B. Wolf Defender locally — no Control API connection

Provide an existing Wolf Defender bundle, including its model and tokenizer,
before starting the proxy. This adapter does not download models.

```yaml
guardrails:
  - guardrail_name: patronus-ark
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
          window_tokens: 256
          overlap_tokens: 0
          batch_size: 1
```

`threshold: 0.8` is an example; calibrate it for your traffic. The adapter warms
one Core scanner and model session during initialization and reuses them.
A configured model that cannot load prevents initialization.

`precision` also accepts `int8_int4` and `auto`. An explicit precision requires
that export locally. Window, overlap, batch and precision settings apply to
**local Wolf**, not the API. The default profile is 256 total tokens including
BOS/EOS, no overlap, and batch size one. Longer documents use additional windows.

### C. Wolf Defender through the API — no local Wolf bundle

Set `PATRONUS_API_KEY` in the proxy environment. Native Core checks still run
locally, so the Core Python package is required in API mode too.

```yaml
guardrails:
  - guardrail_name: patronus-ark
    litellm_params:
      guardrail: patronus_ark
      mode: pre_call
      default_on: true
      timeout: 10
      api_key: os.environ/PATRONUS_API_KEY
      api_base: https://control.patronus.studio
      ark:
        pii: redact
        dlp: redact
        native_pi: true
        wolf:
          backend: api
```

Selected PI text is submitted to `POST /api/v1/scan` with
`categories: [injection]`, L3 enabled and L1/L2 disabled. The adapter uses the
service's final verdict and handles both completed submissions and asynchronous
jobs. Failed or incomplete analysis refuses forwarding; it does not fall back
to a weaker successful result.

### Start the proxy and send a request

Start LiteLLM with your existing configuration:

```sh
litellm --config config.yaml
```

Replace `your-configured-model` with an alias from your `model_list` and provide
your proxy key:

```sh
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"your-configured-model","messages":[{"role":"user","content":"Contact: alice@example.org"}],"guardrails":["patronus-ark"]}'
```

The provider receives masked text when PII redaction succeeds. A block or
analysis failure returns an error before provider access.

## Scan coverage

| Source | PII | DLP | Native PI + configured Wolf |
|---|:---:|:---:|:---:|
| User message text (`user_input`) | Yes | Yes | Yes |
| Tool-call JSON arguments (`tool_call`) | Yes | Yes | No |
| Tool-result text (`tool_result`) | Yes | Yes | Yes |

These are defaults. Select sources independently under `ark`:

```yaml
pii_sources: [user_input, tool_call, tool_result]
dlp_sources: [user_input, tool_call, tool_result]
pi_sources: [user_input, tool_result]
```

Include `tool_call` in `pi_sources` to opt tool arguments into PI scanning. Use
an empty source list to disable that check everywhere. `native_pi: false`
disables native PI while leaving configured Wolf checks available.

System/developer instructions, assistant prose and reasoning are outside these
selected sources. Image/audio payloads and other non-text parts are not scanned
by this adapter. Tool-argument JSON is decoded before PII/DLP rewriting;
escaped values are inspected, numeric values may become strings, and rewritten
JSON remains valid. Key rewrites that would introduce duplicate keys are refused.

Coverage is **Chat Completions input** through `pre_call`, including input to
requests that stream their responses. This adapter does not scan or redact
provider responses. Responses API and Anthropic Messages are outside its scope;
unsupported endpoints are refused when this guardrail applies.

## Configuration reference

Settings below are under `litellm_params.ark`, except where noted.

| Setting | Default | Meaning |
|---|---|---|
| `pii`, `dlp` | `redact` | `off`, `block` or `redact` |
| `native_pi` | `true` | Enable native PI on selected PI sources |
| `pii_sources`, `dlp_sources` | all three sources | Where the corresponding check runs |
| `pi_sources` | `user_input`, `tool_result` | Where native PI and configured Wolf run |
| `pii_mode`, `dlp_mode` | `balanced` | Native scoring mode: `strict`, `balanced` or `recall` |
| `rules` | Core defaults | Rule-ID-to-boolean overrides, for example `pii_email: false` |
| `max_input_bytes` | `1048576` | Positive per-text UTF-8 limit; exceeding it is an error |
| `max_findings` | `10000` | Positive actionable-evidence limit per Core scan |
| `max_queued` | `8` | Waiting inspections in addition to one active inspection; zero disables waiting |
| `queue_timeout` | `0.5` | Positive maximum queue wait in seconds |
| `wolf.backend` | `off` | `off`, `local` or `api` |
| `wolf.bundle_dir` | none | Required existing model directory for local Wolf |
| `wolf.threshold` | `0.8` | Local Wolf probability threshold in `[0,1]` |
| `wolf.precision` | `fp16` | Local export: `fp16`, `int8_int4` or `auto` |
| `wolf.window_tokens` | `256` | Local total window size including BOS/EOS |
| `wolf.overlap_tokens` | `0` | Local content overlap; must be smaller than `window_tokens - 2` |
| `wolf.batch_size` | `1` | Positive local batch size; must be supported by the export |
| `litellm_params.timeout` | `10` | Shared inspection budget in seconds, including queue wait and API work |
| `litellm_params.api_base` | `https://control.patronus.studio` | Control base URL for API Wolf |
| `litellm_params.api_key` | None | Required for API Wolf; supports `os.environ/PATRONUS_API_KEY` through LiteLLM |

Disabling a rule deliberately removes its protection. Unknown Ark settings and
invalid source names are rejected. Local Wolf runs on CPU in this adapter.

## Errors and runtime behavior

| HTTP | Meaning |
|---|---|
| `400` | A configured PII/DLP or prompt-injection policy blocked the request, or the endpoint is unsupported |
| `429` | The active/waiting inspection capacity is full |
| `503` | Analysis could not complete: queue wait, deadline, invalid scan data, unavailable API or detector/model failure |

There is one active inspection per guardrail instance. Native and local-model
work uses a reusable worker pool; the Core scanner and model session stay warm.
Cancellation propagates through `ScanControl` to active local ONNX work, and
capacity is held until that work ends. Each selected field is inspected as its
own document under the inspection's shared deadline. Checks see original text
before redaction, including in API mode.

The integration uses LiteLLM's `apply_guardrail` path, shared HTTP handler and
standard guardrail logging. Successful unchanged input is logged as `allow`,
redaction as `mask`, and a content-policy block as `deny`.
