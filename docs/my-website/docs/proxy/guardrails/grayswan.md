import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gray Swan Cygnal Guardrail

Use [Gray Swan Cygnal](https://docs.grayswan.ai/cygnal/monitor-requests) to continuously monitor conversations for policy violations, indirect prompt injection (IPI), jailbreak attempts, and other safety risks.

Cygnal returns a `violation` score between `0` and `1` (higher means more likely to violate policy), plus metadata such as violated rule indices, mutation detection, and IPI flags. LiteLLM can automatically block or monitor requests based on this signal.

---

## Quick Start

### 1. Obtain Credentials

1. Create a Gray Swan account and generate a Cygnal API key.
2. Configure environment variables for the LiteLLM proxy host:

```bash
export GRAYSWAN_API_KEY="your-grayswan-key"
export GRAYSWAN_API_BASE="https://api.grayswan.ai"
```

### 2. Configure `config.yaml`

Add a guardrail entry that references the Gray Swan integration. Below is a balanced example that monitors both input and output but only blocks once the violation score reaches the configured threshold.

```yaml
model_list:
  - model_name: openai/gpt-4.1-mini
    litellm_params:
      model: openai/gpt-4.1-mini
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "cygnal-monitor"
    litellm_params:
      guardrail: grayswan
      mode: [pre_call, post_call]            # monitor both input and output
      api_key: os.environ/GRAYSWAN_API_KEY
      api_base: os.environ/GRAYSWAN_API_BASE  # optional
      optional_params:
        on_flagged_action: monitor             # or "block"
        violation_threshold: 0.5               # score >= threshold is flagged
        reasoning_mode: hybrid                 # off | hybrid | thinking
        categories:
          safety: "Detect jailbreaks and policy violations"
        policy_id: "your-cygnal-policy-id"
      default_on: true

general_settings:
  master_key: "your-litellm-master-key"

litellm_settings:
  set_verbose: true
```

### 3. Launch the Proxy

```bash
litellm --config config.yaml --port 4000
```

---

## Choosing Guardrail Modes

Gray Swan can run during `pre_call`, `during_call`, and `post_call` stages. Combine modes based on your latency and coverage requirements.

| Mode         | When it Runs      | Protects              | Typical Use Case |
|--------------|-------------------|-----------------------|------------------|
| `pre_call`   | Before LLM call   | User input only       | Block prompt injection before it reaches the model |
| `during_call`| Parallel to call  | User input only       | Low-latency monitoring without blocking |
| `post_call`  | After response    | Full conversation     | Scan output for policy violations, leaked secrets, or IPI |

<Tabs>
<TabItem value="monitor" label="Monitor Only">

```yaml
guardrails:
  - guardrail_name: "cygnal-monitor-only"
    litellm_params:
      guardrail: grayswan
      mode: "during_call"
      api_key: os.environ/GRAYSWAN_API_KEY
      optional_params:
        on_flagged_action: monitor
        violation_threshold: 0.6
      default_on: true
```

Best for visibility without blocking. Alerts are logged via LiteLLM’s standard logging callbacks.

</TabItem>
<TabItem value="block-input" label="Block Input">

```yaml
guardrails:
  - guardrail_name: "cygnal-block-input"
    litellm_params:
      guardrail: grayswan
      mode: "pre_call"
      api_key: os.environ/GRAYSWAN_API_KEY
      optional_params:
        on_flagged_action: block
        violation_threshold: 0.4
        categories:
          pii: "Detect sensitive data"
      default_on: true
```

Stops malicious or sensitive prompts before any tokens are generated.

</TabItem>
<TabItem value="full-coverage" label="Full Coverage">

```yaml
guardrails:
  - guardrail_name: "cygnal-full-coverage"
    litellm_params:
      guardrail: grayswan
      mode: [pre_call, post_call]
      api_key: os.environ/GRAYSWAN_API_KEY
      optional_params:
        on_flagged_action: block
        violation_threshold: 0.5
        reasoning_mode: thinking
        policy_id: "policy-id-from-grayswan"
      default_on: true
```

Provides the strongest enforcement by inspecting both prompts and responses.

</TabItem>
</Tabs>

---

## Configuration Reference

| Parameter                             | Type            | Description |
|---------------------------------------|-----------------|-------------|
| `api_key`                             | string          | Gray Swan Cygnal API key. Reads from `GRAYSWAN_API_KEY` if omitted. |
| `mode`                                | string or list  | Guardrail stages (`pre_call`, `during_call`, `post_call`). |
| `optional_params.on_flagged_action`   | string          | `monitor` (log only) or `block` (raise `HTTPException`). |
| `.optional_params.violation_threshold`| number (0-1)    | Scores at or above this value are considered violations. |
| `optional_params.reasoning_mode`      | string          | `off`, `hybrid`, or `thinking`. Enables Cygnal’s reasoning capabilities. |
| `optional_params.categories`          | object          | Map of custom category names to descriptions. |
| `optional_params.policy_id`           | string          | Gray Swan policy identifier. |
